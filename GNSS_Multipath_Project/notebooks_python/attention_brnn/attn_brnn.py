"""
Attention-aided (partial) bidirectional RNN for GNSS multipath / NLOS detection.

Adapted from:
    X. Liu et al., "Attention-aided partial bidirectional RNN-based nonlinear
    equalizer in coherent optical systems", Opt. Express 30(18), 2022.

The paper equalises an optical symbol from a window of k preceding + k succeeding
symbols using a BiLSTM/BiGRU, then applies a Bahdanau-style additive attention
block (Eq. 7-13) over the BRNN hidden states to discover which positions in the
window actually carry information. Here the same machinery classifies a GNSS
measurement as multipath/NLOS from a window of neighbouring epochs of the *same
satellite track*, so the regression head (MSE) is replaced by a single logit +
BCE-with-logits.

Everything in this module is shared by the three notebooks in this folder:
    01_windowing.ipynb          builds and caches the windowed tensors
    02_sdc_attention_brnn.ipynb trains on SDC-2023, discovers the window
    03_attention_transfer.ipynb transfers to the Warwick SE-NAV simulation
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

def segment_ids(track_code: np.ndarray, time_s: np.ndarray, max_gap_s: float = 1.5) -> np.ndarray:
    """Split time-ordered rows into contiguous observation segments.

    A new segment starts whenever the satellite track changes *or* the sampling
    gap to the previous epoch exceeds ``max_gap_s``. Windows are only ever built
    inside a single segment, so a window can never silently bridge a data gap
    (e.g. a satellite that was lost and re-acquired 40 s later).

    ``track_code`` and ``time_s`` must already be sorted by (track, time).
    """
    track_code = np.asarray(track_code)
    time_s = np.asarray(time_s, dtype=np.float64)
    new_track = np.empty(len(track_code), dtype=bool)
    new_track[0] = True
    new_track[1:] = track_code[1:] != track_code[:-1]
    gap = np.empty(len(time_s), dtype=bool)
    gap[0] = False
    gap[1:] = np.diff(time_s) > max_gap_s
    return np.cumsum(new_track | gap).astype(np.int32) - 1


def valid_centers(seg_id: np.ndarray, k: int) -> np.ndarray:
    """Row indices that can sit at the centre of a full 2k+1 window.

    Windows falling off either edge of a segment are dropped (no padding), as in
    the paper -- a padded window would feed the attention block invented symbols
    and bias the discovered window shape toward the centre.
    """
    seg_id = np.asarray(seg_id)
    starts = np.flatnonzero(np.r_[True, seg_id[1:] != seg_id[:-1]])
    lengths = np.diff(np.r_[starts, len(seg_id)])
    out = [np.arange(s + k, s + L - k) for s, L in zip(starts, lengths) if L >= 2 * k + 1]
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(out).astype(np.int64)


class WindowBatcher:
    """Materialises windows on the fly by gathering rows around each centre.

    Storing every window densely would duplicate each row 2k+1 times (~1.5 GB at
    k=20 for SDC-2023). Instead the per-row feature matrix lives on the device
    once and each batch is gathered with an index arithmetic trick, which also
    means one cached feature matrix serves *every* k in the trimmed-window sweep.
    """

    def __init__(self, F: torch.Tensor, y: torch.Tensor, centers: np.ndarray, k: int,
                 batch_size: int = 512, shuffle: bool = False, generator=None):
        self.F, self.y, self.k = F, y, k
        self.centers = torch.as_tensor(centers, dtype=torch.long, device=F.device)
        self.batch_size, self.shuffle, self.generator = batch_size, shuffle, generator
        self.offsets = torch.arange(-k, k + 1, device=F.device)

    def __len__(self) -> int:
        return math.ceil(len(self.centers) / self.batch_size)

    def __iter__(self):
        n = len(self.centers)
        order = (torch.randperm(n, device=self.F.device, generator=self.generator)
                 if self.shuffle else torch.arange(n, device=self.F.device))
        for i in range(0, n, self.batch_size):
            c = self.centers[order[i:i + self.batch_size]]
            idx = c.unsqueeze(1) + self.offsets.unsqueeze(0)   # (B, 2k+1)
            yield self.F[idx], self.y[c]


# --------------------------------------------------------------------------
# Attention block  (Liu et al. Eq. 7-13)
# --------------------------------------------------------------------------

class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over BRNN hidden states.

        e_t = v^T tanh(W h_t + b)      single-layer-perceptron alignment model
        a   = softmax(e)               over the 2k+1 window positions
        c   = sum_t a_t h_t            context vector

    The alignment model scores each hidden state on its own (no separate query
    vector), which is the form used in the paper: the softmax over positions is
    exactly the quantity plotted in its Fig. 6-7 to read off the useful window.
    """

    def __init__(self, dim: int, attn_dim: int | None = None):
        super().__init__()
        attn_dim = attn_dim or dim
        self.W = nn.Linear(dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H: torch.Tensor):
        e = self.v(torch.tanh(self.W(H))).squeeze(-1)        # (B, T)
        a = torch.softmax(e, dim=1)                          # (B, T)
        c = torch.bmm(a.unsqueeze(1), H).squeeze(1)          # (B, dim)
        return c, a


class AttnBRNN(nn.Module):
    """Bidirectional RNN + attention, ending in one logit.

    ``attn``:
      ``'packed'``   -- §3.2.1 of the paper. Attention is applied to the
                        *concatenated* hidden state h_t = [h_fwd_t ; h_bwd_t],
                        giving one alignment score per window position.
      ``'unpacked'`` -- §3.2.2. The forward and backward hidden states get their
                        own independent alignment models, so the two directions
                        produce two separate attention profiles. This is the
                        variant that can show an asymmetric window.
      ``'none'``     -- ablation: no attention, classify from the centre
                        position's hidden state.
    """

    def __init__(self, n_feat: int, hidden: int = 64, layers: int = 1,
                 rnn: str = 'lstm', attn: str = 'packed', dropout: float = 0.1):
        super().__init__()
        assert rnn in ('lstm', 'gru') and attn in ('packed', 'unpacked', 'none')
        self.rnn_type, self.attn_mode = rnn, attn
        self.hidden, self.layers, self.n_feat = hidden, layers, n_feat

        RNN = nn.LSTM if rnn == 'lstm' else nn.GRU
        self.rnn = RNN(n_feat, hidden, num_layers=layers, batch_first=True,
                       bidirectional=True, dropout=dropout if layers > 1 else 0.0)

        if attn == 'packed':
            self.att = AdditiveAttention(2 * hidden)
        elif attn == 'unpacked':
            self.att_f = AdditiveAttention(hidden)
            self.att_b = AdditiveAttention(hidden)

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        H, _ = self.rnn(x)                                   # (B, T, 2u)
        if self.attn_mode == 'packed':
            c, a = self.att(H)
            attn = a.unsqueeze(1)                            # (B, 1, T)
        elif self.attn_mode == 'unpacked':
            u = self.hidden
            c_f, a_f = self.att_f(H[..., :u])
            c_b, a_b = self.att_b(H[..., u:])
            c = torch.cat([c_f, c_b], dim=-1)
            attn = torch.stack([a_f, a_b], dim=1)            # (B, 2, T)
        else:
            c = H[:, H.shape[1] // 2, :]
            attn = None
        logit = self.head(c).squeeze(-1)
        return (logit, attn) if return_attn else logit


# --------------------------------------------------------------------------
# Complexity: the paper's RMpS, restated per classified epoch
# --------------------------------------------------------------------------

def rmpe(model: AttnBRNN, k: int) -> int:
    """Real multiplications per classified epoch (the paper's RMpS analogue).

    Parameter count is a poor complexity measure here: an RNN shares its weights
    across time, so trimming the window leaves the parameter count *unchanged*.
    What the window actually buys is runtime -- the recurrence is evaluated 2k+1
    times per decision. The paper measures this as RMpS (real multiplications per
    symbol), so we count multiplications per classified epoch the same way.
    """
    T, u, F, L = 2 * k + 1, model.hidden, model.n_feat, model.layers
    gates = 4 if model.rnn_type == 'lstm' else 3

    rnn = 0
    for layer in range(L):
        in_dim = F if layer == 0 else 2 * u
        rnn += 2 * T * gates * (in_dim * u + u * u)          # 2 directions
        rnn += 2 * T * 3 * u                                 # 3 elementwise products
        #   LSTM: i*g, f*c, o*tanh(c)   GRU: r*(Wh), z*h, (1-z)*h~

    if model.attn_mode == 'packed':
        d = 2 * u
        att = T * (d * d + d) + T * d                        # W h_t, v^T(.), weighted sum
    elif model.attn_mode == 'unpacked':
        att = 2 * (T * (u * u + u) + T * u)
    else:
        att = 0

    head = 2 * u
    return int(rnn + att + head)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Training / inference
# --------------------------------------------------------------------------

@torch.no_grad()
def predict_proba(model: AttnBRNN, batcher: WindowBatcher) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(xb)).float().cpu() for xb, _ in batcher]
    return torch.cat(out).numpy()


@torch.no_grad()
def collect_labels(batcher: WindowBatcher) -> np.ndarray:
    return torch.cat([yb.float().cpu() for _, yb in batcher]).numpy()


def train_model(model: AttnBRNN, train_batcher: WindowBatcher, val_batcher: WindowBatcher,
                pos_weight: float, epochs: int = 12, lr: float = 1e-3,
                patience: int = 3, verbose: bool = True, log_every: int = 1):
    """Train with BCE-with-logits, early-stopping on validation ROC-AUC.

    Class imbalance is handled with ``pos_weight`` rather than the SMOTE used by
    the tabular notebooks: interpolating between two *sequences* would fabricate
    satellite tracks that no receiver ever observed, which is exactly the kind of
    artefact the attention analysis is supposed to rule out.
    """
    from sklearn.metrics import roc_auc_score

    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    y_val = collect_labels(val_batcher)
    best_auc, best_state, bad, history = -np.inf, None, 0, []

    for ep in range(1, epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for xb, yb in train_batcher:
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(yb); seen += len(yb)

        auc = roc_auc_score(y_val, predict_proba(model, val_batcher))
        history.append({'epoch': ep, 'train_loss': tot / seen, 'val_auc': auc})
        if verbose and (ep % log_every == 0 or ep == 1):
            print(f'    epoch {ep:>2}  loss {tot/seen:.4f}  val ROC-AUC {auc:.4f}'
                  + ('  *' if auc > best_auc else ''))

        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f'    early stop at epoch {ep} (best val ROC-AUC {best_auc:.4f})')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_auc


# --------------------------------------------------------------------------
# Window discovery
# --------------------------------------------------------------------------

@torch.no_grad()
def attention_profile(model: AttnBRNN, batcher: WindowBatcher, max_batches: int | None = None):
    """Mean attention weight per window position, averaged over samples.

    Returns ``(profile, per_class)`` where ``profile`` has shape (n_heads, T) --
    n_heads is 1 for the packed variant and 2 (forward, backward) for the
    unpacked one -- and ``per_class`` is a dict {0: (n_heads, T), 1: ...}.
    """
    model.eval()
    acc, cnt = None, 0
    acc_c = {0: None, 1: None}
    cnt_c = {0: 0, 1: 0}
    for i, (xb, yb) in enumerate(batcher):
        if max_batches is not None and i >= max_batches:
            break
        _, a = model(xb, return_attn=True)
        if a is None:
            return None, None
        a = a.float()
        acc = a.sum(0) if acc is None else acc + a.sum(0)
        cnt += len(yb)
        for cl in (0, 1):
            m = (yb == cl)
            if m.any():
                s = a[m].sum(0)
                acc_c[cl] = s if acc_c[cl] is None else acc_c[cl] + s
                cnt_c[cl] += int(m.sum())
    prof = (acc / cnt).cpu().numpy()
    per_class = {cl: (acc_c[cl] / cnt_c[cl]).cpu().numpy() if cnt_c[cl] else None for cl in (0, 1)}
    return prof, per_class


def attention_span(profile_1d: np.ndarray, mass: float = 0.90) -> dict:
    """How wide a window the attention profile actually uses.

    ``k_symmetric``  smallest r such that the centre plus +/-r positions holds
                     ``mass`` of the total attention -- the number that decides
                     the trimmed "partial" model.
    ``k_past`` /
    ``k_future``     smallest one-sided extent holding ``mass`` of the attention
                     *on that side alone*, so an asymmetric window (more past than
                     future, or the reverse) shows up instead of being averaged
                     into a single symmetric number.
    ``centre_weight`` vs ``uniform_weight`` says whether attention concentrated at
    all: a profile that learned nothing sits flat at 1/T.
    """
    p = np.asarray(profile_1d, dtype=np.float64)
    p = p / p.sum()
    T = len(p)
    c = T // 2

    cum, r = p[c], 0
    while cum < mass and r < c:
        r += 1
        cum += p[c - r] + p[c + r]

    def one_sided(side):
        tot = side.sum()
        if tot <= 0:
            return 0
        return int(np.searchsorted(side.cumsum() / tot, mass)) + 1

    return {'k_symmetric': r,
            'k_past': one_sided(p[:c][::-1]),      # centre outward into the past
            'k_future': one_sided(p[c + 1:]),      # centre outward into the future
            'mass_target': mass,
            'centre_weight': float(p[c]),
            'uniform_weight': 1.0 / T}


@torch.no_grad()
def occlusion_profile(model: AttnBRNN, batcher: WindowBatcher, k: int,
                      max_batches: int | None = None) -> np.ndarray:
    """Per-position input-occlusion importance -- an attention-free cross-check.

    Attention weights are read off *hidden* states, and a BiRNN hidden state at
    position t has already mixed in every other position, so a high attention
    weight at t is not proof that the *input* at t mattered. Here each position
    is instead overwritten with the (standardised) training mean and the mean
    absolute change in the output logit is recorded. This measures the input's
    influence directly and is what makes the SDC-vs-Warwick window-shape
    comparison a claim about physics rather than about one attention block.
    """
    model.eval()
    T = 2 * k + 1
    tot = np.zeros(T)
    n = 0
    for i, (xb, _) in enumerate(batcher):
        if max_batches is not None and i >= max_batches:
            break
        base = model(xb).float()
        for t in range(T):
            xo = xb.clone()
            xo[:, t, :] = 0.0                      # 0 == training mean after scaling
            tot[t] += (model(xo).float() - base).abs().sum().item()
        n += len(xb)
    return tot / max(n, 1)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def tune_threshold(y_true: np.ndarray, prob: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Pick the decision threshold that maximises F1 on held-out *validation*."""
    from sklearn.metrics import f1_score
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 precision_score, recall_score, accuracy_score)
    pred = (prob >= threshold).astype(int)
    return {
        'roc_auc': roc_auc_score(y_true, prob),
        'pr_auc': average_precision_score(y_true, prob),
        'f1': f1_score(y_true, pred, zero_division=0),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'accuracy': accuracy_score(y_true, pred),
        'threshold': threshold,
    }
