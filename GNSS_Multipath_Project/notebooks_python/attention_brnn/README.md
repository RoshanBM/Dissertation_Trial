# Attention-Aided Partial Bidirectional RNN for Multipath / NLOS Detection

PyTorch adaptation of

> X. Liu et al., *"Attention-aided partial bidirectional RNN-based nonlinear equalizer
> in coherent optical systems"*, **Opt. Express 30**(18), 2022.

The paper equalises an optical symbol from a window of *k* preceding + *k* succeeding
symbols using a BiLSTM/BiGRU, then applies a Bahdanau-style additive attention block
(Eq. 7–13) over the BRNN hidden states to discover which symbols in the window
actually matter. It finds the useful span is much narrower than the trained window and
builds a "partial" BRNN on the trimmed span, matching baseline performance at ~56%
lower complexity (RMpS).

Here the same machinery classifies a GNSS measurement from a window of neighbouring
epochs **of the same satellite track**, replacing the paper's MSE regression head with
a single logit + BCE loss.

## Notebooks

| file | what it does |
|---|---|
| `01_windowing.ipynb` | Builds windowed satellite-track sequences for SDC-2023 and the Warwick SE-NAV simulation. Defines tracks, segments, the held-out-drive split, and the two feature schemas. Writes `data/03_processed/seq_*.npz`. |
| `02_sdc_attention_brnn.ipynb` | Trains BiLSTM/BiGRU × packed/unpacked attention on SDC-2023, reads the useful window off the attention and occlusion profiles, retrains a trimmed "partial" model across a `k` sweep, and compares against RF / HistGBM baselines re-fit on the same split. |
| `03_attention_transfer.ipynb` | Applies the SDC-trained model to Warwick sequences (comparable to `warwick_multipath/sdc_to_warwick_transfer.ipynb`), then repeats the window-discovery analysis on Warwick independently and compares the discovered window shapes. |
| `attn_brnn.py` | Shared library: windowing, the attention block and models, training loop, complexity accounting, window-discovery metrics. |

Run them in order — 02 needs 01's cache, 03 needs 02's checkpoint.

## Model

`AdditiveAttention` implements the paper's alignment model over BRNN hidden states:

```
e_t = v^T tanh(W h_t + b)      single-layer perceptron alignment
a   = softmax(e)               over the 2k+1 window positions
c   = sum_t a_t h_t            context vector
```

`AttnBRNN` wraps it in both variants from the paper:

- `attn='packed'` (§3.2.1) — attention on the concatenated hidden state `[h_fwd ; h_bwd]`.
- `attn='unpacked'` (§3.2.2) — independent alignment models per direction, giving two
  attention profiles and so revealing an asymmetric window.
- `attn='none'` — ablation, classify from the centre position's hidden state.

## Notes carried over from the data

Things established while building this that are easy to get wrong:

- **`sdc2023_epochs.csv` has no `utcTimeMillis`** — the parser drops it at export. Use
  `GpsTimeNanos`; no session spans a GPS week rollover.
- **A satellite track is not `(session, device, Svid)`.** That key leaves 334,455
  duplicate rows: the pixel7pro tracks L1 and L5 simultaneously, and `Svid` repeats
  across constellations. Use
  `(session, device, ConstellationType, Svid, CarrierFrequencyHz)`.
- **`BasebandCn0DbHz`, `SnrInDb`, `AgcDb` are 100% NaN** in this dataset.
- **`AccumulatedDeltaRangeUncertaintyMeters` is a sentinel** (float32 max) on 32.8% of
  rows, meaning "carrier phase unusable" — informative (12.4% multipath vs 4.4%) but
  encoded as a flag plus a value, never log-scaled raw.
- **Parameter count cannot express the partial-model saving.** RNN weights are shared
  across time, so trimming the window leaves parameters unchanged; the meaningful
  complexity axis is the paper's RMpS, reported here as RMpE (real multiplications per
  classified epoch).
- **`03_2023_classification.ipynb` has no stored outputs** and uses a stratified random
  row split, which leaks badly for temporally autocorrelated labels. Its baselines are
  therefore re-fit inside notebook 02 on the identical held-out-drive split rather than
  quoted.

## Environment

Requires `torch` (CUDA build used here) and `torchinfo` on top of the existing project
stack. Both were installed into the Anaconda base environment the other notebooks use.
