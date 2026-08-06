# GNSS Multipath Detection for Reliable Navigation

Detecting GNSS **multipath / NLOS** (signals that reach the receiver via reflections off buildings and terrain) from the signal-quality and geometry of individual measurements — and asking how robust, interpretable, and *useful* that detection is for trustworthy positioning.

> **Research question:** How reliably can GNSS multipath be detected from the signal-quality and geometry of individual measurements, and is that detection robust, interpretable, and useful enough to support trustworthy positioning?

This repository is the code and analysis for an MSc dissertation. The written plan lives in [`dissertation/`](dissertation/).

---

## Key findings

| Question | Result |
|---|---|
| Can multipath be detected from measurements? | Yes — tuned Random Forest / HistGradientBoosting reach **ROC-AUC ≈ 0.90**, far above classical single-threshold rules (**0.84 vs 0.57** on held-out drives). |
| Does it generalise? | Trained on some drives, it holds on **completely unseen drives (AUC 0.84–0.90)** with *no location feature* — it catches multipath by its physical signature, not memorised places. |
| Does it transfer to an independent scenario? | A ray-traced simulation of the Warwick campus is caught at **AUC 0.93–0.99**, but only after **raw-C/N0 + decision-threshold recalibration** (a ~13 dB C/N0 offset must be handled). |
| Do temporal / attention models help? | **No accuracy gain** — the multipath signal is essentially instantaneous. Attention adds **interpretability** (a ~15-epoch useful window) and a **~14× cheaper** partial model. |
| Is the ground-truth label trustworthy? | **No.** The device `MultipathIndicator` is **statistically independent of measurements that carry real errors** (AUC 0.57, r ≈ 0) and is **constellation-biased**. |
| Does detection improve positioning? | **Not by exclusion** — even with oracle labels, on good and bad receivers, excluding multipath satellites does not lower position error (validated WLS solver). Detection is better used as an **integrity signal**. |

Full numbers: [`dissertation/03_figures_numbers_reference.md`](dissertation/03_figures_numbers_reference.md).

---

## Repository structure

```
GNSS_Multipath_Project/
├── data/
│   ├── 01_raw/                     # raw datasets (see below)
│   │   ├── 20*-US-MTV-* , *-US-SF-*   # 2020 SDC drives (Pixel + SPAN NMEA)
│   │   ├── sdc2023/                   # Google SDC 2023 (train/test/metadata; 2020–2023 drives)
│   │   └── Warwick/                   # SE-NAV ray-tracing simulation (.m files)
│   ├── 02_interim/                 # parsed epoch tables (sdc2023_epochs.csv, sdc2022_epochs.csv, mi8_epochs.csv, …)
│   └── 03_processed/               # ML feature tables, sequence caches (seq_*.npz), attn_brnn/ artefacts
├── notebooks_python/               # all analysis (see "Notebooks")
└── dissertation/                   # outline, results skeleton, figures/numbers reference
```

### Datasets
- **Google Smartphone Decimeter Challenge (SDC) 2023** — real smartphone GNSS logs (`device_gnss.csv`) with the device `MultipathIndicator` flag as the label, plus ground-truth trajectories. Drives span 2020–2023; multipath is populated by **pixel6pro, pixel7pro, sm-g955f**.
- **SDC 2022 subset** — same format, a second device generation (mi8, pixel6pro, pixel7/7pro, sm-g988b, samsungs21ultra).
- **Warwick SE-NAV simulation** — physics-based ray-tracing of a drive around the University of Warwick campus (~52.38°N, 1.56°W). `Visibility.m` gives exact **LOS (2) / NLOS (1) / not-visible (0)** ground truth; `Constellation.m`, `Channels.m` supply elevation, azimuth, SNR, power. GPS-only, 2,842 epochs.

---

## Notebooks

Run each pipeline in order (later notebooks read earlier caches).

### Core pipelines (parse → EDA → classify)
| Folder | Contents |
|---|---|
| `notebooks_python/2023_notebooks/` | **SDC-2023** parser → EDA → classification, plus `2023_mapping` and the single-session deep-dive `04_2023_routen_deepdive` |
| `notebooks_python/2022_notebooks/` | **SDC-2022** parser → EDA → classification + mapping |
| `notebooks_python/mi8_multipath/` | Xiaomi Mi8 pipeline (device flag label, no time-sync) |
| `notebooks_python/pixel/` | earlier 2020 Pixel pipeline (SPAN time-sync) |
| `notebooks_python/00_coordinate_maps.ipynb` | route maps for the 2020 collection days |

### Deep-dive & experiments
| Notebook | What it does |
|---|---|
| `2023_notebooks/04_2023_routen_deepdive.ipynb` | one drive end-to-end: map → parse → EDA (per-satellite, elevation×constellation, **skyplot**) → Optuna RF + threshold tuning + LogReg baseline → **SHAP** → cross-route validation → coordinate spot-check |
| `warwick_multipath/warwick_senav_deepdive.ipynb` | SE-NAV parse → map → EDA → skyplot → classification → **SHAP** (in-domain) |
| `warwick_multipath/sdc_to_warwick_transfer.ipynb` | train on SDC → test on the Warwick campus simulation; calibration, threshold recalibration, **campus hotspot map for site visits** |
| `attention_brnn/` | attention-aided BiLSTM/BiGRU: `01_windowing` → `02_sdc_attention_brnn` → `03_attention_transfer` (see its own [README](notebooks_python/attention_brnn/README.md)) |
| `positioning/position_mitigation.ipynb` | validated WLS solver; does excluding/down-weighting multipath satellites improve position? (No.) + integrity model |
| `analysis/interpretability_baselines.ipynb` | classical GNSS baselines, bootstrap CIs, session-grouped CV, PR curves, **cross-domain SHAP** |
| `analysis/residual_label_integrity.ipynb` | physics-based residual label vs the device flag, integrity model, calibration curves, booster/ensemble |

---

## Environment & running

Two Python environments are used:

- **`.venv`** (project virtualenv) — `numpy`, `pandas`, `scikit-learn`. Used for lightweight parsing/data checks.
- **Anaconda base** — the full stack the notebooks execute against: `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `folium`, `optuna`, `shap`, `torch` (CUDA) + `torchinfo`.

> **Important:** `numpy` is pinned to **`<2.0`** in the Anaconda base env. `shap` will try to pull NumPy 2.x, which breaks NumPy-1.x-compiled packages (pandas); keep the pin (`pip install "numpy<2"`).

Notebooks are run with the **Anaconda kernel**. To execute headless:

```bash
python -m nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

Suggested order for a fresh run:
1. `2023_notebooks/01 → 02 → 03` (produces `data/02_interim/sdc2023_epochs.csv`, `data/03_processed/sdc2023_training_features.csv`)
2. `2023_notebooks/04_2023_routen_deepdive`
3. `warwick_multipath/warwick_senav_deepdive` → `sdc_to_warwick_transfer`
4. `attention_brnn/01 → 02 → 03`
5. `positioning/position_mitigation`
6. `analysis/interpretability_baselines`, `analysis/residual_label_integrity`

---

## Caveats (carried throughout the analysis)
1. The device `MultipathIndicator` = *detected multipath* (not strictly NLOS), is **constellation-biased**, and is **uncorrelated with real measurement errors** — a central, acknowledged limitation.
2. The Warwick label is **NLOS** (a different definition); the simulation is idealised and **GPS-only**.
3. Real-world generality rests on **cross-route SDC** evidence; real→sim is the *easy* transfer direction (sim→real fails, AUC 0.62).
4. Cross-dataset transfer needs **raw C/N0 + threshold recalibration**, not per-domain standardisation (which breaks it).
5. Attention/temporal modelling provides **interpretability and efficiency, not accuracy**.
6. Multipath exclusion does not improve positioning here → the useful framing is **integrity**, and a **residual-based label** is the better future target.
