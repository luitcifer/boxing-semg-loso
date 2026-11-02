# Pre-/post-load sEMG state classification in boxers — analysis code

Code accompanying: *Subject-Independent Classification of Pre- and Post-Training-Load Myoelectric States in Boxers Using a Wearable sEMG System: Nested Leave-One-Subject-Out Validation and a Field Pilot* (Technologies, MDPI).

## Data (place in the same folder)
- `segments_all.csv` — 2,440 de-identified raw sEMG segments (31 athletes × 4 muscles × 10 pre-load + 10 post-load repetitions; fs = 1000 Hz). Columns: file, segment (0 = pre-load, 1 = post-load), bmi, uzun (segment length, samples), hrate (session-averaged HR), signal (space-separated samples).
- `features_raw.csv` — 18 features per segment from the unfiltered signal, with athlete_id (1–31, anonymous), label, muscle (LBBCL, RBBCL, LPM, RPM), segment index.

## Requirements
Python ≥ 3.10, numpy, pandas, scipy ≥ 1.11, PyWavelets ≥ 1.4, scikit-learn 1.7.1.

## Steps
1. `python feats.py` — filters every segment (raw, Butterworth, Kalman, Wiener, 5 wavelet variants) and writes `feat_<filter>.csv`.
2. `python abl.py <filter> semg` and `python abl.py <filter> full` — 31-fold LOSO filter ablation (Table 6).
3. `python wil.py` — paired Wilcoxon tests with Holm correction (`filter_wilcoxon.csv`).
4. `python nested.py` — nested LOSO (primary result), one JSON per outer fold.
5. `python nsum.py` — pooled nested-LOSO metrics with athlete-level bootstrap 95% CI.

## Expected results
- Nested LOSO (pure sEMG): F1 = 0.858 [0.821–0.891], accuracy = 0.855, ROC-AUC = 0.919 [0.887–0.948]; filter selected raw 18/31, Butterworth 7/31, D3–D4 wavelet 6/31.
- Filter ablation, sEMG ROC-AUC: raw 0.922, Butterworth 0.926, D3–D4 0.918, D4 0.920, Wiener 0.890, Kalman 0.880, wavelet 4 levels 0.860–0.873, D1–D2 0.873.

All scalers are fitted on training athletes only; feature ranking in nested LOSO uses inner-training athletes only.
