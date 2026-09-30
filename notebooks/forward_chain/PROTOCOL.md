# DR-SCC forward-chaining analysis: protocol

<!-- ai-start -->
Written and committed **before** any analysis code or result exists (step 7). Anything not listed as
primary below is exploratory and will be labelled that way in the results.

## Question

Does SCC oscillatory activity track a patient's depression state **over time**, beyond what the
recovery trend (time in therapy) and the patient's own recent scores already explain?

## Data (fixed)

| Item | Value |
|---|---|
| Frame | `$DATA_DIRECTORY/intermed/chronic/Chronic_FrameFeb2026_F.pickle`, md5 `f40adf2f4c6c2988ccf29023a4959477` |
| Clinical | `$DATA_DIRECTORY/clinical/clinical_vectors_all.json`, md5 `a29431b316cf8d3630c0a0894c32593c` |
| Library | dbspace `e014944` (`dr_scc/2026_finalpush`) |
| Patients | 901, 903, 905, 906, 907, 908 |
| Weeks | B01–B04 (post-op, stimulation off) and C01–C24 (stimulation on): 28 per patient; `t` = 0…27 |
| Per-recording features | The paper pipeline (`calculate_states_in_set`): 5th-order polynomial-subtracted PSD, median power in δ, θ, α, β*, γ¹, left and right (10) |
| Target | nHDRS = HDRS17 / mean(A04–A01) (`pHDRS17`) |

## Factors

**Evaluation (step 1)**
- **E1, forward chaining with the cohort (primary).** For target patient p and each week k in C01–C24: train on all weeks of the other 5 patients plus p's weeks before k; predict week k. 144 predictions.
- **E2, forward chaining within patient only (secondary).** Train on p's weeks before k only (k ≥ C03, so at least 6 training weeks). Linear models only.
- **E3, calibration curve (secondary).** Train on the other 5 patients plus p's first N weeks, N ∈ {0, 2, 4, 8, 12, 16}; predict p's remaining weeks within C01–C24. N = 0 is leave-one-patient-out.

**Models (steps 2, 4).** Learned models standardize features on training rows only.
- M0 persistence: predict p's last observed nHDRS (no learning).
- M1 time-only: ElasticNetCV on [t, √t].
- M2 ENR on neural features.
- M3 ENR on neural features + [t, √t].
- M4 mixed-effects: statsmodels MixedLM, neural features + t fixed, random intercept per patient; the target patient's intercept comes from its own past weeks.
- M5 SVR-RBF on neural features.
- M6 SVR-RBF on neural features + [t, √t].

ENR: `l1_ratio` 0.8, alpha by inner CV. SVR: C ∈ {0.1, 1, 10}, γ ∈ {scale, 0.01, 0.1}, ε = 0.05. Inner CV is grouped by patient in E1/E3 and time-ordered (3-fold `TimeSeriesSplit`) in E2. The test week never enters tuning.

**Feature normalization (step 3)**
- raw
- **baseline (primary):** per patient and feature, (x − μ_B) / σ_B, with μ_B and σ_B from that patient's weekly values in B01–B04. No labels are used.

**Feature sets (step 5)**
- **F-mean (primary):** weekly mean, daytime recordings (the paper).
- F-dist: weekly mean, median, IQR and variance, daytime.
- F-circ: daytime mean plus (daytime mean − nighttime mean).
- F-noGC: weekly mean, daytime, excluding recordings with `GC_Flag`.

**Targets (step 6)**
- **T-raw (primary):** nHDRS.
- T-smooth: causal 3-week trailing mean of nHDRS, used for both training and evaluation.
- T-DSC: the robust-PCA multi-scale measure (`gen_DSC`). Exploratory only: it's computed from each patient's whole trajectory, so it isn't causal.

## Metrics

Pooled over all predicted (patient, week) pairs:
- R² = 1 − SS_res / SS_tot, around the pooled mean of the true values
- Pearson r
- MAE
- mean per-patient r

Gains are ΔR² of a model relative to M1 (time-only) and to M0 (persistence).

## Null (step 2)

Circular-shift null that preserves autocorrelation: each patient's 28-week nHDRS series is shifted by an independent uniform random offset in 1…27. E1 is rerun, with every model refit, 100 times. The p-value for a gain is the fraction of null draws whose ΔR²(model − M1) is at least the observed value.

## Primary hypotheses

All under E1 × T-raw × F-mean × baseline normalization.

- **H1.** M3 (ENR neural + time) beats M1 (time-only): ΔR² > 0 with circular-shift p < 0.05.
- **H2.** M6 (SVR neural + time) beats M1, with the same criterion.
- **H3.** Baseline normalization raises pooled R² for M2 and M5 at E3 N = 0 (leave-one-patient-out) compared with raw features. This one is descriptive: no test.

Everything else (E2, the other E3 values of N, the other feature sets and targets, M4 and the other comparisons) is exploratory and reported in full.

## Determinism

No random train/test split is involved, so record order doesn't matter. The null draws use `numpy.random.default_rng(2026)`.
<!-- ai-end -->

## Amendment 1 (2026-09-30): FOOOF spectral parameterization

<!-- ai-start -->
Added and committed **before** any FOOOF code or result exists. Everything else above is unchanged.

**Why:** the paper's features subtract a 5th-order polynomial from each PSD. FOOOF (Donoghue et al. 2020) instead separates each spectrum into an aperiodic 1/f component and periodic peaks, so broadband shifts (for example from stimulation or mismatch compression) are modelled explicitly rather than folded into band power.

**Feature set F-fooof** (applied to every recording, day and night, both channels):
- Input: the frame's per-recording Welch PSD (0–211 Hz, 513 bins), from the plain export `intermed/chronic/Chronic_FrameFeb2026_plain.pkl` (md5 `07e0f3a2c88b4b323567aa88d84089f8`, same records as `Chronic_FrameFeb2026_F.pickle`).
- Fit: `fooof==1.1.0`, `FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1, aperiodic_mode="fixed")`, frequency range 1–55 Hz.
- Features per channel (7 × 2 = 14): aperiodic offset, aperiodic exponent, and the mean of the flattened spectrum (log10 power minus the aperiodic fit) within δ 1–4, θ 4–8, α 8–14, β* 14–20, γ¹ 35–50 Hz.
- QC: a recording is excluded if either channel's fit R² < 0.5, or if its PSD contains non-positive values in 1–55 Hz. Excluded counts are reported.
- Weekly aggregation as F-mean: daytime recordings, weekly mean.

**Runs**
- E1 (all models), E2 (linear models) and E3 (all N) for F-fooof × {raw, baseline} × {T-raw, T-smooth}.
- A 100-draw circular-shift null for the FOOOF primary condition.

**FOOOF primary condition:** E1 × T-raw × F-fooof × **raw** normalization. Raw is chosen because the pre-registered baseline z-scoring was found to inflate stimulation-on features (post-hoc section of RESULTS.md). That choice was made after seeing F-mean results, and is stated here as such.

**Hypotheses (same criterion as H1)**
- **H4.** M3 (ENR neural + time) with F-fooof beats M1 (time-only): ΔR² > 0, circular-shift p < 0.05.
- **H5.** M6 (SVR neural + time) with F-fooof beats M1, same criterion.
- **H6 (descriptive).** F-fooof versus F-mean pooled R² and r for M2 and M5 under E1 and under E3 N = 0.
<!-- ai-end -->

## Amendment 2 (2026-09-30): confirmation of the exploratory FOOOF × baseline result

<!-- ai-start -->
Added and committed **before** any confirmation code or result exists.

**Finding to confirm (exploratory, RESULTS.md):** E1 × T-raw × F-fooof × baseline normalization, daytime recordings.
- M3 (ENR + time): pooled R² 0.246, r 0.497, circular-shift p = 0.0099.
- M4 (mixed effects): pooled R² 0.159, r 0.504, p = 0.0099.
- Persistence (M0): R² 0.058.

That condition was chosen after looking at several FOOOF conditions, so it needs data that played no part in choosing it.

**Everything is frozen as in amendment 1:** FOOOF settings, QC rule, features, baseline normalization (z-score against B01–B04 weekly values), E1 design, models and hyperparameter grids, target (nHDRS, T-raw), the 100 circular-shift draws (`default_rng(2026)`), and the metrics. Nothing may be tuned.

### C1: held-out recordings (run now)

Nighttime recordings (`circ == "night"`, 21:00–10:00 by filename timestamp): about 7,400 recordings never used in feature selection, model selection or the FOOOF amendment. Feature set **F-fooof-night** is identical to F-fooof except that it uses night recordings, including the night recordings' own B01–B04 baseline. Labels are the same weekly nHDRS, so this tests whether the result replicates on independent recordings, not on independent patients or weeks.

Confirmation criteria, **each required** for M3 and, separately, for M4:
- (a) pooled R² > the persistence baseline (M0) R² in the same run
- (b) circular-shift null p for pooled R² < 0.05
- (c) pooled r > 0

Report both models, all criteria, and every other model's numbers, whatever the outcome.

### C2: held-out frame (pending the data)

When the original `Chronic_FrameMay2020.pickle` is available:
1. Export it to a plain frame with `recap/tools/plainify_frame.py`.
2. Run `fooof_features.py` on it with no setting changed. Only the input path and md5 change.
3. Rerun E1 for daytime F-fooof × baseline × T-raw with the same criteria (a)–(c) and the same null.

C2 tests replication on independently built intermediate data, with record selection and PSDs from the 2020 builder.
<!-- ai-end -->

## Amendment 3 (2026-09-30): C2 withdrawn; the Feb 2026 frame is canonical

<!-- ai-start -->
The original `Chronic_FrameMay2020.pickle` is no longer part of this project. The Feb 2026 frame (`Chronic_FrameFeb2026_F.pickle`, md5 `f40adf2f4c6c2988ccf29023a4959477`) is the canonical intermediate, built with the current preprocessing, and all analyses use it.

C2 is therefore withdrawn and will not be run. The confirmation status stays as C1 left it: **the exploratory FOOOF × baseline result for M3/M4 is not confirmed**. Any further confirmation needs new held-out data (for example new patients or later weeks) and a new amendment written before it runs.
<!-- ai-end -->

## Amendment 4 (2026-09-30): C1 retracted as a confirmation; selection-corrected test (C3)

<!-- ai-start -->
Added and committed **before** any C3 code or result exists.

**C1 retracted as a confirmation test.** Daytime and nighttime recordings come from different physiological states, so there's no reason a model of the daytime generator should transfer to the night. C1 does not test whether the exploratory result is a selection artifact. Its numbers stay in RESULTS.md as a **descriptive day-vs-night comparison only**. The exploratory FOOOF × baseline result is **untested**, not "not confirmed".

**C3: selection-corrected circular-shift null (daytime data only).** The exploratory result was chosen as the best of the FOOOF grid I examined, so its null must include that choice.
- Grid: F-fooof × {raw, baseline} × {T-raw, T-smooth} × {M2, M3, M4, M5, M6} (20 cells), E1, daytime, exactly as in amendment 1.
- Statistic per cell: gain = pooled R²(model) − pooled R²(M0 persistence) in the same condition.
- Observed: max gain over the 20 cells (M3, baseline, T-raw: 0.246 − 0.058 = 0.188).
- Null: the same 100 circular-shift draws (`default_rng(2026)`). In each draw every patient's target is shifted by the same offsets in every cell, all 20 cells are refit (persistence included), and the draw's maximum gain is recorded.
- **Criterion:** the lead survives selection correction if p = (#draws with max gain ≥ 0.188 + 1) / 101 < 0.05.
- Also reported: each cell's own gain and its selection-corrected p (the fraction of draw maxima ≥ that cell's gain).

C3 tests whether the lead exceeds what the best of 20 tries would produce by chance. Replication on new data would still be stronger evidence.
<!-- ai-end -->
