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
