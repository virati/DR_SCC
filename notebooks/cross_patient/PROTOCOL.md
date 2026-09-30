# DR-SCC cross-patient decoding: protocol (phase 1)

<!-- ai-start -->
Written and committed **before** any code or result for this analysis exists. Branch `extend/cross_patient`
(DR_SCC and umbrella). It doesn't touch the dissertation, the bioRxiv text or the fixed-preprint work.

## Question

Can a model trained on some patients track depression in a **new patient without any of that patient's
clinical scores**, once patient-specific measurement effects are removed?

Model: x_p(t) = g_p(z(t)) + n_p(t)
- z(t): the shared mood state.
- g_p: the patient-specific measurement (lead angle and position, gain, impedance mismatch).
- n_p(t): nuisance (stimulation, gain compression, time since implant).

## Data (fixed)

| Item | Value |
|---|---|
| Frame | canonical `Chronic_FrameFeb2026_F.pickle` (md5 `f40adf2f…`), as the plain export `Chronic_FrameFeb2026_plain.pkl` (md5 `07e0f3a2…`) |
| Raw recordings | the `.txt` files named in the frame (`Filename`), used for cross-spectra |
| Clinical | `clinical_vectors_all.json` (md5 `a29431b3…`); target nHDRS = HDRS17 / mean(A04–A01) |
| Stimulation | `voltage_changes.mat` `StimMatrix` (6 × 32 phases) |
| Recordings | daytime (10:00–21:00), weeks B01–C24 (`t` = 0…27), patients 901, 903, 905, 906, 907, 908 |
| Library | dbspace `e014944`, unmodified |

## Step 1: confound measurements (label-free)

- **GCr** per recording and channel (Ch 4, eq. 4.17): log₁₀ P(66 Hz) − log₁₀ P(64 Hz), taking each value at the PSD bin nearest to that frequency. Weekly mean per side.
- **Stimulation voltage** per patient-week, from `StimMatrix`.
- **Mismatch-compression recording screen:** exclude recordings whose GCr (either side) is in the top decile of all daytime recordings pooled. The threshold is computed without labels.
- **Mismatch-compression feature screen:** within each training patient, Spearman ρ between each weekly feature and weekly mean GCr (same side; for bilateral features, the mean of both sides). A feature is dropped if FDR-corrected p < 0.05 (Benjamini–Hochberg across features × patients) in ≥ 3 of the training patients. Recomputed inside every leave-one-patient-out fold, from training patients only.

## Step 2: feature families

| Family | Definition | Invariant to |
|---|---|---|
| F-band | the paper's 10 band powers (weekly mean) | — (reference) |
| F-fooof-per | FOOOF periodic band power, 10 (amendment 1 settings; no offset or exponent) | aperiodic tilt / slope flattening |
| F-rel | per channel, each band's power minus the mean of that channel's 5 bands (10) | broadband gain |
| F-asym | left minus right for each band, from F-rel (5) | gain shared by both sides |
| F-riem | per band, the real 2×2 left/right co-spectral matrix (Welch CSD, same settings as the frame: Blackman-Harris window, 512 samples/segment, 128 overlap, nfft 1024, fs 422; last 10 s of channels 0 and 2), averaged over the band's bins. Represented in tangent space after alignment (15) | per-patient reference (after re-centering) |

## Step 3: alignment of the held-out patient, using no labels

- **none**
- **z**: z-score against that patient's own B01–B04 weekly values (as in forward chaining)
- **riem** (F-riem only): Riemannian re-centering. Whiten each weekly matrix by the patient's reference C_ref^(−1/2), where C_ref is the Riemannian (geometric) mean of the patient's B01–B04 weekly matrices. Map to tangent space at the identity with the matrix log and take the upper triangle (3 per band). Then apply **stretching**: divide by the patient's mean Riemannian distance of B01–B04 matrices to C_ref. There is no rotation step, because rotation alignment needs labels.

## Step 4: models

Trained on 5 patients, predicting the held-out patient. Features are standardized on training rows.
- **ENR**: ElasticNetCV, `l1_ratio` 0.8, alpha by GroupKFold over the training patients.
- **SVR-RBF**: C ∈ {0.1, 1, 10}, γ ∈ {scale, 0.01, 0.1}, ε = 0.05, grouped inner CV.
- **Anchor regression** (Rothenhäusler et al. 2021), γ = 5, fixed and not tuned. Anchors: training-patient one-hot, weekly GCr (both sides) and stimulation voltage. Implementation: transform X and y by W = I + (√γ − 1)·P_A (P_A = projection onto the anchors, with an intercept), then fit the ENR above.
- **ICP** (Peters et al. 2016), linear. Preselect the 6 features with the largest absolute ENR coefficients on the training patients. For every subset S, fit OLS on the pooled training patients and test invariance of the residuals across patients: equal means (Kruskal–Wallis) and equal variances (Levene), both at α = 0.05, Bonferroni-combined. The ICP set is the intersection of the accepted subsets. The prediction is OLS on that set, or the training mean if the set is empty.
- **Reference predictors (always reported, raw):**
  - the training mean (a constant): the natural leave-one-patient-out null
  - time-only ENR on [t, √t]

## Step 5: evaluation

- **Primary: calibration-free leave-one-patient-out** over the C01–C24 weeks of the held-out patient. No label from that patient is used anywhere: not in alignment, tuning, feature screening or model selection.
- **Secondary:** calibration curve. Training adds the held-out patient's first N weeks' labels, N ∈ {4, 8, 12}, and evaluation is on the remaining C weeks (as E3 in forward chaining).
- **Grid:** mismatch-compression screen {off, on} × feature family × alignment (none and z for all families; riem for F-riem only) × model. That's 2 × (4 × 2 + 1 × 3) × 4 = 88 cells.
- **Metrics, all raw:**
  - pooled R² and pooled r over all held-out patient-weeks (144)
  - MAE
  - per-patient R² and r
  - the same metrics for the training-mean and time-only references

  Any gain or difference is always shown next to the raw values on both sides.

## Inference

Circular-shift null: 100 draws with `default_rng(2026)`. Each patient's 28-week nHDRS is shifted by an independent offset in 1…27, and all 88 cells are refit, including screening. The statistic is pooled R².
- **Selection-corrected p** for a cell is the fraction of draws whose **maximum pooled R² over all 88 cells** is at least that cell's observed pooled R².
- **Pre-specified single cell ("first pick"):** F-riem × riem alignment × mismatch-compression screen on × ENR. Its own uncorrected p (fraction of draws where *that cell* reaches its observed pooled R²) is reported too.

## Hypotheses

- **X1 (primary, selection-corrected).** At least one cell reaches calibration-free leave-one-patient-out pooled R² > 0 with selection-corrected p < 0.05.
- **X2 (pre-specified single cell).** F-riem × riem × screen on × ENR: pooled R² > 0 with uncorrected p < 0.05.
- **X3 (descriptive).** The mismatch-compression screen on versus off, for each family, in raw pooled R² and r.
- **X4 (descriptive).** The calibration curve for the best selection-corrected cell.

Per-patient results are always reported. A positive X1 or X2 driven by one or two patients will be called that.

## Phase 2 (not run now; its own amendment first)

- Lead-geometry modelling (CT, tractography, per-contact tissue tables) is **out of scope for this repository**: it belongs to the separate SCCwm-DBS umbrella project. Its outputs (for example per-patient lead-field gains) may later be imported here as fixed covariates, through a new amendment.
- Shared metric learning and pullback from the clinical space (nHDRS, MADRS, BDI, GAF).
- Stitching models (shared latent, patient-specific read-in).
- Behaviour-contrastive embeddings (CEBRA) with a cross-subject consistency metric.
- Gromov–Wasserstein trajectory alignment.
- The DBS910 prospective blind test, which needs implant and stimulation-on dates.
<!-- ai-end -->
