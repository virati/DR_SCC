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

## Amendment P2 (2026-09-30): phase 2, geometric and latent-variable methods

<!-- ai-start -->
Added and committed **before** any phase-2 code or result exists. Phase 1 settings are unchanged. DBS910 and lead geometry stay out of phase 2 (no dates or labels for DBS910; geometry belongs to the SCCwm-DBS project).

**Evaluation (unchanged from phase 1):** primary is calibration-free leave-one-patient-out on C01–C24 (144 patient-weeks), using no labels from the held-out patient anywhere. Raw pooled R², r and MAE, plus per-patient R² and r, alongside the phase-1 reference predictors.

**Inputs.** The mismatch-compression screen is **on** throughout (recording decile exclusion plus the per-fold feature screen from training patients), because the preprint-v2 plan uses it. Each family uses its label-free alignment:
- F-band (z)
- F-fooof-per (z)
- F-rel (z)
- F-asym (z)
- F-riem (riem)

All are weekly daytime features exported from phase 1 code.

**Clinical space (for pullback):** (pHDRS17, pBDI, pGAF), each divided by the patient's A04–A01 mean. MADRS is excluded: its arrays don't align with the 32 phases for 901 (34 entries) and 903 (33).

### Methods (primary family, 3 methods × 5 inputs = 15 cells)

**P2-A, shared metric pullback.**
- Learn one linear map W (k = 2 × d), shared across training patients. It minimizes stress Σ_p Σ_{i<j} (‖W(x_pi − x_pj)‖ − ‖c_pi − c_pj‖)², where c is the clinical vector and pairs are within patient.
- Optimizer: L-BFGS from a PCA initialization plus 4 random initializations (`default_rng(2026)`); keep the lowest stress.
- Decoder: ridge regression from Wx to nHDRS on training patients, alpha by GroupKFold over {0.01, 0.1, 1, 10, 100}.
- The held-out patient is projected with the shared W, after its own label-free alignment.

**P2-B, stitched linear dynamical system (shared latent, patient-specific read-in).**
- Model, latent dimension k = 2:
  - z_{t+1} = F z_t + w, w ~ N(0, Q)
  - x_t = A_p z_t + b_p + v, v ~ N(0, diag R_p)
  - y_t = c·z_t + d + e, e ~ N(0, s²); c and d are shared
- Fit by EM on the training patients (x and y), 200 iterations.
- For the held-out patient, F, Q, c, d and the initial state are fixed. A_p, b_p and R_p are fit by EM on x alone, starting from the mean of the training A_p and b_p. Then predict y_t = c·E[z_t | x] + d using the **Kalman smoother**, which uses features only. Causal Kalman-filter predictions are also reported, descriptively.

**P2-D, Gromov–Wasserstein label transport.**
- For the held-out patient p and each training patient q: Euclidean intra-trajectory distance matrices D_p and D_q on weekly features (B01–C24), each scaled by its own maximum.
- GW coupling T_pq with square loss and uniform marginals (POT `ot.gromov.gromov_wasserstein`).
- Prediction for held-out week i = Σ_j T_ij y_qj / Σ_j T_ij, averaged over the 5 training patients.
- This uses only the trajectories' shapes in feature space, not week indices or labels of p.

### Exploratory (outside the corrected family)

**P2-C, CEBRA-Behavior.**
- Recording-level daytime features for F-band, F-fooof-per, F-rel and F-asym (4 inputs), z-aligned per patient against that patient's B01–B04 recordings. Label: the recording's weekly nHDRS (training patients only).
- `cebra` with model `offset1-model`, output dimension 3, 2000 iterations, batch 512, temperature 1, time offset 1, conditional "time_delta".
- Fit on training patients. Embed the held-out patient's recordings. Decode with kNN regression (k = 25) fitted on training embeddings. Average per week.
- Cross-patient consistency of embeddings (CEBRA consistency score between training patients) is reported descriptively.
- Null: 20 circular-shift draws (the first 20 of the standard sequence), uncorrected.

### Inference

- Circular-shift null with the standard 100 draws (`default_rng(2026)`). The nHDRS and clinical-vector series are shifted together, with the same per-patient offsets. All 15 primary cells are refit per draw.
- Selection-corrected p for a cell: the fraction of draws whose maximum pooled R² over the 15 cells is at least that cell's R².
- Also reported: p against the combined family, phase 1's 88 cells plus these 15, using the same draws.

### Hypotheses

- **X5.** At least one phase-2 primary cell reaches calibration-free leave-one-patient-out pooled R² > 0 with selection-corrected p < 0.05 within phase 2.
- **X6.** The same, judged against the combined phase 1 + phase 2 family. This is the stricter test.
- **X7 (descriptive).** CEBRA cross-patient consistency, and its leave-one-patient-out R² with the 20-draw p.

Per-patient results are always reported. A cell driven by one or two patients will be called that.
<!-- ai-end -->
