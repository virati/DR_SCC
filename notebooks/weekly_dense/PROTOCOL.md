# DR-SCC weekly in-clinic ("dense") recordings: protocol

<!-- ai-start -->
Written and committed **before** any analysis code or result exists. Branch `extend/weekly_dense` (DR_SCC and
umbrella). Only label-free inspection preceded this document: file counts, durations, and the 130 Hz stimulation
line within sessions.

## Aims

- **(a)** Reproduce the analysis of Alagapan et al., *Nature* 2023 (doi:10.1038/s41586-023-06541-3), which used these recordings.
- **(b)** Check whether its complexity is needed and whether its result survives controls for time and drift.
- **(c)** Use the stimulation-off sessions to strengthen the DR-SCC analysis of the hourly/daily at-home recordings.

## Data (fixed)

| Item | Value |
|---|---|
| Patients | 905, 906, 907, 908. 901 and 903 have no weekly sessions. |
| Sessions | `.txt` recordings > 10 MB in `$DATA_DIRECTORY/neural/lfp/<pt>/`, dated within ±1 day of a clinic-visit date for B01–C24 (`clinical_vectors_all.json`) |
| Signals | columns 0 (left) and 2 (right), 422 Hz |
| Primary session set | visit-weeks with exactly **one** such file. Weeks with several files are experiment days (for example OnTarget/OffTarget) and enter only a sensitivity analysis that pools all their stimulation-off time. |
| Clinical | nHDRS = HDRS17 / mean(A04–A01) |
| Chronic comparison data | the canonical frame `Chronic_FrameFeb2026_F.pickle` and its derived feature tables (forward-chaining and cross-patient branches) |

The original paper's participant IDs can't be mapped to these patients, so every patient is reported; there's no "typical responder" subset.

## Preprocessing (label-free)

1. **Stimulation state** per 2-s window and channel: on if log₁₀[P(129–131 Hz) / P(120–127 ∪ 133–140 Hz)] > 1. Median filter over 5 windows; a window is on if either channel is on.
2. **Stimulation-off data:** contiguous off runs, dropping the first 10 s after any on→off transition (as the paper), the last 4 s before any off→on transition, and the first 2 s of the file. **Stimulation-on data** is kept the same way (guards of 4 s either side) for aim (c).
3. **Segments:** non-overlapping 10 s.
4. **Artifact rejection:** drop a segment if, in either channel, max |x − median| exceeds 10 × 1.4826 × MAD of that channel over the session's same-state data.
5. A session needs ≥ 6 clean stimulation-off segments.

Deviation from the paper: no robust-PCA removal of device drift. If its absence matters, that's itself informative for (b).

## Features per segment

**Paper set (20):**
- multitaper PSD (DPSS, time–bandwidth 4, 7 tapers, equal weights): log₁₀ mean power in δ 1–4, θ 4–8, α 8–13, low β 13–20, high β 20–30, γ 30–40 Hz, per hemisphere (12)
- multitaper magnitude-squared left–right coherence in the same 6 bands (6)
- phase–amplitude coupling per hemisphere (2): Tort modulation index, 18 phase bins, phase 1.5–3 Hz, amplitude 20–35 Hz

**DR-SCC set (10):** the paper pipeline used for the chronic recordings, applied identically to each segment (Welch: Blackman-Harris, 512/128/1024; 5th-order polynomial subtraction; median power in δ, θ, α, β*, γ¹), through dbspace `e014944`.

**FOOOF set (14):** forward-chaining amendment 1 settings.

Session-level features are the median across the session's segments.

## (a) Replication

- Labels: "sick" = C01–C04, "stable response" = C21–C24.
- Features: paper set, each scaled to [0, 1] on training data.
- Model: neural network (scikit-learn `MLPClassifier`, hidden layers 32 and 16, ReLU, α = 10⁻³, up to 2000 iterations, `random_state` 2026). The paper's exact architecture isn't available in text.
- Validation: leave one patient out (4 folds), trained on segments.
- Report AUROC per held-out patient and its mean, at **segment level** (as the paper) and at **session level** (mean predicted probability per session). The paper's value is 0.87 ± 0.09 over five participants.

## (b) Checks

- **B1, simpler models.** L2 logistic regression (C = 1) on the same 20 features. Then every single feature alone, with its direction learned on training patients.
- **B2, time and drift controls,** using the same pipeline as (a):
  - B01–B04 vs C01–C04: both clinically "sick"; they differ in chronic stimulation and time.
  - C13–C16 vs C21–C24: two late blocks.
  - C01–C04 vs C05–C08: two early blocks.
  - nuisance-only features: log power at 100–120 Hz and at 58–62 Hz per hemisphere (4), on sick vs stable.
  - For each contrast, the mean nHDRS difference between the blocks is reported beside the AUROC.
  - **Pre-registered reading:** if contrasts with little clinical difference reach AUROC ≥ 0.8, or the nuisance features do, then sick-vs-stable classification can't be attributed to clinical state.
- **B3, pseudo-replication.** Session-level AUROC, with a permutation null: session labels permuted within patient, 200 permutations, logistic regression.
- **B4, continuous tracking.** Session-level paper-set features in the forward-chaining design: each C-week predicted from the other patients' weeks plus the patient's own past weeks, with models M0–M6 from `forward_chain/fc.py`, raw and baseline-z (B-week sessions) normalization. Circular-shift null, 100 standard draws, selection-corrected over the 10 neural cells. Raw R², r and MAE for every model **and** for persistence and time-only.

## (c) Extension to the hourly/daily analysis

- **C1, clean-reference agreement.** For each patient-week with both a stimulation-off session and at-home daytime recordings: the within-patient Spearman ρ, across weeks, between the stimulation-off value and the at-home (stimulation-on) weekly mean of each DR-SCC feature (10) and each FOOOF feature (14). A feature is "validated" if ρ > 0 with BH-FDR q < 0.05 in ≥ 3 of 4 patients. Reported beside the gain-compression screen's verdict for the same feature, with the paper's three coefficients (left β*, right β*, right δ) called out.
- **C2, the stimulation effect within a session.** Per week and feature, stimulation-on minus stimulation-off within the same session: its size relative to the week-to-week SD of the stimulation-off value, and its relation to stimulation voltage.
- **C3, transfer and comparison,** on the same four patients and weeks, forward chaining:
  - DR-SCC features from stimulation-off sessions
  - at-home chronic features (F-mean)
  - both concatenated
  - the chronic-trained paper ENR applied unchanged to stimulation-off features
- **C4, coherence in the chronic recordings.** Add per-band left–right co-coherence, (Re P_LR)² / (P_LL · P_RR), from the cross-patient feature table to the chronic feature set, in forward chaining, for all six patients.

## Hypotheses

- **W1.** (a) reproduces: mean leave-one-patient-out segment-level AUROC > 0.5 with permutation p < 0.05.
- **W2.** Logistic regression's mean AUROC is within 0.05 of the neural network's. This one is descriptive.
- **W3.** Drift controls, read as stated under B2.
- **W4.** In B4, at least one neural model beats persistence in pooled R², selection-corrected p < 0.05.
- **W5.** C1: how many chronic features are validated, and whether left β*, right β* and right δ are.
- **W6.** C3: stimulation-off features give higher forward-chaining pooled R² than at-home features on the same weeks. Descriptive, raw values.

All metrics are reported raw. Any difference or gain is shown beside the values it's computed from. Per-patient values are always included. With four patients, every cross-patient statement is limited, and the results will say so.
<!-- ai-end -->

## Amendment W-A1 (2026-09-30): clinical-state labels, the relapse patient, and time versus state

<!-- ai-start -->
**Written after the first run of (a) and (b) B1–B3, and prompted by it.** Those results stay in the record as run (`outputs/a_b_contrasts.csv`, all four patients, calendar-block labels).

**What the first run showed.** The calendar labels ("sick" = C01–C04, "stable" = C21–C24) are clinically wrong for 905. Its mean HDRS17 is 3.2 in C01–C04 and 20.5 in C21–C24: well early, relapsed late. That matches the paper's held-out relapse participant (P001), which the paper excluded from classifier training. The pooled nHDRS difference between the blocks across 905–908 is 0.001, so the pre-registered contrast isn't a clinical contrast for this group. Logistic regression was inverted in 905 (segment AUROC 0.07) and high in 906–908 (0.97, 0.93, 0.84).

**Added analyses** (code committed before they run):

- **A2-1, the paper's design.** Classifier trained and validated leave-one-patient-out on the patients whose calendar labels are clinically valid: **906, 907, 908**. Segment- and session-level AUROC, neural network and logistic regression. (901 and 903 have no weekly sessions.)
- **A2-2, held-out relapse patient.** Models trained on all of 906–908 (sick-vs-stable blocks), applied to every 905 session. AUROC is reported against (i) the calendar labels and (ii) clinical-state labels, where sick is nHDRS ≥ 0.5 over all C-week sessions.
- **A2-3, time versus state.** For each patient, a session score = mean predicted P(stable). It comes from a model that never saw that patient: the leave-one-out fold model for 906–908, and the 906–908 model for 905. Over all C-week sessions, the within-patient Spearman ρ of the score with **nHDRS** and with **week index**.

**Hypotheses**
- **W7.** In 905 the score follows clinical state, not time: ρ(score, nHDRS) < 0 with p < 0.05, while ρ(score, week) has the opposite sign from the typical responders.
- **W8 (descriptive).** In 906–908, ρ(score, nHDRS) is reported next to ρ(score, week). In those patients time and state are confounded, so only 905 separates them.

The (b) drift controls and (b) B4 and (c) are unchanged. Readers should weigh every (a)/(b) result knowing this amendment followed the first look.
<!-- ai-end -->

## Amendment W-A2 (2026-09-30): removing the stimulation effect from the at-home recordings

<!-- ai-start -->
Written and committed **before** any W-A2 code or result exists. **Why:** C1 and C2 showed that at-home features (stimulation on during C weeks) don't agree with the same week's stimulation-off values, and that stimulation shifts the DR-SCC features by many week-to-week SDs. Each weekly session has both states, minutes apart, so it measures the stimulation effect for that week.

**Scope:** patients 905–908; daytime at-home recordings (weekly means, as F-mean); weekly sessions as in the primary set, additionally requiring ≥ 3 clean stimulation-on segments to define an "on" value. Two feature sets: DR-SCC (10) and FOOOF (14). Corrections apply to C weeks only (t ≥ 4). B weeks have no stimulation and are left unchanged.

### Diagnostic (label-free)

- **D1.** For each feature and patient, Spearman ρ across C weeks of (i) session-on vs at-home, (ii) session-off vs at-home, (iii) session-on vs session-off. If at-home agrees with session-on no better than with session-off, the disagreement in C1 is about context (clinic vs home), not stimulation, and corrections shouldn't be expected to help.

### Corrections (all label-free)

- **K0:** none (reference).
- **K1, weekly additive offset:** home_w − (on_w − off_w), using that week's session. Weeks without a usable session take the offset linearly interpolated over t within the patient, using the nearest value at the ends.
- **K3, on→off map:** per patient and feature, off = a + b·on, fitted by least squares across that patient's session weeks **excluding week w**. The at-home value of week w is passed through the map.
- **K4, projecting out the stimulation axis:** u = unit vector of the patient's mean (on − off) across session weeks **excluding week w**, in features scaled by the patient's at-home SD over C weeks. The at-home vector (centred on the patient's C-week at-home mean) has its component along u removed.

K1 uses week w's own stimulation-off value, so K1 is **not** evaluated by agreement with the stimulation-off session; K3 and K4 are.

### Evaluation

- **V1, agreement (K3 and K4 only).** As C1: within-patient Spearman ρ across C weeks between the corrected at-home value and the same week's stimulation-off value. "Validated" = ρ > 0 with BH-FDR q < 0.05 in ≥ 3 of 4 patients. K0 is shown beside each.
- **V2, depression tracking.** Forward chaining as B4/C3 (other patients' weeks plus the patient's own past weeks → next week). Models M0–M6. Normalization raw and baseline-z (against the patient's B-week at-home values). Cells: {K1, K3, K4} × {DR-SCC, FOOOF} × {raw, baseline-z} × 5 neural models = **60 cells**, plus K0 and the stimulation-off-session features as references on the same weeks.
- **Null:** circular shift, 100 standard draws, all 60 cells refit. Statistic = pooled R²(cell) − pooled R²(persistence). Selection-corrected p = fraction of draws whose maximum over the 60 cells is at least the cell's observed gain.

### Hypotheses

- **S1 (descriptive).** D1: mean ρ of at-home with session-on versus with session-off.
- **S2.** V1: K3 or K4 yields at least one validated feature where K0 has none.
- **S3 (primary).** V2: at least one corrected cell beats persistence with selection-corrected p < 0.05.
- **S4 (descriptive).** V2: corrected at-home features (best cell) versus uncorrected (K0) and versus the stimulation-off session features, raw R², r and MAE, same weeks.

All metrics raw, with both sides of every gain shown and per-patient values in the outputs. Four patients: any generalization is limited and will be stated as such.
<!-- ai-end -->
