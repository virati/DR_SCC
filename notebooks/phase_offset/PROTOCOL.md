# Left–right SCC phase offset: protocol

<!-- ai-start -->
Written and committed **before** any code or result for this analysis exists. Branch `extend/phase_offset`
(DR_SCC and umbrella). It doesn't touch the dissertation, the bioRxiv text or the earlier branches' results.

## Question

What is the phase offset between left and right SCC in each oscillation band, and how does it depend on
time of day (day vs night), stimulation (on vs off), and mismatch-compression correction (corrected vs
uncorrected)? A secondary question is whether it relates to depression state.

## Data

| Source | Patients | Recordings | Stimulation state | Time of day |
|---|---|---|---|---|
| At-home | 901, 903, 905, 906, 907, 908 | the canonical frame's 10-s recordings, weeks B01–C24 | determined per recording from the 130 Hz line (below); B weeks expected off, C weeks on | day (10:00–21:00) and night, as in the frame |
| Weekly in-clinic sessions | 905, 906, 907, 908 | the clean 10-s segments from `weekly_dense` (primary sessions) | on and off within the same session | day |

## Quantities (per 10-s recording or segment, per band)

- Welch cross-spectrum S_LR(f) and auto-spectra with the frame's settings (Blackman-Harris, 512 samples/segment, 128 overlap, nfft 1024, fs 422), channels 0 (left) and 2 (right).
- Band coherency c_b = mean_f S_LR / sqrt(mean_f S_LL · mean_f S_RR), a complex number.
- **Phase offset** φ_b = arg(c_b). Convention: S_LR = conj(L)·R, so **φ > 0 means the right channel leads the left**.
- Coherency magnitude |c_b|, and imaginary coherency Im(c_b), which is insensitive to zero-lag common signals such as shared artefact.
- A **surrogate** coherency from the same recording with the right channel circularly shifted by 5 s, which keeps both spectra and destroys phase locking.
- Stimulation state per recording: on if log₁₀[P(129–131 Hz) / P(120–127 ∪ 133–140 Hz)] > 1 in either channel.
- Gain-compression ratio per channel: log₁₀ P(66 Hz) − log₁₀ P(64 Hz).

**Bands**
- Uncorrected (traditional): δ 1–4, θ 4–8, α 8–14, β 14–30, γ 30–50 Hz.
- Corrected (the paper's mismatch-compression correction): δ, θ, α as above, **β\* 14–20**, **γ¹ 35–50** Hz.

**"Corrected" versus "uncorrected"** means:
- uncorrected = traditional bands, all recordings;
- corrected = corrected bands, and for at-home recordings, excluding those whose gain-compression ratio (either side) is in the top decile of all at-home recordings pooled. Session segments are never excluded on this basis.

## Aggregation

Weekly value per patient × condition × band: the mean coherency over that week's recordings (complex mean of c_b). Its angle is the weekly phase; its magnitude is the weekly coherency. A week is **reliable** for a condition and band if its coherency magnitude exceeds the 95th percentile, across that patient's weeks, of the weekly surrogate magnitude for the same condition and band. Tests use reliable weeks; all weeks are reported as a sensitivity analysis.

Conditions:
- at-home: {day, night} × {stimulation off, stimulation on}
- session: {stimulation off, stimulation on}

## Analyses

- **PH1, description.** Per patient, band, condition and correction: circular mean of weekly phase (degrees), resultant length across weeks, median weekly coherency magnitude, median imaginary coherency, number of weeks and fraction reliable.
- **PH2, contrasts**, within patient and paired by week:
  - (i) night − day, at-home, same stimulation state;
  - (ii) stimulation on − off within the same in-clinic session;
  - (iii) at-home stimulation on (C weeks) vs off (B weeks), same time of day, unpaired;
  - (iv) corrected − uncorrected: β\* vs β and γ¹ vs γ in the same condition, and corrected vs uncorrected recording sets for δ, θ, α.
  - Test for paired contrasts: sign-flip permutation (2,000) on mean sin(Δφ). For (iii): label permutation (2,000) on the circular difference of means. BH-FDR within each contrast across patients × bands. A contrast is **consistent** for a band if q < 0.05 with the same sign in ≥ 3 of 4 patients (sessions) or ≥ 4 of 6 (at-home).
- **PH3, relation to depression (secondary).**
  - Within patient: nHDRS regressed on [cos φ, sin φ] of the weekly phase (corrected bands) for session-off and at-home day. R², with p from 200 circular shifts of nHDRS. FDR across patients × bands.
  - Stimulation-off sessions: logistic regression for sick (C01–C04) vs stable (C21–C24), leave one patient out among 906–908, 905 held out and scored against clinical-state labels (as `weekly_dense` W-A1), with features (a) [cos φ, sin φ, |c|] for the 5 corrected bands (15) and (b) those plus the 12 band-power features. Segment- and session-level AUROC, with a session-label permutation p (200).

## Hypotheses

- **H-P1.** Stimulation changes the left–right phase offset: contrast (ii) is consistent in at least one band.
- **H-P2.** Day and night differ: contrast (i) is consistent in at least one band.
- **H-P3.** Mismatch-compression correction changes the estimate: contrast (iv) is consistent for β\* vs β or γ¹ vs γ.
- **H-P4.** Phase features classify sick vs stable in the stimulation-off sessions (mean leave-one-patient-out segment AUROC > 0.5, permutation p < 0.05), and whether they add to band power is reported.

All results are reported raw, per patient, with the coherency magnitude beside every phase. A phase from a band with negligible coherency is noise, and the tables will say which ones those are. The channels are bipolar, so an offset near ±180° can reflect electrode polarity rather than physiology; that caveat applies throughout.
<!-- ai-end -->
