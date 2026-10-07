# Pure inference on three new patients (DBS910, DBS912, DBS913): protocol

<!-- ai-start -->
Written 2026-10-07, before any model is fitted or any feature of the new patients is computed. Branch `extend/new_patients` (from `extend/weekly_dense`).
Belongs to: the "Forward chain" paper (does a model built on earlier patients apply to a new one?) and, for the in-clinic part, the "Decoding" paper.

## What this is, and what it can't be

Models are trained on the six original patients (901, 903, 905–908), frozen, and applied to recordings from 910, 912 and 913. Nothing from the new patients enters training, feature scaling or any model choice.

**There are no depression scores for 910, 912 or 913 on this machine.** So nothing here can say whether a prediction is right. This protocol produces:
1. **sealed predictions**, committed with a checksum, to be scored once scores are supplied;
2. the **accuracy to expect**, from leaving each original patient out in turn with the identical pipeline;
3. checks of whether the new recordings **look like the training recordings at all**;
4. checks of whether the different models **agree with each other** in 910.

Prior expectation, stated before running: earlier cross-patient tests on the at-home recordings were close to zero (`notebooks/cross_patient/RESULTS.md`). The at-home predictions are therefore expected to be uninformative, and point 2 will quantify that.

## Data

Seen before writing this (file inventory and one header only, no signal analysis):

| Patient | At-home recordings | In-clinic session files (> 10 MB) | Span |
|---|---|---|---|
| 910 | about 2,050 | 47 | 23 Apr – 11 Dec 2018 |
| 912 | about 110 | 0 | under one day (10–11 Dec 2018) |
| 913 | about 120 | 2 | 30 Nov – 10 Dec 2018 |

- Raw files are unpacked to `intermed/new_patients/raw/`; the archives in `neural/lfp/` are untouched.
- Same device family and settings as before in the one header inspected (422 Hz, channels 1 and 3, 910 sensing E1–E3 and E9–E11). The sensing contacts of every recording are read from its header and reported.
- **Unknown for the new patients:** implant date, week of therapy, stimulation settings, clinical course. Time is counted in 7-day bins from each patient's first recording.
- **Training set, at home:** every recording in the February 2026 frame (`Chronic_FrameFeb2026_plain.pkl`), day and night, weeks B01–C24, with that week's nHDRS (HDRS17 divided by the mean of the four pre-implant weeks) as the target.
- **Training set, in clinic:** the existing stimulation-off segment table (`intermed/weekly_dense/segments.csv.gz`).

## Features

At-home recordings, last 4,220 samples of each file, channels 0 (left) and 2 (right), computed by one function for all nine patients:

| Block | Count | Content |
|---|---|---|
| band power | 10 | the DR-SCC code path: Welch (Blackman-Harris, 512/128/1024), 5th-order polynomial subtraction, band median, in δ 1–4, θ 4–8, α 8–13, β\* 13–20, γ¹ 35–60 Hz (the edges in the released code) |
| coherence and phase | 15 | left–right coherence, and cosine and sine of the phase difference, in five bands |
| phase-amplitude coupling | 10 | θ and α phase to γ¹ amplitude within and across sides; δ (1.5–3 Hz) phase to 20–35 Hz amplitude per side |
| FOOOF | 14 | offset, exponent and five flattened band powers per side; set to missing when the fit R² < 0.5, then filled with the training median |
| catch22 | 44 | 22 time-series features per side, after a 50 Hz low-pass |

Also per recording: stimulation line size (130 Hz against its neighbours, log10; "on" if above 1 on either side) and the gain-compression ratio. These describe the recording and are not model inputs.

In-clinic files of 910 and 913: the functions of `notebooks/weekly_dense/wd_extract.py`, imported unchanged (stimulation on/off detection, guards, 10-s segments, amplitude rejection, the 20 features in the style of Alagapan et al. 2023).

## Frozen models

| Name | Inputs | Model | Output |
|---|---|---|---|
| M1, DR-SCC-like | band power (10) | features standardised on training data; elastic net (L1 ratio 0.5), penalty chosen by leaving one training patient out | nHDRS |
| M2, full set | band power, coherence and phase, coupling, FOOOF (49) | as M1 | nHDRS |
| M3, catch22 | catch22 (44) | as M1 | nHDRS |
| M4, in-clinic marker | the 20 session features, stimulation-off segments only | features scaled to 0–1; logistic regression (C = 1) trained on 906–908, first four therapy weeks ("sick") against C21–C24 ("stable") | probability of "sick" |

M1 follows the feature path of the 2022 DR-SCC pipeline but is **not** the published decoder object: it is refitted here on all recordings so it can be frozen and applied. M4 is the simplified version of the Nature analysis, with the differences already listed (three training patients, no drift removal, no explainer).

## Analyses

**P1. Accuracy to expect in an unseen patient.** M1–M3 refitted six times, each time leaving one original patient out entirely (the penalty chosen within the remaining five). For the left-out patient: R², Pearson r and mean absolute error at the recording level and at the week level (weekly mean prediction against that week's nHDRS). Beside them, the same three numbers for predicting the training mean for everyone. For M4, the existing leave-one-patient-out values among 906–908 and the 905 result are quoted from `notebooks/weekly_dense/RESULTS.md`, not recomputed.

**P2. Do the new recordings resemble the training recordings?** Per new patient and feature block:
- the share of feature values outside the pooled training 1st–99th percentile range;
- how well a classifier (standardised features, logistic regression, 5-fold) separates that patient's recordings from the pooled training recordings (AUROC).
The same two numbers for each original patient against the other five are the reference: a new patient is "no more unusual than an original patient" if it falls within their range.
The same is done for the session features of 910 and 913 against 906–908.

**P3. The predictions.** For every recording of the new patients: M1–M3 predictions, stimulation state, time of day. Reported: counts by stimulation state; the distribution of predictions per patient beside the distribution of training predictions; the share of predictions outside the range of nHDRS seen in training; for 910 the weekly means over time. For M4: per-session mean probability of "sick" over time. A single number per patient is given for 912 and 913, which have no time course to speak of.

**P4. Do the models agree in 910?** Across weeks, Spearman correlation between the weekly means of M1, M2 and M3; between each of them and the same week's M4 session score; and each against week number. p from all circular shifts of the weeks. Agreement between M1 and M2 is partly built in (M2 contains M1's inputs); agreement of either with M3 or M4 is not.

**Sealing.** `outputs/predictions_home.csv` and `outputs/predictions_sessions.csv` are committed with their SHA-256 in `RESULTS.md`.

**Scoring, when scores arrive.** For 910: R², r and mean absolute error of weekly predictions against weekly nHDRS, beside a constant (training mean) and beside 910's own mean; the M4 score against clinical state. For 912 and 913: the absolute error of the single pooled prediction, beside the training mean's error. No model may be changed between sealing and scoring.

## What will not be claimed

- Without scores, no statement that any model "works" or "fails" in these patients.
- A trend over weeks in 910 is not evidence of tracking: every earlier analysis found that time alone produces such trends.
- Day against night is not treated as a replication.

## Known limitations

- The heartbeat artefact is not detected or removed here.
- nHDRS is relative to each patient's own pre-implant level, which the model cannot know for a new patient.
- 912 covers one night and 913 ten days.
- Lead position and sensing contacts differ between patients and are not modelled.
<!-- ai-end -->

## Amendment N-A1 (2026-10-07, after feature extraction, before any model is fitted)

<!-- ai-start -->
Seen so far: recording counts, lengths, sensing contacts and stimulation-line presence. No model has been fitted and no prediction made.

1. **Recording length.** About half of the at-home recordings of 901 and 903 are roughly 4,200 samples long, just under the 4,220 the protocol required, and were dropped. That would bias the training set. Change: use the last min(length, 4,220) samples of each recording and require at least 4,096.
2. **Sensing contacts.** 910's at-home recordings are mostly E1–E3 and E8–E10 (1,836), with small sets on seven other contact pairs (about 28 each), which look like a contact survey. 913 is mostly E0–E2 and E8–E10 (108 of 123); 912 is all E1–E3 and E9–E11. Change: predictions use each new patient's main contact pair only. The others are counted and left out.
3. **Stimulation state.** The stimulation line is present in 59% of 910's recordings but only 6% of 912's and 2% of 913's, so 912 and 913 were recorded almost entirely with stimulation off. In training, stimulation-off recordings come mostly from the four weeks before stimulation started, when patients were unwell. Changes:
    - P2 (resemblance) is computed within stimulation state: new stimulation-off recordings against training stimulation-off recordings, and likewise for on.
    - Secondary frozen models M1-off, M2-off, M3-off (trained on training stimulation-off recordings only) and M1-on, M2-on, M3-on, each applied to the new recordings in that state. P1 is reported for these too. The models trained on all recordings stay primary.
    - A caution for reading 912 and 913: a model that has mostly learned "stimulation off means unwell" will call them unwell whatever the signal says. The state-matched models are the check on that.
<!-- ai-end -->
