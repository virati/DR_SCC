# Cohort-initialised, patient-calibrated next-week prediction: scaling with cohort size

<!-- ai-start -->
Written and committed **before** any code or result for this analysis. Branch `extend/fc_scaling` of `virati/DR_SCC` (from `extend/forward_chain`).
It is the analysis for the planned "forward chain" paper: a proof of principle for a foundation-model-style approach, in which other patients
supply a prior and the new patient's own weeks calibrate it.

## Questions

1. Does next-week prediction of nHDRS improve as more other patients are added to the cohort?
2. Does the cohort add anything over the patient's own history?
3. Do the recordings add anything over the patient's own HDRS history?
4. How many of the patient's own weeks are needed?

## Fixed choices

- **Primary condition**, taken unchanged from `extend/forward_chain`: FOOOF features from daytime at-home recordings, normalised to each patient's pre-stimulation weeks, target = weekly nHDRS (raw), elastic net on neural features plus time (model M3; `l1_ratio` 0.8, penalty by inner cross-validation).
    - **This condition was selected after looking at results in that branch (R² 0.246).** Its headline value is therefore not re-tested here. The questions above are new, but they are conditional on a feature set that was chosen with knowledge of this data.
- **Sensitivity conditions**, reported beside the primary so the result isn't shown only for a selected condition:
    - FOOOF features without baseline normalisation;
    - the 10 DR-SCC band powers with baseline normalisation (the originally pre-registered primary).
- **Test weeks:** C03–C24 for every analysis in S1, S3 and S4 (22 weeks × 6 patients = 132 predictions), so that every cohort size is scored on the same weeks. Own-history-only models need 6 earlier weeks.
- **Pooled R²** uses one total sum of squares, about the pooled mean of those 132 values.

## Models

| ID | Inputs | Uses recordings? |
|---|---|---|
| H0 | last observed nHDRS | no |
| H1 | running mean of all earlier nHDRS | no |
| H2 | exponentially weighted mean of earlier nHDRS (weight 0.5) | no |
| H3 | elastic net on the two previous nHDRS values, t and √t | no |
| M1 | elastic net on t and √t | no |
| M2 | elastic net on neural features | yes |
| M3 | elastic net on neural features, t and √t (primary) | yes |
| M3h | elastic net on neural features, t, √t and the two previous nHDRS values | yes |

Where a previous value doesn't exist (first weeks), 1.0 is used, the pre-implant level by definition of nHDRS.

## Analyses

- **S1, scaling curve.**
    - For each target patient and each cohort size k = 0 … 5, every subset of the other patients of that size (32 subsets per patient).
    - Training data: all weeks of the cohort plus the target's own earlier weeks. Prediction: the target's next week.
    - For each k: the pooled R², with each patient's squared error averaged over its subsets of that size. Models M1, M2, M3. The spread across subsets is also reported.
    - From the k = 4 subsets: the effect of removing each cohort member.
- **S2, calibration curve.**
    - Cohort of five plus the target's first N weeks, N ∈ {0, 2, 4, 6, 8, 12, 16, 20}.
    - Scored on a fixed test set: the target's weeks C17–C24.
    - Models M1 and M3, and H0 from the last calibration week.
- **S3, beyond clinical history.** With the full cohort (k = 5): H0–H3, M1, M3 and M3h, on raw and on 3-week-smoothed nHDRS.
- **S4, robustness (k = 5, M3 and M3h):**
    - recordings with a detected heartbeat dropped before weekly averaging;
    - stimulation-on recordings only during therapy weeks.
- **S5, the relapse patient.** 905's week-by-week predictions from H0, M1, M3 and M3h are saved and summarised (r, mean absolute error).

## Null and tests

**Neural-shift null:** each patient's weekly neural features are circularly shifted against that patient's nHDRS, time and history (independent offsets), and the analysis
is recomputed. This keeps every non-neural input intact, so it asks specifically whether the recordings carry the information.

- **H-F1 (scaling).** The pooled R² of M3 rises with cohort size: the least-squares slope over k = 0 … 5 exceeds the null slope distribution (50 draws).
- **H-F2 (cohort beats own history).** M3 at k = 5 minus M3 at k = 0 exceeds the null (50 draws).
- **H-F3 (recordings add to a time model).** M3 minus M1 at k = 5 exceeds the null (200 draws).
- **H-F4 (recordings add to clinical history).** M3h minus H3 at k = 5 exceeds the null (200 draws), on raw nHDRS. The smoothed target is reported beside it.

All are reported whether or not they hold, for the primary and both sensitivity conditions. A scaling curve for the time-only model M1 is reported too: if M1
improves with cohort size as much as M3 does, the cohort is teaching the average recovery course, not a neural mapping.

## Not in this run

The prospective test on DBS910, 912 and 913 with a frozen model. It is the confirmation this design needs and can't be done with the six patients used to choose the condition.
<!-- ai-end -->
