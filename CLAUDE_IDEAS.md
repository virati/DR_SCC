# Ideas for Improving DR-SCC Decoder Performance

Predicting depression scale (HDRS) from bilateral subcallosal cingulate LFPs.

## Feature Engineering (low effort, high potential)

- **Inter-hemispheric coherence/phase coupling** between L and R SCC — currently treated as independent features but their relationship likely carries signal
- **Cross-frequency coupling** (e.g., theta-gamma PAC) — well-established in mood circuit literature
- **Skewness/kurtosis** of within-week distributions — we now have mean and variance, but non-Gaussian shape of the within-week distribution could be informative (especially if variance of a few recordings is driven by outlier sessions)
- **Log-transform power** before band extraction — band power distributions are typically right-skewed, which violates ElasticNet assumptions

## Preprocessing

- **FOOOF/specparam** instead of polynomial subtraction — separates periodic peaks from aperiodic (1/f) component more principally. The aperiodic exponent itself is a feature worth decoding from
- Compare `polyord=5` against other orders — 5th order polynomial is flexible enough to eat real signal

## Modeling (moderate effort)

- **Mixed-effects model** (e.g., `MixedLM` or `MERF`) — CV groups by patient combos, but a mixed-effects model explicitly accounts for patient-level intercepts/slopes, which is more statistically appropriate for repeated measures from 6 patients
- **Temporal structure** — weeks are currently treated as i.i.d. but depression has strong autocorrelation. A state-space model or even just adding delta-features (week-over-week change) could help
- **SVR with RBF kernel** — if the brain-behavior mapping is nonlinear, ElasticNet will miss it. Lose direct coefficient interpretability but can use SHAP values instead

## Target Variable

- **HDRS subscale scores** (sleep, anxiety, cognitive) rather than total — different SCC oscillatory bands may map to different symptom dimensions, and summing into total HDRS adds noise from dimensions SCC doesn't track

## Validation

- **Temporal split** instead of random split — current `train_test_split(shuffle=True)` can leak temporal autocorrelation. Try training on earlier phases, testing on later ones

## Priority

Inter-hemispheric coherence and log-transforming power are probably the highest bang-for-buck. The mixed-effects framing is the most important methodological improvement given N=6 patients.
