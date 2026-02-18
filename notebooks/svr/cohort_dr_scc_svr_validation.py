# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Cohort DR-SCC: SVR RBF — Generalization Analysis
# Quantifies the gap between within-patient and cross-patient generalization:
# 1. Hold out ~20% of recordings as a final validation set
# 2. On the ~80% development set, train SVR with patient-level CV for hyperparameters
# 3. Report both recording-level and patient-level CV R2 on the dev set
#    (the gap reveals how much performance depends on patient-specific patterns)
# 4. Final evaluation on the held-out validation set

# %%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from dbspace.readout import ClinVect, decoder

# Logging stuff here
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Plotting settings here
plt.rcParams["image.cmap"] = "tab10"
sns.set_context("paper")
sns.set_style("white")

# %%
base_data_dir = "/home/virati/Data/phd_vrt_2013/"
frame_to_analyse = 'Chronic_FrameFeb2026_F'

do_pts = [
    "901",
    "903",
    "905",
    "906",
    "907",
    "908",
]
test_scale = "pHDRS17"

# %%
# Initialize our Clinical Frame and load in our BR Frame
ClinFrame = ClinVect.CStruct(Path(notebook_setup.DATADIR + "/clinical/clinical_vectors_all.json"))
if test_scale == "mHDRS":
    ClinFrame.gen_mHDRS()
elif test_scale == "DSC":
    ClinFrame.gen_DSC()

frame_to_analyse = 'Chronic_FrameFeb2026_F'
BRFrame = pickle.load(open(Path(notebook_setup.DATADIR) / f"{frame_to_analyse}.pickle","rb"))

# %% [markdown]
# # Split recordings: 80% development, 20% validation

# %%
data_extractor = decoder.weekly_decoder(
    BRFrame=BRFrame,
    ClinFrame=ClinFrame,
    pts=do_pts,
    clin_measure=test_scale,
    algo="ENR",
    shuffle_null=False,
    FeatureSet="main",
    variance="both",
)
data_extractor.global_plotting = False
data_extractor.filter_recs(rec_class="main_study")
data_extractor.split_train_set(0.8)
data_extractor.train_setup()
data_extractor.test_setup()

dev_y = data_extractor.train_set_y
dev_c = data_extractor.train_set_c.squeeze()
dev_pt = data_extractor.train_set_pt

val_y = data_extractor.test_set_y
val_c = data_extractor.test_set_c.squeeze()
val_pt = data_extractor.test_set_pt
feat_labels = data_extractor.feat_labels

print(f"Features: {dev_y.shape[1]}")
print(f"Development set: {dev_y.shape[0]} weeks")
print(f"Validation set:  {val_y.shape[0]} weeks (held out)")
print(f"\nDev patients: {np.unique(dev_pt)} ({len(np.unique(dev_pt))} unique)")
print(f"Val patients: {np.unique(val_pt)} ({len(np.unique(val_pt))} unique)")

# %% Standardize using development set only
scaler = StandardScaler()
dev_y_scaled = scaler.fit_transform(dev_y)
val_y_scaled = scaler.transform(val_y)

# %% [markdown]
# # Select hyperparameters via patient-level CV
# Use the more conservative (patient-grouped) CV for hyperparameter selection.

# %%
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.05, 0.1],
}

n_groups = len(np.unique(dev_pt))
pt_cv = GroupKFold(n_splits=min(n_groups, 5))
pt_splits = list(pt_cv.split(dev_y_scaled, dev_c, groups=dev_pt))

svr_search = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid,
    cv=pt_splits,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)
svr_search.fit(dev_y_scaled, dev_c)

print(f"\nBest params (patient-level CV): {svr_search.best_params_}")
print(f"Patient-level CV R2: {svr_search.best_score_:.4f}")

svr_model = svr_search.best_estimator_

# %% [markdown]
# ---
# # Generalization gap: recording-level vs patient-level CV
# Evaluate the **same fitted model** under both CV strategies.
# - Recording-level CV: random folds, same patient in train+val → within-patient generalization
# - Patient-level CV: entire patients held out → cross-patient generalization
#
# The gap between these two numbers quantifies how much the model relies on
# learning patient-specific patterns vs a universal SCC→HDRS mapping.

# %%
rec_cv = KFold(n_splits=5, shuffle=True, random_state=2011)

# Use the best hyperparameters, evaluate under both CV schemes
best_svr = SVR(kernel="rbf", **svr_search.best_params_)

rec_scores = cross_val_score(best_svr, dev_y_scaled, dev_c, cv=rec_cv, scoring="r2")
pt_scores = cross_val_score(best_svr, dev_y_scaled, dev_c, cv=pt_cv, scoring="r2", groups=dev_pt)

print(f"Recording-level CV R2: {rec_scores.mean():.4f} +/- {rec_scores.std():.4f}  (per fold: {np.round(rec_scores, 4)})")
print(f"Patient-level CV R2:   {pt_scores.mean():.4f} +/- {pt_scores.std():.4f}  (per fold: {np.round(pt_scores, 4)})")
print(f"\nGap (rec - pt): {rec_scores.mean() - pt_scores.mean():.4f}")
if rec_scores.mean() - pt_scores.mean() > 0.05:
    print("  -> Substantial gap: model benefits from patient-specific patterns.")
elif rec_scores.mean() - pt_scores.mean() < -0.05:
    print("  -> Patient-level CV is better: random folds may be overfitting to recording noise.")
else:
    print("  -> Small gap: model generalizes similarly within and across patients.")

# %% Plot per-fold scores
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(max(len(rec_scores), len(pt_scores)))
width = 0.35
ax.bar(x[:len(rec_scores)] - width/2, rec_scores, width, label=f"Recording-level (mean={rec_scores.mean():.3f})")
ax.bar(x[:len(pt_scores)] + width/2, pt_scores, width, label=f"Patient-level (mean={pt_scores.mean():.3f})")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Fold")
ax.set_ylabel("R2")
ax.set_title("CV R2 by Fold: Recording-level vs Patient-level")
ax.legend()

# %% [markdown]
# ---
# # Final evaluation on held-out validation set

# %%
predicted_c = svr_model.predict(val_y_scaled)

r2 = svr_model.score(val_y_scaled, val_c)
mse = mean_squared_error(val_c, predicted_c)
pearson = sp_stats.pearsonr(predicted_c, val_c)
slope = sp_stats.linregress(predicted_c, val_c)

print(f"Validation R2:        {r2:.4f}")
print(f"Validation MSE:       {mse:.4f}")
print(f"Validation Pearson r: {pearson[0]:.4f} (p={pearson[1]:.4e})")
print(f"Validation Slope:     {slope.slope:.4f}")

# %% Plot predicted vs actual
plt.figure()
plt.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
for pt in do_pts:
    mask = val_pt == pt
    if np.any(mask):
        plt.scatter(predicted_c[mask], val_c[mask], label=pt, alpha=0.7)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"SVR (RBF) — Held-Out Validation\nR2={r2:.3f}  MSE={mse:.3f}  Pearson={pearson[0]:.3f}")
plt.legend()

# %% [markdown]
# # Summary table

# %%
summary = pd.DataFrame([{
    "Metric": "Recording-level CV R2",
    "Value": f"{rec_scores.mean():.4f} +/- {rec_scores.std():.4f}",
}, {
    "Metric": "Patient-level CV R2",
    "Value": f"{pt_scores.mean():.4f} +/- {pt_scores.std():.4f}",
}, {
    "Metric": "Generalization gap",
    "Value": f"{rec_scores.mean() - pt_scores.mean():.4f}",
}, {
    "Metric": "Validation R2",
    "Value": f"{r2:.4f}",
}, {
    "Metric": "Validation Pearson r",
    "Value": f"{pearson[0]:.4f}",
}])
print(summary.to_string(index=False))

# %% [markdown]
# # Per-patient validation timecourses

# %%
for pt in do_pts:
    mask = val_pt == pt
    if not np.any(mask):
        continue
    plt.figure()
    plt.plot(val_c[mask], label="Actual")
    plt.plot(predicted_c[mask], label="Predicted")
    plt.xlabel("Week")
    plt.ylabel("nHDRS")
    plt.title(f"Patient {pt} (validation set)")
    plt.legend()

# %% [markdown]
# # Hyperparameter landscape (patient-level CV)

# %%
results_df = pd.DataFrame(svr_search.cv_results_)
best_eps = svr_search.best_params_["epsilon"]
subset = results_df[results_df["param_epsilon"] == best_eps].copy()
subset["param_gamma"] = subset["param_gamma"].astype(str)

pivot = subset.pivot_table(
    values="mean_test_score",
    index="param_C",
    columns="param_gamma",
    aggfunc="mean",
)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn")
plt.title(f"Patient-level CV R2: C vs gamma (epsilon={best_eps})")
plt.ylabel("C")
plt.xlabel("gamma")

# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")
