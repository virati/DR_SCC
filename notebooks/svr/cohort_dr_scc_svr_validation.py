# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Cohort DR-SCC: SVR RBF — Recording-level and Patient-level CV with Validation Set
# Mirrors the structure of cohort_dr_scc.py:
# 1. Hold out ~20% of recordings as a final validation set (untouched until the end)
# 2. On the remaining ~80%, fit SVR with two CV strategies:
#    - **Recording-level CV**: random folds (same patient can appear in train+val)
#    - **Patient-level CV**: GroupKFold by patient (no patient leakage)
# 3. Evaluate both models on the held-out validation set

# %%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV, train_test_split
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
# The validation set is held out completely — no model sees it until final evaluation.

# %%
# Use the decoder to get filtered recordings and aggregate into weekly features
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

# Split recordings 80/20, then aggregate each into weekly features
data_extractor.split_train_set(0.8)
data_extractor.train_setup()
data_extractor.test_setup()

dev_y = data_extractor.train_set_y
dev_c = data_extractor.train_set_c.squeeze()
dev_pt = data_extractor.train_set_pt
dev_ph = data_extractor.train_set_ph

val_y = data_extractor.test_set_y
val_c = data_extractor.test_set_c.squeeze()
val_pt = data_extractor.test_set_pt
val_ph = data_extractor.test_set_ph
feat_labels = data_extractor.feat_labels

print(f"Features: {dev_y.shape[1]}")
print(f"Development set: {dev_y.shape[0]} weeks")
print(f"Validation set:  {val_y.shape[0]} weeks (held out)")
print(f"\nDevelopment patients: {np.unique(dev_pt)}")
print(f"Validation patients:  {np.unique(val_pt)}")

# %% [markdown]
# # Standardize using development set only

# %%
scaler = StandardScaler()
dev_y_scaled = scaler.fit_transform(dev_y)
val_y_scaled = scaler.transform(val_y)

# %% [markdown]
# # SVR hyperparameter grid

# %%
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.05, 0.1],
}

# %% [markdown]
# ---
# # Strategy 1: Recording-level CV (random 5-fold)
# Same patient can appear in both train and validation folds.
# This tests within-patient generalization to new timepoints.

# %%
rec_cv = KFold(n_splits=5, shuffle=True, random_state=2011)

svr_rec = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid,
    cv=rec_cv,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)
svr_rec.fit(dev_y_scaled, dev_c)

print(f"\n[Recording-level CV]")
print(f"Best params: {svr_rec.best_params_}")
print(f"Best CV R2: {svr_rec.best_score_:.4f}")

# %% [markdown]
# ---
# # Strategy 2: Patient-level CV (GroupKFold)
# Each fold holds out entire patients. Tests cross-patient generalization
# during hyperparameter selection.

# %%
n_groups = len(np.unique(dev_pt))
pt_cv = GroupKFold(n_splits=min(n_groups, 5))
pt_splits = list(pt_cv.split(dev_y_scaled, dev_c, groups=dev_pt))

svr_pt = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid,
    cv=pt_splits,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)
svr_pt.fit(dev_y_scaled, dev_c)

print(f"\n[Patient-level CV]")
print(f"Best params: {svr_pt.best_params_}")
print(f"Best CV R2: {svr_pt.best_score_:.4f}")

# %% [markdown]
# ---
# # Evaluate both models on the held-out validation set

# %%
def evaluate_model(model, name, test_y, test_c):
    pred = model.predict(test_y)
    r2 = model.score(test_y, test_c)
    mse = mean_squared_error(test_c, pred)
    pearson = sp_stats.pearsonr(pred, test_c)
    slope = sp_stats.linregress(pred, test_c)
    print(f"\n[{name}] Validation Set:")
    print(f"  R2:        {r2:.4f}")
    print(f"  MSE:       {mse:.4f}")
    print(f"  Pearson r: {pearson[0]:.4f} (p={pearson[1]:.4e})")
    print(f"  Slope:     {slope.slope:.4f}")
    return {"name": name, "r2": r2, "mse": mse, "pearson_r": pearson[0],
            "pearson_p": pearson[1], "slope": slope.slope, "predicted": pred}

rec_eval = evaluate_model(svr_rec.best_estimator_, "Recording-level CV", val_y_scaled, val_c)
pt_eval = evaluate_model(svr_pt.best_estimator_, "Patient-level CV", val_y_scaled, val_c)

# %% [markdown]
# # Side-by-side predicted vs actual

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

for ax, ev in zip(axes, [rec_eval, pt_eval]):
    ax.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
    for pt in do_pts:
        mask = val_pt == pt
        if np.any(mask):
            ax.scatter(ev["predicted"][mask], val_c[mask], label=pt, alpha=0.7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{ev['name']}\nR2={ev['r2']:.3f}  Pearson={ev['pearson_r']:.3f}")
    ax.legend(fontsize=7)

plt.suptitle("Held-Out Validation Set")
plt.tight_layout()

# %% [markdown]
# # Summary comparison

# %%
summary = pd.DataFrame([
    {"CV Strategy": "Recording-level", "CV R2": svr_rec.best_score_,
     "Val R2": rec_eval["r2"], "Val MSE": rec_eval["mse"],
     "Val Pearson": rec_eval["pearson_r"],
     "C": svr_rec.best_params_["C"], "gamma": svr_rec.best_params_["gamma"],
     "epsilon": svr_rec.best_params_["epsilon"]},
    {"CV Strategy": "Patient-level", "CV R2": svr_pt.best_score_,
     "Val R2": pt_eval["r2"], "Val MSE": pt_eval["mse"],
     "Val Pearson": pt_eval["pearson_r"],
     "C": svr_pt.best_params_["C"], "gamma": svr_pt.best_params_["gamma"],
     "epsilon": svr_pt.best_params_["epsilon"]},
])
print(summary.to_string(index=False))

# %% [markdown]
# # Per-patient validation timecourses (patient-level CV model)

# %%
best_pred = pt_eval["predicted"]
for pt in do_pts:
    mask = val_pt == pt
    if not np.any(mask):
        continue
    plt.figure()
    plt.plot(val_c[mask], label="Actual")
    plt.plot(best_pred[mask], label="Predicted")
    plt.xlabel("Week")
    plt.ylabel("nHDRS")
    plt.title(f"Patient {pt} (validation set)")
    plt.legend()

# %% [markdown]
# # Hyperparameter landscape (patient-level CV)

# %%
results_df = pd.DataFrame(svr_pt.cv_results_)
best_eps = svr_pt.best_params_["epsilon"]
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
