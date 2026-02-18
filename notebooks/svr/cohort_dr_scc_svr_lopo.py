# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Cohort DR-SCC: SVR RBF — Leave-One-Patient-Out
# Tests whether the nonlinear SCC→HDRS mapping generalizes to entirely unseen
# patients. Each fold holds out one patient for testing, trains on the remaining
# five, and selects hyperparameters via grouped CV on the training patients.
# Results are aggregated across all 6 held-out patients.

# %%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GridSearchCV, LeaveOneGroupOut
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
# # Extract ALL weekly features (no train/test split yet)
# We use the decoder pipeline to aggregate recordings into weekly features,
# then handle the patient-level split ourselves.

# %%
# Set up decoder for feature extraction only — use all recordings as "train"
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

# Put all recordings into train_set so aggregate_weeks sees everything
data_extractor.train_set = data_extractor.active_rec_list
data_extractor.train_setup()

all_y = data_extractor.train_set_y
all_c = data_extractor.train_set_c.squeeze()
all_pt = data_extractor.train_set_pt
all_ph = data_extractor.train_set_ph
feat_labels = data_extractor.feat_labels

print(f"Total weekly observations: {all_y.shape[0]}")
print(f"Features: {all_y.shape[1]}")
print(f"Patients: {np.unique(all_pt)}")
for pt in do_pts:
    print(f"  {pt}: {np.sum(all_pt == pt)} weeks")

# %% [markdown]
# # Leave-One-Patient-Out cross-validation
# For each held-out patient:
# 1. Standardize features (fit on train patients only)
# 2. Grid search SVR hyperparameters with grouped CV on training patients
# 3. Predict held-out patient's weeks

# %%
logo = LeaveOneGroupOut()

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.05, 0.1],
}

lopo_results = []

for fold, (train_idx, test_idx) in enumerate(logo.split(all_y, all_c, groups=all_pt)):
    held_out_pt = all_pt[test_idx[0]]
    print(f"\n--- Fold {fold + 1}: Holding out patient {held_out_pt} ({len(test_idx)} weeks) ---")

    # Split
    train_y, test_y = all_y[train_idx], all_y[test_idx]
    train_c, test_c = all_c[train_idx], all_c[test_idx]
    train_pt_fold = all_pt[train_idx]

    # Standardize (fit on training patients only)
    scaler = StandardScaler()
    train_y = scaler.fit_transform(train_y)
    test_y = scaler.transform(test_y)

    # Inner CV for hyperparameter selection (grouped by remaining patients)
    inner_cv = GroupKFold(n_splits=min(len(np.unique(train_pt_fold)), 5))
    inner_splits = list(inner_cv.split(train_y, train_c, groups=train_pt_fold))

    svr_search = GridSearchCV(
        SVR(kernel="rbf"),
        param_grid,
        cv=inner_splits,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    svr_search.fit(train_y, train_c)

    svr_model = svr_search.best_estimator_
    predicted_c = svr_model.predict(test_y)

    # Stats for this fold
    fold_r2 = svr_model.score(test_y, test_c)
    fold_mse = mean_squared_error(test_c, predicted_c)
    fold_pearson = sp_stats.pearsonr(predicted_c, test_c) if len(test_c) > 2 else (np.nan, np.nan)

    print(f"  Best params: {svr_search.best_params_}")
    print(f"  R2={fold_r2:.4f}  MSE={fold_mse:.4f}  Pearson={fold_pearson[0]:.4f}")

    lopo_results.append({
        "patient": held_out_pt,
        "r2": fold_r2,
        "mse": fold_mse,
        "pearson_r": fold_pearson[0],
        "pearson_p": fold_pearson[1],
        "best_params": svr_search.best_params_,
        "actual": test_c,
        "predicted": predicted_c,
    })

# %% [markdown]
# # Aggregate LOPO results

# %%
results_df = pd.DataFrame([{
    "Patient": r["patient"],
    "R2": r["r2"],
    "MSE": r["mse"],
    "Pearson r": r["pearson_r"],
    "Pearson p": r["pearson_p"],
} for r in lopo_results])

print(results_df.to_string(index=False))
print(f"\nMean R2:  {results_df['R2'].mean():.4f} +/- {results_df['R2'].std():.4f}")
print(f"Mean MSE: {results_df['MSE'].mean():.4f} +/- {results_df['MSE'].std():.4f}")
print(f"Mean Pearson r: {results_df['Pearson r'].mean():.4f} +/- {results_df['Pearson r'].std():.4f}")

# %% Plot per-patient R2
plt.figure()
plt.bar(results_df["Patient"], results_df["R2"])
plt.axhline(results_df["R2"].mean(), color="gray", linestyle="dotted", label=f'Mean={results_df["R2"].mean():.3f}')
plt.axhline(0, color="black", linewidth=0.5)
plt.ylabel("R2")
plt.xlabel("Held-Out Patient")
plt.title("Leave-One-Patient-Out: SVR RBF")
plt.legend()

# %% [markdown]
# # Pooled predicted vs actual (all held-out patients)

# %%
all_actual = np.concatenate([r["actual"] for r in lopo_results])
all_predicted = np.concatenate([r["predicted"] for r in lopo_results])
all_pt_labels = np.concatenate([[r["patient"]] * len(r["actual"]) for r in lopo_results])

pooled_pearson = sp_stats.pearsonr(all_predicted, all_actual)
pooled_mse = mean_squared_error(all_actual, all_predicted)
ss_res = np.sum((all_actual - all_predicted) ** 2)
ss_tot = np.sum((all_actual - all_actual.mean()) ** 2)
pooled_r2 = 1 - ss_res / ss_tot

print(f"Pooled R2: {pooled_r2:.4f}")
print(f"Pooled MSE: {pooled_mse:.4f}")
print(f"Pooled Pearson: {pooled_pearson[0]:.4f} (p={pooled_pearson[1]:.4e})")

plt.figure()
plt.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
for pt in do_pts:
    mask = all_pt_labels == pt
    plt.scatter(all_predicted[mask], all_actual[mask], label=pt, alpha=0.7)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"LOPO SVR (RBF)\nPooled R2={pooled_r2:.3f}  Pearson={pooled_pearson[0]:.3f}")
plt.legend()

# %% [markdown]
# # Per-patient timecourses (held-out predictions)

# %%
for r in lopo_results:
    plt.figure()
    plt.plot(r["actual"], label="Actual")
    plt.plot(r["predicted"], label="Predicted")
    plt.xlabel("Week")
    plt.ylabel("nHDRS")
    plt.title(f"Patient {r['patient']} (held out) — R2={r['r2']:.3f}")
    plt.legend()

# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")
