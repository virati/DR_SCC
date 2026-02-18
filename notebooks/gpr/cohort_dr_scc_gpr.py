# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Cohort DR-SCC: Gaussian Process Regression
# Nonlinear mapping from SCC oscillatory power to depression score using
# Gaussian Process Regression. Key advantages over SVR:
# - Provides **uncertainty estimates** on each prediction (clinically valuable)
# - Learns the kernel hyperparameters from data via marginal likelihood
# - Scales fine for this dataset size (~168 weeks, 20 features)
#
# Uses the same validation structure: 80/20 recording split, with both
# recording-level and patient-level CV for kernel selection.

# %%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
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
# # Extract weekly features: 80% development, 20% validation

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

# %% Standardize using development set only
scaler = StandardScaler()
dev_y_scaled = scaler.fit_transform(dev_y)
val_y_scaled = scaler.transform(val_y)

# %% [markdown]
# # Define candidate kernels
# GPR learns kernel hyperparameters via marginal likelihood optimization.
# We compare a few kernel families to see which best describes the data.

# %%
kernels = {
    "RBF": ConstantKernel() * RBF() + WhiteKernel(),
    "Matern_1.5": ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
    "Matern_2.5": ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
}

# %% [markdown]
# ---
# # Strategy 1: Recording-level CV

# %%
rec_cv = KFold(n_splits=5, shuffle=True, random_state=2011)

print("Recording-level CV:")
rec_results = {}
for name, kernel in kernels.items():
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        alpha=1e-6,
    )
    scores = cross_val_score(gpr, dev_y_scaled, dev_c, cv=rec_cv, scoring="r2")
    rec_results[name] = {"mean_r2": scores.mean(), "std_r2": scores.std(), "scores": scores}
    print(f"  {name}: R2 = {scores.mean():.4f} +/- {scores.std():.4f}")

best_rec_kernel_name = max(rec_results, key=lambda k: rec_results[k]["mean_r2"])
print(f"\nBest kernel (recording-level): {best_rec_kernel_name}")

# %% [markdown]
# ---
# # Strategy 2: Patient-level CV

# %%
n_groups = len(np.unique(dev_pt))
pt_cv = GroupKFold(n_splits=min(n_groups, 5))

print("Patient-level CV:")
pt_results = {}
for name, kernel in kernels.items():
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        alpha=1e-6,
    )
    scores = cross_val_score(gpr, dev_y_scaled, dev_c, cv=pt_cv, scoring="r2", groups=dev_pt)
    pt_results[name] = {"mean_r2": scores.mean(), "std_r2": scores.std(), "scores": scores}
    print(f"  {name}: R2 = {scores.mean():.4f} +/- {scores.std():.4f}")

best_pt_kernel_name = max(pt_results, key=lambda k: pt_results[k]["mean_r2"])
print(f"\nBest kernel (patient-level): {best_pt_kernel_name}")

# %% [markdown]
# # CV comparison summary

# %%
cv_summary = []
for name in kernels:
    cv_summary.append({
        "Kernel": name,
        "Rec-CV R2": f"{rec_results[name]['mean_r2']:.4f} +/- {rec_results[name]['std_r2']:.4f}",
        "Pt-CV R2": f"{pt_results[name]['mean_r2']:.4f} +/- {pt_results[name]['std_r2']:.4f}",
    })
print(pd.DataFrame(cv_summary).to_string(index=False))

# %% [markdown]
# ---
# # Fit final models on full development set and evaluate on validation

# %%
def fit_and_evaluate(kernel, name, dev_y, dev_c, val_y, val_c, val_pt):
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        alpha=1e-6,
    )
    gpr.fit(dev_y, dev_c)

    pred_mean, pred_std = gpr.predict(val_y, return_std=True)

    r2 = gpr.score(val_y, val_c)
    mse = mean_squared_error(val_c, pred_mean)
    pearson = sp_stats.pearsonr(pred_mean, val_c)
    slope = sp_stats.linregress(pred_mean, val_c)

    print(f"\n[{name}] Validation Set:")
    print(f"  Learned kernel: {gpr.kernel_}")
    print(f"  Log-marginal-likelihood: {gpr.log_marginal_likelihood_value_:.2f}")
    print(f"  R2:        {r2:.4f}")
    print(f"  MSE:       {mse:.4f}")
    print(f"  Pearson r: {pearson[0]:.4f} (p={pearson[1]:.4e})")
    print(f"  Slope:     {slope.slope:.4f}")
    print(f"  Mean pred uncertainty (std): {pred_std.mean():.4f}")

    return {"name": name, "model": gpr, "r2": r2, "mse": mse,
            "pearson_r": pearson[0], "predicted": pred_mean, "uncertainty": pred_std}

# Fit with best kernel from each CV strategy
rec_eval = fit_and_evaluate(
    kernels[best_rec_kernel_name], f"Recording-CV ({best_rec_kernel_name})",
    dev_y_scaled, dev_c, val_y_scaled, val_c, val_pt,
)
pt_eval = fit_and_evaluate(
    kernels[best_pt_kernel_name], f"Patient-CV ({best_pt_kernel_name})",
    dev_y_scaled, dev_c, val_y_scaled, val_c, val_pt,
)

# %% [markdown]
# # Side-by-side predicted vs actual (with uncertainty)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

for ax, ev in zip(axes, [rec_eval, pt_eval]):
    ax.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
    for pt in do_pts:
        mask = val_pt == pt
        if np.any(mask):
            ax.errorbar(
                ev["predicted"][mask], val_c[mask],
                xerr=ev["uncertainty"][mask],
                fmt="o", alpha=0.6, label=pt, capsize=2,
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{ev['name']}\nR2={ev['r2']:.3f}  Pearson={ev['pearson_r']:.3f}")
    ax.legend(fontsize=7)

plt.suptitle("Held-Out Validation Set (GPR with uncertainty)")
plt.tight_layout()

# %% [markdown]
# # Uncertainty calibration
# If the GP is well-calibrated, ~68% of true values should fall within 1 std
# of the predicted mean, ~95% within 2 std.

# %%
best_eval = pt_eval  # use patient-level CV model
residuals = np.abs(val_c - best_eval["predicted"])
within_1std = np.mean(residuals < best_eval["uncertainty"])
within_2std = np.mean(residuals < 2 * best_eval["uncertainty"])

print(f"Fraction within 1 std: {within_1std:.2%} (ideal: ~68%)")
print(f"Fraction within 2 std: {within_2std:.2%} (ideal: ~95%)")

plt.figure()
plt.hist(residuals / best_eval["uncertainty"], bins=15, density=True, alpha=0.7)
plt.axvline(1, color="red", linestyle="dotted", label="1 std")
plt.axvline(2, color="red", linestyle="dashed", label="2 std")
plt.xlabel("|Residual| / Predicted Std")
plt.ylabel("Density")
plt.title("Uncertainty Calibration")
plt.legend()

# %% [markdown]
# # Per-patient validation timecourses with uncertainty bands

# %%
best_pred = best_eval["predicted"]
best_unc = best_eval["uncertainty"]

for pt in do_pts:
    mask = val_pt == pt
    if not np.any(mask):
        continue
    idx = np.arange(np.sum(mask))
    plt.figure()
    plt.plot(idx, val_c[mask], "k-o", label="Actual", markersize=4)
    plt.plot(idx, best_pred[mask], "b-o", label="Predicted", markersize=4)
    plt.fill_between(
        idx,
        best_pred[mask] - 2 * best_unc[mask],
        best_pred[mask] + 2 * best_unc[mask],
        alpha=0.2, color="blue", label="95% CI",
    )
    plt.xlabel("Week")
    plt.ylabel("nHDRS")
    plt.title(f"Patient {pt} (validation set)")
    plt.legend()

# %% [markdown]
# # Summary

# %%
summary = pd.DataFrame([
    {"Model": ev["name"], "Val R2": ev["r2"], "Val MSE": ev["mse"],
     "Val Pearson": ev["pearson_r"], "Mean Uncertainty": ev["uncertainty"].mean()}
    for ev in [rec_eval, pt_eval]
])
print(summary.to_string(index=False))

# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")
