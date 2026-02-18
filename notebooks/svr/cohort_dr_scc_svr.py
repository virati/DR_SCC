# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Cohort DR-SCC: SVR with RBF Kernel
# Nonlinear mapping from SCC oscillatory power to depression score using
# Support Vector Regression with a radial basis function kernel.
# Uses patient-grouped CV for hyperparameter selection and SHAP for
# feature importance (since direct coefficient interpretation is lost).

# %%
import pickle
import numpy as np
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GridSearchCV
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
# # Extract weekly features using the existing decoder pipeline

# %%
data_extractor = decoder.weekly_decoderCV(
    BRFrame=BRFrame,
    ClinFrame=ClinFrame,
    pts=do_pts,
    clin_measure=test_scale,
    algo="ENR",
    alpha=-4,
    shuffle_null=False,
    FeatureSet="main",
    variance="both",
    standardize=True,
)
data_extractor.global_plotting = False
data_extractor.filter_recs(rec_class="main_study")
data_extractor.split_train_set(0.8)

# %%
data_extractor.train_setup()
data_extractor.test_setup()

train_y = data_extractor.train_set_y
train_c = data_extractor.train_set_c.squeeze()
train_pt = data_extractor.train_set_pt
test_y = data_extractor.test_set_y
test_c = data_extractor.test_set_c.squeeze()
test_pt = data_extractor.test_set_pt
feat_labels = data_extractor.feat_labels

print(f"Features: {train_y.shape[1]}")
print(f"Train: {train_y.shape[0]} weeks, Test: {test_y.shape[0]} weeks")

# %% [markdown]
# # Hyperparameter search with patient-grouped CV
# SVR has two key hyperparameters:
# - **C**: regularization (lower = more regularized)
# - **gamma**: RBF kernel width (lower = smoother decision boundary)
#
# We search over a grid using leave-patient-out CV.

# %%
group_cv = GroupKFold(n_splits=min(len(np.unique(train_pt)), 5))
cv_splits = list(group_cv.split(train_y, train_c, groups=train_pt))

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.05, 0.1],
}

svr_search = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid,
    cv=cv_splits,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)
svr_search.fit(train_y, train_c)

print(f"\nBest parameters: {svr_search.best_params_}")
print(f"Best CV R2: {svr_search.best_score_:.4f}")

svr_model = svr_search.best_estimator_

# %% [markdown]
# # Evaluate on test set

# %%
predicted_c = svr_model.predict(test_y)

pearson = sp_stats.pearsonr(predicted_c, test_c)
slope = sp_stats.linregress(predicted_c, test_c)
mse = mean_squared_error(test_c, predicted_c)
r2 = svr_model.score(test_y, test_c)

print(f"R2: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Pearson r: {pearson[0]:.4f} (p={pearson[1]:.4e})")
print(f"Slope: {slope.slope:.4f}")

# %% Plot predicted vs actual
plt.figure()
plt.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
ax = sns.regplot(x=predicted_c, y=test_c)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"SVR (RBF)\nR2={r2:.3f}  MSE={mse:.3f}  Pearson={pearson[0]:.3f}")

# %% [markdown]
# # SHAP feature importance
# Since SVR coefficients aren't directly interpretable, use SHAP to understand
# which features drive predictions.

# %%
try:
    import shap

    explainer = shap.KernelExplainer(svr_model.predict, train_y)
    shap_values = explainer.shap_values(test_y)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, test_y, feature_names=feat_labels, show=False)
    plt.title("SHAP Feature Importance (SVR RBF)")
    plt.tight_layout()

    # Bar plot of mean |SHAP|
    plt.figure()
    shap.summary_plot(shap_values, test_y, feature_names=feat_labels, plot_type="bar", show=False)
    plt.title("Mean |SHAP| Values")
    plt.tight_layout()

except ImportError:
    print("SHAP not installed. Run: pip install shap")
    print("Falling back to permutation importance...")

    from sklearn.inspection import permutation_importance

    perm_result = permutation_importance(
        svr_model, test_y, test_c, n_repeats=30, scoring="r2"
    )

    sorted_idx = perm_result.importances_mean.argsort()[::-1]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(feat_labels)), perm_result.importances_mean[sorted_idx])
    plt.xticks(range(len(feat_labels)), np.array(feat_labels)[sorted_idx], rotation=45, ha="right")
    plt.ylabel("Permutation Importance (R2 decrease)")
    plt.title("Permutation Feature Importance (SVR RBF)")
    plt.tight_layout()

# %% [markdown]
# # Per-patient timecourses

# %%
for pt in do_pts:
    pt_mask = test_pt == pt
    if not np.any(pt_mask):
        continue
    plt.figure()
    plt.plot(test_c[pt_mask], label="Actual")
    plt.plot(predicted_c[pt_mask], label="Predicted")
    plt.xlabel("Week")
    plt.ylabel("nHDRS")
    plt.title(f"Patient {pt}")
    plt.legend()

# %% [markdown]
# # Hyperparameter landscape
# Visualize how R2 varies across C and gamma to check for sensitivity.

# %%
import pandas as pd

results_df = pd.DataFrame(svr_search.cv_results_)

# Pivot for C vs gamma (at best epsilon)
best_eps = svr_search.best_params_["epsilon"]
subset = results_df[results_df["param_epsilon"] == best_eps]

# Convert gamma values to strings for pivoting
subset = subset.copy()
subset["param_gamma"] = subset["param_gamma"].astype(str)

pivot = subset.pivot_table(
    values="mean_test_score",
    index="param_C",
    columns="param_gamma",
    aggfunc="mean",
)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn")
plt.title(f"CV R2: C vs gamma (epsilon={best_eps})")
plt.ylabel("C")
plt.xlabel("gamma")

# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")
