# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Cohort DR-SCC: Polynomial Features + ElasticNet
# Expand the oscillatory feature set with degree-2 polynomial terms
# (squared terms + pairwise interactions), then let ElasticNet's L1 penalty
# select which nonlinear terms matter. This tests whether the SCC power → HDRS
# mapping has important nonlinear or interaction structure.

# %%
import pickle
import numpy as np
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
base_feat_labels = data_extractor.feat_labels

print(f"Base features: {train_y.shape[1]}")
print(f"Train: {train_y.shape[0]} weeks, Test: {test_y.shape[0]} weeks")

# %% [markdown]
# # Expand features with degree-2 polynomial terms
# This creates: original features + squared terms + all pairwise interactions.
# With 20 base features (10 mean + 10 variance), this produces 230 features
# (20 linear + 20 squared + 190 interactions).

# %%
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
train_y_poly = poly.fit_transform(train_y)
test_y_poly = poly.transform(test_y)

poly_feat_names = np.array(poly.get_feature_names(base_feat_labels))
print(f"Expanded features: {train_y_poly.shape[1]}")
print(f"  Linear: {len(base_feat_labels)}")
print(f"  Squared + interactions: {train_y_poly.shape[1] - len(base_feat_labels)}")

# %% Standardize the expanded features (important since squared/interaction terms have different scales)
scaler = StandardScaler()
train_y_poly = scaler.fit_transform(train_y_poly)
test_y_poly = scaler.transform(test_y_poly)

# %% [markdown]
# # Fit ElasticNetCV on polynomial features
# Use patient-grouped CV so the same patient never appears in both train and
# validation folds. This prevents optimistic alpha selection that leads to
# overfitting on interaction terms.

# %%
# Build patient-grouped CV splits
group_cv = GroupKFold(n_splits=min(len(np.unique(train_pt)), 5))
cv_splits = list(group_cv.split(train_y_poly, train_c, groups=train_pt))

en_model = ElasticNetCV(
    l1_ratio=[0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
    n_alphas=100,
    cv=cv_splits,
    max_iter=10000,
)
en_model.fit(train_y_poly, train_c)

print(f"Selected alpha: {en_model.alpha_:.6f}")
print(f"Selected l1_ratio: {en_model.l1_ratio_:.2f}")
print(f"Non-zero coefficients: {np.sum(en_model.coef_ != 0)} / {len(en_model.coef_)}")

# %% [markdown]
# # Evaluate on test set

# %%
predicted_c = en_model.predict(test_y_poly)

pearson = sp_stats.pearsonr(predicted_c, test_c)
slope = sp_stats.linregress(predicted_c, test_c)
mse = mean_squared_error(test_c, predicted_c)
r2 = en_model.score(test_y_poly, test_c)

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
plt.title(f"Poly Features + ElasticNet\nR2={r2:.3f}  MSE={mse:.3f}  Pearson={pearson[0]:.3f}")

# %% [markdown]
# # Which features survived regularization?
# Show the non-zero coefficients — these are the linear, squared, and interaction
# terms that ElasticNet selected as predictive.

# %%
nonzero_mask = en_model.coef_ != 0
surviving_features = poly_feat_names[nonzero_mask]
surviving_coeffs = en_model.coef_[nonzero_mask]

# Sort by absolute coefficient value
sort_idx = np.argsort(np.abs(surviving_coeffs))[::-1]
surviving_features = surviving_features[sort_idx]
surviving_coeffs = surviving_coeffs[sort_idx]

print(f"\n{len(surviving_features)} surviving features:")
for name, coef in zip(surviving_features, surviving_coeffs):
    print(f"  {coef:+.4f}  {name}")

# %% Plot surviving coefficients
fig, ax = plt.subplots(figsize=(14, 6))
x_pos = np.arange(len(surviving_features))
colors = ['tab:blue' if c > 0 else 'tab:red' for c in surviving_coeffs]
ax.bar(x_pos, surviving_coeffs, color=colors, alpha=0.7)
ax.axhline(0, color="gray", linestyle="dotted")
ax.set_xticks(x_pos)
ax.set_xticklabels(surviving_features, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Coefficient")
ax.set_title(f"Surviving Coefficients ({len(surviving_features)} / {len(en_model.coef_)})")
plt.tight_layout()

# %% [markdown]
# # Compare: how many surviving features are nonlinear (squared/interaction) vs linear?

# %%
n_base = len(base_feat_labels)
linear_surviving = np.sum(nonzero_mask[:n_base])
nonlinear_surviving = np.sum(nonzero_mask[n_base:])

print(f"Linear features surviving: {linear_surviving} / {n_base}")
print(f"Nonlinear features surviving: {nonlinear_surviving} / {train_y_poly.shape[1] - n_base}")

plt.figure(figsize=(5, 4))
plt.bar(["Linear", "Squared/Interaction"], [linear_surviving, nonlinear_surviving])
plt.ylabel("Count")
plt.title("Surviving Feature Types")

# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")
