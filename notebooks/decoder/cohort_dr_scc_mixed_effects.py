# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Cohort DR-SCC: Mixed-Effects Model
# Instead of treating weekly observations as i.i.d., use a linear mixed-effects model
# with patient-level random intercepts to account for repeated measures structure.
# Fixed effects are the oscillatory band features; random effects capture
# patient-specific baselines.

# %%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
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
# # Use the existing decoder to extract weekly features and depression scores
# We use weekly_decoderCV to get the data pipeline (aggregate_weeks, poly_subtr, etc.)
# but fit our own mixed-effects model instead of ElasticNet.

# %%
# Set up the decoder just for data extraction
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
# Extract train and test data
data_extractor.train_setup()
data_extractor.test_setup()

# %% [markdown]
# # Build DataFrames with patient labels for mixed-effects modeling

# %%
def build_dataframe(y, c, pt, feat_labels):
    """Build a pandas DataFrame from decoder arrays with patient grouping."""
    df = pd.DataFrame(y, columns=feat_labels)
    df["depression"] = c.squeeze()
    df["patient"] = pt
    return df

feat_labels = data_extractor.feat_labels
train_df = build_dataframe(
    data_extractor.train_set_y,
    data_extractor.train_set_c,
    data_extractor.train_set_pt,
    feat_labels,
)
test_df = build_dataframe(
    data_extractor.test_set_y,
    data_extractor.test_set_c,
    data_extractor.test_set_pt,
    feat_labels,
)

print(f"Training set: {len(train_df)} weeks from {train_df['patient'].nunique()} patients")
print(f"Test set: {len(test_df)} weeks from {test_df['patient'].nunique()} patients")
train_df.head()

# %% [markdown]
# # Fit Mixed-Effects Model
# Random intercept per patient, fixed effects for all oscillatory features.

# %%
# Sanitize column names for statsmodels formula (replace * and - with _)
rename_map = {col: col.replace("*", "star").replace("-", "_") for col in feat_labels}
train_df = train_df.rename(columns=rename_map)
test_df = test_df.rename(columns=rename_map)
clean_feat_names = [rename_map[f] for f in feat_labels]

# Build formula: depression ~ feature1 + feature2 + ...
fixed_effects = " + ".join(clean_feat_names)
formula = f"depression ~ {fixed_effects}"
print(f"Formula: {formula}")

# %%
# Fit the mixed-effects model with random intercept per patient
me_model = smf.mixedlm(formula, train_df, groups=train_df["patient"])
me_result = me_model.fit()
print(me_result.summary())

# %% [markdown]
# # Evaluate on test set

# %%
from scipy import stats as sp_stats
from sklearn.metrics import mean_squared_error

# Predict on test set
test_df["predicted"] = me_result.predict(test_df)

# Stats
pearson = sp_stats.pearsonr(test_df["predicted"], test_df["depression"])
slope = sp_stats.linregress(test_df["predicted"], test_df["depression"])
mse = mean_squared_error(test_df["depression"], test_df["predicted"])
ss_res = np.sum((test_df["depression"] - test_df["predicted"]) ** 2)
ss_tot = np.sum((test_df["depression"] - test_df["depression"].mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"R2: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Pearson r: {pearson[0]:.4f} (p={pearson[1]:.4e})")
print(f"Slope: {slope.slope:.4f}")

# %% Plot predicted vs actual
plt.figure()
plt.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
ax = sns.regplot(x="predicted", y="depression", data=test_df)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Mixed-Effects Model\nR2={r2:.3f}  MSE={mse:.3f}  Pearson={pearson[0]:.3f}")

# %% [markdown]
# # Fixed-effect coefficients

# %%
# Plot the fixed-effect coefficients (excluding intercept)
fe_params = me_result.fe_params.drop("Intercept")
fe_ci = me_result.conf_int().loc[fe_params.index]

fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(fe_params))
ax.bar(x_pos, fe_params.values, yerr=[
    fe_params.values - fe_ci.iloc[:, 0].values,
    fe_ci.iloc[:, 1].values - fe_params.values,
], capsize=3, alpha=0.7)
ax.axhline(0, color="gray", linestyle="dotted")
ax.set_xticks(x_pos)
ax.set_xticklabels(fe_params.index, rotation=45, ha="right")
ax.set_ylabel("Coefficient")
ax.set_title("Fixed-Effect Coefficients (with 95% CI)")
plt.tight_layout()

# %% [markdown]
# # Random effects (patient intercepts)

# %%
re_df = pd.DataFrame({
    "patient": me_result.random_effects.keys(),
    "intercept": [v["Group"] for v in me_result.random_effects.values()],
})
print(re_df)

plt.figure()
plt.bar(re_df["patient"], re_df["intercept"])
plt.axhline(0, color="gray", linestyle="dotted")
plt.xlabel("Patient")
plt.ylabel("Random Intercept")
plt.title("Patient-Level Random Intercepts")

# %% [markdown]
# # Compare: per-patient predicted vs actual timecourses

# %%
for pt in do_pts:
    pt_data = test_df[test_df["patient"] == pt].copy()
    if len(pt_data) == 0:
        continue
    plt.figure()
    plt.plot(pt_data["depression"].values, label="Actual")
    plt.plot(pt_data["predicted"].values, label="Predicted")
    plt.xlabel("Week")
    plt.ylabel("nHDRS")
    plt.title(f"Patient {pt}")
    plt.legend()

# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")
