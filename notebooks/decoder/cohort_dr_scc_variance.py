# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Cohort DR-SCC (with Variance Features)
# This notebook covers the main cohort-level results of the DR-SCC from my dissertation.
# In addition to mean oscillatory power, variance of oscillatory power within a week is included as features.

# %%
import pickle
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from dbspace.readout import ClinVect, decoder
import dbspace.readout.BR_DataFrame as BR_DataFrame

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

## Patients and Depression Score to use
do_pts = [
    "901",
    "903",
    "905",
    "906",
    "907",
    "908",
]
test_scale = "pHDRS17"  # Which scale are we using as the measurement of the depression state? pHDRS17 = nHDRS (from paper) and is a patient-specific normalized HDRS


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
# # Train, test, validate the weekly decoder (with variance features)

# %%
main_readout = decoder.weekly_decoderCV(
    BRFrame=BRFrame,
    ClinFrame=ClinFrame,
    pts=do_pts,
    clin_measure=test_scale,
    algo="ENR",
    alpha=-4,
    shuffle_null=False,
    FeatureSet="main",
    variance="both",
)  # main analysis is -3.4
main_readout.global_plotting = True
main_readout.filter_recs(rec_class="main_study")
main_readout.split_train_set(0.8)

# %% Plot distribution of oscillatory power across all observations
from dbspace.readout.OBands import OBands

obands = OBands(BRFrame, do_patients=do_pts)
obands.plot_power_distributions()


# %%
main_readout.train_setup()
optimal_alpha = (
    main_readout._path_slope_regression()
)  # suppress_vars=1/40)#,override_alpha=2**-3) #suppress_vars = 1 works well if we're not doing THarm analysis
main_readout.train_model()


# %% Plot the coefficients of the trained model
main_readout.plot_decode_CV()


# %% Test setup and model, plot timecourses
main_readout.test_setup()
main_readout.test_model()

main_readout.plot_test_timecourse()

# %% Plot the test statistics
main_readout.plot_test_stats()

# %% Plot the regression
main_readout.plot_test_regression_figure()
# main_readout.plot_combo_paths()


# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")


