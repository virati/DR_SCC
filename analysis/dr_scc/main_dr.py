"""
Created on Tue Apr 21 09:32:00 2020
asdf
@author: virati
THIS IS THE SCRIPT THAT RUNS THE PARSIMONIOUS READOUT from dissertation!

Intermediate file is formed by
"""

#%%
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
from dbspace.readout import ClinVect, decoder
import dbspace.utils.frames.BR_DataFrame as BR_DataFrame

plt.rcParams["image.cmap"] = "tab10"

sns.set_context("paper")
sns.set(font_scale=4)
sns.set_style("white")

## MAJOR PARAMETERS for our partial biometric analysis
do_pts = [
    "901",
    "903",
    "905",
    "906",
    "907",
    "908",
]  # Which patients do we want to include in this entire analysis?
test_scale = "pHDRS17"  # Which scale are we using as the measurement of the depression state? pHDRS17 = nHDRS (from paper) and is a patient-specific normalized HDRS

#%%
# Initialize our Clinical Frame and load in our BR Frame

ClinFrame = ClinVect.CStruct()
if test_scale == "mHDRS":
    ClinFrame.gen_mHDRS()
elif test_scale == "DSC":
    ClinFrame.gen_DSC()

# BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame.pickle')
BRFrame = pickle.load(
    open(
        "/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/SCC_Readout/assets/intermediate_data/ChronicFrame_April2022.pickle",
        "rb",
    )
)

#%% [markdown]
## Train, test, validate the weekly decoder

#%%
main_readout = decoder.weekly_decoderCV(
    BRFrame=BRFrame,
    ClinFrame=ClinFrame,
    pts=do_pts,
    clin_measure=test_scale,
    algo="ENR",
    alpha=-4,
    shuffle_null=False,
    FeatureSet="main",
    variance=False,
)  # main analysis is -3.4
main_readout.global_plotting = True
main_readout.filter_recs(rec_class="main_study")
main_readout.split_train_set(0.6)

#%%
main_readout.train_setup()
optimal_alpha = (
    main_readout._path_slope_regression()
)  # suppress_vars=1/40)#,override_alpha=2**-3) #suppress_vars = 1 works well if we're not doing THarm analysis
main_readout.train_model()

#%%
main_readout.plot_decode_CV()

#%%
main_readout.test_setup()
main_readout.test_model()

main_readout.plot_test_timecourse()
#%%
main_readout.plot_test_stats()
#%%
main_readout.plot_test_regression_figure()
# main_readout.plot_combo_paths()
