#%%
"""
Created on Tue Apr 21 09:32:00 2020
asdf
@author: virati
THIS IS THE SCRIPT THAT RUNS THE PARSIMONIOUS READOUT from dissertation!

Intermediate file is formed by 
"""

from dbspace.readout import BR_DataFrame as BRDF
from dbspace.readout import ClinVect, decoder
from dbspace.readout import decoder as decoder
from dbspace.readout.BR_DataFrame import BR_Data_Tree

import pickle
import matplotlib.pyplot as plt

plt.rcParams["image.cmap"] = "tab10"
plt.close("all")

import seaborn as sns

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

# Initial
# Now we set up our dbspace environment
# ClinFrame = ClinVect.CFrame(norm_scales=True)
ClinFrame = ClinVect.CStruct()
if test_scale == "mHDRS":
    ClinFrame.gen_mHDRS()
elif test_scale == "DSC":
    ClinFrame.gen_DSC()

# BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame.pickle')
BRFrame = pickle.load(
    open(
        "/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Chronic_FrameMay2020.pickle",
        "rb",
    )
)

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
#%%
# Now we move on to the classifier analysis
threshold_c = decoder.controller_analysis(
    main_readout, bin_type="stim_changes"
)  #'stim_changes')#'threshold')
# threshold_c.classif_runs()
threshold_c.controller_runs()
#%%
test = [
    (a, b)
    for a, b in zip(main_readout.test_set_pt, main_readout.test_set_ph)
    if a == "901"
]
print(test)
print(ClinFrame.Stim_Change_Table())
