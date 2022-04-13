#%%
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
from dbspace.readout import ClinVect, decoder

plt.rcParams["image.cmap"] = "tab10"

sns.set_context("paper")
sns.set(font_scale=4)
sns.set_style("white")

#%%
do_pts = [
    "901",
    "903",
    "905",
    "906",
    "907",
    "908",
]
test_scale = "pHDRS17"  # Which scale are we using as the measurement of the depression state? pHDRS17 = nHDRS (from paper) and is a patient-specific normalized HDRS

#%%
ClinFrame = ClinVect.CStruct()

# BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame.pickle')
BRFrame = pickle.load(
    open(
        "../../assets/intermediate_data/ChronicFrame_April2022.pickle",
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

main_readout.test_setup()
main_readout.test_model()

#%%
threshold_c = decoder.controller_analysis(main_readout, bin_type="stim_changes")
threshold_c.controller_runs()
#%%
threshold_c.controller_runs_plot(
    plot_controllers=["empirical", "empirical+readout"], plot_pr_aucs=True
)
#%%
test = [
    (a, b)
    for a, b in zip(main_readout.test_set_pt, main_readout.test_set_ph)
    if a == "901"
]
print(test)
print(ClinFrame.Stim_Change_Table())
