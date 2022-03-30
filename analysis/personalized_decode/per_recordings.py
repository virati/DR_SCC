#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:27:48 2020

@author: virati

Patient specific readouts for a *PREDICTIVE* model without need for parsimony
"""

#%%

from dbspace.readout import ClinVect, decoder
from dbspace.utils.structures import nestdict
from dbspace.readout import BR_DataFrame as BRDF

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True
import numpy as np

# Misc libraries
import pickle

# Debugging
## MAJOR PARAMETERS for our partial biometric analysis
test_scale = (
    "nHDRS17"  # Which scale are we using as the measurement of the depression state?
)
do_pts = [
    "901",
    "903",
    "905",
    "906",
    "907",
    "908",
]  # Which patients do we want to include in this entire analysis?


""" DETRENDING
Which detrending scheme are we doing
This is important. Block goes into each patient and does zero-mean and linear detrend across time
None does not do this
All does a linear detrend across all concatenated observations. This is dumb and should not be done. Will eliminate this since it makes no sense
"""

# Initial
# Now we set up our dbspace environment
# ClinFrame = ClinVect.CFrame(norm_scales=True)
ClinFrame = ClinVect.CStruct()
# BRFrame = BRDF.BR_Data_Tree(
#    clin_vector_file="../../assets/intermediate_data/ClinVec.json",
# )

BRFrame = pickle.load(
    open(
        "../../assets/intermediate_data/ChronicFrame_April2022.pickle",
        "rb",
    )
)
do_shuffled_null = False
#%%
pt_coeff = {pt: [] for pt in do_pts}
for do_pt in do_pts:
    main_readout = decoder.base_decoder(
        BRFrame=BRFrame,
        ClinFrame=ClinFrame,
        pts=do_pt,
        clin_measure=test_scale,
        shuffle_null=False,
        FeatureSet="main",
    )
    main_readout.filter_recs(rec_class="main_study")
    main_readout.split_train_set(0.6)

    null_slopes, null_r2 = main_readout.model_analysis(do_null=True, n_iter=100)
    main_slope, main_r2 = main_readout.model_analysis()

    print(do_pt + " Slope: " + str(main_slope))
    print("p<" + str(np.sum(null_slopes > main_slope[0]) / 100))
    print("R2: " + str(main_r2))
    print("p<" + str(np.sum(null_r2 > main_r2[0]) / 100))

    plt.figure()
    plt.hist(null_slopes)
    plt.vlines(main_slope[0], 0, 10, linewidth=20)
    plt.title(do_pt + " predictive readout")

    pt_coeff[do_pt] = main_readout.decode_model.coef_
#%%
# Plot all the coeffs
plt.figure()
[plt.plot(pt_coeff[pt], alpha=0.7, linewidth=5, label=pt) for pt in do_pts]
plt.vlines(4.5, -1, 1, linewidth=20)
plt.hlines(0, 0, 9, linestyle="dotted")
plt.xticks(
    range(10),
    [
        r"$\delta$",
        r"$\theta$",
        r"$\alpha$",
        r"$\beta*$",
        r"$\gamma1$",
        r"$\delta$",
        r"$\theta$",
        r"$\alpha$",
        r"$\beta*$",
        r"$\gamma1$",
    ],
)
plt.ylim((-0.15, 0.15))
plt.legend()