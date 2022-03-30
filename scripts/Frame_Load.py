#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:27:32 2017

@author: virati
Quick file to load in Data Frame
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import ElasticNet, ElasticNetCV
import pdb

# The frame is made by the old scripts in Integrated Ephys (for now)

# Let's load in that frame, which should contain all the data we want
# Only do Chronics with this analysis
exp = "Chronics"
DataFrame = np.load("/home/virati/MDD_Data/Data_frame_ChronicsMed.npy").item()
# Load in the stim change times, though this should be merged into DataFrame itself in the generation script
StimChange = scipy.io.loadmat("/home/virati/MDD_Data/stim_changes.mat")["StimMatrix"]
f_focus = (0, 50)

#%%
# Nitty gritty time
f_vect = np.linspace(0, 211, 512)
f_trunc = np.where(np.logical_and(f_vect < f_focus[1], f_vect > f_focus[0]))[0]
f_dict = {"FreqVector": f_vect, "FreqTrunc": f_trunc}

# fixed 3test x 3train approach (for now)
training_patients = ["905", "907", "906"]
test_patients = ["901", "903", "908"]

# Need to make our design matrix now
# get the list of ephys-related clinical phases
def Phase_StateMatrix(
    DataFrame,
    Phases,
    max_freq=211,
    training_patients=["901", "903", "905", "906", "907", "908"],
):
    max_fidx = int(max_freq / 211 * 512)

    dsgnX = []
    dsgnY = []
    for pt, patient in enumerate(training_patients):
        for pp, phase in enumerate(DataFrame["DBS" + patient].keys()):
            if phase[0] != "A":
                print(patient + " " + phase + " up now")

                try:
                    state_vector = np.hstack(
                        (
                            DataFrame["DBS" + patient][phase]["MeanPSD"]["LOGPxx"][
                                :max_fidx, 0
                            ],
                            DataFrame["DBS" + patient][phase]["MeanPSD"]["LOGPxx"][
                                :max_fidx, 1
                            ],
                        )
                    )
                    # state_vector = np.hstack((DataFrame['DBS'+patient][phase]['MeanPSD']['LOGPxx'][:max_fidx,0],DataFrame['DBS'+patient][phase]['MeanPSD']['LOGPxx'][:max_fidx,1]))
                    dsgnX.append(state_vector)
                    dsgnY.append(DataFrame["DBS" + patient][phase]["HDRS17"])
                except:
                    print(patient + " " + phase + " has a problem")

    # confirm dimensional conguence
    Xout = np.array(dsgnX)
    Yout = np.array(dsgnY)

    # return the design tensors and flags for NaN presence
    return Xout, Yout


DsgnX, clinY = Phase_StateMatrix(
    DataFrame,
    Phase_List(exprs="ephys"),
    max_freq=70,
    training_patients=training_patients,
)

# The actual regression code below
EN_alpha = 0.2
# Generate our elastic net model now
DM = ElasticNet(alpha=EN_alpha, tol=0.001, normalize=True, positive=False)
DM.fit(DsgnX, clinY)

plt.figure()
coefs = DM.coef_
csize = coefs.shape[0]
plt.plot(coefs[0 : int(np.ceil(csize / 2))], color="blue")
plt.plot(coefs[int(np.ceil(csize / 2) + 1) :], color="red")
