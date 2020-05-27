#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:32:00 2020

@author: virati
This is the script that runs the 'standard' SCC Readout training and testing.
"""

from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, decoder
from DBSpace.readout.BR_DataFrame import BR_Data_Tree

# Misc libraries
import copy
import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np

#Debugging
import ipdb


#
## MAJOR PARAMETERS for our partial biometric analysis
do_pts = ['901','903','905','906','907','908'] # Which patients do we want to include in this entire analysis?
# We need to split out the above one to have different training and testing sets
#do_pts = ['901','903','906','907','908'] # Which patients do we want to include in the training set?
#do_pts = ['901']
test_scale = 'pHDRS17' # Which scale are we using as the measurement of the depression state?

''' DETRENDING
Which detrending scheme are we doing
This is important. Block goes into each patient and does zero-mean and linear detrend across time
None does not do this
All does a linear detrend across all concatenated observations. This is dumb and should not be done. Will eliminate this since it makes no sense
'''

# Initial
# Now we set up our DBSpace environment
#ClinFrame = ClinVect.CFrame(norm_scales=True)
ClinFrame = ClinVect.CStruct()
#BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame.pickle')
BRFrame = pickle.load(open('/home/virati/Chronic_Frame.pickle',"rb"))

#%%
main_readout = decoder.weekly_decoder(BRFrame=BRFrame,ClinFrame=ClinFrame,pts=do_pts,clin_measure=test_scale,algo='ENR_all')
main_readout.filter_recs(rec_class='main_study')
main_readout.split_train_set(0.6)

#%%
main_readout.train_setup()
main_readout.train_model()
#%%
main_readout.test_setup()
_,stats = main_readout.test_model()
print(stats)
#%%
main_readout.plot_coeff_sig_path(do_plot=True)
main_readout.plot_test_predictions()
main_readout.plot_decode_coeffs(main_readout.decode_model)
