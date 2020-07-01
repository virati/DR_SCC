#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:14:20 2020

@author: virati
Script that tests for correlations between any given band and stim artifacts
"""

from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, decoder
from DBSpace.readout.decoder import feat_check as feat_check
from DBSpace.readout.BR_DataFrame import BR_Data_Tree

# Misc libraries
import copy
import pickle


#Debugging
import ipdb

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'tab10'
plt.close('all')


## MAJOR PARAMETERS for our partial biometric analysis
do_pts = ['901','903','905','906','907','908'] # Which patients do we want to include in this entire analysis?
test_scale = 'pHDRS17' # Which scale are we using as the measurement of the depression state? pHDRS17 = nHDRS (from paper) and is a patient-specific normalized HDRS

# Initial
BRFrame = pickle.load(open('/home/virati/Dropbox/Data/Chronic_FrameMay2020.pickle',"rb"))

checker = feat_check(BRFrame = BRFrame)

#%%
checker.check_stim_corr(band='Beta*',artifact='GCratio',pt='ALL')