#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:32:00 2020

@author: virati
"""

from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, DSV
from DBSpace.readout.BR_DataFrame import BR_Data_Tree

# General python libraries
import scipy.signal as sig
import numpy as np
from scipy import interp

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
#Do some cleanup of the plotting space
plt.close('all')
sns.set_context('paper')
sns.set_style('white')
sns.set(font_scale=1)

# Misc libraries
import copy
import itertools
import scipy.stats as stats
import pickle

#Debugging
import ipdb


#%%
## MAJOR PARAMETERS for our partial biometric analysis
do_pts = ['901','903','905','906','907','908'] # Which patients do we want to include in this entire analysis?
# We need to split out the above one to have different training and testing sets
#do_pts = ['901','903','906','907','908'] # Which patients do we want to include in the training set?
#do_pts = ['901']
test_scale = 'HDRS17' # Which scale are we using as the measurement of the depression state?

''' DETRENDING
Which detrending scheme are we doing
This is important. Block goes into each patient and does zero-mean and linear detrend across time
None does not do this
All does a linear detrend across all concatenated observations. This is dumb and should not be done. Will eliminate this since it makes no sense
'''

do_detrend = 'Block' 
rmethod = 'ENR_Osc'
    

#%% Initial
# Now we set up our DBSpace environment
ClinFrame = ClinVect.CFrame(norm_scales=True)
#BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame.pickle')
BRFrame = pickle.load(open('/home/virati/Chronic_Frame.pickle',"rb"))

#%%
main_readout = DSV.RO(BRFrame,ClinFrame,pts=do_pts)
main_readout.filter_recs(rec_class='main_study')
main_readout.split_validation_set(0.6)
