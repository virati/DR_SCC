#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:19:13 2019

@author: virati
"""

#DBSpace libraries and sublibraries here
from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, DSV

# General python libraries
import scipy.signal as sig
import numpy as np

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
#Do some cleanup of the plotting space
plt.close('all')
sns.set_context('paper')
sns.set_style('white')
sns.set(font_scale=4)

# Misc libraries
import copy
import itertools
import scipy.stats as stats

#%%
#Debugging
import pdb


#%%
## MAJOR PARAMETERS for our partial biometric analysis
do_pts = ['901','903','905','906','907','908'] # Which patients do we want to include in this entire analysis?
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
    


#%%
# Now we set up our DBSpace environment
ClinFrame = ClinVect.CFrame(norm_scales=True)
BRFrame = BRDF.BR_Data_Tree()

# Run our main sequence for populating the BRFrame
BRFrame.full_sequence(data_path='/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Chronic_Frame_july.npy')
BRFrame.check_empty_phases() # Check to see if there are any empty phases. This should be folded into full_sequence soon TODO

readout = DSV.on_demand(BRFrame,ClinFrame,validation=0.3)

