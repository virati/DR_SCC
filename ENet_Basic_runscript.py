#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:51:57 2018

@author: virati
Elastic Net Analysis on BrainRadio Data
THIS USES THE NEW BR_DATAFRAME class
"""

from BR_DataFrame import *
from ClinVect import CFrame
from DSV import *

ClinFrame = CFrame()
#ClinFrame.plot_scale(pts='all',scale='HDRS17')
#ClinFrame.plot_scale(pts=['901'],scale='MADRS')

#%%
BRFrame = BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame.npy')
BRFrame.check_empty_phases()
#%%
analysis = DSV(BRFrame,CFrame)

analysis.run_EN()

#then call the premade methods for analysing EN results for this analysis

#%%

#ready for elastic net setup
#analysis.ENet_Construct()
