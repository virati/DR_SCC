#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:39:35 2018

@author: virati
This file does all the regressions on the oscillatory states over chronic timepoints
"""

from BR_DataFrame import *
from ClinVect import CFrame

ClinFrame = CFrame()
#ClinFrame.plot_scale(pts='all',scale='HDRS17')
#ClinFrame.plot_scale(pts=['901'],scale='MADRS')

#%%
BRFrame = BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame.npy')
BRFrame.check_empty_phases()


#%%
from DSV import *

analysis = DSV(BRFrame,CFrame)
analysis.O_feat_extract()

#%%

#X,Y = analysis.get_dsgns()
Otest,Ctest = analysis.dsgn_O_C(['901','903'])


#ready for elastic net setup
#analysis.ENet_Construct()
