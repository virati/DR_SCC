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

import matplotlib.pyplot as plt
plt.close('all')

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


analysis.run_EN()

#%%
plt.figure()
coeff_len = int(analysis.ENet.ENet.coef_.shape[0]/2)

plt.plot(analysis.trunc_fvect,analysis.ENet.ENet.coef_[0:coeff_len],label='Left Feats')
plt.plot(analysis.trunc_fvect,analysis.ENet.ENet.coef_[coeff_len:],label='Right Feats')
plt.legend()

#then call the premade methods for analysing EN results for this analysis

#%%

#ready for elastic net setup
#analysis.ENet_Construct()
