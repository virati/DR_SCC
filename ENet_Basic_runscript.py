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

#%%
ClinFrame = CFrame(norm_scales=True)
#ClinFrame.plot_scale(pts='all',scale='HDRS17')
#ClinFrame.plot_scale(pts=['901'],scale='MADRS')

#%%
BRFrame = BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_March.npy')
BRFrame.check_empty_phases()
#%%
from DSV import *
analysis = DSV(BRFrame,ClinFrame,lim_freq=30)

ENet_params = {'Alpha':(0.2,0.5),'Lambda':(0.5,0.6)}

analysis.run_EN()

#%%
#aanalysis.plot_dsgn_matrix()

analysis.plot_tests()
analysis.plot_EN_coeffs()

#%%
analysis.plot_dsgn_matrix()


#then call the premade methods for analysing EN results for this analysis

#%%

#ready for elastic net setup
#analysis.ENet_Construct()
