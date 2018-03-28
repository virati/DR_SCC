#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:39:35 2018

@author: virati
This file does all the regressions on the oscillatory states over chronic timepoints
"""

import BR_DataFrame as BRDF
#from BR_DataFrame import *
from ClinVect import CFrame
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#%%

ClinFrame = CFrame(norm_scales=True)
#ClinFrame.plot_scale(pts='all',scale='HDRS17')
#ClinFrame.plot_scale(pts=['901'],scale='MADRS')

BRFrame = BRDF.BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_March.npy')
BRFrame.check_empty_phases()


#%%
from DSV import DSV, ORegress

analysis = ORegress(BRFrame,ClinFrame)
analysis.O_feat_extract()

#%%

analysis.O_regress(method='RANSAC',doplot=True,inpercent=0.7,avgweeks=False,ranson=False,plot_indiv=False,circ='day') 
analysis.O_models(plot=True,models=['RANSAC'])
#%%
dorsac = True

#analysis.O_regress(method='OLS',doplot=True,inpercent=0.6,avgweeks=True)
#analysis.O_regress(method='OLS',doplot=True,inpercent=0.6,avgweeks=True,ignore_flags=True)
analysis.O_regress(method='RIDGE',doplot=True,avgweeks=True,ranson=dorsac,ignore_flags=False,circ='night',plot_indiv=True)
analysis.O_models(plot=True,models=['RIDGE'])
#%%
analysis.O_regress(method='LASSO',doplot=True,avgweeks=True,ranson=dorsac,ignore_flags=False,circ='',plot_indiv=True)
analysis.O_models(plot=True,models=['LASSO'])
#analysis.O_regress(method='RIDG_Zmis',doplot=True,inpercent=0.6)


#%%
#Plot model coefficients


#%%

#O1,C1 = analysis.dsgn_O_C(['901','903'],week_avg=False)

#%%

#X,Y = analysis.get_dsgns()
#analysis.run_OLS(doplot=True)
#analysis.run_RANSAC(inpercent=0.4)


#%%
# Quick check of the bad flags to see what the thresholds should be

#%%

#This should be folded into the class

#QUICK PLOT

#ready for elastic net setup
#analysis.ENet_Construct()