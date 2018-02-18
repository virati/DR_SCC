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


ClinFrame = CFrame()
#ClinFrame.plot_scale(pts='all',scale='HDRS17')
#ClinFrame.plot_scale(pts=['901'],scale='MADRS')

BRFrame = BRDF.BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame.npy')
BRFrame.check_empty_phases()


#%%
from DSV import DSV, ORegress

analysis = ORegress(BRFrame,CFrame)
analysis.O_feat_extract()

#%%
analysis.O_regress(method='RANSAC',doplot=True,inpercent=0.6,avgweeks=False)
analysis.O_regress(method='OLS',doplot=True,inpercent=0.6,avgweeks=True)

#%%

O1,C1 = analysis.dsgn_O_C(['901','903'],week_avg=False)

#%%

#X,Y = analysis.get_dsgns()
#analysis.run_OLS(doplot=True)
#analysis.run_RANSAC(inpercent=0.4)



#%%

#This should be folded into the class

#QUICK PLOT

#ready for elastic net setup
#analysis.ENet_Construct()