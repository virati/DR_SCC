#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:24:26 2018

@author: virati
Traditional Analyses
"""


from BR_DataFrame import *
from collections import defaultdict

import matplotlib.pyplot as plt
plt.close('all')

#Bring in our data first

BRFrame = BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame.npy')

#do a check to see if any PSDs are entirely zero; bad sign
BRFrame.check_meta()


#%%
#Move forward with traditional oscillatory band analysis
from OBands import *
analysis = OBands(BRFrame)
analysis.feat_extract()

##PLOTS

focus_feats =['Delta','Theta','Alpha','Beta']
#Now we're ready for the plots

#First thing is the per-patient weekly averages plotted for left and right
_ = analysis.mean_psds(weeks=["C01","C24"],patients='all')


#%%
#analysis.scatter_state(week='all',pt=['908'],feat='SHarm')
#analysis.scatter_state(week='all',pt=['908'],feat='Stim')
for ff in ['Delta','Theta','Alpha','Beta']:
    print('Plotting feature ' + ff)
    _=analysis.scatter_state(weeks=["C01","C24"],pt='all',feat=ff,circ='',plot_type='scatter')
#analysis.scatter_state(week=['C01','C23'],pt='all',feat='Alpha',circ='night',plot_type='boxplot')

#%%
#plot the week's PSDs