#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:24:26 2018

@author: virati
Traditional Analyses
"""

from OBands import *
from BR_DataFrame import *

#Bring in our data first
BRFrame = BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame.npy')

#do a check to see if any PSDs are entirely zero; bad sign
BRFrame.check_meta()


#%%
#Move forward with analysis
analysis = OBands(BRFrame)
analysis.feat_extract()

#%%
#analysis.scatter_state(week='all',pt=['908'],feat='SHarm')
#analysis.scatter_state(week='all',pt=['908'],feat='Stim')
analysis.scatter_state(week=['C01','C23'],pt='all',feat='Alpha',circ='day',plot_type='scatter')
analysis.scatter_state(week=['C01','C23'],pt='all',feat='Alpha',circ='night',plot_type='scatter')