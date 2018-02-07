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

analysis = OBands(BRFrame)
analysis.feat_extract()

#%%
analysis.scatter_state(week=['C01','C05','C15','C24'],pt=['901','905','908'],feat='Alpha')