#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 06:36:55 2018

@author: virati
Script to generate plots analysing the distribution from chronic recordings
"""

#Standard libraries
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns
sns.set_context('paper')
import copy

#Custom libraries
import DBS_Osc as dbo
import BR_DataFrame as BRDF


BRFrame = BRDF.BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_july.npy')

#%%
# Here we'll plot general distributions

band = 'Alpha'





