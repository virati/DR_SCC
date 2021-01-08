#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:44:02 2020

@author: virati
Check input matrix for ECG
"""

from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, decoder
from DBSpace.readout import decoder as decoder
from DBSpace.readout.BR_DataFrame import BR_Data_Tree

# Misc libraries
import copy
import pickle


#Debugging
import ipdb

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'tab10'
plt.close('all')

import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')
#%%
# Load in the frame we're analysing
BRFrame = pickle.load(open('/home/virati/Dropbox/Data/Chronic_FrameDec2020_T.pickle',"rb"))


#%% Look at the PSDs for all recordings in a given patient
for pt in ['901']:
    BRFrame.plot_TS()