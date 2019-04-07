#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:50:40 2019

@author: virati
Local biometric script

Partial Biometric REWORK
"""
#DBSpace libraries and sublibraries here
from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, DSV

# General python libraries
import scipy.signal as sig
import numpy as np

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
#Do some cleanup of the plotting space
plt.close('all')
sns.set_context('paper')
sns.set_style('white')
sns.set(font_scale=4)

# Misc libraries
import copy
import itertools
import scipy.stats as stats

#%%


ClinFrame=ClinVect.CFrame(norm_scales=True)
BRFrame = BRDF.BR_Data_Tree()

BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_july.npy')
BRFrame.check_empty_phases()

local_ro = DSV.ORegress

