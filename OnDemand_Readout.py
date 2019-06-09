#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:41:52 2019

@author: virati
File to build and validate ONDEMAND SCC readouts
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

import ipdb

#%% Initial
# Now we set up our DBSpace environment
ClinFrame = ClinVect.CFrame(norm_scales=True)
#BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame_2019.pickle')
BRFrame = 

readout = DSV.DMD_RO(BRFrame,ClinFrame)
readout.default_run()