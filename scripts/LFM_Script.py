#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:02:49 2018

@author: virati
Script to actually do the LFM model
"""

from DSV import LFM
import BR_DataFrame as BRDF
from ClinVect import CFrame

BRFrame = BRDF.BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_july.npy')
ClinFrame = CFrame(norm_scales=True)
#%%
model = LFM(BRFrame,ClinFrame,patient='901')