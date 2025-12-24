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
sns.set(font_scale=2)
sns.set_style('white')

import copy

#Custom libraries
from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, DSV

from DBSpace.readout import Impedances as Zs

import scipy.stats as stats


#%%
BRFrame = BRDF.BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_july.npy')

#%%
# Here we'll plot general distributions
ClinFrame = ClinVect.CFrame(norm_scales=True)
BRFrame = BRDF.BR_Data_Tree()

# Run our main sequence for populating the BRFrame
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_july.npy')
BRFrame.check_empty_phases() # Check to see if there are any empty phases. This should be folded into full_sequence soon TODO

#%%
analysis = DSV.ORegress(BRFrame,ClinFrame)

analysis.split_validation_set(do_split = True) #Split our entire dataset into a validation and training set
analysis.O_feat_extract() #Do a feature extraction of our entire datset
#%%
#print the features available
#print(analysis.YFrame.file_meta[0]['FeatVect'].keys())
analysis.plot_band_distributions(band='fSlope')
analysis.plot_timecourse(feat='Clock')
analysis.plot_timecourse(feat='GCratio',ylim=(-2,2))
#%%
# do any writeouts here
clock_tcourse = analysis.get_timecourse(feat='Clock')

ratio_tcourse = analysis.get_timecourse(feat='GCratio')

#%%

## Can do regressions with impedance here
print('Doing Zs')
Z_lib = Zs.Z_class()
z_tcourse = Z_lib.get_recZs()

'''
The two timecourse we care about have very different data structures. This is annoying, but what happens when merging methods from various years together
'''

pt_list = ['901','903','905','906','907','908']
ch_list = ['Left','Right']

plot_feat = ratio_tcourse

for pp,pt in enumerate(pt_list):
    plt.figure()
    plt.title(pt)
    for cc,ch in enumerate(ch_list):
        plt.subplot(1,2,cc+1)
        zs = (np.diff(z_tcourse[ch][pp,:,:],axis=0)).T
        clock = (plot_feat[pt][:,cc]).reshape(-1,1)
        tidxs = np.arange(0,28)
        
        #Normalize stuff here
        nonan_idxs = ~np.isnan(zs)
        zs = (zs - np.mean(zs[nonan_idxs])) / np.max(np.abs(zs[nonan_idxs]))
        clock = (clock - np.mean(clock[nonan_idxs])) / np.max(np.abs(clock[nonan_idxs]))
        
        plt.plot(tidxs,zs)
        plt.plot(tidxs,clock)
        
        #find nan in zs
        
        
        corr_stat = stats.spearmanr(zs[nonan_idxs],clock[nonan_idxs])
        print(corr_stat)



