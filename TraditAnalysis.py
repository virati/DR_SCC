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

import sys
#Need to important DBSpace and DBS_Osc Libraries
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/SigProc/CFC-Testing/Python CFC/')
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo
from DBS_Osc import nestdict

import seaborn as sns
import seaborn as sns
#sns.set_context("paper")

sns.set(font_scale=4)
sns.set_style("white")
#Bring in our data first

BRFrame = BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_March.npy')

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
#%%
#First thing is the per-patient weekly averages plotted for left and right
_ = analysis.mean_psds(weeks=["C01","C24"],patients='all')


#%%
#analysis.scatter_state(week='all',pt=['908'],feat='SHarm')
#analysis.scatter_state(week='all',pt=['908'],feat='Stim')
ks_stats = nestdict()
for pt in ['901','903','905','906','907','908']:
    for ff in ['Delta','Theta','Alpha','Beta','Gamma']:
        print('Plotting feature ' + ff)
        _,ks_stats[pt][ff]=analysis.scatter_state(weeks=["C01","C24"],pt=pt,feat=ff,circ='',plot_type='scatter')
#analysis.scatter_state(week=['C01','C23'],pt='all',feat='Alpha',circ='night',plot_type='boxplot')


for ii in plt.get_fignums():
    plt.figure(ii)
    
    plt.savefig('/tmp/' + str(ii) + '.svg')
#%%
#plot the week's PSDs