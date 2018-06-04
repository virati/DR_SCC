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
pts = ['901','903','905','906','907','908']
bands = ['Delta','Theta','Alpha','Beta*','Gamma1']

circ = 'day'
for pt in pts:
    for ff in bands:
        print('Computing ' + ' ' + ff)
        _,ks_stats[pt][ff]=analysis.scatter_state(weeks=["C01","C24"],pt=pt,feat=ff,circ=circ,plot=False,plot_type='scatter')
        #plt.title('Plotting feature ' + ff)
    #analysis.scatter_state(week=['C01','C23'],pt='all',feat='Alpha',circ='night',plot_type='boxplot')

# Make our big stats 2d grid for all features across all patients
K_stats = np.array([[[ks_stats[pt][band][side][0] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
P_val = np.array([[[ks_stats[pt][band][side][1] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
    
plt.figure()
#plt.subplot(2,1,1)
plt.pcolormesh(K_stats);plt.colorbar()
plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
plt.yticks(np.arange(6)+0.5,pts)
plt.figure()
#plt.subplot(2,1,2)
#plt.pcolormesh((P_val < (0.05)).astype(np.int));plt.colorbar()
#P_val[P_val > (0.05/10)] = 1
plt.pcolormesh((P_val < (0.05/10)).astype(np.float32),cmap=plt.get_cmap('Set1_r'));plt.colorbar();
plt.yticks(np.arange(6)+0.5,pts)
plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)

    
for ii in plt.get_fignums():
    plt.figure(ii)
    
    plt.savefig('/tmp/' + str(ii) + circ + '.svg')
#%%
#Do day-night comparisons across all bands
analysis.day_vs_nite()