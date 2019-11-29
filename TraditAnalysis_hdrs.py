#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:24:26 2018

@author: virati
Traditional Analysis, comparing extremes of HDRS17
"""

from DBSpace.readout.BR_DataFrame import BR_Data_Tree
from collections import defaultdict
import scipy.stats as stats

import matplotlib.pyplot as plt
plt.close('all')

import sys
#Need to important DBSpace and DBS_Osc Libraries
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/SigProc/CFC-Testing/Python CFC/')
import DBSpace as dbo
from DBSpace import nestdict
from DBSpace.readout import ClinVect

import seaborn as sns
#sns.set_context("paper")

sns.set(font_scale=4)
sns.set_style("white")
#Bring in our data first

import numpy as np
import pickle
from matplotlib.patches import Rectangle, Circle
from numpy import ndenumerate

## Parameters for the analysis
ks_stats = nestdict()
pts = ['901','903','905','906','907','908']
bands = ['Delta','Theta','Alpha','Beta*','Gamma1']
all_feats = ['L-' + band for band in bands] + ['R-' + band for band in bands]


# for each patient, let's find the highest and lowest HDRS17 value and the week we find it
ClinFrame = ClinVect.CFrame(norm_scales=True)

hdrs_info = nestdict()
week_labels = ClinFrame.week_labels()

for pt in pts:
    pt_hdrs_traj = [a for a in ClinFrame.DSS_dict['DBS'+pt]['HDRS17raw']][8:]
    
    hdrs_info[pt]['max']['index'] = np.argmax(pt_hdrs_traj)
    hdrs_info[pt]['min']['index'] = np.argmin(pt_hdrs_traj)
    hdrs_info[pt]['max']['week'] = week_labels[np.argmax(pt_hdrs_traj)+8]
    hdrs_info[pt]['min']['week'] = week_labels[np.argmin(pt_hdrs_traj)+8]
    
    hdrs_info[pt]['max']['HDRSr'] = pt_hdrs_traj[hdrs_info[pt]['max']['index']]
    hdrs_info[pt]['min']['HDRSr'] = pt_hdrs_traj[hdrs_info[pt]['min']['index']]
    hdrs_info[pt]['traj']['HDRSr'] = pt_hdrs_traj

    

BRFrame = pickle.load(open('/home/virati/Chronic_Frame.pickle',"rb"))
#do a check to see if any PSDs are entirely zero; bad sign
BRFrame.check_meta()


#Move forward with traditional oscillatory band analysis
from OBands import *
feat_frame = OBands(BRFrame)
feat_frame.feat_extract(do_corrections=True)

#main_readout = naive_readout(feat_frame,ClinFrame)
#%%
#week_1 = 'C01'
#week_2 = 'C24'

week_distr = nestdict()
sig_stats = nestdict()
for pt in pts:
    for ff in bands:
        # if we want to compare max vs min hdrs
        feats,sig_stats,week_distr[pt][ff] = feat_frame.compare_states([hdrs_info[pt]['max']['week'],hdrs_info[pt]['min']['week']],feat=ff,circ='day',stat='ks')
        # if we want to compare early vs late
        #feats,sig_stats[pt][ff],week_distr[pt][ff] = feat_frame.compare_states(["C01","C24"],pt=pt,feat=ff,circ='day',stat='ks')
    

depr_list = []
notdepr_list = []
aggr_week_distr = nestdict()

for ff in bands:
    aggr_week_distr[ff] = {dz:{side:[item for sublist in [week_distr[pt][ff][side][dz] for pt in pts] for item in sublist] for side in ['Left','Right']} for dz in ['depr','notdepr']}
    
    plt.figure()
    for ss,side in enumerate(['Left','Right']):
        
        print(ff + ' ' + side)
        
        plt.subplot(1,2,ss+1)
        ax = sns.violinplot(y=aggr_week_distr[ff]['depr'][side],alpha=0.2)
        plt.setp(ax.collections,alpha=0.3)
        ax = sns.violinplot(y=aggr_week_distr[ff]['notdepr'][side],color='red',alpha=0.2)
        plt.setp(ax.collections,alpha=0.3)
        print(stats.ks_2samp(np.array(aggr_week_distr[ff]['depr'][side]),np.array(aggr_week_distr[ff]['notdepr'][side])))
        
    plt.suptitle(ff + ' ' + ' min/max HDRS')

for side in ['Left','Right']:
    for ff in bands:
        depr_list.append(aggr_week_distr[ff]['depr'][side])
        notdepr_list.append(aggr_week_distr[ff]['notdepr'][side])
    
    #%%
depr_list_flat = [item for sublist in depr_list for item in sublist]
notdepr_list_flat = [item for sublist in notdepr_list for item in sublist]

# Plot them all in same plot
plt.figure()
ax = sns.violinplot(data=depr_list,color='blue')
ax = sns.violinplot(data=notdepr_list,color='red',alpha=0.3)
_ = plt.setp(ax.collections,alpha=0.3)

#%%
