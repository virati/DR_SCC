#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:24:26 2018

@author: virati
Traditional Analyses
"""

from DBSpace.readout.BR_DataFrame import BR_Data_Tree
from collections import defaultdict

import matplotlib.pyplot as plt
plt.close('all')

import sys
#Need to important DBSpace and DBS_Osc Libraries
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/SigProc/CFC-Testing/Python CFC/')
import DBSpace as dbo
from DBSpace import nestdict

import seaborn as sns
#sns.set_context("paper")

sns.set(font_scale=4)
sns.set_style("white")
#Bring in our data first

import numpy as np
import pickle
from matplotlib.patches import Rectangle, Circle
from numpy import ndenumerate


#%%
BRFrame = pickle.load(open('/home/virati/Chronic_Frame.pickle',"rb"))
#BRFrame.full_sequence(data_path='/home/virati/Dropbox/Chronic_Frame_June.npy')

#do a check to see if any PSDs are entirely zero; bad sign
BRFrame.check_meta()


#%%
#Move forward with traditional oscillatory band analysis
from OBands import *
analysis = OBands(BRFrame)
analysis.feat_extract(do_corrections=True)

##PLOTS

#focus_feats =['Delta','Theta','Alpha','Beta']
#Now we're ready for the plots
#%%
#First thing is the per-patient weekly averages plotted for left and right
#_ = analysis.mean_psds(weeks=["C01","C24"],patients='all')


#%%
# Do comparison of two timepoints here
#analysis.scatter_state(week='all',pt=['908'],feat='SHarm')
#analysis.scatter_state(week='all',pt=['908'],feat='Stim')
ks_stats = nestdict()
pts = ['901','903','905','906','907','908']
bands = ['Delta','Theta','Alpha','Beta*','Gamma1']
all_feats = ['L-' + band for band in bands] + ['R-' + band for band in bands]

circ = 'day'
week_distr = nestdict()
for pt in pts:
    for ff in bands:
        print('Computing ' + ' ' + ff)
        _,ks_stats[pt][ff],week_distr[pt][ff] =analysis.scatter_state(weeks=["C01","C24"],pt=pt,feat=ff,circ=circ,plot=False,plot_type='scatter',stat='ks')
        #plt.title('Plotting feature ' + ff)
    #analysis.scatter_state(week=['C01','C23'],pt='all',feat='Alpha',circ='night',plot_type='boxplot')

# Make our big stats 2d grid for all features across all patients
K_stats = np.array([[[ks_stats[pt][band][side][0] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
P_val = np.array([[[ks_stats[pt][band][side][1] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
pre_feat_vals = np.array([[[week_distr[pt][band][side]['C01'] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
post_feat_vals = np.array([[[week_distr[pt][band][side]['C24'] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
#%%
    
def get_distr_change(patient,band,plot=True):
    
    pp = pts.index(patient)
    ff = all_feats.index(band)
    if plot:
        plt.figure()
        sns.violinplot(y=pre_feat_vals[pp,ff])
        sns.violinplot(y=post_feat_vals[pp,ff],color='red',alpha=0.3)

    return (np.mean(post_feat_vals[pp,ff]) - np.mean(pre_feat_vals[pp,ff]))

change_grid = np.zeros((len(pts),len(all_feats)))
for pp,pt in enumerate(pts):
    for ff,freq in enumerate(all_feats):
        change_grid[pp,ff] = get_distr_change(pt,freq,plot=False)



#%%
def all_ensemble_change(plot=True):
    
    pre_flat_list = []
    post_flat_list = []
    
    for ff,freq in enumerate(all_feats):
        pre_flat_list.append([item for sublist in pre_feat_vals[:,ff] for item in sublist])
        post_flat_list.append([item for sublist in post_feat_vals[:,ff] for item in sublist])
        
    if plot:
        plt.figure()
        ax = sns.violinplot(data=pre_flat_list,color='blue')
        ax = sns.violinplot(data=post_flat_list,color='red',alpha=0.3)
        
        
        plt.setp(ax.collections,alpha=0.3)

all_ensemble_change()


#%%
# want to marginalize across patients and compare all band observations in C01 with all from C24
def get_ensemble_change(band,plot=True):
    ff = all_feats.index(band)
    pre_flat_list = [item for sublist in pre_feat_vals[:,ff] for item in sublist]
    post_flat_list = [item for sublist in post_feat_vals[:,ff] for item in sublist]
    if plot:
        plt.figure()
        ax = sns.violinplot(x=pre_flat_list)
        ax = sns.violinplot(x=post_flat_list,color='red',alpha=0.3)
        plt.title(band)
        plt.xlim(-10,10)
        plt.setp(ax.collections,alpha=0.3)
    #do some stats here
    stat_check = stats.ks_2samp(pre_flat_list,post_flat_list)
    return {'diff':(np.mean(pre_flat_list) - np.mean(post_flat_list)),'var':(np.var(pre_flat_list) - np.var(post_flat_list)),'ks':stat_check}


def get_ensemble_dist(band,plot=True):
    ff = all_feats.index(band)
    pre_flat_list = [item for sublist in pre_feat_vals[:,ff] for item in sublist]
    post_flat_list = [item for sublist in post_feat_vals[:,ff] for item in sublist]
    
    return pre_flat_list, post_flat_list

test = nestdict()#np.zeros(len(all_feats))

for ff,freq in enumerate(all_feats):
    test[ff] = get_ensemble_change(freq,plot=True)
#%%
list_of_p = [test[num]['ks'][1] for num,band in enumerate(all_feats)]
plt.figure()
#plt.pcolormesh(list_of_p)
plt.plot(list_of_p)
plt.hlines(0.005,0,10)
plt.hlines(0.05,0,10,linestyle='dotted')
        
#%%
plt.figure()
plt.pcolormesh(change_grid,cmap=plt.cm.get_cmap('viridis'))
plt.colorbar()
plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
plt.yticks(np.arange(6)+0.5,pts)
plt.title('Band Change over Timepoints')

ax = plt.axes()

for index,value in ndenumerate(P_val):
    if value < (0.05/10):
        if change_grid[index] > 0:
            usecolor='red'
        else:
            usecolor='blue'
            
        print(index)
        ax.add_patch(Rectangle((index[1], index[0]), 1, 1, fill=False, edgecolor='red', lw=5))
        ax.add_patch(Circle((index[1]+0.5, index[0]+0.5), 0.2, fill=True, facecolor=usecolor, edgecolor='white', lw=2))
plt.show()




plt.figure()
#plt.subplot(2,1,1)
plt.pcolormesh(K_stats);plt.colorbar()
plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
plt.yticks(np.arange(6)+0.5,pts)
plt.title('Using K Stat variable')

plt.figure()
#plt.subplot(2,1,2)
#plt.pcolormesh((P_val < (0.05)).astype(np.int));plt.colorbar()
#P_val[P_val > (0.05/10)] = 1
plt.pcolormesh((P_val < (0.05/10)).astype(np.float32),cmap=plt.get_cmap('Set1_r'));plt.colorbar();
plt.yticks(np.arange(6)+0.5,pts)
plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
plt.title('Significant P-value')
#%%
P_val_num = (P_val<(0.05/10)).astype(np.int)
sig_feats = P_val_num.sum(axis=0) >= 5
sig_feats_name = (d for d,s in zip(bands,sig_feats))
#print(np.array(bands)[sig_feats.astype(np.int)])
    
for ii in plt.get_fignums():
    plt.figure(ii)
    
    plt.savefig('/tmp/' + str(ii) + circ + '.svg')

#%%
#Do day-night comparisons across all bands
#analysis.day_vs_nite()