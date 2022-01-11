#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:32:36 2018

@author: virati
AUC analysis of ensemble-level data
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sns
sns.set(font_scale=5)
sns.set_style('white')

from DBSpace import nestdict
import scipy.stats as stats



#%%
with open('/home/virati/AUCs.pickle','rb') as handle:
    get_aucs = pickle.load(handle)
auc_curves_from_run = get_aucs['Algos']
null_curves_from_run = get_aucs['Nulls']
aacus_from_run = get_aucs['AUCval']
    
test_aucs = np.array(auc_curves_from_run)
hdrs_aucs = np.array([[(r['HDRS'][0],r['HDRS'][1]) for r in run] for run in auc_curves_from_run]).reshape(-1,2)
cb_aucs = np.array([[(r['CB'][0],r['CB'][1]) for r in run] for run in auc_curves_from_run]).reshape(-1,2)
rand_aucs = np.array([[(r['Random'][0],r['Random'][1]) for r in run] for run in auc_curves_from_run]).reshape(-1,2)

#%%

mean_blah = np.linspace(0,1,100)

color_code = {'HDRS':'red','CB':'blue','Random':'green'}
hdrs_list = []
curve_lib = {key:[] for key in color_code.keys()}
algo_list = ['HDRS','CB','Random']

plt.figure()
for rr in range(8):
    for itr in range(5):
        for algo in algo_list:
            hdrs_func = interp1d(auc_curves_from_run[rr][itr][algo][1],auc_curves_from_run[rr][itr][algo][0],kind='zero')
            hdrs_do = hdrs_func(mean_blah)
            curve_lib[algo].append(hdrs_do)
            
            #pdb.set_trace()
            hdrs_rec = auc_curves_from_run[rr][itr][algo][0]
            plt.plot(mean_blah,hdrs_do,color=color_code[algo],alpha=0.1)
        
        #plt.plot(auc_curves_from_run[rr][itr]['HDRS'][1],auc_curves_from_run[rr][itr]['HDRS'][0],color='red',alpha=0.1)
        #plt.plot(auc_curves_from_run[rr][itr]['CB'][1],auc_curves_from_run[rr][itr]['CB'][0],color='blue',alpha=0.1)
        #plt.plot(auc_curves_from_run[rr][itr]['Random'][1],auc_curves_from_run[rr][itr]['Random'][0],color='green',alpha=0.1)
#%%
# Plot of AUCs with shaded errors
plt.figure()
for algo in algo_list:
    algo_res = np.array(curve_lib[algo])
    mean_auc = np.mean(algo_res,axis=0)
    std_auc = np.std(algo_res,axis=0)
    
    auc_lower = np.minimum(mean_auc + std_auc,1)
    auc_upper = np.maximum(mean_auc - std_auc,0)
    
    plt.plot(mean_blah,np.mean(algo_res,axis=0),color=color_code[algo],linewidth=2)
    plt.fill_between(mean_blah,auc_lower,auc_upper,color=color_code[algo],alpha=0.05)
plt.legend(algo_list)
        
#%%
#plot the null distribution and AUC
#stack nulls
plt.figure()
all_nulls = np.array(null_curves_from_run).reshape(-1,)

flat_aacus = [item for sublist in aacus_from_run for item in sublist]
algo_aucs = np.array(flat_aacus)

bins = np.linspace(0,0.5,25)
#plt.hist(aucs[:,2],stacked=False,color=['purple'],label=['CoinFlip Null'],bins=bins)
#plt.hist(all_nulls,stacked=False,color=['green'],label=['Sparse Null'],bins=bins)
plt.xlim((0,0.5))
line_height=50

use_normal_null = True
if use_normal_null:
    end_algo = 3
else:
    end_algo = 2
    
    
for aa, algo in enumerate(algo_list[0:end_algo]):
    plt.hist(algo_aucs[:,aa],stacked=False,color=color_code[algo],bins=bins,label=algo,alpha=0.5)
    plt.vlines(np.median(algo_aucs[:,aa]),0,line_height+10*aa,color=color_code[algo],label=algo,linewidth=5)
    algo_std = hdrs_sem = np.sqrt(np.var(algo_aucs[:,aa]))
    plt.hlines(line_height + 10 * aa,np.median(algo_aucs[:,aa]) - algo_std,np.median(algo_aucs[:,aa]) + algo_std,color=color_code[algo],linewidth=5)

plt.hist(all_nulls,stacked=False,color='green',bins=bins,label=algo,alpha=0.5)
plt.vlines(np.median(all_nulls),0,line_height+20,color='green',label='Null',linewidth=5)
algo_std = hdrs_sem = np.sqrt(np.var(all_nulls))
plt.hlines(line_height + 20,np.median(all_nulls) - algo_std,np.median(all_nulls) + algo_std,color='green',linewidth=5)

#%%
#pairwise hypothe testing
ks_results = nestdict()
for aa, algo in enumerate(algo_list[0:end_algo]):
    for bb, algo2 in enumerate(algo_list[0:end_algo]):
        ks_results[algo][algo2] = stats.ks_2samp(algo_aucs[:,aa],algo_aucs[:,bb])





#%%
#plt.hist(algo_aucs[:,0],stacked=False,color='red',bins=bins,label='HDRS')
#plt.hist(algo_aucs[:,1],stacked=False,color='blue',bins=bins,label='CB')
#plt.hist(algo_aucs[:,2],stacked=False,color='green',bins=bins,label='Uniform')
#
#
#plt.vlines(np.median(algo_aucs[:,0]),0,line_height,color='red',label='HDRS',linewidth=5)
#plt.vlines(np.median(algo_aucs[:,1]),0,line_height,color='blue',label='CB',linewidth=5)
#plt.vlines(np.median(algo_aucs[:,2]),0,line_height,color='green',label='CB',linewidth=5)
#
#hdrs_sem = np.sqrt(np.var(aucs[:,0])) / np.sqrt(n_iterations)
#plt.hlines(line_height,np.median(aucs[:,0]) - hdrs_sem,np.median(aucs[:,0]) + hdrs_sem,color='red')
