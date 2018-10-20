#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:50:40 2018

@author: virati
Partial Biometric RunScript. Taken from Regress_Runscript, but without the logic to do other regression approaches
This is the *GO TO* for the partial oscillatory biometric reported in Paper 2 - cortical response
"""

#DBSpace libraries and sublibraries here
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys')
import DBSpace as dbo
from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect

#import BR_DataFrame as BRDF # Library for taking in BR data from the folder structure and populating phases
#from ClinVect import CFrame # Cliical frame data
#from DSV import ORegress # Biometric regression library


# General python libraries
import scipy.signal as sig
import numpy as np

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
#Do some cleanup of the plotting space
plt.close('all')
sns.set_context('paper')

# Misc libraries
import copy
import itertools


#%%
do_pts = ['901','903','906','907','908'] # Which patients do we want to include in this entire analysis?
#do_pts=['901']

test_scale = 'HDRS17' # Which scale are we using as the measurement of the depression state?
''' DETRENDING
Which detrending scheme are we doing
This is important. Block goes into each patient and does zero-mean and linear detrend across time
None does not do this
All does a linear detrend across all concatenated observations. This is dumb and should not be done. Will eliminate this since it makes no sense
'''

do_detrend = 'Block' 


#%%
# Now we set up our DBSpace environment
ClinFrame = ClinVect.CFrame(norm_scales=True)
BRFrame = BRDF.BR_Data_Tree()

# Run our main sequence for populating the BRFrame
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_july.npy')
BRFrame.check_empty_phases() # Check to see if there are any empty phases. This should be folded into full_sequence soon TODO


#%%
# This sets up the regression
analysis = ORegress(BRFrame,ClinFrame)
analysis.split_validation_set(do_split = True) #Split our entire dataset into a validation and training set
analysis.O_feat_extract() #Do a feature extraction of our entire datset


#%%

dorsac = True
print('DOING CV RIDGE REGRESSION NOW....................................................................')
#all_pairs = list(itertools.product(all_pts,all_pts))
#all_pairs = [cc for cc in all_pairs if cc[0] != cc[1]]

all_pairs = list(itertools.combinations(do_pts,3))
#if you want to remove DBS905:
#all_pairs = list(itertools.combinations(['901','903','906','907','908'],3))

num_pairs = len(list(all_pairs))
coeff_runs = [0] * num_pairs
summ_stats_runs  = [0] * num_pairs

all_model_pairs = list(all_pairs)
#%%
for run,pt_pair in enumerate(all_model_pairs):
    print(pt_pair)
    analysis.O_regress(method='ENR_Osc',doplot=False,avgweeks=True,ignore_flags=False,circ='day',scale=test_scale,lindetrend=do_detrend,train_pts=pt_pair)
    coeff_runs[run] = analysis.O_models(plot=False,models=['ENR_Osc'])
    summ_stats_runs[run] = analysis.Clinical_Summary('ENR_Osc',plot_indiv=False,ranson=dorsac,doplot=False)
    #analysis.shuffle_summary('RIDGE')


#%%
# Figures time
    
#summary stats
# this one plots the permutation based results
#plt.figure()
#plt.suptitle('Permutation')
#plt.subplot(2,1,1)
#plt.hist(np.array([cc['DProd']['Dot']/cc['DProd']['Perfect'] for cc in summ_stats_runs]))
#plt.title('Correlations distribution')
#
#plt.subplot(2,1,2)
#plt.hist(np.array([cc['DProd']['pval'] for cc in summ_stats_runs]))
#plt.title('p-values distribution')

#%%
#left_coeffs = np.median(np.array([cc['Left'] for cc in coeff_runs]),axis=0)
#right_coeffs = np.median(np.array([cc['Right'] for cc in coeff_runs]),axis=0)
    
    
def plot_coeffs(coeff_runs):
    left_coeffs = np.array([cc['Left'] for cc in coeff_runs])
    right_coeffs = np.array([cc['Right'] for cc in coeff_runs])
    
    
    #Plot our coefficients
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.mean(left_coeffs,axis=0))
    for bb in range(5):
        plt.scatter(bb*np.ones((num_pairs)),left_coeffs[:,bb],alpha=0.5,s=200)
    
    plt.ylim((-0.05,0.05))
    plt.xlim((-0.1,4.1))
    plt.hlines(0,0,5)
    
    plt.subplot(1,2,2)
    plt.plot(np.mean(right_coeffs,axis=0))
    for bb in range(5):
        plt.scatter(bb*np.ones((num_pairs)),right_coeffs[:,bb],alpha=0.5,s=200)
    
    plt.ylim((-0.05,0.05))
    plt.xlim((-0.1,4.1))
    plt.hlines(0,0,5)
    plt.suptitle('Mean Coefficients')
    
    return left_coeffs, right_coeffs
left_coeffs,right_coeffs = plot_coeffs(coeff_runs)

#%%
    
def plot_corr(corr_measure='PearsCorr'):
    plt.figure()
    
    plt.subplot(2,1,1)
    plt.hist(np.array([cc[corr_measure][0] for cc in summ_stats_runs]))
    #plt.annotate(xy=([summ_stats_runs[mod]['SpearCorr'] for mod in range(20)],[1 for mod in range(20)]),text='test')
    plt.title('Correlations distribution')
    
    plt.subplot(2,1,2)
    plt.hist(np.array([cc[corr_measure][1] for cc in summ_stats_runs]))
    plt.title('p-values distribution')
    plt.suptitle(corr_measure)
    
    #Select the final model
    ## THIS ONE PLOTS OUR SPEARMAN CORR
    plt.figure()
    models_perf = np.array([cc[corr_measure][0] for cc in summ_stats_runs])
    plt.plot(np.arange(len(all_model_pairs)),models_perf)
    plt.xticks(np.arange(len(all_model_pairs)),all_model_pairs,rotation=90)
    plt.suptitle(corr_measure)
plot_corr('SpearCorr')
#%%


#%%
# Rerun the best model
choose_best = False
if choose_best:
    #which one has max performance?
    #idx_max = np.argmax(models_perf)
    #max_perf = models_perf[idx_max]
    #best_model = all_model_pairs[idx_max]
    for run,pt_pair in enumerate([best_model]):
        print(pt_pair)
        analysis.O_regress(method='ENR_Osc',doplot=False,avgweeks=True,ignore_flags=False,circ='day',scale=test_scale,lindetrend=do_detrend,train_pts=pt_pair)
        coeff_runs[run] = analysis.O_models(plot=False,models=['ENR_Osc'])
        summ_stats_runs[run] = analysis.Clinical_Summary('ENR_Osc',plot_indiv=False,ranson=dorsac,doplot=False)
    
    
    analysis.Model['FINAL']['Model'] =  copy.deepcopy(analysis.Model['ENR_Osc']['Model'])
    analysis.Model['RANDOM']['Model'] = copy.deepcopy(analysis.Model['ENR_Osc']['Model'])
    #analysis.O_regress(method='RIDGE',doplot=False,avgweeks=True,ignore_flags=False,circ='day',scale=test_scale,lindetrend=do_detrend,train_pts=['903','906','907'],finalWrite=True)

else:
    analysis.Model['FINAL']['Model'] =  copy.deepcopy(analysis.Model['ENR_Osc']['Model'])
    analysis.Model['RANDOM']['Model'] = copy.deepcopy(analysis.Model['ENR_Osc']['Model'])
     
    
    #Here, we force the coefficients to be the median or mean or whatever
    analysis.Model['FINAL']['Model'].coef_ = np.hstack((np.mean(left_coeffs,axis=0),np.mean(right_coeffs,axis=0)))
    #analysis.Model['FINAL']['Model'].coef_ = np.array([0.1,0,2,0.3,0,4,0.1,0.2,-0.3,0.1])
    

#%%
#Plot out final model's coefficients
#Plot our coefficients
avgleft_coeffs = analysis.Model['FINAL']['Model'].coef_[:5]
avgright_coeffs = analysis.Model['FINAL']['Model'].coef_[5:]
plt.figure()
plt.subplot(1,2,1)
plt.plot((avgleft_coeffs))


plt.ylim((-0.05,0.05))
plt.xlim((-0.1,4.1))
plt.hlines(0,0,5)

plt.subplot(1,2,2)
plt.plot((avgright_coeffs))

plt.ylim((-0.05,0.05))
plt.xlim((-0.1,4.1))
plt.hlines(0,0,5)

#%%
# What do our prediction curves look like with the final model?
_ = analysis.Model_Validation(method='FINAL',do_detrend='Block',randomize=0.7,do_plots=True,show_clin=False)

#%%

# Now we apply this model to the validation set to see what's what
#We should have a model right now. Now we're going to do a final validation set on ALL PATIENTS using the held out validation set
aucs = []
null_distr = []
_ = analysis.Model_Validation(method='FINAL',do_detrend='Block',randomize=0.7,do_plots=True,show_clin=True)

#%%
#Here we're going to do iterations of randomness
n_iterations = 100
if n_iterations > 1: plot_algos = False
for ii in range(n_iterations):
    analysis.Model['RANDOM']['Model'].coef_ = np.random.uniform(-0.04,0.04,size=(1,10));
    algo_list,null_algo = analysis.Model_Validation(method='FINAL',do_detrend='Block',do_plots=plot_algos,randomize=0.7);
    aucs.append(algo_list)
    null_distr.append(null_algo)
    
aucs = np.array(aucs)

#%%
# Display the surrogate results
plt.figure()
#plt.hist(aucs[:,2],label='Nulls')
#plt.hist(aucs[:,0],label='HDRS')
#plt.hist(aucs[:,1],label='Candidate')
#plt.hist(aucs[:,3],label='RandMod')
#plt.subplot(2,1,1)

#plt.hist(aucs[:,2:4],stacked=False,color=['green','purple'],label=['Null','RandM'],bins=bins)
bins = np.linspace(0,0.5,40)
plt.hist(aucs[:,2],stacked=False,color=['purple'],label=['CoinFlip Null'],bins=bins)
plt.hist(null_distr,stacked=False,color=['green'],label=['Sparse Null'],bins=bins)
plt.xlim((0,0.5))

line_height = 600

plt.vlines(np.median(aucs[:,0]),0,line_height,color='red',label='HDRS',linewidth=5)
hdrs_sem = np.sqrt(np.var(aucs[:,0])) / np.sqrt(n_iterations)
plt.hlines(line_height,np.median(aucs[:,0]) - hdrs_sem,np.median(aucs[:,0]) + hdrs_sem,color='red')

plt.vlines(np.median(aucs[:,1]),0,line_height,color='blue',label='Candidate')
cb_sem = np.sqrt(np.var(aucs[:,1])) / np.sqrt(n_iterations)
plt.hlines(line_height,np.median(aucs[:,1]) - cb_sem,np.median(aucs[:,1]) + cb_sem,color='blue')



#plt.vlines(np.median(aucs[:,4]),0,line_height,color='yellow',label='Proposed')
#pc_sem = np.sqrt(np.var(aucs[:,4])) / np.sqrt(n_iterations)
#plt.hlines(20,np.median(aucs[:,4]) - pc_sem,np.median(aucs[:,4]) + pc_sem,color='yellow')
#
#plt.vlines(np.median(aucs[:,5]),0,line_height,color='cyan',label='CenterOff')
#co_sem = np.sqrt(np.var(aucs[:,5])) / np.sqrt(n_iterations)
#plt.hlines(20,np.median(aucs[:,5]) - co_sem,np.median(aucs[:,5]) + co_sem,color='cyan')


#How many above?
phdrs = np.sum(aucs[:,2] > np.median(aucs[:,0])) / n_iterations # Our HDRS
pcb= np.sum(aucs[:,2] > np.median(aucs[:,1])) / n_iterations # our Candidate
#pmin = np.sum(aucs[:,2] > np.median(aucs[:,4])) / n_iterations
print('HDRS over coin flip' + str(phdrs))
print('Candidate over coin flip' + str(pcb))

psparsehdrs = np.sum(null_distr > np.median(aucs[:,0])) / n_iterations
psparsecb = np.sum(null_distr> np.median(aucs[:,1])) / n_iterations 
#print(phdrs)
#print(pcb)

print('HDRS over sparse' + str(psparsehdrs))
print('Candidate over sparse' + str(psparsecb))



plt.legend()
