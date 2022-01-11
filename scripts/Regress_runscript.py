#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:39:35 2018

@author: virati
This file does all the regressions on the oscillatory states over chronic timepoints
USE THIS to look at OLS for all recordings as well as CV_EN
# Main script for thesis work
This is getting obsolete!!
Pretty much only use this for the old school OLS/Ridge/LASSO approaches
"""

#DBSpace libraries and sublibraries here
from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, DSV
from DBSpace.readout.BR_DataFrame import BR_Data_Tree

#import BR_DataFrame as BRDF
#from ClinVect import CFrame


# General python libraries
import scipy.signal as sig
import numpy as np

#Plotting libraries
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns
sns.set_context('paper')


# Misc tools
import copy
import itertools
import pickle

from DBSpace.readout import DSV


#%%
#This sets up our clinical frame for the regression

print('Loading Clinical and BrainRadio Frames...')
ClinFrame = ClinVect.CFrame(norm_scales=True)

BRFrame = pickle.load(open('/home/virati/Chronic_Frame.pickle',"rb"))

print('Starting the readout analysis...')
analysis = DSV.ORegress(BRFrame,ClinFrame)

#%%
analysis.split_validation_set(do_split = True)
analysis.O_feat_extract()

all_pts = ['901','903','905','906','907','908']

#%%

regr_type = 'RIDGE'
test_scale = 'HDRS17'
do_detrend='Block'

ranson = True
if regr_type == 'OLSnite':
    circ = 'night'
    print('DOING OLS' + circ + ' REGRESSION NOW....................................................................')
    analysis.O_regress(method='OLS',doplot=True,inpercent=0.7,avgweeks=False,circ=circ,scale=test_scale,lindetrend=do_detrend)
    analysis.O_models(plot=True,models=['OLS'+circ])
    analysis.Clinical_Summary('OLS'+circ,ranson=ranson,plot_indiv=False)
    analysis.shuffle_summary('OLS'+circ)
    
elif regr_type == 'OLSday':
    circ = 'day'
    print('DOING OLS' + circ + ' REGRESSION NOW....................................................................')
    analysis.O_regress(method='OLS',doplot=True,inpercent=0.7,avgweeks=False,circ=circ,lindetrend=do_detrend)
    analysis.O_models(plot=True,models=['OLS'+circ])
    analysis.Clinical_Summary('OLS'+circ,ranson=ranson)
    analysis.shuffle_summary('OLS'+circ)

elif regr_type == 'OLSall':
    circ = ''
    print('DOING OLS' + circ + ' REGRESSION NOW....................................................................')
    analysis.O_regress(method='OLS',doplot=True,inpercent=0.7,avgweeks=False,circ=circ,lindetrend=do_detrend)
    analysis.O_models(plot=True,models=['OLS'+circ])
    analysis.Clinical_Summary('OLS'+circ,ranson=ranson)
    analysis.shuffle_summary('OLS'+circ)

#NO MORE RANSAC ON THE BIOMETRIC MODEL, JUST ON THE PERFORMANCE
#circ = 'day'
#print('DOING RANSAC ' + circ + ' REGRESSION NOW....................................................................')
#analysis.O_regress(method='RANSAC',doplot=True,inpercent=0.7,avgweeks=False,ranson=False,plot_indiv=False,circ=circ)
#analysis.O_models(plot=True,models=['RANSAC'])
#analysis.Clinical_Summary('RANSAC',ranson=False)
#analysis.shuffle_summary('RANSAC')




elif regr_type == 'RIDGE':
    dorsac = True
    print('DOING RIDGE REGRESSION NOW....................................................................')
    #analysis.O_regress(method='OLS',doplot=True,inpercent=0.6,avgweeks=True)
    #analysis.O_regress(method='OLS',doplot=True,inpercent=0.6,avgweeks=True,ignore_flags=True)
    analysis.O_regress(method='RIDGE',doplot=True,avgweeks=True,ignore_flags=False,circ='',scale=test_scale,lindetrend=do_detrend,finalWrite=True,train_pts=['901','903'])
    analysis.O_models(plot=True,models=['RIDGE'])
    analysis.Clinical_Summary('RIDGE',plot_indiv=True,ranson=dorsac)
    analysis.shuffle_summary('RIDGE')
    #plt.figure();plt.hist(analysis.Model['RIDGE']['Performance']['DProd']['Distr'])
    #print(np.sum(analysis.Model['RIDGE']['Performance']['DProd']['Distr'] > analysis.Model['RIDGE']['Performance']['DProd']['Dot'])/len(analysis.Model['RIDGE']['Performance']['DProd']['Distr']))

    
    analysis.Model['FINAL']['Model'] =  copy.deepcopy(analysis.Model['RIDGE']['Model'])
    analysis.Model['RANDOM']['Model'] = copy.deepcopy(analysis.Model['RIDGE']['Model'])

elif regr_type == 'LASSO':
    dorsac = False
    print('DOING LASSO REGRESSION NOW....................................................................')
    analysis.O_regress(method='LASSO',doplot=True,avgweeks=True,ranson=dorsac,ignore_flags=False,circ='night',scale=test_scale,lindetrend=do_detrend)
    analysis.O_models(plot=True,models=['LASSO'])
    analysis.Clinical_Summary('LASSO',plot_indiv=False,ranson=dorsac)
    analysis.shuffle_summary('LASSO')
    
    
elif regr_type == 'CV_RIDGE':
    dorsac = True
    print('DOING CV RIDGE REGRESSION NOW....................................................................')
    #all_pairs = list(itertools.product(all_pts,all_pts))
    #all_pairs = [cc for cc in all_pairs if cc[0] != cc[1]]
    
    all_pairs = list(itertools.combinations(['901','903','905','906','907','908'],3))
    
    num_pairs = len(list(all_pairs))
    coeff_runs = [0] * num_pairs
    summ_stats_runs  = [0] * num_pairs
    
    all_model_pairs = list(all_pairs)
    
    for run,pt_pair in enumerate(all_model_pairs):
        print(pt_pair)
        analysis.O_regress(method='RIDGE',doplot=False,avgweeks=True,ignore_flags=False,circ='day',scale=test_scale,lindetrend=do_detrend,train_pts=pt_pair)
        coeff_runs[run] = analysis.O_models(plot=False,models=['RIDGE'])
        summ_stats_runs[run] = analysis.Clinical_Summary('RIDGE',plot_indiv=False,ranson=dorsac,doplot=False)
        #analysis.shuffle_summary('RIDGE')
        
    #summary stats
    plt.figure()
    plt.suptitle('Permutation')
    plt.subplot(2,1,1)
    plt.hist(np.array([cc['DProd']['Dot']/cc['DProd']['Perfect'] for cc in summ_stats_runs]))
    plt.title('Correlations distribution')
    
    plt.subplot(2,1,2)
    plt.hist(np.array([cc['DProd']['pval'] for cc in summ_stats_runs]))
    plt.title('p-values distribution')
    
    plt.figure()
    plt.suptitle('Spearman')
    plt.subplot(2,1,1)
    plt.hist(np.array([cc['SpearCorr'][0] for cc in summ_stats_runs]))
    #plt.annotate(xy=([summ_stats_runs[mod]['SpearCorr'] for mod in range(20)],[1 for mod in range(20)]),text='test')
    plt.title('Correlations distribution')
    
    plt.subplot(2,1,2)
    plt.hist(np.array([cc['SpearCorr'][1] for cc in summ_stats_runs]))
    plt.title('p-values distribution')
    
    #left_coeffs = np.median(np.array([cc['Left'] for cc in coeff_runs]),axis=0)
    #right_coeffs = np.median(np.array([cc['Right'] for cc in coeff_runs]),axis=0)
    left_coeffs = np.array([cc['Left'] for cc in coeff_runs])
    right_coeffs = np.array([cc['Right'] for cc in coeff_runs])
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.median(left_coeffs,axis=0))
    for bb in range(5):
        plt.scatter(bb*np.ones((num_pairs)),left_coeffs[:,bb],alpha=0.5,s=200)

    plt.ylim((-0.05,0.05))
    plt.xlim((-0.1,4.1))
    plt.hlines(0,0,5)
    
    plt.subplot(1,2,2)
    plt.plot(np.median(right_coeffs,axis=0))
    for bb in range(5):
        plt.scatter(bb*np.ones((num_pairs)),right_coeffs[:,bb],alpha=0.5,s=200)
    
    plt.ylim((-0.05,0.05))
    plt.xlim((-0.1,4.1))
    plt.hlines(0,0,5)
    
    #Select the final model
    plt.figure()
    plt.plot(np.arange(len(all_model_pairs)),np.array([cc['SpearCorr'][0] for cc in summ_stats_runs]))
    plt.xticks(np.arange(len(all_model_pairs)),all_model_pairs,rotation=90)
    
    #analysis.O_regress(method='RIDGE',doplot=False,avgweeks=True,ignore_flags=False,circ='day',scale=test_scale,lindetrend=do_detrend,train_pts=['903','906','907'],finalWrite=True)
    
    analysis.Model['FINAL']['Model'] =  copy.deepcopy(analysis.Model['RIDGE']['Model'])
    analysis.Model['RANDOM']['Model'] = copy.deepcopy(analysis.Model['RIDGE']['Model'])
    
    
    #Choose the coefficients we want
    #Median here
    #final_l_coefs = 
    
    analysis.Model['FINAL']['Model'].coef_ = np.hstack((np.median(left_coeffs,axis=0),np.median(right_coeffs,axis=0)))
    #analysis.Model['FINAL']['Model'].coef_ = np.vstack((final_l_coefs,final_r_coefs))


#%%
#We should have a model right now. Now we're going to do a final validation set on ALL PATIENTS using the held out validation set
aucs = []
#Here we're going to do iterations of randomness
n_iterations = 100
for ii in range(n_iterations):
    analysis.Model['RANDOM']['Model'].coef_ = np.random.uniform(-2,2,size=(1,10))
    aucs.append(analysis.Model_Validation(method='FINAL',do_detrend='None',do_plots=False,randomize=0.7))
aucs = np.array(aucs)
#%%
# EVERYTHING BELOW IS RELATED TO THE PRC-AUC curve analysis

plt.figure()
#plt.hist(aucs[:,2],label='Nulls')
#plt.hist(aucs[:,0],label='HDRS')
#plt.hist(aucs[:,1],label='Candidate')
#plt.hist(aucs[:,3],label='RandMod')
#plt.subplot(2,1,1)
plt.hist(aucs[:,2:4],stacked=False,color=['green','purple'],label=['Null','RandM'],bins=20)

plt.vlines(np.median(aucs[:,0]),0,21,color='red',label='HDRS',linewidth=5)
hdrs_sem = np.sqrt(np.var(aucs[:,0])) / np.sqrt(n_iterations)
plt.hlines(21,np.median(aucs[:,0]) - hdrs_sem,np.median(aucs[:,0]) + hdrs_sem,color='red')

plt.vlines(np.median(aucs[:,1]),0,20,color='blue',label='Candidate')
cb_sem = np.sqrt(np.var(aucs[:,1])) / np.sqrt(n_iterations)
plt.hlines(20,np.median(aucs[:,1]) - cb_sem,np.median(aucs[:,1]) + cb_sem,color='blue')


plt.vlines(np.median(aucs[:,4]),0,20,color='yellow',label='Proposed')
pc_sem = np.sqrt(np.var(aucs[:,4])) / np.sqrt(n_iterations)
plt.hlines(20,np.median(aucs[:,4]) - pc_sem,np.median(aucs[:,4]) + pc_sem,color='yellow')

plt.vlines(np.median(aucs[:,5]),0,20,color='cyan',label='CenterOff')
co_sem = np.sqrt(np.var(aucs[:,5])) / np.sqrt(n_iterations)
plt.hlines(20,np.median(aucs[:,5]) - co_sem,np.median(aucs[:,5]) + co_sem,color='cyan')


#How many above?
phdrs = np.sum(aucs[:,2] > np.median(aucs[:,0])) / n_iterations
pcb= np.sum(aucs[:,2] > np.median(aucs[:,1])) / n_iterations
pmin = np.sum(aucs[:,2] > np.median(aucs[:,4])) / n_iterations
#print(phdrs)
#print(pcb)

#Model Comparison
print('Likelihood ratio of two clinical algos: ' + str(phdrs/pcb))
print('Likelihood ratio for Proposed algo: ' + str(phdrs/pmin))

#plt.subplot(2,1,2)
#plt.hist(aucs[:,0],stacked=False,alpha=0.9)
#plt.hist(aucs[:,1],stacked=False,alpha=0.9)

#plt.legend(['HDRS','Cand','Null','RandMod'])
#plt.legend(['HDRS','Candidate','Nulls','RandMod'])

plt.legend()

#plt.bar([0,1,2,3,4],left_coeffs)
#plt.figure();plt.hist(analysis.Model['LASSO']['Performance']['DProd']['Distr'])
#print(np.sum(analysis.Model['LASSO']['Performance']['DProd']['Distr'] > analysis.Model['LASSO']['Performance']['DProd']['Dot'])/len(analysis.Model['LASSO']['Performance']['DProd']['Distr']))
#analysis.O_regress(method='RIDG_Zmis',doplot=True,inpercent=0.6)

#%%

#X,Y = analysis.get_dsgns()
#analysis.run_OLS(doplot=True)
#analysis.run_RANSAC(inpercent=0.4)


#%%
# Quick check of the bad flags to see what the thresholds should be

#%%

#This should be folded into the class

#QUICK PLOT

#ready for elastic net setup
#analysis.ENet_Construct()