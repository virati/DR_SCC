#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:39:35 2018

@author: virati
This file does all the regressions on the oscillatory states over chronic timepoints
"""

import BR_DataFrame as BRDF
#from BR_DataFrame import *
from ClinVect import CFrame
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import seaborn as sns
sns.set_context('paper')


ClinFrame = CFrame(norm_scales=True)
#ClinFrame.plot_scale(pts='all',scale='HDRS17')
#ClinFrame.plot_scale(pts=['901'],scale='MADRS')

BRFrame = BRDF.BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_April.npy')
BRFrame.check_empty_phases()



from DSV import DSV, ORegress

analysis = ORegress(BRFrame,ClinFrame)
analysis.O_feat_extract()


#%%

regr_type = 'LASSO'
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
    analysis.O_regress(method='RIDGE',doplot=True,avgweeks=True,ignore_flags=False,circ='day',scale=test_scale,lindetrend=do_detrend)
    analysis.O_models(plot=True,models=['RIDGE'])
    analysis.Clinical_Summary('RIDGE',plot_indiv=True,ranson=dorsac)
    analysis.shuffle_summary('RIDGE')
    #plt.figure();plt.hist(analysis.Model['RIDGE']['Performance']['DProd']['Distr'])
    #print(np.sum(analysis.Model['RIDGE']['Performance']['DProd']['Distr'] > analysis.Model['RIDGE']['Performance']['DProd']['Dot'])/len(analysis.Model['RIDGE']['Performance']['DProd']['Distr']))


elif regr_type == 'LASSO':
    dorsac = False
    print('DOING LASSO REGRESSION NOW....................................................................')
    analysis.O_regress(method='LASSO',doplot=True,avgweeks=True,ranson=dorsac,ignore_flags=False,circ='night',scale=test_scale,lindetrend=do_detrend)
    analysis.O_models(plot=True,models=['LASSO'])
    analysis.Clinical_Summary('LASSO',plot_indiv=False,ranson=dorsac)
    analysis.shuffle_summary('LASSO')
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