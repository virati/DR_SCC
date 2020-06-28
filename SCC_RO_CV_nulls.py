#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:07:26 2020

@author: virati
The synthetic null script for SCC_RO_CV
"""
from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, decoder
from DBSpace.readout import decoder as decoder
from DBSpace.readout.BR_DataFrame import BR_Data_Tree

# Misc libraries
import copy
import pickle


#Debugging
import ipdb

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'tab10'
plt.close('all')


## MAJOR PARAMETERS for our partial biometric analysis
do_pts = ['901','903','905','906','907','908'] # Which patients do we want to include in this entire analysis?
test_scale = 'pHDRS17' # Which scale are we using as the measurement of the depression state? pHDRS17 = nHDRS (from paper) and is a patient-specific normalized HDRS

# Initial
# Now we set up our DBSpace environment
#ClinFrame = ClinVect.CFrame(norm_scales=True)
ClinFrame = ClinVect.CStruct()
#BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame.pickle')


null_distribution = []
for ii in range(100):
    #%%
    BRFrame = pickle.load(open('/home/virati/Dropbox/Data/Chronic_FrameMay2020.pickle',"rb"))
    main_readout = decoder.weekly_decoderCV(BRFrame=BRFrame,ClinFrame=ClinFrame,pts=do_pts,clin_measure=test_scale,algo='ENR',alpha=-4,shuffle_null=True) #main analysis is -3.4
    main_readout.global_plotting = False
    main_readout.filter_recs(rec_class='main_study')
    main_readout.split_train_set(0.6)
    
    #%%
    main_readout.train_setup()
    optimal_alpha = main_readout._path_slope_regression()
    main_readout.train_model()
    #%%
    #main_readout.p/lot_decode_CV()
    
    #%%
    main_readout.test_setup()
    #main_readout.test_model()
    #get the R^2 and slope
    
    null_distribution.append(main_readout.one_shot_test())
#main_readout.plot_combo_paths()

#%%
BRFrame = pickle.load(open('/home/virati/Dropbox/Data/Chronic_FrameMay2020.pickle',"rb"))
main_readout = decoder.weekly_decoderCV(BRFrame=BRFrame,ClinFrame=ClinFrame,pts=do_pts,clin_measure=test_scale,algo='ENR',alpha=-4,shuffle_null=False) #main analysis is -3.4
main_readout.global_plotting = False
main_readout.filter_recs(rec_class='main_study')
main_readout.split_train_set(0.6)

main_readout.train_setup()
optimal_alpha = main_readout._path_slope_regression()
main_readout.train_model()

main_readout.test_setup()

#get the R^2 and slope

main_model = main_readout.one_shot_test()

#%%
for stat in ['Slope','Score']:
    plt.figure()
    null_distr = np.array([a[stat] for a in null_distribution]).squeeze()
    plt.hist(null_distr)
    plt.vlines(main_model[stat],0,10,linewidth=10)
    plt.title(stat + ' || Number above model: ' + str(np.sum(null_distr > main_model[stat])) + ' p:' + str(np.sum(null_distr > main_model[stat])/100))