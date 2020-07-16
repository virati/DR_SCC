#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:20:52 2020

@author: virati
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
#%%

## MAJOR PARAMETERS for our partial biometric analysis
do_pts = ['901','903','905','906','907','908'] # Which patients do we want to include in this entire analysis?
test_scale = 'pHDRS17' # Which scale are we using as the measurement of the depression state? pHDRS17 = nHDRS (from paper) and is a patient-specific normalized HDRS

# Initial
# Now we set up our DBSpace environment
#ClinFrame = ClinVect.CFrame(norm_scales=True)
ClinFrame = ClinVect.CStruct()
#BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame.pickle')
BRFrame = pickle.load(open('/home/virati/Dropbox/Data/Chronic_FrameMay2020.pickle',"rb"))

#%%
main_readout = decoder.weakly_decoderCV_Lasso(BRFrame=BRFrame,ClinFrame=ClinFrame,pts=do_pts,clin_measure=test_scale,algo='ENR',alpha=-4,shuffle_null=False,FeatureSet='main') #main analysis is -3.4
main_readout.global_plotting = True
main_readout.filter_recs(rec_class='main_study')
main_readout.split_train_set(0.6)

#%%
main_readout.train_setup()
optimal_alpha = main_readout._path_slope_regression()
main_readout.train_model()
#%%
main_readout.plot_decode_CV()

#%%
main_readout.test_setup()
main_readout.test_model()

main_readout.plot_test_timecourse()
#%%
main_readout.plot_test_stats()
#%%
main_readout.plot_test_regression_figure()
#main_readout.plot_combo_paths()
#%%
# Now we move on to the classifier analysis
threshold_c = decoder.controller_analysis(main_readout,bin_type='threshold')
threshold_c.classif_runs()