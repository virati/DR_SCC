#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:32:00 2020

@author: virati
This is the script that runs the patient-CV SCC Readout training and testing.
"""

from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, decoder
from DBSpace.readout.BR_DataFrame import BR_Data_Tree

# Misc libraries
import copy
import pickle


#Debugging
import ipdb

#
## MAJOR PARAMETERS for our partial biometric analysis
do_pts = ['901','903','905','906','907','908'] # Which patients do we want to include in this entire analysis?
test_scale = 'pHDRS17' # Which scale are we using as the measurement of the depression state?

''' DETRENDING
Which detrending scheme are we doing
This is important. Block goes into each patient and does zero-mean and linear detrend across time
None does not do this
All does a linear detrend across all concatenated observations. This is dumb and should not be done. Will eliminate this since it makes no sense
'''

# Initial
# Now we set up our DBSpace environment
#ClinFrame = ClinVect.CFrame(norm_scales=True)
ClinFrame = ClinVect.CStruct()
#BRFrame = BRDF.BR_Data_Tree(preFrame='Chronic_Frame.pickle')
BRFrame = pickle.load(open('/home/virati/Dropbox/Data/Chronic_FrameMay2020.pickle',"rb"))

#%%
main_readout = decoder.weekly_decoderCV(BRFrame=BRFrame,ClinFrame=ClinFrame,pts=do_pts,clin_measure=test_scale,algo='ENR')
main_readout.filter_recs(rec_class='main_study')
main_readout.split_train_set(0.6)

#%%
main_readout.train_setup()
main_readout.train_model()
main_readout.plot_decode_CV()

#%%
main_readout.test_setup()
main_readout.test_model()
#%%
main_readout.plot_test_stats()
#%%
main_readout.plot_test_regression_figure()

#%%
# Now we move on to the classifier analysis
