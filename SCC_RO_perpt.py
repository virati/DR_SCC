#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:27:48 2020

@author: virati

Patient specific readouts for a *PREDICTIVE* model without need for parsimony
"""

from DBSpace.readout import BR_DataFrame as BRDF
from DBSpace.readout import ClinVect, decoder
from DBSpace.readout.BR_DataFrame import BR_Data_Tree
from DBSpace import nestdict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
import numpy as np

# Misc libraries
import copy
import pickle


#Debugging
import ipdb

## MAJOR PARAMETERS for our partial biometric analysis
test_scale = 'pHDRS17' # Which scale are we using as the measurement of the depression state?
do_pts = ['901','903','905','906','907','908'] # Which patients do we want to include in this entire analysis?

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
do_shuffled_null = False

pt_coeff = nestdict()
for do_pt in do_pts:
    main_readout = decoder.base_decoder(BRFrame = BRFrame,ClinFrame = ClinFrame,pts=do_pt,clin_measure=test_scale)
    main_readout.filter_recs(rec_class='main_study')
    main_readout.split_train_set(0.6)
    
    null_slopes = main_readout.model_analysis(do_null=True,n_iter=100)
    main_slope = main_readout.model_analysis()
    
    print(np.sum(null_slopes > main_slope[0])/100)
    
    pt_coeff[do_pt] = main_readout.decode_model.coef_

# Plot all the coeffs
plt.figure()
[plt.plot(pt_coeff[pt],alpha=0.7,linewidth=5,label=pt) for pt in do_pts]
plt.vlines(4.5,-1,1,linewidth=20)
plt.hlines(0,0,9,linestyle='dotted')
plt.xticks(range(10),[r'$\delta$',r'$\theta$',r'$\alpha$',r'$\beta*$',r'$\gamma1$',r'$\delta$',r'$\theta$',r'$\alpha$',r'$\beta*$',r'$\gamma1$'])
plt.ylim((-0.15,0.15))
plt.legend()