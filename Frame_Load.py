#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:27:32 2017

@author: virati
Quick file to load in Data Frame
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import ElasticNet, ElasticNetCV

def Phase_List(exprs='all',nmo=-1):
    all_phases = ['A04','A03','A02','A01','B01','B02','B03','B04']
    for aa in range(1,25):
        if aa < 10:
            numstr = '0' + str(aa)
        else:
            numstr = str(aa)
        all_phases.append('C'+numstr)
        
        ephys_phases = all_phases[4:]
    if exprs=='all':
        return all_phases
    elif exprs=='ephys':
        return ephys_phases
    elif exprs == 'Nmo_ephys':
        #nmo = 3
        return ephys_phases[0:4*(nmo+1)-1]
    elif exprs == 'Nmo_onStim':
        #nmo = 5
        return ephys_phases[4:4*(nmo+1)-1]

#The frame is made by the old scripts in Integrated Ephys (for now)

#Let's load in that frame, which should contain all the data we want
#Only do Chronics with this analysis
exp = 'Chronics'
DataFrame = np.load('/home/virati/Data_frame_' + exp + '.npy').item()
#Load in the stim change times, though this should be merged into DataFrame itself in the generation script
StimChange = scipy.io.loadmat('/home/virati/MDD_Data/stim_changes.mat')['StimMatrix']
f_focus = ((0,50))

#%%
#Nitty gritty time
f_vect = np.linspace(0,211,512)
f_trunc = np.where(np.logical_and(f_vect < f_focus[1], f_vect > f_focus[0]))[0]
f_dict = {'FreqVector': f_vect, 'FreqTrunc': f_trunc}

#fixed 3test x 3train approach (for now)
training_patients = ['DBS905','DBS907','DBS906']
test_patients = ['DBS901','DBS903','DBS908']

#Need to make our design matrix now
#get the list of ephys-related clinical phases
def Phase_Matrix(DataFrame,Phases):    
    dsgnX = []
    dsgnY = []
    for pt, patient in enumerate(['901','903','905','906','907','908']):
        for pp,phase in enumerate(DataFrame['DBS'+patient].keys()):
            if phase[0] != 'A':
                try:
                    state_vector = np.hstack((DataFrame['DBS'+patient][phase]['MeanPSD']['LOGPxx'][:512,0],DataFrame['DBS'+patient][phase]['MeanPSD']['LOGPxx'][:,1]))
                    dsgnX.append(state_vector)
                    dsgnY.append(DataFrame['DBS'+patient][phase]['HDRS17'])
                except:
                    print(patient + ' ' + phase + ' has a problem')
                
    #confirm dimensional conguence
    Xout = np.array(dsgnX)
    Yout = np.array(dsgnY)
    
    #return the design tensors and flags for NaN presence
    return Xout, Yout
    
DsgnX, clinY = Phase_Matrix(DataFrame,Phase_List(exprs='ephys'))

#The actual regression code below
EN_alpha = 0.05
#Generate our elastic net model now
DM = ElasticNet(alpha=EN_alpha,tol=0.001,normalize=True,positive=False)
DM.fit(DsgnX,clinY)

plt.figure()
coefs = DM.coef_
csize = coefs.shape[0]
plt.plot(coefs[0:int(np.ceil(csize/2))],color='blue')
plt.plot(coefs[int(np.ceil(csize/2) + 1):],color='red')