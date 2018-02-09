#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:40:07 2017

@author: virati
Module meant to be wrapper for DSV related stuff and scikitlearn related functions/procedures/analysis flows
MAIN Library for the DSV methodology
"""
import sklearn
from sklearn.linear_model import ElasticNet, ElasticNetCV

from collections import defaultdict

import json

import pdb

import numpy as np

default_params = {'CrossValid':10}            
        
class DSV:
    def __init__(self, BRFrame,CFrame):
        #load in the BrainRadio DataFrame we want to work with
        self.YFrame = BRFrame
        
        #Load in the clinical dataframe we will work with
        self.CFrame = CFrame()

    def gen_DV(self,basis=['HDRS17','MADRS','GAF','BDI']):
        #this method generates the Depression Vector
        #this is taken from an rPCA of the full scale clusters
        nweeks = 34
        
        if len(basis) == 1:
            pass
            
        else:
            c_vect = np.zeros((len(basis),nweeks))
            for ss,scales in enumerate(basis):
                c_vect[ss,:] = 0    
    
    def dsgn_X_Y(self,pts,scale='HDRS17'):
        #generate the X and Y needed for the regression
        fmeta = self.YFrame.file_meta
        ptcdict = self.CFrame.clin_dict
        
        fullfilt_data = [(rr['Data']['Left'],rr['Data']['Right'],rr['Phase'],rr['Patient']) for rr in fmeta if rr['Patient'] in pts]
        
        #go search the clin vect and replace the last element of the tuple (phase) with the actual score
        X_dsgn = np.array([np.vstack((a.reshape(-1,1),b.reshape(-1,1))) for a,b,c,d in fullfilt_data])
        Y_dsgn = np.zeros((X_dsgn.shape[0],1))
        
#        for rr,(aa,bb,cc,dd) in enumerate(fullfilt_data):
#            try:
#                Y_dsgn[rr] = ptcdict['DBS'+dd][cc][scale]
#            except:
#                pdb.set_trace()
#            
        Y_dsgn = [ptcdict['DBS'+d][c][scale] for a,b,c,d in fullfilt_data]
    
        self.X_dsgn = X_dsgn
        self.Y_dsgn = Y_dsgn
        return X_dsgn,Y_dsgn
        
      
    def ENet_Construct(self):
        #parameters are done in here
        l1_rat = 1
        alphas = []
        cv=5
        
        self.ENet = ElasticNetCV(l1_ratio=l_ratio,alphas=alpha_list,tol=0.01,normalize=True,positive=False,cv=k_fold)
        
    ## Ephys shaping methods
    def extract_DayNit(self):
        #stratify recordings based on Day/Night
        pass
    
    #This is the rPCA based method that generates the actual clinical measure on DSV and adds it to our CVect
    def gen_D_latent(self):
        pass
    
    ## Elastic Net Methods
    def SetupNet(self,params):
        self.CoreEN = ElasticNet()
        
        
    def TrainNet(self,X,Y):
        #Check structures
        #X is going to be a (512bins x2chann) x W weeks
        
        #Y is going to be a (C clinical measures) x W weeks vector
        pass
        
