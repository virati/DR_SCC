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

default_params = {'CrossValid':10}

class DSV:
    def __init__(self):
        self.Fephys = []
        self.Cclin = []
        
        self.Fephys = self.load_Ephys()
        self.Cclin = self.load_Cvect()

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
            
    def load_Ephys(self,preload=True):
        #Import the data structure needed for the Ephys
        if preload:
            #the data is brought in from a preloaded .npy file
            pass
        else:
            #do the raw load of the data and F-transform
            DataFrame = BR_Data_Tree()
            DataFrame.full_sequence()
    
    def load_Cvect(self):
        #Import the data structure needed for the CVect
        pass
    
    ## Ephys shaping methods
    def extract_DN(self):
        #stratify recordings based on Day/Night
        pass
    
    
    ## Elastic Net Methods
    def SetupNet(self,params):
        self.CoreEN = ElasticNet()
        
    def TrainNet(self,X,Y):
        #Check structures
        #X is going to be a (512bins x2chann) x W weeks
        
        #Y is going to be a (C clinical measures) x W weeks vector
        pass
        
