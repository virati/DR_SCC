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

default_params = {'CrossValid':10}

class CFrame:
    do_pts = ['901','903','905','906','907','908']
    
    def __init__(self,incl_scales = ['HDRS17','MADRS','BDI','GAF']):
        #load in our JSON file
        #Import the data structure needed for the CVect
        ClinVect = json.load(open('/home/virati/Dropbox/ClinVec.json'))
        clin_dict = defaultdict(dict)
        for ss,scale in enumerate(incl_scales):
            for pp in range(len(ClinVect['HAMDs'])):
                ab = ClinVect['HAMDs'][pp]
                clin_dict[ab['pt']][scale] = ab[scale]
        
        self.clin_dict = clin_dict
        
    def gen_subdim(self,latentdim=1):
        #just do it for all patients
        for pp,pat in enumerate(self.do_pts):
            all_cscores = (' '.join(w) for w in self.clin_dict[pat])
            sC = np.vstack((big_dict[pat]['HDRS17'],big_dict[pat]['HDRS17'],big_dict[pat]['HDRS17'],big_dict[pat]['HDRS17']))

        self.bvect = {'LR': lrvect,'SC': srvect}
        
    def pt_scale_dict(self,pt):
        #return dictionary with all scales, and each element of that dictionary should be a NP vector
        return self.clin_dict[pt]

    def pt_scale_vect(self,pt):
        for ss in self.clin_dict[pt]:
            pass
        
    def c_vect(self):
        #each patient will be a dict key
        c_vects = {el:0 for el in self.do_pts}
        for pp,pt in enumerate(self.do_pts):
            #vector with all clinical measures in the thing
            #return will be phase x clinscores
            c_vect[pt] = 0
            
        
class DSV:
    def __init__(self, BRFrame=[]):
        #load in the BrainRadio DataFrame we want to work with
        if BRFrame == []:
            self.YFrame = BR_Data_Tree()
            print('Populating the BR Frame')
            self.YFrame.full_sequence()
        else:
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
            
    def DEPRload_Ephys(self,preload=True):
        #Import the data structure needed for the Ephys
        if preload:
            #the data is brought in from a preloaded .npy file
            pass
        else:
            #do the raw load of the data and F-transform
            DataFrame = BR_Data_Tree()
            DataFrame.full_sequence(datapath='/home/virati/')
    
    
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
        
