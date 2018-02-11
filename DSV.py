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
import itertools as itt

import json

import pdb

import numpy as np
import scipy.stats as stats
import scipy.signal as sig

from sklearn.linear_model import ElasticNet, ElasticNetCV

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo


default_params = {'CrossValid':10}            
        
class PSD_EN:
    def __init__(self,cv=True,alpha=0.5):

        if cv:
            self.ENet = ElasticNetCV()
        else:
            self.ENet = ElasticNet()
            
    def Train(self,X,Y):
        #get the shape of the X and Y
        assert X.shape[0] == Y.shape[0]
        
        self.n_obs = Y.shape[0]
        
        self.ENet.fit(X,Y)
        self.performance['Train_Error'] = ENet.score(X,Y)
    
    def Test(self,X,Y_true):
        assert X.shape[0] == Y.shape[0]
        
        self.ENet.predict(X)
        
        

class DSV:
    def __init__(self, BRFrame,CFrame):
        #load in the BrainRadio DataFrame we want to work with
        self.YFrame = BRFrame
        
        #Load in the clinical dataframe we will work with
        self.CFrame = CFrame()
        self.dsgn_shape_params = ['logged']#,'detrendX','detrendY','zscoreX','zscoreY']

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
                
    #This function will generate the full OSCILLATORY STATE for all desired observations/weeks
    
    def O_feat_extract(self):
        big_list = self.YFrame.file_meta
        for rr in big_list:
            feat_dict = {key:[] for key in dbo.feat_dict.keys()}
            for featname,dofunc in dbo.feat_dict.items():
                datacontainer = {ch: rr['Data'][ch] for ch in rr['Data'].keys()}
                feat_dict[featname] = dofunc['fn'](datacontainer,self.YFrame.data_basis,dofunc['param'])
            rr.update({'FeatVect':feat_dict})
    
    def dsgn_O_C(self,pts,scale='HDRS17',week_avg=True):
        fmeta = self.YFrame.file_meta
        ptcdict = self.CFrame.clin_dict
        
        fullfilt_data = np.array([(dbo.featDict_to_Matr(rr['FeatVect']),ptcdict['DBS'+rr['Patient']][rr['Phase']][scale]) for rr in fmeta if rr['Patient'] in pts])
        if week_avg:
            #collapse fullfilt_data along the desired week axis, which I don't know at this moment....
            pass
        
        #fullfilt_data is still a bit weird here, need to listcompr to fix it
        O_dsgn_intermed = np.array([ff[0] for ff in fullfilt_data])
        #THIS LOOKS LIKE IT WORKS; it stacks the right vector UNDERNEIGHT the left; so we cycle through all left LFP features THEN the right LFP features
        #the order of features is determined by dbo.feat_order
        O_dsgn = O_dsgn_intermed.reshape(5819,12,order='F')
        
        C_dsgn = np.array([ff[1] for ff in fullfilt_data])
        
        
        #O_dsgn = np.squeeze(np.array(fullfilt_data)[:,0:-1])
        #C_dsgn = np.squeeze(np.array(fullfilt_data)[:,-1])
        
        return O_dsgn, C_dsgn
    
    
    
        
    def dsgn_F_C(self,pts,scale='HDRS17',week_avg=True):
        #generate the X and Y needed for the regression
        fmeta = self.YFrame.file_meta
        ptcdict = self.CFrame.clin_dict
        
        if week_avg == False:
            fullfilt_data = [(rr['Data']['Left'],rr['Data']['Right'],rr['Phase'],rr['Patient']) for rr in fmeta if rr['Patient'] in pts]
            
            #go search the clin vect and replace the last element of the tuple (phase) with the actual score
            ALL_dsgn = np.array([np.vstack((a.reshape(-1,1),b.reshape(-1,1),ptcdict['DBS'+d][c][scale])) for a,b,c,d in fullfilt_data])
        else:
            phases = dbo.Phase_List(exprs='ephys')
            #bigdict = {key:{phs:(0,0) for phs in phases} for key in pts}
            biglist = []
            
            for pp in pts:
                for ph in phases:
                    leftavg = np.mean(np.array([rr['Data']['Left'] for rr in fmeta if rr['Patient'] == pp and rr['Phase'] == ph]),axis=0).reshape(-1,1)
                    rightavg = np.mean(np.array([rr['Data']['Right'] for rr in fmeta if rr['Patient'] == pp and rr['Phase'] == ph]),axis=0).reshape(-1,1)
                    
                    #bigdict[pp][ph] = (leftavg,rightavg)
                    biglist.append(np.vstack((leftavg,rightavg,ptcdict['DBS'+pp][ph][scale])))
            ALL_dsgn = np.array(biglist)
        
        X_dsgn = np.squeeze(ALL_dsgn)[:,0:-2]
        Y_dsgn = np.squeeze(ALL_dsgn)[:,-1].reshape(-1,1)
            
        #if we want to reshape, do it here!
        F_dsgn,C_dsgn = self.shape_F_C(X_dsgn,Y_dsgn,self.dsgn_shape_params)
        
        return F_dsgn, C_dsgn

    
    def shape_F_C(self,X,Y,params):
        if 'logged' in params:
            X = np.log10(X)
            
        if 'detrendX' in params:
            X = sig.detrend(X.T).T
        
        if 'detrendY' in params:
            Y = sig.detrend(Y)
            
        if 'zscoreX' in params:
            X = stats.zscore(X.T).T
            
        if 'zscoreY' in params:
            Y = stats.zscore(Y)
            
        return X,Y
            
    def get_dsgns(self):
        assert self.X_dsgn.shape[1] == 1025
        assert self.Y_dsgn.shape[1] == 1
        
        return self.X_dsgn, self.Y_dsgn
        
    def learn_ENcoeffs(self):
        self.train_F,self.train_C = self.dsgn_F_C(['901','903','905'],week_avg=True)
        #setup our Elastic net here
        Ealg = PSD_EN()
        Ealg.Train(self.train_F,self.train_C)
        
        #coefficients available
        
        #test phase
        Ftest,Ctest = self.dsgn_F_C(['906','907','908'],week_avg=True)
        Ealg.Test(Ftest,Ctest)
        
        
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
        
