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

import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet, ElasticNetCV

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

from sklearn import linear_model

default_params = {'CrossValid':10}            
        
class PSD_EN:
    def __init__(self,cv=True,alpha=0.5):

        if cv:
            self.ENet = ElasticNetCV()
        else:
            self.ENet = ElasticNet()
            
        self.performance = {'Train_Error':0}
            
    def Train(self,X,Y):
        #get the shape of the X and Y
        assert X.shape[0] == Y.shape[0]
        
        self.n_obs = Y.shape[0]
        
        self.ENet.fit(X,Y)
        self.performance['Train_Error'] = self.ENet.score(X,Y)
    
    def Test(self,X,Y_true):
        assert X.shape[0] == Y_true.shape[0]
        
        self.ENet.predict(X)
        
        

class DSV:
    def __init__(self, BRFrame,CFrame):
        #load in the BrainRadio DataFrame we want to work with
        self.YFrame = BRFrame
        
        #Load in the clinical dataframe we will work with
        self.CFrame = CFrame()
        self.dsgn_shape_params = ['logged','polyrem']#,'detrendX','detrendY','zscoreX','zscoreY']
        
        self.Model = {}
            
    #This function will generate the full OSCILLATORY STATE for all desired observations/weeks
    def O_feat_extract(self):
        big_list = self.YFrame.file_meta
        for rr in big_list:
            feat_dict = {key:[] for key in dbo.feat_dict.keys()}
            for featname,dofunc in dbo.feat_dict.items():
                datacontainer = {ch: rr['Data'][ch] for ch in rr['Data'].keys()}
                feat_dict[featname] = dofunc['fn'](datacontainer,self.YFrame.data_basis,dofunc['param'])
            rr.update({'FeatVect':feat_dict})
            
    def plot_feat_scatters(self):
        #Go to each feature in the feat_order
        Otest,Ctest = self.dsgn_O_C(self.YFrame.do_pts,collapse_chann=False)
        
        for ff,feat in enumerate(dbo.feat_order):
            if feat == 'fSlope' or feat == 'nFloor':
                dispfunc = dbo.unity
            else:
                dispfunc = np.log10
        
            plt.figure()
            plt.subplot(1,2,1)
            plt.scatter(Ctest,dispfunc(Otest[:,ff,0]))
            plt.subplot(1,2,2)
            plt.scatter(Ctest,dispfunc(Otest[:,ff,1]))
            plt.suptitle(feat)
    
    def dsgn_O_C(self,pts,scale='HDRS17',week_avg=True,collapse_chann=True):
        #hardcoded for now, remove later
        nchann = 2
        nfeats = len(dbo.feat_order)
        
        fmeta = self.YFrame.file_meta
        ptcdict = self.CFrame.clin_dict
        
        #This one gives the FULL STACK
        #fullfilt_data = np.array([(dbo.featDict_to_Matr(rr['FeatVect']),ptcdict['DBS'+rr['Patient']][rr['Phase']][scale]) for rr in fmeta if rr['Patient'] in pts])
        
        
        #FURTHER SHAPING WILL HAPPEN HERE, for example Z-scoring within each patient, within each channel; averaging week, etc.
        
        if week_avg:
            #collapse fullfilt_data along the desired week axis, which I don't know at this moment....
            pt_dict = {pt:{phase:np.mean(np.array([dbo.featDict_to_Matr(rec['FeatVect']) for rec in fmeta if rec['Patient'] == pt and rec['Phase'] == phase]),axis=0) for phase in dbo.all_phases} for pt in pts}
            
            #pt_dict forms the core of where we will draw data from
            #now we go to the fullfilt_data structure from the pt_dict structure
            
            #we want a matrix that is PTs x WEEKS x FEATS x Channs
            BigMatrix = np.array([np.array([np.array(fmatr) for ph,fmatr in dic.items() if ph in dbo.Phase_List(exprs='ephys')]) for key,dic in pt_dict.items()])
            BigClinVect = np.array([[ptcdict['DBS'+ pt][ph][scale] for ph in dbo.Phase_List(exprs='ephys')] for pt in pts])
            #Now we want to zscore INSIDE a patient
            
            #Have to log10 first here
            try:
                BigMatrix = np.log10(BigMatrix.astype(np.float64))
                BigClinVect = BigClinVect/self.CFrame.scale_max['HDRS17']
            except:
                pdb.set_trace()
            
            normscheme = 'zscore'
            if normscheme=='zscore':
                #This zscores all the features along the phases axis
                BigMatrix = stats.zscore(BigMatrix,axis=1)
            elif normscheme=='polyfit':
                pass
        
        
            #This collapses over patients so npts x obs becomes one dimension of obs without differentiating patients of size (npts x obs)
            O_dsgn_prelim = BigMatrix.reshape((-1,nfeats,nchann),order='C')
            C_dsgn_prelim = BigClinVect.reshape(-1,order='C')
            
        else:
            pt_dict = {pt:{phase:np.array([dbo.featDict_to_Matr(rec['FeatVect']) for rec in fmeta if rec['Patient'] == pt and rec['Phase'] == phase]) for phase in dbo.all_phases} for pt in pts}
            
            #THIs BLOCK GETS US END without the patients being split out into dimensions
            fullfilt_data = [[(fvect,ptcdict['DBS'+pt][ph][scale])for ph,fvect in pt_dict[pt].items() if ph in dbo.Phase_List(exprs='ephys')] for pt in pts]
            
            O_dsgn_prelim = np.array([ff[week][0] for ff,week in itt.product(fullfilt_data,range(28))])
            C_dsgn_prelim = np.array([ff[week][1] for ff,week in itt.product(fullfilt_data,range(28))])
            
            #fullfilt_data is still a bit weird here, need to listcompr to fix it
            O_dsgn_prelim= np.array([ff[0] for ff in fullfilt_data])
            #THIS LOOKS LIKE IT WORKS; it stacks the right vector UNDERNEIGHT the left; so we cycle through all left LFP features THEN the right LFP features
            #the order of features is determined by dbo.feat_order
            
        
        pdb.set_trace()
        
        if collapse_chann:
            O_dsgn = O_dsgn_prelim.reshape(-1,nfeats*nchann,order='F')
        else:
            O_dsgn = O_dsgn_prelim
        
        C_dsgn = C_dsgn_prelim

        
        
        #C_dsgn = np.array([ff[1] for ff in fullfilt_data])
        
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
        
        if 'polyrem' in params:
            
            for obs in range(X.shape[0]):
                Pl = np.polyfit(self.YFrame.data_basis,X[obs,0:513],5)
                Pr = np.polyfit(self.YFrame.data_basis,X[obs,512:],5)
                pdb.set_trace()
            
        
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
    
    def O_regress(self,method='OLS',inpercent=1,doplot=False,avgweeks=False):
        Otrain,Ctrain = self.dsgn_O_C(['901','903'],week_avg=avgweeks)
       
        #Then detrend the ENTIRE SET
        Otrain = sig.detrend(Otrain,axis=-1)
        Ctrain = sig.detrend(Ctrain)
        
        if method == 'OLS':
            regmodel = linear_model.LinearRegression()
        elif method == 'RANSAC':
            regmodel = linear_model.RANSACRegressor(min_samples=inpercent,max_trials=1000)
        
        #Do the model's fit
        #For this method, we'll do it on ALL available features
        regmodel.fit(Otrain,Ctrain.reshape(-1,1))
        
        #Test the model's performance in the other patients
        #Generate the testing set data
        Otest,Ctest = self.dsgn_O_C(['905','906','907','908'],week_avg=avgweeks)
        #Shape the input oscillatory state vectors
        Otest = sig.detrend(Otest,axis=-1)
        Ctest = sig.detrend(Ctest)
                
        #Generate the predicted clinical states
        Cpredictions = regmodel.predict(Otest)
        
        #Adding a bit of random noise may actually help, doesn't hurt
        #DITHERING STEP?!?!?!?!?!?!?!
        noise = 1
        Ctest = Ctest  + np.random.uniform(-noise,noise,Ctest.shape)
        Ctest = sig.detrend(Ctest)
        Ctest = stats.zscore(Ctest)
        
        #generate the statistical correlation of the prediction vs the empirical HDRS17 score
        #statistical correlation
        res = stats.pearsonr(Cpredictions,Ctest.astype(float).reshape(-1,1))
        self.Model.update({method:{'Model':regmodel,'Performance':{'PCorr_Zscore':res}}})
        
        if doplot:
            plt.figure()
            plt.plot(Cpredictions)
            plt.plot(Ctest)
            
            plt.figure()
            plt.scatter(Ctest,Cpredictions)
    
    #primary entry for OLS regression
    def run_OLS(self,doplot=False):
        #assumes we're running on \vec{O}
        Otrain,Ctrain = self.dsgn_O_C(['901','903'])
       
        # Can probably just do a regression now...
        Otrain = np.log10(Otrain)
        Otrain = sig.detrend(Otrain,axis=-1)
        Otrain = stats.zscore(Otrain)
        
        Ctrain = sig.detrend(Ctrain)
        Ctrain = stats.zscore(Ctrain)
        
        OLSapproach = linear_model.LinearRegression()
        
        side = 'all'
        noise = 1
        if side == 'left':
            featidxs = range(0,5)
        elif side == 'right':
            featidxs = range(5,10)
        elif side == 'all':
            featidxs = range(0,10)
        
        OLSapproach.fit(Otrain[:,featidxs],Ctrain.reshape(-1,1))
        
        Otest,Ctest = self.dsgn_O_C(['905','906','907','908'])
        Otest = sig.detrend(np.log10(Otest),axis=-1)
        Otest = stats.zscore(Otest)
        Cpredictions = OLSapproach.predict(Otest[:,featidxs])
        
        #DITHERING STEP?!?!?!?!?!?!?!
        Ctest = Ctest  + np.random.uniform(-noise,noise,Ctest.shape)
        Ctest = sig.detrend(Ctest)
        Ctest = stats.zscore(Ctest)
        
        #statistical correlation
        res = stats.pearsonr(Cpredictions,Ctest.astype(float).reshape(-1,1))
        self.OLS = {'Model':OLSapproach,'Performance':{'PCorr_Zscore':res}}
        
        if doplot:
            plt.figure()
            plt.plot(Cpredictions)
            plt.plot(Ctest)
            
            plt.figure()
            plt.scatter(Ctest,Cpredictions)
        
        
    def run_RANSAC(self,inpercent=0.4,doplot=False):
        Otrain,Ctrain = self.dsgn_O_C(['901','903'])
       
        # Can probably just do a regression now...
        Otrain = np.log10(Otrain)
        Otrain = sig.detrend(Otrain,axis=-1)
        Otrain = stats.zscore(Otrain)
        
        Ctrain = sig.detrend(Ctrain)
        Ctrain = stats.zscore(Ctrain)
        
        Rapproach = linear_model.RANSACRegressor(min_samples=inpercent,max_trials=1000)
        
        side = 'left'
        noise = 1
        if side == 'left':
            featidxs = range(0,5)
        else:
            featidxs = range(5,10)
        
        Rapproach.fit(Otrain[:,featidxs],Ctrain.reshape(-1,1))
        
        Otest,Ctest = self.dsgn_O_C(['905','906','907','908'])
        Otest = sig.detrend(np.log10(Otest),axis=-1)
        Otest = stats.zscore(Otest)
        Cpredictions = Rapproach.predict(Otest[:,featidxs])
        
        #DITHERING STEP?!?!?!?!?!?!?!
        Ctest = Ctest  + np.random.uniform(-noise,noise,Ctest.shape)
        Ctest = sig.detrend(Ctest)
        Ctest = stats.zscore(Ctest)
        
        #statistical correlation
        res = stats.pearsonr(Cpredictions,Ctest.astype(float).reshape(-1,1))
        self.RANSAC = {'Model':Rapproach,'Performance':{'PCorr_Zscore':res}}
        
        if doplot:
            plt.figure()
            plt.plot(Cpredictions)
            plt.plot(Ctest)
            
            plt.figure()
            plt.scatter(Ctest,Cpredictions)
            
        
    #primary 
    def run_EN(self):
        #first, learn the coefficients by training and testing
        self.learn_ENcoeffs()
        
        
    def learn_ENcoeffs(self):
        self.train_F,self.train_C = self.dsgn_F_C(['901','903','905'],week_avg=True)
        #setup our Elastic net here
        Ealg = PSD_EN()
        
        print("Training Elastic Net...")
        pdb.set_trace()
        Ealg.Train(self.train_F,self.train_C)
        
        #coefficients available
        
        #test phase
        Ftest,Ctest = self.dsgn_F_C(['906','907','908'],week_avg=True)
        print("Testing Elastic Net...")
        Ealg.Test(Ftest,Ctest)
        
        self.ENet = Ealg
        
        
    ## Ephys shaping methods
    def extract_DayNit(self):
        #stratify recordings based on Day/Night
        pass
    
    #This is the rPCA based method that generates the actual clinical measure on DSV and adds it to our CVect
    def gen_D_latent(self):
        pass    
