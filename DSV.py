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
            self.ENet = ElasticNetCV(cv=10,fit_intercept=True,l1_ratio=np.linspace(0.1,0.1,20),alphas=np.linspace(0.1,0.15,20))
        else:
            self.ENet = ElasticNet(alpha=0.9,l1_ratio=0.5,max_iter=1000,normalize=False,positive=False,fit_intercept=True)
            
        self.performance = {'Train_Error':0}
            
    def Train(self,X,Y):
        #get the shape of the X and Y
        try:
            assert X.shape[0] == Y.shape[0]
        except:
            pdb.set_trace()
        
        self.n_obs = Y.shape[0]
        
        try:
            self.ENet.fit(X,Y)
        except:
            pdb.set_trace()
        self.performance['Train_Error'] = self.ENet.score(X,Y)
    
    def Test(self,X,Y_true):
        assert X.shape[0] == Y_true.shape[0]
        
        Y_Pred = self.ENet.predict(X)
        plt.figure()
        plt.plot(stats.zscore(sig.detrend(Y_Pred)),label='Predicted')
        plt.plot(stats.zscore(sig.detrend(Y_true)),label='Actual')
        plt.legend()
        

class DSV:
    def __init__(self, BRFrame,CFrame):
        #load in the BrainRadio DataFrame we want to work with
        self.YFrame = BRFrame
        
        #Load in the clinical dataframe we will work with
        self.CFrame = CFrame()
        self.dsgn_shape_params = ['logged']#,'detrendX','detrendY','zscoreX','zscoreY']
        
        self.Model = {}
            
    
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
            big_score = []
            
            for pp in pts:
                pt_list = []
                pt_score = []
                for ph in phases:
                    leftavg = np.mean(np.array([rr['Data']['Left'] for rr in fmeta if rr['Patient'] == pp and rr['Phase'] == ph]),axis=0).reshape(-1,1)
                    rightavg = np.mean(np.array([rr['Data']['Right'] for rr in fmeta if rr['Patient'] == pp and rr['Phase'] == ph]),axis=0).reshape(-1,1)
                    
                    #bigdict[pp][ph] = (leftavg,rightavg)
                    pt_list.append(np.vstack((leftavg,rightavg)))
                    pt_score.append(ptcdict['DBS'+pp][ph][scale])
                    
                #after all the phases are done
                pt_list = np.squeeze(np.array(pt_list))
                
                #THIS IS WHERE SHAPING WILL HAPPEN
                pt_list = self.shape_PSD_stack(pt_list)
                
                biglist.append(pt_list)
                big_score.append(pt_score)
                
            ALL_dsgn = np.array(biglist)
            ALL_score = np.array(big_score)
        
        F_dsgn = np.squeeze(ALL_dsgn).reshape(-1,self.freq_bins,order='C')
        C_dsgn = np.squeeze(ALL_score).reshape(-1,1,order='C')
            
        #if we want to reshape, do it here!
        #F_dsgn,C_dsgn = self.shape_F_C(X_dsgn,Y_dsgn,self.dsgn_shape_params)
        
        return F_dsgn, C_dsgn

    def shape_PSD_stack(self,pt_list,polyord=6,plot=False):
        #input list is the per-patient stack of all PSDs for all phases, along with the HDRS
        #Do log transform of all of it
        
        preproc = ['log','zscore','limfreq']
        
        fix_pt_list = pt_list
        
        if 'log' in preproc:
            pt_list = 20 * np.log10(pt_list[:,:].T)
        
            fit_pt_list = pt_list
        
        #just subtract the FIRST recording from all of them
        print(pt_list.shape)
        
        #To subtract the average
        #base_subtr = np.mean(pt_list,axis=1).reshape(-1,1)
        base_subtr = np.zeros_like(pt_list)
        
        if 'polysub' in preproc:
            #to take a polynomial fit to all and then subtract it from each week's avg
            for ph in range(pt_list.shape[1]):
                pLeft = np.poly1d(np.polyfit(np.linspace(0,211,513),pt_list[0:513,ph],polyord))
                pRight = np.poly1d(np.polyfit(np.linspace(0,211,513),pt_list[513:,ph],polyord))
                base_subtr[0:513,ph] = pLeft(np.linspace(0,211,513))
                base_subtr[513:,ph] = pRight(np.linspace(0,211,513))
            
                fix_pt_list = pt_list - base_subtr
        elif 'zscore' in preproc:
            fix_pt_list = stats.zscore(pt_list,1)
        #finally, detrend the WHOLE STACK
        #fix_pt_list = sig.detrend(fix_pt_list,axis=0)
        #fix_pt_list = stats.zscore(fix_pt_list,axis=0)
        
        #do we want to start cutting frequencies out???
        self.freq_bins = 1026
        if 'limfreq' in preproc:
            freq_idx = np.tile(np.linspace(0,211,513),2)
            fmax = 90
            keep_idx = np.where(freq_idx < fmax)
            
            fix_pt_list = fix_pt_list[keep_idx,:]
            
            self.freq_bins = len(keep_idx[0])
        
        
        if plot:
            plt.figure()
            #plt.plot(fix_pt_list)
            plt.subplot(211)
            plt.plot(base_subtr)
            plt.plot(pt_list,alpha=0.2)
            plt.subplot(212);
            plt.plot(fix_pt_list)
            
        return fix_pt_list
    
    def shape_F_C(self,X,Y,params):
        
        if 'logged' in params:
            X = np.log10(X)
        
        if 'polyrem' in params:
            
            for obs in range(X.shape[0]):
                Pl = np.polyfit(self.YFrame.data_basis,X[obs,0:513],5)
                Pr = np.polyfit(self.YFrame.data_basis,X[obs,512:],5)
                
            
        
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
              
        
    #primary 
    def run_EN(self):
        #first, learn the coefficients by training and testing
        self.learn_ENcoeffs()
        
        
    def learn_ENcoeffs(self):
        self.train_F,self.train_C = self.dsgn_F_C(['901','903','905'],week_avg=True)
        #setup our Elastic net here
        Ealg = PSD_EN(cv=False)
        
        print("Training Elastic Net...")
        #pdb.set_trace()
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


class ORegress:
    def __init__(self,BRFrame,CFrame):
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
            
    def plot_feat_scatters(self,week_avg=False,patients='all'):
        #Go to each feature in the feat_order
        if patients == 'all':
            patients = self.YFrame.do_pts
        
        Otest,Ctest = self.dsgn_O_C(patients,collapse_chann=False,week_avg=week_avg)
        
        if week_avg:
            plotalpha = 1
        else:
            plotalpha = 0.1
        for ff,feat in enumerate(dbo.feat_order):
            if feat == 'fSlope' or feat == 'nFloor':
                dispfunc = dbo.unity
            else:
                dispfunc = np.log10
        
            plt.figure()
            plt.subplot(1,2,1)
            plt.scatter(Ctest,dispfunc(Otest[:,ff,0]),alpha=plotalpha)
            plt.subplot(1,2,2)
            plt.scatter(Ctest,dispfunc(Otest[:,ff,1]),alpha=plotalpha)
            plt.suptitle(feat)
    
    def dsgn_O_C(self,pts,scale='HDRS17',week_avg=True,collapse_chann=True,ignore_flags=False):
        #hardcoded for now, remove later
        nchann = 2
        nfeats = len(dbo.feat_order)
        
        fmeta = self.YFrame.file_meta
        ptcdict = self.CFrame.clin_dict
        
        ePhases = dbo.Phase_List(exprs='ephys')
        
        #This one gives the FULL STACK
        #fullfilt_data = np.array([(dbo.featDict_to_Matr(rr['FeatVect']),ptcdict['DBS'+rr['Patient']][rr['Phase']][scale]) for rr in fmeta if rr['Patient'] in pts])
        
        
        #FURTHER SHAPING WILL HAPPEN HERE, for example Z-scoring within each patient, within each channel; averaging week, etc.
        
        if ignore_flags:
            pt_dict = {pt:{phase:np.array([dbo.featDict_to_Matr(rec['FeatVect']) for rec in fmeta if rec['Patient'] == pt and rec['Phase'] == phase and rec['GC_Flag'] == False]) for phase in dbo.all_phases} for pt in pts}
        else:
            pt_dict = {pt:{phase:np.array([dbo.featDict_to_Matr(rec['FeatVect']) for rec in fmeta if rec['Patient'] == pt and rec['Phase'] == phase]) for phase in dbo.all_phases} for pt in pts}
        
        if week_avg:
            #if we want the week average we now want to go into the deepest level here, which has an array, and just take the average across observations
            pt_dict = {pt:{phase:[np.mean(featvect,axis=0)] for phase,featvect in pt_dict[pt].items()} for pt in pts}
        #RIGHT NOW we should have a solid pt_dict
        
        #Let's do a clugy zscore and dump it back into the pt_dict
        #get all recordings for a given patient across all phases
        
        #This forms a big list where we make tuples with all our feature vectors for all pt x ph
        #rr goes through the number of observations in each week; which may be 1 if we are averaging
        big_list = [[(rr,ptcdict['DBS'+pt][ph][scale],pt) for rr in pt_dict[pt][ph]] for pt,ph in itt.product(pts,ePhases)]
        #Fully flatten now for all observations
        #this is a big list that works great!
        obs_list = [item for sublist in big_list for item in sublist]
        
        #Piece out the obs_list
        O_dsgn_intermed= np.array([ff[0] for ff in obs_list])
        C_dsgn_intermed = np.array([ff[1] for ff in obs_list]) #to bring it into [0,1]
        
        try:
            if collapse_chann:
                O_dsgn = O_dsgn_intermed.reshape(-1,nfeats*nchann,order='F')
            else:
                O_dsgn = O_dsgn_intermed
        except:
            pdb.set_trace()
        
        #pdb.set_trace()
        C_dsgn = sig.detrend(C_dsgn_intermed,axis=-1)
        
        #Final shaping of both outputs
        O_dsgn = np.log10(O_dsgn)
        O_dsgn = sig.detrend(O_dsgn,axis=0)
        O_dsgn = sig.detrend(O_dsgn,axis=1)
        
        return O_dsgn, C_dsgn
    
    
        
        # #The below is for zscoring
        # O_dsgn_prelim = []
        # C_dsgn_prelim = []
        # for pt in pts:
        #     pt_matr = np.array([rr[0] for rr in obs_list if rr[2] == pt])
        #     pt_clin = np.array([rr[1] for rr in obs_list if rr[2] == pt])
            
        #     pt_matr = stats.zscore(pt_matr,axis=0)
        #     O_dsgn_prelim.append(pt_matr)
        #     C_dsgn_prelim.append(pt_clin)
        
        # O_dsgn_prelim = np.concatenate([matr for matr in O_dsgn_prelim],0)
        # C_dsgn_prelim = np.concatenate([matr for matr in C_dsgn_prelim],0)
        # #O_dsgn_prelim = [item for sublist in O_dsgn_prelim for item in sublist]
                
        # try:
        #     O_dsgn_intermed = np.array(O_dsgn_prelim).reshape((-1,5,2),order='C')
        #     C_dsgn_intermed = np.array(C_dsgn_prelim).reshape((-1,1),order='C')
        # except:
        #     pdb.set_trace()
            

    def O_regress(self,method='OLS',inpercent=1,doplot=False,avgweeks=False,ignore_flags=False):
        Otrain,Ctrain = self.dsgn_O_C(['901','903'],week_avg=avgweeks,ignore_flags=ignore_flags)
       
        Ctrain = sig.detrend(Ctrain) #this is ok to zscore here given that it's only across phases
        
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
                        
        #Generate the predicted clinical states
        Cpredictions = regmodel.predict(Otest)
        
        #Adding a bit of random noise may actually help, doesn't hurt
        #DITHERING STEP?!?!?!?!?!?!?!
        noise = 1
        Ctest = Ctest  + np.random.uniform(-noise,noise,Ctest.shape)
        Ctest = sig.detrend(Ctest)
        Ctest = Ctest
        
        #generate the statistical correlation of the prediction vs the empirical HDRS17 score
        #statistical correlation
        res = stats.pearsonr(Cpredictions,Ctest.astype(float).reshape(-1,1))
        self.Model.update({method:{'Model':regmodel,'Performance':{'PCorr':res,'Internal':0}}})
        
        #let's do internal scoring for a second
        self.Model[method]['Performance']['Internal'] = regmodel.score(Otest,Ctest)
        
        #what if we do a final "logistic" part here...
        
        
        if doplot:
            plt.figure()
            
            plt.plot(Cpredictions,label='Predicted')
            plt.plot(Ctest,label='Actual')
            plt.legend()
            plt.xlabel('Week')
            plt.ylabel('Normalized Disease Severity')
            plt.suptitle(method)
            
            plt.figure()
            plt.scatter(Ctest,Cpredictions)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.axis('equal')
            plt.suptitle(method)
    
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
