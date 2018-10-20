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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

from collections import defaultdict
import itertools as itt
from itertools import compress

import json

import ipdb

import numpy as np
import scipy.stats as stats
import scipy.signal as sig

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import random

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo
from DBS_Osc import nestdict

from sklearn import linear_model

default_params = {'CrossValid':10}

import seaborn as sns
#sns.set_context("paper")

sns.set(font_scale=4)
sns.set_style("white")

import time
import copy

#%%
#OLD ELASTIC NET METHODS HERE


#%%
#general methods here

def L1_dist(x,y):
    return np.sum(np.abs(a-b) for a,b in zip(x,y))

#%%            
        
class PSD_EN:
    def __init__(self,cv=True,alphas=np.linspace(0.7,0.8,10),alpha=0.5):

        if cv:
            print('Running ENet CV')
            #This works great!
            #l_ratio = np.linspace(0.2,0.5,20)
            #alpha_list = np.linspace(0.7,0.9,10)
            
            #play around here
            #THIS WORKS WELL#l_ratio = np.linspace(0.2,0.3,20)
            #l_ratio = np.linspace(0.3,0.5,30)
            l_ratio = np.linspace(0.2,0.5,30)
            alpha_list = alphas
            
            
            #self.ENet = ElasticNetCV(cv=10,tol=0.01,fit_intercept=True,l1_ratio=np.linspace(0.1,0.1,20),alphas=np.linspace(0.1,0.15,20))
            self.ENet = ElasticNetCV(l1_ratio=l_ratio,alphas=alpha_list,cv=15,tol=0.001,normalize=True,positive=False,copy_X=True,fit_intercept=True)
        else:
            raise ValueError
            alpha = 0.12
            print('Running Normal ENet w/ alpha ' + str(alpha))
            self.ENet = ElasticNet(alpha=alpha,l1_ratio=0.1,max_iter=1000,normalize=True,positive=False,fit_intercept=True,precompute=True,copy_X=True)
            
        self.performance = {'Train_Error':0,'Test_Error':0}
        
            
    def Train(self,X,Y):
        #get the shape of the X and Y
        try:
            assert X.shape[0] == Y.shape[0]
        except:
            ipdb.set_trace()
        
        self.n_obs = Y.shape[0]
        
        self.ENet.fit(X,Y.reshape(-1))
        
        #do a predict to see what the fit is for
        Y_train_pred = self.ENet.predict(X).reshape(-1,1)
        
        self.train_Ys = (Y_train_pred,Y)
        
        self.performance['Train_Error'] = self.ENet.score(X,Y)
        print('ENet CV Params: Alpha: ' + str(self.ENet.alpha_) + ' l ratio: ' + str(self.ENet.l1_ratio_))
    
    def Test(self,X,Y_true):
        
        assert X.shape[0] == Y_true.shape[0]
        
        Y_Pred = self.ENet.predict(X).reshape(-1,1)
        
        self.Ys = (Y_Pred,Y_true)
        self.performance['Test_Error'] = self.ENet.score(X,Y_true)
        self.performance['PearsonR'] = stats.pearsonr(self.norm_func(Y_Pred),self.norm_func(Y_true))
        self.performance['SpearmanR'] = stats.spearmanr(self.norm_func(Y_Pred),self.norm_func(Y_true))


#Class for the 'Linear Frequency Model'
# This model is PATIENT SPECIFIC and is not meant to be a generalized model, rather a METHODOLOGY
class LFM:
    def __init__(self,BRFrame,ClinFrame,patient='901',lim_freq=50):
        print('Returning Recording List for ' + patient)
        self.YFrame = BRFrame
        self.CFrame = ClinFrame
        self.patient = patient
        
        #general parameters here
        
        
        # Split out the dataset to train and validate
        self.Patient_Data = BRFrame.grab_recs(reqs={'Patient':['901']})

    def F_C_Matrix(self):
        pass

class DSV:
    def __init__(self, BRFrame,ClinFrame,lim_freq=50):
        #load in the BrainRadio DataFrame we want to work with
        self.YFrame = BRFrame
        
        #Load in the clinical dataframe we will work with
        self.CFrame = ClinFrame
        self.dsgn_shape_params = ['logged']#,'detrendX','detrendY','zscoreX','zscoreY']
        
        self.Model = {}
        
        self.lim_freq = lim_freq
        self.train_pts = ['901','903']
        self.test_pts = ['906','907','905','908']
    
    def dsgn_F_C(self,pts,scale='HDRS17',week_avg=True):
        #generate the X and Y needed for the regression
        fmeta = self.YFrame.file_meta
        ptcdict = self.CFrame.clin_dict
        
        if week_avg == False:
            print('Taking ALL Recordings')
            fullfilt_data = [(rr['Data']['Left'],rr['Data']['Right'],rr['Phase'],rr['Patient']) for rr in fmeta if rr['Patient'] in pts]
            
            #go search the clin vect and replace the last element of the tuple (phase) with the actual score
            ALL_dsgn = np.array([np.vstack((a.reshape(-1,1),b.reshape(-1,1),ptcdict['DBS'+d][c][scale])) for a,b,c,d in fullfilt_data])
        else:
            print('Taking Weekly Averages')
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
                pt_list,polyvect = self.shape_PSD_stack(pt_list,plot=True,polyord=4)
                
                biglist.append(pt_list)
                big_score.append(pt_score)
                
            ALL_dsgn = np.array(biglist)
            ALL_score = np.array(big_score)
        
        self.all_dsgn = ALL_dsgn
        self.all_score = ALL_score
        
        F_dsgn = np.swapaxes(np.squeeze(ALL_dsgn),1,2).reshape(-1,self.freq_bins,order='C')
        C_dsgn = np.squeeze(ALL_score).reshape(-1,1,order='C').astype(np.float64)
            
        #if we want to reshape, do it here!
        #F_dsgn,C_dsgn = self.shape_F_C(X_dsgn,Y_dsgn,self.dsgn_shape_params)
        
        
        return F_dsgn, 100*C_dsgn

    def OBSshape_PSD_stack(self,pt_list,polyord=4,plot=False):
        #input list is the per-patient stack of all PSDs for all phases, along with the HDRS
        #Do log transform of all of it
        
        preproc = ['log','polysub','limfreq']
        fmax = self.lim_freq
        
        fix_pt_list = pt_list
        
        pLeft = []
        pRight = []
        
        if 'log' in preproc:
            pt_list = 20 * np.log10(pt_list[:,:].T)
        
            fix_pt_list = pt_list
        
        #just subtract the FIRST recording from all of them
        
        #To subtract the average
        #base_subtr = np.mean(pt_list,axis=1).reshape(-1,1)
        base_subtr = np.zeros_like(pt_list)
        
        if 'polysub' in preproc:
            print('Polynomial Subtraction of order ' + str(polyord))
            #to take a polynomial fit to all and then subtract it from each week's avg
            for ph in range(pt_list.shape[1]):
                pLeft = np.poly1d(np.polyfit(np.linspace(0,211,513),pt_list[0:513,ph],polyord))
                pRight = np.poly1d(np.polyfit(np.linspace(0,211,513),pt_list[513:,ph],polyord))
                base_subtr[0:513,ph] = pLeft(np.linspace(0,211,513))
                base_subtr[513:,ph] = pRight(np.linspace(0,211,513))
            
                fix_pt_list = pt_list - base_subtr
                
        elif 'zscore' in preproc:
            fix_pt_list = self.norm_func(pt_list,axis=1)
        
        #finally, detrend the WHOLE STACK
        #fix_pt_list = sig.detrend(fix_pt_list,axis=0)
        #fix_pt_list = stats.zscore(fix_pt_list,axis=0)
        
        #do we want to start cutting frequencies out???
        self.freq_bins = 1026
        if 'limfreq' in preproc:
            print('Limiting Frequency')
            freq_idx = np.tile(np.linspace(0,211,513),2)
            
            keep_idx = np.where(freq_idx <= fmax)[0]
            
            fix_pt_list = fix_pt_list[keep_idx,:]
            
            self.freq_bins = len(keep_idx)
            self.trunc_fvect = np.linspace(0,fmax,len(keep_idx)/2)
            
        
        if 'detrend':
            fix_pt_list = sig.detrend(fix_pt_list,axis=0)
        
        fix_pt_list = np.squeeze(fix_pt_list)
        
        if plot:
            plt.figure()
            #plt.plot(fix_pt_list)
            plt.subplot(221)
            plt.plot(np.linspace(0,211,513),base_subtr[0:513])
            plt.plot(np.linspace(0,211,513),pt_list[0:513],alpha=0.2)
            
            plt.subplot(222)
            plt.plot(np.linspace(0,211,513),base_subtr[513:])
            plt.plot(np.linspace(0,211,513),pt_list[513:],alpha=0.2)
            
            
            plt.subplot(223);
            plt.plot(self.trunc_fvect,fix_pt_list[:146])
            
            plt.subplot(224);
            plt.plot(self.trunc_fvect,fix_pt_list[146:])
            
        return fix_pt_list,(pLeft,pRight)
            
    def get_dsgns(self):
        assert self.X_dsgn.shape[1] == 1025
        assert self.Y_dsgn.shape[1] == 1
        
        return self.X_dsgn, self.Y_dsgn
              
    def plot_dsgn_matrix(self):
        one_side_bins = int(self.freq_bins/2)
        plt.figure()
        plt.subplot(1,2,1)
        
        plt.plot(self.trunc_fvect, self.train_F.T[:one_side_bins])#,color=cm.hot(self.train_C/70).squeeze())
        plt.title('Left Channel')
        
        plt.subplot(1,2,2)
        plt.plot(self.trunc_fvect,self.train_F.T[one_side_bins:])#,color=cm.hot(self.train_C/70).squeeze())
        plt.title('Right Channel')
        
    #primary 
    def run_EN(self,alpha_list,scale='HDRS17'):
        self.train_F,self.train_C = self.dsgn_F_C(self.train_pts,week_avg=True)
        
        #setup our Elastic net here
        Ealg = PSD_EN(cv=True,alphas=alpha_list)
        
        print("Training Elastic Net...")
        
        Ealg.Train(self.train_F,self.train_C)
        
        #test phase
        
        Ftest,Ctest = self.dsgn_F_C(self.test_pts,week_avg=True,scale=scale)
        print("Testing Elastic Net...")
        Ealg.Test(Ftest,Ctest)
        
        self.ENet = Ealg
        
    def plot_EN_coeffs(self):
        plt.figure()
        coeff_len = int(self.ENet.ENet.coef_.shape[0]/2)
        
        plt.plot(self.trunc_fvect,self.ENet.ENet.coef_[0:coeff_len],label='Left Feats')
        plt.plot(self.trunc_fvect,self.ENet.ENet.coef_[coeff_len:],label='Right Feats')
        plt.legend()
        
    def plot_tests(self):
        
        #Now plot them if we'd like
        num_pts = len(self.test_pts)
        total_obs = len(self.ENet.Ys[0])
        per_pt_obs = int(total_obs / num_pts)
    
        pt_zscored = defaultdict(dict)
        norm_func = stats.zscore
        for pp,pt in enumerate(self.test_pts):
            plt.figure()
            pt_zscored[pt] = {'Predicted':norm_func(self.ENet.Ys[0][pp*per_pt_obs:per_pt_obs*(pp+1)],axis=0),'HDRS17':norm_func(self.ENet.Ys[1][pp*per_pt_obs:per_pt_obs*(pp+1)],axis=0)}
            
            plt.plot(pt_zscored[pt]['Predicted'],label='Predicted')
            plt.plot(pt_zscored[pt]['HDRS17'],label='HDRS17')
            plt.legend()
            plt.title(pt)
            
            spearm = stats.spearmanr(pt_zscored[pt]['Predicted'],pt_zscored[pt]['HDRS17'])
            print(pt + ' Spearman:')
            print(spearm)
            
            #output the PER PATIENT results plotted here
            
            
        self.Zscore_Results = pt_zscored
        
    #def plot_performance(self):
        
    def plot_performance(self,plot_indiv=False,doplot = True,ranson=True):
        ###############
        #Old way
        #Cpredictions = (self.ENet.Ys[0])
        #Ctest = (self.ENet.Ys[1])
        
        #Cpredictions = stats.zscore((Cpredictions.reshape(-1,1)))
        #Ctest = stats.zscore((Ctest.reshape(-1,1)))
        ###############
        
        # NEW WAY
        #preCpredictions = self.Zscore_Results['Predicted']
        #preCtest = self.Zscore_Results['HDRS17']
        #These need to be numpied
        pt_list = ['905','906','907','908']
        Cpredictions = np.array([self.Zscore_Results[pt]['Predicted'] for pt in pt_list]).reshape((-1,1))
        Ctest = np.array([self.Zscore_Results[pt]['HDRS17'] for pt in pt_list]).squeeze().reshape((-1,1))
        
        
        
        scatter_alpha = 0.3
        
        #cpred_msub = sig.detrend(Cpredictions,type='linear')
        #ctest_msub = sig.detrend(Ctest,type='linear')

        plt.figure()
        plt.plot(Cpredictions)
        plt.plot(Ctest)
        
        spearm = stats.spearmanr(Cpredictions,Ctest)
        
        
        print('ENR has Spearman: ' + str(spearm))
        
        #post-process the test and predicted things
        
        
        if plot_indiv:
        #do a plot for each patient on this?
            for pt in test_pts:
                plt.figure()
                #check if the sizes are right
                assert len(Cpredictions) == len(labels['Patient'])
                
                pt_preds = [cpred for cpred,pat in zip(Cpredictions,labels['Patient']) if pat == pt]
                pt_actuals = [ctest for ctest,pat in zip(Ctest,labels['Patient']) if pat == pt]
                
                
                plt.plot(pt_preds,label='Predicted')
                plt.plot(pt_actuals,label='HDRS17')
                plt.legend()
                
                plt.xlabel('Week')
                plt.ylabel('Normalized Disease Severity')
                plt.suptitle(pt + ' ' + 'ENR')
                sns.despine()
        
        if doplot:
            
            x,y = (1,1)
            if ranson:
                assesslr = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=True),min_samples=0.8)
            else:
                assesslr = linear_model.LinearRegression(fit_intercept=True)
                
            
            #assesslr.fit(Ctest.reshape(-1,1),Cpredictions.reshape(-1,1))
            #THIS ELIMATES BROAD DECREASES OVER TIME, a linear detrend
            assesslr.fit(Ctest,Cpredictions)
            
            line_x = np.linspace(0,1,20).reshape(-1,1)
            line_y = assesslr.predict(line_x)
            
            if ranson:
                inlier_mask = assesslr.inlier_mask_
                corrcoef = assesslr.estimator_.coef_[0]
                
            else:
                inlier_mask = np.ones_like(Ctest).astype(bool)
                corrcoef = assesslr.coef_[0]
                
            outlier_mask = np.logical_not(inlier_mask)
            
            #FINALLY just do a stats package linear regression
            if ranson:
                slsl,inin,rval,pval,stderr = stats.mstats.linregress(Ctest[inlier_mask].reshape(-1,1),Cpredictions[inlier_mask].reshape(-1,1))
            else:
                slsl,inin,rval,pval,stderr = stats.mstats.linregress(Ctest.reshape(-1,1),Cpredictions.reshape(-1,1))
            
            
            # if ranson:
            #     print(method + ' model has ' + str(corrcoef) + ' correlation with real score')
            # else:
            print('ENR' + ' model has ' + str(slsl) + ' correlation with real score (p < ' + str(pval) + ')')
            
            plt.figure()
            plt.scatter(Ctest[outlier_mask],Cpredictions[outlier_mask],alpha=scatter_alpha,color='gray')
            plt.scatter(Ctest[inlier_mask],Cpredictions[inlier_mask],alpha=scatter_alpha)
            plt.plot(np.linspace(0,x,2),np.linspace(0,y,2),alpha=0.2,color='gray')
            plt.plot(line_x,line_y,color='red')
            plt.ylim((-4,4))
            plt.xlim((-2.5,2.5))
            plt.axes().set_aspect('equal')
            
            plt.xlabel('HDRS17')
            plt.ylabel('Predicted')
            #plt.xlim((0,1))
            #plt.ylim((-0.2,1))
            
            plt.suptitle('ENR')
            #plt.title('All Observations')
            sns.despine()
            
        plt.figure()
        predicted = stats.zscore(self.ENet.Ys[0],axis=0)
        actuals = stats.zscore(self.ENet.Ys[1],axis=0)
        plt.scatter(actuals,predicted)
        
    def plot_trains(self):
        
        #Now plot them if we'd like
        num_pts = len(self.train_pts)
        total_obs = len(self.ENet.train_Ys[0])
        per_pt_obs = int(total_obs / num_pts)
    
        norm_func = stats.zscore
        for pp,pt in enumerate(self.train_pts):
            plt.figure()
            
            plt.plot(norm_func(self.ENet.train_Ys[0][pp*per_pt_obs:per_pt_obs*(pp+1)],axis=0),label='Predicted')
            plt.legend()
            plt.title(pt)
                
            
    ## Ephys shaping methods
    def extract_DayNit(self):
        #stratify recordings based on Day/Night
        pass
    
    #This is the rPCA based method that generates the actual clinical measure on DSV and adds it to our CVect
    def gen_D_latent(self):
        pass    


class ORegress:
    def __init__(self,BRFrame,inCFrame):
        self.YFrame = BRFrame
        
        #Load in the clinical dataframe we will work with
        self.CFrame = inCFrame
        self.dsgn_shape_params = ['logged','polyrem']#,'detrendX','detrendY','zscoreX','zscoreY']
        
        self.Model = {}
        
        self.norm_func = stats.zscore
        #self.norm_func = dbo.unity
        
    #This function will generate the full OSCILLATORY STATE for all desired observations/weeks
    def O_feat_extract(self):
        print('Extracting Oscillatory Features')
        #print(dbo.feat_dict)
        big_list = self.YFrame.file_meta
        for rr in big_list:
            feat_dict = {key:[] for key in dbo.feat_dict.keys()}
            for featname,dofunc in dbo.feat_dict.items():
                #Choose the zero index of the poly_subtr return because we are not messing with the polynomial vector itself
                #rr['data'] is NOT log10 transformed, so makes no sense to do the poly subtr
                datacontainer = {ch: self.poly_subtr(rr['Data'][ch],polyord=5)[0] for ch in rr['Data'].keys()} #THIS RETURNS a PSD, un-log transformed
                
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
    
    def split_validation_set(self,do_split = True):
        if do_split:
            print('Splitting out validation set')
            print('Pre splot YFrame ' + str(len(self.YFrame.file_meta)))
            self.train_set, self.valid_set = train_test_split(self.YFrame.file_meta,train_size=0.6,shuffle=True)
            
            print('Train set' + str(len(self.train_set)))
            print('Validation set ' + str(len(self.valid_set)))
        else:
            print('NOT Splitting out a validation set')
            self.train_set = self.YFrame.file_meta
            self.valid_set = self.YFrame.file_meta
    
    
    
    def dsgn_O_C(self,pts,scale='HDRS17',week_avg=True,collapse_chann=True,ignore_flags=False,circ='',from_set='TRAIN',randomize=0.0):
        #hardcoded for now, remove later
        nchann = 2
        nfeats = len(dbo.feat_order)
        label_dict={'Patient':[]}
        
        #fmeta = self.YFrame.file_meta
        if from_set == 'TRAIN':
            print('Using Train Set ' + str(len(self.train_set)))
            fmeta = self.train_set
        elif from_set == 'VALIDATION':
            print('DIPPING INTO VALIDATION SET '  + str(len(self.valid_set)))
            fmeta = self.valid_set
        else:
            raise ValueError
            
        ptcdict = self.CFrame.clin_dict
        
        ePhases = dbo.Phase_List(exprs='ephys')
        
        #This one gives the FULL STACK
        #fullfilt_data = np.array([(dbo.featDict_to_Matr(rr['FeatVect']),ptcdict['DBS'+rr['Patient']][rr['Phase']][scale]) for rr in fmeta if rr['Patient'] in pts])
        
        #RANDOMIZE??
        if randomize != 0.0:
            fmeta = np.random.choice(fmeta,int(np.floor(randomize * len(fmeta))),replace=False)
        
        ###THIS IS NEW
        #generate our stack of interest, with all the flags and all
        pt_dict_flags = {pt:{phase:[rec for rec in fmeta if rec['Patient'] == pt and rec['Phase'] == phase] for phase in ePhases} for pt in pts}
        
        #How many recordings are going in for now?
        
        
        
        if ignore_flags:
            print('Ignoring GC Flags')
            pt_dict_flags = {pt:{phase:[rec for rec in fmeta if rec['Patient'] == pt and rec['Phase'] == phase and rec['GC_Flag']['Flag'] == False] for phase in ePhases} for pt in pts}
            pt_dict_gc = pt_dict_flags
        else:
            pt_dict_gc = pt_dict_flags
        
        if circ != '':
            print('Filtering circadian ' + circ)
            pt_dict_flags = {pt:{phase:[rec for rec in pt_dict_gc[pt][phase] if rec['Circadian'] == circ] for phase in ePhases} for pt in pts}
            pt_dict_circ = pt_dict_flags
        else:
            pt_dict_circ = pt_dict_gc
        
        
        #tot_recs = ([[len(rec) for phz,rec in phase.items()] for pt,phase in pt_dict_flags.items()])
        tot_recs = sum([len(pt_dict_circ[pt][ph]) for pt,ph in itt.product(pts,ePhases)])
        print(str(tot_recs) + ' recordings!!!!!!!!!!!!!')
        
        
        
        ##THIS converts our recordings to a feature vector matrix that is the basis for the design matrix. Averaging happens immediately after as a result
        print('Generating Pre-filtered Recordings - GC ignore:' + str(ignore_flags) + ' circadian:' + circ)
        pt_dict = {pt:{phase:np.array([dbo.featDict_to_Matr(rec['FeatVect']) for rec in pt_dict_circ[pt][phase]]) for phase in ePhases} for pt in pts}
 
        
        
        #first step, find feat vect for all
        if week_avg:
            #if we want the week average we now want to go into the deepest level here, which has an array, and just take the average across observations
            pt_dict = {pt:{phase:[np.median(featvect,axis=0)] for phase,featvect in pt_dict[pt].items()} for pt in pts}
        #RIGHT NOW we should have a solid pt_dict
        
        #Let's do a clugy zscore and dump it back into the pt_dict
        #get all recordings for a given patient across all phases
        
        #This forms a big list where we make tuples with all our feature vectors for all pt x ph
        #rr goes through the number of observations in each week; which may be 1 if we are averaging
        big_list = [[(rr,ptcdict['DBS'+pt][ph][scale],pt,ph) for rr in pt_dict[pt][ph]] for pt,ph in itt.product(pts,ePhases)]
        
        #Fully flatten now for all observations
        #this is a big list that works great!
        obs_list = [item for sublist in big_list for item in sublist if not np.isnan(np.array(item[0])).any()]
        
        
        #Piece out the obs_list
        O_dsgn_intermed= np.array([ff[0] for ff in obs_list])
        C_dsgn_intermed = np.array([ff[1] for ff in obs_list]) #to bring it into [0,1]
        label_dict['Patient'] = [ff[2] for ff in obs_list]
        label_dict['Phase'] = [ff[3] for ff in obs_list]
        
        #ipdb.set_trace()
        if collapse_chann:
            try:
                O_dsgn = O_dsgn_intermed.reshape(-1,nfeats*nchann,order='F')
            except:
                ipdb.set_trace()
        else:
            O_dsgn = O_dsgn_intermed
    
        
        #We will detrend the CLINICAL SCORES along the -1 axis: SHOULD BE Phase
        #C_dsgn = sig.detrend(C_dsgn_intermed,axis=-1)
        C_dsgn = C_dsgn_intermed
        
        #O_dsgn = sig.detrend(O_dsgn,axis=0)
        #O_dsgn = sig.detrend(O_dsgn,axis=1)
        
        return O_dsgn, C_dsgn, label_dict
    
    
        
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

    def poly_subtr(self,inp_psd,polyord=4):
        #log10 in_psd first
        log_psd = 10*np.log10(inp_psd)
        pfit = np.polyfit(self.YFrame.data_basis,log_psd,polyord)
        pchann = np.poly1d(pfit)
        
        bl_correction = pchann(self.YFrame.data_basis)
        
        return 10**((log_psd - bl_correction)/10), pfit

    def O_models(self,plot=True,models=['RANSAC','RIDGE']):
        sns.set_style("ticks")
        
        
        sides = ['Left','Right']
        sides_idxs = {'Left':np.arange(0,5),'Right':np.arange(5,10)}
        
        Coefs = {key:{sid:[] for sid in sides} for key in models}
        
        for mtype in models:
            mod = self.Model[mtype]
        
            for sid in sides:
                if mtype == 'RANSAC': 
                    Coefs[mtype][sid] = mod['Model'].estimator_.coef_[0][sides_idxs[sid]]
                elif mtype == 'LASSO':
                    Coefs[mtype][sid] = mod['Model'].coef_[sides_idxs[sid]]
                elif mtype == 'ENR_Osc':
                    Coefs[mtype][sid] = mod['Model'].coef_[sides_idxs[sid]]
                else: 
                    try: Coefs[mtype][sid] = mod['Model'].coef_[0][sides_idxs[sid]]
                    except: ipdb.set_trace()
            
#            
#        mod = self.Model['RIDGE']
#        ridge_cs = {key:0 for key in sides}
#        rans_cs = {key:0 for key in sides}
#        ridge_cs['Left'] = mod['Model'].coef_[0][:5]
#        ridge_cs['Right'] = mod['Model'].coef_[0][5:]
#        
#        mod = self.Model['RANSAC']
#        rans_cs['Left'] = mod['Model'].estimator_.coef_[0][:5]
#        rans_cs['Right'] = mod['Model'].estimator_.coef_[0][5:]
#        
        
        if plot:
            plt.figure()
            for ss,side in enumerate(sides):
                plt.subplot(1,2,ss+1)
                for mtype in models:
                    plt.plot(np.arange(5),Coefs[mtype][side],label=mtype)
                    
                #plt.plot(np.arange(5),ridge_cs[side],label='Ridge')
                #plt.plot(np.arange(5),rans_cs[side],label='RANSAC')
                
                plt.xticks(np.arange(5),['Delta','Theta','Alpha','Beta','Gamma*'],rotation=70)
                plt.xlim((0,4))
                
                plt.xlabel('Feature')
                plt.ylim((-0.03,0.03))
                
            plt.subplot(1,2,1)
            plt.ylabel('Coefficient Value')
                
            plt.legend()
        
        return Coefs[mtype]
        
    def shuffle_dprods(self,regmodel,Otest,Ctest,numshuff=100):
        #print('Starting Shuffle Test...')
        res_dot = np.zeros((numshuff,1))
        
        
        for ss in range(numshuff):
            Oshuff,Cshuff = shuffle(Otest,Ctest,random_state=ss)
            #compare our Oshuff to Ctest
            
            Cspred = regmodel.predict(Oshuff).reshape(-1,1)
            
            #DETREND HERE
            #Cspred = sig.detrend(Cspred,axis=0,type='linear')
            #Ctest = sig.detrend(Ctest.reshape(-1,1),axis=0,type='linear')
            
            #res_dot[ss] = np.dot(Cspred.T,Ctest)
            res_dot[ss] = L1_dist(Cspred,Ctest)
            
                        
        return res_dot

    def shuffle_summary(self,method='RIDGE'):
        #print('Shuffle Assessment')
        Cpredictions = (self.Model[method]['Cpredictions'].reshape(-1,1))
        Ctest = (self.Model[method]['Ctest'].reshape(-1,1))
        #do some shuffling here and try to see how well the model does
        shuff_distr = self.shuffle_dprods(self.Model[method]['Model'],self.Model[method]['Otest'],self.Model[method]['Ctest'],numshuff=1000)
        #What's the similarity of the model output, or "actual_similarity"
        act_sim = L1_dist(Cpredictions,Ctest)
        self.Model[method]['Performance']['DProd'] = {'Dot':act_sim,'Distr':shuff_distr,'Perfect':L1_dist(Ctest,np.zeros_like(Ctest)),'pval':0}
        self.Model[method]['Performance']['DProd']['pval'] = np.sum(np.abs(self.Model[method]['Performance']['DProd']['Distr']) > self.Model[method]['Performance']['DProd']['Dot'])/len(self.Model[method]['Performance']['DProd']['Distr'])

        #First, let's do shuffled version of IPs
        #print('Shuffle: IP similarity is:' + str(self.Model[method]['Performance']['DProd']['Dot']) + ' | Percentage of surrogate IPs larger: ' + str(np.sum(np.abs(self.Model[method]['Performance']['DProd']['Distr']) > self.Model[method]['Performance']['DProd']['Dot'])/len(self.Model[method]['Performance']['DProd']['Distr'])) + '|| Perfect: ' + str(self.Model[method]['Performance']['DProd']['Perfect']))
        #plt.figure();plt.hist(self.Model[method]['Performance']['DProd']['Distr'])
        return copy.deepcopy(self.Model[method]['Performance']['DProd'])
        

    def new_regress(self,method):
        pass

    def O_regress(self,method='OLS',inpercent=1,doplot=False,avgweeks=False,ignore_flags=False,ranson=True,circ='',plot_indiv=False,scale='HDRS17',lindetrend = 'Block',train_pts = ['903','906','907'],train_all=False,finalWrite=False):

        print('Doing DETREND: ' + lindetrend)
        
        #Test/Train patient separation
        if not train_all:
            test_pts = [pt for pt in dbo.all_pts if pt not in train_pts]
        else:
            train_pts = dbo.all_pts
            test_pts = dbo.all_pts
        
        #test_pts = ['905','906','907','908']
        #ALWAYS train on the HDRS17
        print('Making Training Set Data ' + str(train_pts))
        Otrain,Ctrain,_ = self.dsgn_O_C(train_pts,week_avg=avgweeks,ignore_flags=ignore_flags,circ=circ,scale='HDRS17')
        
        #Ctrain = sig.detrend(Ctrain) #this is ok to zscore here given that it's only across phases
                
        if method[0:3] == 'OLS':
            regmodel = linear_model.LinearRegression(normalize=True,copy_X=True,fit_intercept=True)
            scatter_alpha = 0.9
            method = method + circ
        elif method == 'RANSAC':
            regmodel = linear_model.RANSACRegressor(base_estimator=linear_model.Ridge(alpha=1.0,normalize=False,fit_intercept=True,copy_X=True),min_samples=inpercent,max_trials=1000)
            scatter_alpha = 0.1
        elif method == 'RIDGE':
            #regmodel = linear_model.Ridge(alpha=0.36,copy_X=True,fit_intercept=True,normalize=True)
            regmodel = linear_model.RidgeCV(alphas=np.linspace(0.1,0.7,50),fit_intercept=True,normalize=True,cv=10)
            scatter_alpha = 0.9
        elif method == 'LASSO':
            #regmodel = linear_model.LassoCV(alphas=np.linspace(0.0005,0.0015,50), copy_X=True,fit_intercept=True,normalize=True)
            regmodel = linear_model.Lasso(alpha=0.0095, copy_X=True,fit_intercept=True,normalize=True)
            scatter_alpha=0.9
        elif method == 'ENR_Osc':
            # Below works decently well
            #regmodel = linear_model.ElasticNetCV(l1_ratio=np.linspace(0.1,0.5,50), copy_X=True,fit_intercept=True,normalize=True,cv=10)
            
            regmodel = linear_model.ElasticNetCV(l1_ratio=np.linspace(0.1,0.8,50), copy_X=True,fit_intercept=True,normalize=True,cv=10)
            scatter_alpha=0.9
            
        
        
        #Do the model's fit
        #For this method, we'll do it on ALL available features
        regmodel.fit(Otrain,Ctrain.reshape(-1,1))
        
        #Test the model's performance in the other patients
        #Generate the testing set data
        self.Model = nestdict()
        
        self.test_MEAS = scale
        print('Making Testing Set Data: ' + str(test_pts))
        Otest,Ctest,labels = self.dsgn_O_C(test_pts,week_avg=avgweeks,circ=circ,ignore_flags=ignore_flags,scale=scale)
        #Shape the input oscillatory state vectors
        
        #Generate the predicted clinical states
        Cpredictions = regmodel.predict(Otest)
        
        #reshape to vector
        Ctest = Ctest.reshape(-1,1)
        Cpredictions = Cpredictions.reshape(-1,1)
        
        
        
        #generate the statistical correlation of the prediction vs the empirical HDRS17 score
        #statistical correlation
        
        #Detrend for stats
        #cpred_stats = sig.detrend(Cpredictions.reshape(-1,1),axis=0,type='linear')
        #ctest_stats = sig.detrend(Ctest.reshape(-1,1),axis=0,type='linear')

        #what if we do a final "logistic" part here...
        self.Model[method]['Model'] = regmodel
        if finalWrite:
            self.Model['FINAL']['Model'] = copy.deepcopy(regmodel)
        self.Model[method]['OTrain'] = Otrain
        self.Model[method]['Ctrain'] = Ctrain
        
        self.Model[method]['Otest'] = Otest
        self.Model[method]['TestPts'] = test_pts
        self.Model[method]['TrainPts'] = train_pts
        self.Model[method]['Circ'] = circ
        self.Model[method]['Labels'] = labels
        
        if lindetrend == 'None':
            self.Model[method]['Ctest'] = Ctest
            self.Model[method]['Cpredictions'] = Cpredictions
        elif lindetrend == 'Block':
            #go through each patient and detrend for each BLOCK
            for pp,pt in enumerate(test_pts):
                ctest_block = Ctest[28*pp:28*(pp+1)]
                Ctest[28*pp:28*(pp+1)] = sig.detrend(ctest_block,axis=0,type='linear')
                cpred_block = Cpredictions[28*pp:28*(pp+1)]
                Cpredictions[28*pp:28*(pp+1)] = sig.detrend(cpred_block,axis=0,type='linear')

            self.Model[method]['Ctest'] = Ctest
            self.Model[method]['Cpredictions'] = Cpredictions
        
        elif lindetrend == 'All':
            self.Model[method]['Ctest'] = sig.detrend(Ctest,axis=0,type='linear')
            self.Model[method]['Cpredictions'] = sig.detrend(Cpredictions,axis=0,type='linear')
        
        
        
        self.Model[method]['NORMALIZED'] = lindetrend
        
        
        #post-process the test and predicted things
        #Cpredictions = cpred_msub
        #Ctest = ctest_msub
        
        #now we can do other stuff I suppose...
    
    def Clinical_Summary(self,method='RIDGE',plot_indiv=False,ranson=True,doplot=True):
        print('Clinical Summary')
        Cpredictions = (self.Model[method]['Cpredictions'].reshape(-1,1))
        Ctest = (self.Model[method]['Ctest'].reshape(-1,1))

        
        #Do the stats here
        self.Model[method]['Performance']['PearsCorr'] = stats.pearsonr(Cpredictions.reshape(-1,1),Ctest.astype(float).reshape(-1,1))
        self.Model[method]['Performance']['SpearCorr'] = stats.spearmanr(Cpredictions.reshape(-1,1),Ctest.astype(float))
        self.Model[method]['Performance']['Permutation'] = self.shuffle_summary(method)
        
        #Then we do Spearman's R
        print('Spearmans R: ' + str(self.Model[method]['Performance']['SpearCorr']))
        print('Pearsons R: ' + str(self.Model[method]['Performance']['PearsCorr']))

        #self.Model.update({method:{'Performance':{'SpearCorr':spearcorr,'PearsCorr':pecorr,'Internal':0,'DProd':0}}})
        #self.Model['Performance'] = {'SpearCorr':spearcorr,'PearsCorr':pecorr,'Internal':0,'DProd':0}
        
        #just do straight up inner prod on detrended data
        
        
        #let's do internal scoring for a second
        self.Model[method]['Performance']['Internal'] = self.Model[method]['Model'].score(self.Model[method]['Otest'],self.Model[method]['Ctest'])
        #self.Model[method]['Performance']['DProd'] = np.dot(cpred_msub.T,ctest_msub)
        
        # Get the Test Patients used
        test_pts = self.Model[method]['TestPts']
        #Get the labels        
        labels = self.Model[method]['Labels']
        
        
        #Plot individually
        if plot_indiv:
        #do a plot for each patient on this?
            for pt in test_pts:
                plt.figure()
                #check if the sizes are right
                try:
                    assert len(Cpredictions) == len(labels['Patient'])
                except:
                    ipdb.set_trace()
                
                pt_preds = [cpred for cpred,pat in zip(Cpredictions,labels['Patient']) if pat == pt]
                pt_actuals = [ctest for ctest,pat in zip(Ctest,labels['Patient']) if pat == pt]
                
                
                plt.plot(pt_preds,label='Predicted')
                plt.plot(pt_actuals,label=self.test_MEAS)
                plt.legend()
                
                plt.xlabel('Week')
                plt.ylabel('Normalized Disease Severity')
                plt.suptitle(pt + ' ' + method)
                sns.despine()
                
                
        #Do summary plots
        
        if doplot:
            
            x,y = (1,1)
            if ranson:
                #THIS WORKS THE BEST!!
                #assesslr = linear_model.RANSACRegressor(base_estimator=linear_model.Ridge(alpha=0.2,fit_intercept=False),residual_threshold=0.16)
                #Find TOTAL MAD
                #Then find threshold ~20% of that
                
                assesslr = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=False),residual_threshold=0.16) #0.15 residual threshold works great!
            else:
                assesslr = linear_model.LinearRegression(fit_intercept=True)
                #assesslr = linear_model.TheilSenRegressor()
                
            
            #assesslr.fit(Ctest.reshape(-1,1),Cpredictions.reshape(-1,1))

            #Maybe flip this? Due to paper: https://www.researchgate.net/publication/230692926_How_to_Evaluate_Models_Observed_vs_Predicted_or_Predicted_vs_Observed
            assesslr.fit(Ctest,Cpredictions)
            
            
            line_x = np.linspace(-1,1,20).reshape(-1,1)
            line_y = assesslr.predict(line_x)
            
            
            #HANDLING OUTLIERS TO THE BIOMETRIC MODEL
            if ranson:
                inlier_mask = assesslr.inlier_mask_
                corrcoef = assesslr.estimator_.coef_[0]
            else:
                inlier_mask = np.ones_like(Ctest).astype(bool)
                corrcoef = assesslr.coef_[0]
                
            outlier_mask = np.logical_not(inlier_mask)
            
            #FINALLY just do a stats package linear regression
            if ranson:
                slsl,inin,rval,pval,stderr = stats.mstats.linregress(Ctest[inlier_mask].reshape(-1,1),Cpredictions[inlier_mask].reshape(-1,1))
                print('OUTLIER: ' + method + ' model has RANSAC ' + str(slsl) + ' correlation with real score (p < ' + str(pval) + ')')
            
            else:
                slsl,inin,rval,pval,stderr = stats.mstats.linregress(Ctest.reshape(-1,1),Cpredictions.reshape(-1,1))
                 
                print(method + ' model has OLS ' + str(slsl) + ' correlation with real score (p < ' + str(pval) + ')')
            
            self.Model[method]['Performance']['Regression'] = assesslr
            
            #do the permutation test
            
            
            #THESE TWO ARE THE SAME!!
            #print(method + ' model has ' + str(corrcoef) + ' correlation with real score (p < ' + str(pval) + ')')
            
            #PLOT the outlier points and their patient + phase
            outlier_phases = list(compress(labels['Phase'],outlier_mask))
            outlier_pt = list(compress(labels['Patient'],outlier_mask))
            #plotting work
            if method[0:3] != 'OLS':
                scatter_alpha = 0.8
            else:
                scatter_alpha = 0.05
                
            plt.figure()
            plt.scatter(Ctest[outlier_mask],Cpredictions[outlier_mask],alpha=scatter_alpha,color='gray')
            for ii, txt in enumerate(outlier_phases):
                plt.annotate(txt+'\n'+outlier_pt[ii],(Ctest[outlier_mask][ii],Cpredictions[outlier_mask][ii]),fontsize=8,color='gray')
                
            
            #Plot all the inliers now
            plt.scatter(Ctest[inlier_mask],Cpredictions[inlier_mask],alpha=scatter_alpha)
            plt.plot(np.linspace(-x,x,2),np.linspace(-y,y,2),alpha=0.2,color='gray')
            
            #This is the regression line itself
            plt.plot(line_x,line_y,color='green')
            plt.axes().set_aspect('equal')
            
            
            
            #Finally, let's label the clinician's changes
            #find out the points that the stim was changed
            stim_change_list = self.CFrame.Stim_Change_Table()
            ostimchange = []
            
            for ii in range(Ctest.shape[0]):
                if (labels['Patient'][ii],labels['Phase'][ii]) in stim_change_list:                
                    ostimchange.append(ii)
            
            if method[0:3] != 'OLS':
                
                for ii in ostimchange:
                    plt.annotate(labels['Patient'][ii] + ' ' + labels['Phase'][ii],(Ctest[ii],Cpredictions[ii]),fontsize=10,color='red')
                
            plt.scatter(Ctest[ostimchange],Cpredictions[ostimchange],alpha=scatter_alpha,color='red',marker='^',s=130)
            
            plt.xlabel(self.test_MEAS)
            plt.ylabel('Predicted')
            plt.xlim((-0.5,0.5))
            plt.ylim((-0.5,0.5))
            
            plt.suptitle(method + ' | recordings: ' + self.Model[method]['Circ'])
            #plt.title('All Observations')
            sns.despine()
            print('There are ' + str(sum(outlier_mask)/len(outlier_mask)*100) + '% outliers')
            plt.suptitle(method)
            
            #let's do a quick SWEEP on both CLIN MEASURE and our putative biometric to see which one yields a more congruent response
            
            #doppv = 'PPV'
            for doppv in ['PPV','NPV']:
            #ALL THIS DOES PPV
                if doppv == 'PPV':
                    thresh = np.linspace(-0.4,0.4,100)
                    cs_above = np.zeros_like(thresh)
                    putbm_above = np.zeros_like(thresh)
                    both_above = np.zeros_like(thresh)
                    cons_above = np.zeros_like(thresh)
                    
                    nochange = [x for x in range(Ctest.shape[0]) if x not in ostimchange]
                    disagree = 0.02
                    for tt,thr in enumerate(thresh):
                        cs_above[tt] = np.sum(Ctest[ostimchange] > thr) #TRUE POSITIVES
                        putbm_above[tt] = np.sum(Cpredictions[ostimchange] > thr)
                        both_above[tt] = np.sum(np.logical_and(Cpredictions[ostimchange] > thr,Ctest[ostimchange] > thr))
                        cons_above[tt] = np.sum(np.logical_and((Cpredictions[ostimchange] - Ctest[ostimchange])**2 < disagree,Ctest[ostimchange] > thr))
                        
                    nccs_above = np.zeros_like(thresh)
                    ncputbm_above = np.zeros_like(thresh)
                    ncboth_above = np.zeros_like(thresh)
                    nccons_above = np.zeros_like(thresh)
                    
                    for tt,thr in enumerate(thresh):
                        nccs_above[tt] = np.sum(Ctest[nochange] > thr) #FALSE POSITIVES
                        ncputbm_above[tt] = np.sum(Cpredictions[nochange] > thr)
                        ncboth_above[tt] = np.sum(np.logical_and(Cpredictions[nochange] > thr,Ctest[nochange] > thr))
                        nccons_above[tt] = np.sum(np.logical_and((Cpredictions[nochange] - Ctest[nochange])**2 < disagree,Ctest[nochange] > thr))
                        
                    plt.figure()
                    plt.plot(thresh,cs_above,label='Standard')
                    plt.plot(thresh,putbm_above,label='Putative Alone')
                    plt.plot(thresh,both_above,label='Proposed')
                    #plt.plot(thresh,cons_above,label='ForFun')
                    plt.legend()
                    
                    plt.figure()
                    #TRUE POSITIVE / (ALL POSITIVES)
                    plt.plot(thresh,cs_above/(cs_above + nccs_above),label='Standard')
                    plt.plot(thresh,putbm_above/(putbm_above+ncputbm_above),label='Putative Alone')
                    plt.plot(thresh,both_above/(both_above+ncboth_above),label='Proposed')
                    #plt.plot(thresh,cons_above/(cons_above+nccons_above),label='ForFun')
                    plt.legend()
                    plt.suptitle('PPV')
                    #print('Outlier phases are: ' + str(outlier_phases))
                elif doppv == 'NPV':
                    thresh = np.linspace(-0.4,0.4,100)
                    cs_above = np.zeros_like(thresh)
                    putbm_above = np.zeros_like(thresh)
                    both_above = np.zeros_like(thresh)
                    cons_above = np.zeros_like(thresh)
                    
                    nochange = [x for x in range(Ctest.shape[0]) if x not in ostimchange]
                    disagree = 0.02
                    for tt,thr in enumerate(thresh):
                        cs_above[tt] = np.sum(Ctest[ostimchange] < thr) #THESE ARE FALSE NEGATIVES
                        putbm_above[tt] = np.sum(Cpredictions[ostimchange] < thr)
                        both_above[tt] = np.sum(np.logical_and(Cpredictions[ostimchange] < thr,Ctest[ostimchange] < thr))
                        #cons_above[tt] = np.sum(np.logical_and((Cpredictions[ostimchange] - Ctest[ostimchange])**2 < disagree,Ctest[ostimchange] > thr))
                        
                    nccs_above = np.zeros_like(thresh)
                    ncputbm_above = np.zeros_like(thresh)
                    ncboth_above = np.zeros_like(thresh)
                    nccons_above = np.zeros_like(thresh)
                    
                    for tt,thr in enumerate(thresh):
                        nccs_above[tt] = np.sum(Ctest[nochange] < thr) #TRUE NEGATIVES
                        ncputbm_above[tt] = np.sum(Cpredictions[nochange] < thr)
                        ncboth_above[tt] = np.sum(np.logical_and(Cpredictions[nochange] < thr,Ctest[nochange] < thr))
                        nccons_above[tt] = np.sum(np.logical_and((Cpredictions[nochange] - Ctest[nochange])**2 < disagree,Ctest[nochange] > thr))
                        
                    plt.figure()
                    plt.plot(thresh,cs_above,label='Standard')
                    plt.plot(thresh,putbm_above,label='Putative Alone')
                    plt.plot(thresh,both_above,label='Proposed')
                    #plt.plot(thresh,cons_above,label='ForFun')
                    plt.legend()
                    
                    plt.figure()
                    plt.plot(thresh,nccs_above/(cs_above + nccs_above),label='Standard')
                    plt.plot(thresh,ncputbm_above/(putbm_above+ncputbm_above),label='Putative Alone')
                    plt.plot(thresh,ncboth_above/(both_above+ncboth_above),label='Proposed')
                    #plt.plot(thresh,cons_above/(cons_above+nccons_above),label='ForFun')
                    plt.legend()
                    plt.suptitle('NPV')
                    #print('Outlier phases are: ' + str(outlier_phases))
                elif doppv == 'Sens':
                    thresh = np.linspace(-0.4,0.4,100)
                    cs_above = np.zeros_like(thresh)
                    putbm_above = np.zeros_like(thresh)
                    both_above = np.zeros_like(thresh)
                    cons_above = np.zeros_like(thresh)
                    
                    nochange = [x for x in range(Ctest.shape[0]) if x not in ostimchange]
                    disagree = 0.02
                    for tt,thr in enumerate(thresh):
                        cs_above[tt] = np.sum(Ctest[ostimchange] > thr) #TRUE POSITIVES
                        putbm_above[tt] = np.sum(Cpredictions[ostimchange] > thr)
                        both_above[tt] = np.sum(np.logical_and(Cpredictions[ostimchange] > thr,Ctest[ostimchange] > thr))
                        #cons_above[tt] = np.sum(np.logical_and((Cpredictions[ostimchange] - Ctest[ostimchange])**2 < disagree,Ctest[ostimchange] > thr))
                        
                    nccs_above = np.zeros_like(thresh)
                    ncputbm_above = np.zeros_like(thresh)
                    ncboth_above = np.zeros_like(thresh)
                    nccons_above = np.zeros_like(thresh)
                    
                    for tt,thr in enumerate(thresh):
                        nccs_above[tt] = np.sum(Ctest[ostimchange] < thr) #FALSE NEGATIVES
                        ncputbm_above[tt] = np.sum(Cpredictions[ostimchange] < thr)
                        ncboth_above[tt] = np.sum(np.logical_and(Cpredictions[ostimchange] < thr,Ctest[ostimchange] < thr))
                        #nccons_above[tt] = np.sum(np.logical_and((Cpredictions[nochange] - Ctest[nochange])**2 < disagree,Ctest[nochange] > thr))
                        
                    plt.figure()
                    plt.plot(thresh,cs_above,label='Standard')
                    plt.plot(thresh,putbm_above,label='Putative Alone')
                    plt.plot(thresh,both_above,label='Proposed')
                    #plt.plot(thresh,cons_above,label='ForFun')
                    plt.legend()
                    
                    plt.figure()
                    plt.plot(thresh,cs_above/(cs_above + nccs_above),label='Standard')
                    plt.plot(thresh,putbm_above/(putbm_above+ncputbm_above),label='Putative Alone')
                    plt.plot(thresh,both_above/(both_above+ncboth_above),label='Proposed')
                    #plt.plot(thresh,cons_above/(cons_above+nccons_above),label='ForFun')
                    plt.legend()
                    plt.suptitle('')
                    #print('Outlier phases are: ' + str(outlier_phases))
                elif doppv == 'Spec':
                    thresh = np.linspace(-0.4,0.4,100)
                    cs_above = np.zeros_like(thresh)
                    putbm_above = np.zeros_like(thresh)
                    both_above = np.zeros_like(thresh)
                    cons_above = np.zeros_like(thresh)
                    
                    nochange = [x for x in range(Ctest.shape[0]) if x not in ostimchange]
                    disagree = 0.02
                    for tt,thr in enumerate(thresh):
                        cs_above[tt] = np.sum(Ctest[nochange] > thr) #FALSE POSITIVES
                        putbm_above[tt] = np.sum(Cpredictions[nochange] > thr)
                        both_above[tt] = np.sum(np.logical_and(Cpredictions[nochange] > thr,Ctest[nochange] > thr))
                        #cons_above[tt] = np.sum(np.logical_and((Cpredictions[ostimchange] - Ctest[ostimchange])**2 < disagree,Ctest[ostimchange] > thr))
                        
                    nccs_above = np.zeros_like(thresh)
                    ncputbm_above = np.zeros_like(thresh)
                    ncboth_above = np.zeros_like(thresh)
                    nccons_above = np.zeros_like(thresh)
                    
                    for tt,thr in enumerate(thresh):
                        nccs_above[tt] = np.sum(Ctest[nochange] < thr) #True NEGATIVES
                        ncputbm_above[tt] = np.sum(Cpredictions[nochange] < thr)
                        ncboth_above[tt] = np.sum(np.logical_and(Cpredictions[nochange] < thr,Ctest[nochange] < thr))
                        #nccons_above[tt] = np.sum(np.logical_and((Cpredictions[nochange] - Ctest[nochange])**2 < disagree,Ctest[nochange] > thr))
                        
                    plt.figure()
                    plt.plot(thresh,cs_above,label='Standard')
                    plt.plot(thresh,putbm_above,label='Putative Alone')
                    plt.plot(thresh,both_above,label='Proposed')
                    #plt.plot(thresh,cons_above,label='ForFun')
                    plt.legend()
                    
                    plt.figure()
                    plt.plot(thresh,nccs_above/(cs_above + nccs_above),label='Standard')
                    plt.plot(thresh,ncputbm_above/(putbm_above+ncputbm_above),label='Putative Alone')
                    plt.plot(thresh,ncboth_above/(both_above+ncboth_above),label='Proposed')
                    #plt.plot(thresh,cons_above/(cons_above+nccons_above),label='ForFun')
                    plt.legend()
                    plt.suptitle('')
                    #print('Outlier phases are: ' + str(outlier_phases))
                    
        return copy.deepcopy(self.Model[method]['Performance'])

    def Model_Validation(self,method,scale='HDRS17',do_detrend='None',do_plots=False,randomize=0.0,show_clin=True):
        #This function does the final model validation on the held out validation set
        Oval,Cval_base,labels_val = self.dsgn_O_C(['901','903','905','906','907','908'],week_avg=True,circ='day',ignore_flags=True,scale=scale,from_set='VALIDATION',randomize=randomize)
        
        #HARD CHANGE MODEL COEFFICIENTS
        #self.Model[method]['Model'].coef_ = np.array([[-0.00583578, -0.00279751,  0.00131825,  0.01770169,  0.01166687],[-1.06586005e-02,  2.42700023e-05,  7.31445236e-03,  2.68723035e-03,-3.90440108e-06]]).reshape(-1)
        #print(Oval.shape)
        
        Cpred_base = self.Model[method]['Model'].predict(Oval)
        self.Model['RANDOM']['Cpred'] = self.Model['RANDOM']['Model'].predict(Oval)
        
        Cval = copy.deepcopy(Cval_base)
        Cpred = copy.deepcopy(Cpred_base)
        
        ## Normalize here
        Cval = self.norm_func(Cval)
        Cpred = self.norm_func(Cpred)
        
        
        ### Do we want to plot the scatter here?
        
        ### DETREND CODE HERE
        if do_detrend == 'None':
            pass
        elif do_detrend == 'Block':
            #go through each patient and detrend for each BLOCK
            try:
                for pp,pt in enumerate(dbo.all_pts):
                    cval_block = Cval[28*pp:28*(pp+1)]
                    Cval[28*pp:28*(pp+1)] = sig.detrend(cval_block,axis=0,type='linear')
                    cpred_block = Cpred[28*pp:28*(pp+1)]
                    Cpred[28*pp:28*(pp+1)] = sig.detrend(cpred_block,axis=0,type='linear')
            except:
                ipdb.set_trace()
        
        elif do_detrend == 'All':
            Cval = sig.detrend(Cval,axis=0,type='constant')
            Cpred = sig.detrend(Cpred,axis=0,type='constant')
        
        
        
        ### Plotting and actual summary results
        if do_plots:
            self.Pred_vs_Meas_NEW(Cpred,Cval,labels_val,show_clin=show_clin)
        
        pr_aucs = self.algo_perfs(Cpred,Cval,labels_val,do_plots,Crand=True)
        pr_null = self.null_algo(Cpred,Cval,labels_val)
        plt.suptitle('Actual Model')
        return pr_aucs, pr_null
        #self.algo_perfs(Cpred_random,Cval,labels_val)
        #plt.suptitle('Random Model')
        
    def rans_assess_PvM(self,Cpred,Cmeas,labels,usefig):
        assesslr = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=True),residual_threshold=1.5,min_samples=0.8,loss='absolute_loss') #0.15 residual threshold works great!
        
        assesslr.fit(Cmeas.reshape(-1,1),Cpred.reshape(-1,1))
        
        #Plot the correlation line
        line_x = np.linspace(-1,1,20).reshape(-1,1)
        line_y = assesslr.predict(line_x)
        
        plt.figure(usefig.number)
        plt.plot(line_x,line_y)
        
        #Find inliers and outliers
        inlier_mask = assesslr.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        
        slsl,inin,rval,pval,stderr = stats.mstats.linregress(Cmeas[inlier_mask].reshape(-1,1),Cpred[inlier_mask].reshape(-1,1),)
        
        rsac_stats = {'Slope':slsl,'Intercept':inin,'Rval':rval,'Pval':pval,'SE':stderr}
        
        return outlier_mask, inlier_mask, rsac_stats
        
    def clin_changes(self,Cpred,Cmeas,labels,usefig=[],method='RIDGE',measure='HDRS17',doplot=True):
        #THIS NEEDS TO BE SHIFTED TO THE CVECT HANDLING FUNCTIONS FOR STIM CHANGES THAT CORRECT FOR SHIFTS
        stim_change_list = self.CFrame.Stim_Change_Table()
        ostimchange = []
        
        for ii in range(Cmeas.shape[0]):
            if (labels['Patient'][ii],labels['Phase'][ii]) in stim_change_list:
                ostimchange.append(ii)
        
        if doplot:
            plt.figure(usefig.number)
            for ii in ostimchange:
                plt.annotate(labels['Patient'][ii] + ' ' + labels['Phase'][ii],(Cmeas[ii],Cpred[ii]),fontsize=10,color='red')
            
            plt.scatter(Cmeas[ostimchange],Cpred[ostimchange],alpha=0.3,color='red',marker='^',s=130)
            plt.xlabel(measure)
            plt.ylabel('Predicted')
        
        return ostimchange
        
    
    def Pred_vs_Meas_NEW(self,Cpred,Cmeas,labels,show_clin=True):
        Cpredictions = Cpred.reshape(-1,1)
        Ctest = Cmeas.reshape(-1,1)

        #self.Model.update({method:{'Performance':{'SpearCorr':spearcorr,'PearsCorr':pecorr,'Internal':0,'DProd':0}}})
        #self.Model['Performance'] = {'SpearCorr':spearcorr,'PearsCorr':pecorr,'Internal':0,'DProd':0}
        
        #just do straight up inner prod on detrended data
        
        
        #let's do internal scoring for a secon
        #self.Model[method]['Performance']['DProd'] = np.dot(cpred_msub.T,ctest_msub)
        
        # Get the Test Patients used
        test_pts = dbo.all_pts
        #Get the labels        
        labels = labels
        
        
        #Plot individually
        for pt in test_pts:
            plt.figure()
            #check if the sizes are right
            try:
                assert len(Cpredictions) == len(labels['Patient'])
            except:
                ipdb.set_trace()
            
            pt_preds = [cpred for cpred,pat in zip(Cpredictions,labels['Patient']) if pat == pt]
            pt_actuals = [ctest for ctest,pat in zip(Ctest,labels['Patient']) if pat == pt]
            
            
            plt.plot(pt_preds,label='Predicted')
            plt.plot(pt_actuals,label=self.test_MEAS)
            #plt.plot(np.vstack((np.array([0]).reshape(-1,1),np.diff(np.array(pt_preds),axis=0))),label='Diff')
            plt.legend()
            
            plt.xlabel('Week')
            plt.ylabel('Normalized Disease Severity')
            plt.suptitle(pt)
            sns.despine()
                
   
        x,y = (1,1)
        
        
        ###
        #sweep threshold
        if 1:

            outlier_percent = []
            sweep_thresh = np.linspace(0.1,5,100)
            corr_level = []
            
            for rthresh in sweep_thresh:
                assesslr = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=False),residual_threshold=rthresh) #0.15 residual threshold works great!
                assesslr.fit(Ctest,Cpredictions)
                inlier_mask = assesslr.inlier_mask_      
                
                slsl,inin,rval,pval,stderr = stats.mstats.linregress(Ctest[inlier_mask].reshape(-1,1),Cpredictions[inlier_mask].reshape(-1,1))
                
                outlier_mask = np.logical_not(inlier_mask)
                outlier_percent.append(100*np.sum(outlier_mask)/len(outlier_mask))
                corr_level.append(slsl)
            ###
            plt.figure()
            fig, ax1=plt.subplots()
            
            ax1.plot(sweep_thresh,outlier_percent,color='blue',label='Outlier %')
            ax2 = ax1.twinx()
            ax2.plot(sweep_thresh,corr_level,color='green',label='Correlation')
            
        assesslr = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=False),residual_threshold=1) #0.15 residual threshold works great!
        #Maybe flip this? Due to paper: https://www.researchgate.net/publication/230692926_How_to_Evaluate_Models_Observed_vs_Predicted_or_Predicted_vs_Observed
        assesslr.fit(Ctest,Cpredictions)
        
        
        line_x = np.linspace(-1,1,20).reshape(-1,1)
        line_y = assesslr.predict(line_x)
        
        
        #HANDLING OUTLIERS TO THE BIOMETRIC MODEL
        inlier_mask = assesslr.inlier_mask_
        corrcoef = assesslr.estimator_.coef_[0]
      
            
        outlier_mask = np.logical_not(inlier_mask)
        

        slsl,inin,rval,pval,stderr = stats.mstats.linregress(Ctest.reshape(-1,1),Cpredictions.reshape(-1,1))
        print('ALL: ' + ' model has RANSAC ' + str(slsl) + ' correlation with real score (p < ' + str(pval) + ')')
        
        slsl,inin,rval,pval,stderr = stats.mstats.linregress(Ctest[inlier_mask].reshape(-1,1),Cpredictions[inlier_mask].reshape(-1,1))
        print('OUTLIER: ' + ' model has RANSAC ' + str(slsl) + ' correlation with real score (p < ' + str(pval) + ')')
        
        #self.Model[method]['Performance']['Regression'] = assesslr
        
        #do the permutation test
        
        
        #THESE TWO ARE THE SAME!!
        #print(method + ' model has ' + str(corrcoef) + ' correlation with real score (p < ' + str(pval) + ')')
        
        #PLOT the outlier points and their patient + phase
        outlier_phases = list(compress(labels['Phase'],outlier_mask))
        outlier_pt = list(compress(labels['Patient'],outlier_mask))
        
        ##What weeks are the outliers in?
        outlier_week_num = [int(weeknum[-2:]) for weeknum in outlier_phases]
        plt.figure()
        plt.hist(outlier_week_num)
        
        #plotting work

        scatter_alpha = 0.2

            
        plt.figure()
        plt.scatter(Ctest[outlier_mask],Cpredictions[outlier_mask],alpha=scatter_alpha,color='gray')
        for ii, txt in enumerate(outlier_phases):
            plt.annotate(txt+'\n'+outlier_pt[ii],(Ctest[outlier_mask][ii],Cpredictions[outlier_mask][ii]),fontsize=12,color='gray')
            
        
        #Plot all the inliers now
        plt.scatter(Ctest[inlier_mask],Cpredictions[inlier_mask],alpha=scatter_alpha)
        plt.plot(np.linspace(-x,x,2),np.linspace(-y,y,2),alpha=0.2,color='gray')
        
        #This is the regression line itself
        plt.plot(line_x,line_y,color='green')
        plt.axes().set_aspect('equal')
        plt.annotate(s=str(100*np.sum(outlier_mask)/len(outlier_mask)) + '% outliers',xy=(-3,0),fontsize=12)
        
        
        #Finally, let's label the clinician's changes
        #find out the points that the stim was changed
        if show_clin:
            stim_change_list = self.CFrame.Stim_Change_Table()
            ostimchange = []
            
            for ii in range(Ctest.shape[0]):
                if (labels['Patient'][ii],labels['Phase'][ii]) in stim_change_list:                
                    ostimchange.append(ii)
            
            for ii in ostimchange:
                plt.annotate(labels['Patient'][ii] + ' ' + labels['Phase'][ii],(Ctest[ii],Cpredictions[ii]),fontsize=12,color='red')
                
            plt.scatter(Ctest[ostimchange],Cpredictions[ostimchange],alpha=scatter_alpha,color='red',marker='^',s=130)
        
        plt.xlabel(self.test_MEAS)
        plt.ylabel('Predicted')
        #plt.xlim((-4,4))
        #plt.ylim((-4,4))
        #plt.title('All Observations')
        sns.despine()
        print('There are ' + str(sum(outlier_mask)/len(outlier_mask)*100) + '% outliers')
            
        
    def plot_Pred_vs_Meas(self,Cpred,Cmeas,labels,plot_type='scatter'):
        main_clin_fig = plt.figure()
        
        
        if plot_type == 'scatter':
            outlier_mask, inlier_mask, rsac_stats = self.rans_assess_PvM(Cpred,Cmeas,labels,usefig=main_clin_fig)
            outlier_phases = list(compress(labels['Phase'],outlier_mask))
            outlier_pt = list(compress(labels['Patient'],outlier_mask))
            
            #print(rsac_stats)
            
            plt.annotate(s=str(100*np.sum(outlier_mask)/len(outlier_mask)) + '% outliers',xy=(0,0.5),fontsize=8)
            
            plt.scatter(Cmeas[outlier_mask],Cpred[outlier_mask],alpha=0.3,color='gray')
            for ii, txt in enumerate(outlier_phases):
                plt.annotate(txt+'\n'+outlier_pt[ii],(Cmeas[outlier_mask][ii],Cpred[outlier_mask][ii]),fontsize=8,color='gray')
            plt.scatter(Cmeas[inlier_mask],Cpred[inlier_mask],alpha=0.3)
            #plt.ylim((0,0.5))
            #plt.xlim((0,0.5))
            plt.plot(np.linspace(-1,1,20),np.linspace(-1,1,20),alpha=0.2,color='gray')
            
            _ = self.clin_changes(Cpred,Cmeas,labels,usefig=main_clin_fig)
            
            plt.axes().set_aspect('equal')
            print(stats.spearmanr(Cmeas,Cpred))
            print(rsac_stats['Slope'])
        else:
            plt.plot(Cmeas)
            plt.plot(Cpred)
            
    
    def algo_pers_shuffle_change(self,Cpred,Cmeas,labels):
        ostimchange = self.clin_changes(Cpred,Cmeas,labels,doplot=False)
        stimstay = [x for x in range(Cmeas.shape[0]) if x not in ostimchange]
        stimchange = [x for x in range(Cmeas.shape[0]) if x in ostimchange]
        
        #Now, generate some Random stay and change vectors, with the same sparsity
        randstim_change = random.shuffle(ostimchange)
        rand_stimstay = [x for x in range(Cmeas.shape[0]) if x not in randstim_change]
        rand_stimchange = [x for x in range(Cmeas.shape[0]) if x in randstim_change]
        
    def null_algo(self,Cpred,Cmeas,labels):
        #how many changes are there?
        # Choose a set of ~9
        
        ostimchange = self.clin_changes(Cpred,Cmeas,labels,doplot=False)
        stimstay = [x for x in range(Cmeas.shape[0]) if x not in ostimchange]
        stimchange = [x for x in range(Cmeas.shape[0]) if x in ostimchange]
        
        true_change = np.zeros(Cmeas.shape)
        true_change[stimchange] = 1
        
        fake_change = np.zeros(Cmeas.shape)
        fake_change_idxs = np.random.randint(Cmeas.shape[0],size=(len(stimchange)))
        fake_change[fake_change_idxs] = 1
        
        precision,recall,_=precision_recall_curve(true_change,fake_change)
        prauc = auc(precision,recall,reorder=True)
        avg_precision = average_precision_score(true_change,fake_change)
        
        return avg_precision
        
    def algo_perfs(self,Cpred,Cmeas,labels,do_plot=True,Crand = False):
        ostimchange = self.clin_changes(Cpred,Cmeas,labels,doplot=False)
        stimstay = [x for x in range(Cmeas.shape[0]) if x not in ostimchange]
        stimchange = [x for x in range(Cmeas.shape[0]) if x in ostimchange]
        
        #output the PR curves using this dict
        pr_curves = nestdict()
        
        true_change = np.zeros(Cmeas.shape)
        true_change[stimchange] = 1
        
        thresh = np.linspace(-1,1,100)
        
        Cmeas = Cmeas.reshape(-1,1)
        Cpred = Cpred.reshape(-1,1)
        #now make one that has, per observation, the MINIMUM
        Cmin = np.min(np.hstack((Cmeas,Cpred)),axis=1).reshape(-1,1)
        Coff = np.abs(Cpred - Cmeas) * np.sin(np.pi/4)
        tpfn_plot = False
        
        if do_plot:
            if tpfn_plot: summary = plt.figure()
            roc_plots = plt.figure()
        unif = np.random.uniform(-2.0,2.0,size=Cmeas.size)
        
        if Crand:
            do_algos = [Cmeas,Cpred,unif,self.Model['RANDOM']['Cpred'],Cmin,Coff]
            algo_list = ['HDRS','CB','Random','RandomLin','Minimum','OffDiag']
            #Fixed rand is the one where we go in and randomly select an N number of indices to say stim was changed on, and then choose N based off of how many actual changes there were
            
        else:
            do_algos = [Cmeas,Cpred,unif]
            algo_list = ['HDRS','CB','Random']
        
        #trutru = Cmeas
        algo_avg_prec = []
        
        for algon,trutru in enumerate(do_algos):#,np.max(np.vstack((Cmeas.T,Cpred.T)),axis=0),(Cmeas + Cpred)]):
            change_above = []
            norm_above = []
            change_below = []
            norm_below = []
            
            for tt,thr in enumerate(thresh):
                #count above
                change_above.append(np.sum(trutru[ostimchange] > thr))
                norm_above.append(np.sum(trutru[stimstay] > thr))
                
                #count below
                change_below.append(np.sum(trutru[ostimchange] < thr))
                norm_below.append(np.sum(trutru[stimstay] < thr))
                
                #ROC stuff
            
            #fpr,tpr,thresholds = roc_curve(true_change,trutru)
            #print(metrics.auc(fpr,tpr))
            #plt.figure(roc_plots.number)
            #plt.plot(fpr,tpr)
            
            #WHAT IF WE SHUFFLE THE TRUE CHANGE
            rand_change = np.zeros_like(Cmeas)
            
            #rand_idxs = np.random.randint(0,Cmeas.size,size=(8,1))
            
            rand_idxs = random.sample(range(Cmeas.shape[0]),8)
            rand_change[rand_idxs] = 1
            
            
            precision,recall,_=precision_recall_curve(true_change,trutru)
            pr_curves[algo_list[algon]] = (precision,recall)
            
            #prauc = auc(precision,recall,reorder=True)
            avg_precision = average_precision_score(true_change,trutru)
            
            if do_plot:
                plt.figure(roc_plots.number)            
                plt.step(recall,precision,alpha=1,where='post',label=algon)
                plt.legend(algo_list)
                plt.title('Precision Recall Plots')
            #print('Algo: ' + algo_list[algon] + ' Precision-Recall Average: ' + str(avg_precision) + ' AUC: ' + str(prauc))
            
            #HERE is where you decide to do either avg_precision or prauc
            algo_avg_prec.append(avg_precision)
            
            change_above = np.array(change_above)
            norm_above = np.array(norm_above)
            change_below = np.array(change_below)
            norm_below = np.array(norm_below)
            
            
            if tpfn_plot:
                plt.figure(summary.number)
                plt.subplot(2,2,1)
                plt.plot(thresh,change_above)
                plt.title('Change Above (TP)')
                plt.subplot(2,2,2)
                plt.plot(thresh,change_above / np.float64(change_above + norm_above))
                plt.title('PPV')
                plt.subplot(2,2,3)
                plt.plot(thresh,(change_below))
                plt.title('Change Below (FN)')
                plt.subplot(2,2,4)
                plt.plot(thresh,norm_below / np.float64(norm_below + change_below),label=algon)
                plt.title('NPV')
                plt.suptitle(str(algon))
                plt.legend()
        
        return algo_avg_prec, pr_curves
            