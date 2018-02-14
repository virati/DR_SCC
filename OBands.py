#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:05:07 2018

@author: virati
Traditional Analyses
This provides the main class used to do analysis and plot analyses related to "traditional" approaches
Linear Regression approaches will extend this class
"""
import sys

sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

from DBS_Osc import unity,displog
import scipy.stats as stats

import pdb
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

#sns.set()
sns.set_style('white')
sns.set_context('talk')

class OBands:
    def __init__(self,BRFrame):
        #Bring in the BR Data Frame
        self.BRFrame = BRFrame
        
    def OBSfeat_extract(self):
        big_list = self.BRFrame.file_meta
        for rr in big_list:
            feat_dict = {key:[] for key in dbo.feat_dict.keys()}
            for featname,dofunc in dbo.feat_dict.items():
                datacontainer = {ch: rr['Data'][ch] for ch in rr['Data'].keys()}
                feat_dict[featname] = dofunc['fn'](datacontainer,self.BRFrame.data_basis,dofunc['param'])
            rr.update({'FeatVect':feat_dict})
         

#Standard two state/categorical analyses HERE
    def mean_psds(self,patients='all',weeks=['C01','C23']):
        if patients == 'all':
            patients = dbo.all_pts
            
        if weeks[0] == 'DEPR' or weeks[0] == 'noDEPR':
            #FIND the weeks that are being plotted
            pass
        
        fmeta = self.BRFrame.file_meta
        
        pt_weeks = defaultdict(dict)
        for pt in patients:
            pt_weeks[pt] = {weeks[0]:0,weeks[1]:0}
            
            for ww in weeks:
                all_psds = [(rr['Data']['Left'],rr['Data']['Right']) for rr in fmeta if rr['Phase'] == ww and rr['Patient'] == pt]
                all_psds = np.array(all_psds)
                pt_weeks[pt][ww] = np.mean(all_psds,0)
            
        #now go in and vectorize/make matrix
        #pdb.set_trace()
        week_avg = {key:0 for key in weeks}
        for ww in weeks:
            week_avg[ww] = np.array([val[ww] for key,val in pt_weeks.items()])
        
        #pdb.set_trace()
        #MAIN PLOTTING
        fvect = self.BRFrame.data_basis
        plt.figure()
        for ch,chann in enumerate(['Left','Right']):
            plt.subplot(1,2,ch+1)
            w0psds = np.squeeze(np.array([10*np.log10(val[:,ch,:].T) for key,val in week_avg.items() if key == weeks[0]]))
            w1psds = np.squeeze(np.array([10*np.log10(val[:,ch,:].T) for key,val in week_avg.items() if key == weeks[1]]))
            
            w0ptavg = np.mean(w0psds,1)
            w1ptavg = np.mean(w1psds,1)
            
            [plt.plot(fvect,10*np.log10(val[:,ch,:].T),color='black',alpha=0.1,label=weeks[0]) for key,val in week_avg.items() if key == weeks[0]]
            plt.plot(fvect,w0ptavg,color='black',linewidth=2)            
            [plt.plot(fvect,10*np.log10(val[:,ch,:].T),color='green',alpha=0.1,label=weeks[1]) for key,val in week_avg.items() if key == weeks[1]]
            plt.plot(fvect,w1ptavg,color='green',linewidth=2)
            plt.xlim((0,50))
            plt.ylim((-80,0))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')
            plt.title(chann)
            sns.despine(top=True,right=True)
            
            #diffPSDs = pt x 513 x 2chann x 2timepoints
            #to do per-patient baseline subtraction
            
            
        plt.suptitle('Per Patient PSDs for: ' + str(weeks))
        
    def scatter_state(self,weeks='all',pt='all',feat='Alpha',circ='',plot=True,plot_type='scatter'):
        #generate our data to visualize
        if weeks == 'all':
            weeks = dbo.Phase_List('ephys')
        if pt == 'all':
            pt = dbo.all_pts
        
        fmeta = self.BRFrame.file_meta
        feats = {'Left':0,'Right':0}
        
        if feat == 'fSlope' or feat == 'nFloor':
            dispfunc = unity
        else:
            dispfunc = displog
        
        #do day and night here
        if circ != '':
            fdnmeta = [rr for rr in fmeta if rr['Circadian'] == circ]
            #circ_recs = -(len(fdnmeta) - len(fmeta))
            #print(len(fmeta))
            #print(circ_recs)
        else:
            fdnmeta = fmeta
        
        feats['Left'] = [(dispfunc(rr['FeatVect'][feat]['Left']),rr['Phase']) for rr in fdnmeta if rr['Patient'] in pt and rr['Phase'] in weeks]
        feats['Right'] = [(dispfunc(rr['FeatVect'][feat]['Right']),rr['Phase']) for rr in fdnmeta if rr['Patient'] in pt and rr['Phase'] in weeks]

        
        #plot the figure
        if plot:
            plt.figure()
            for cc,ch in enumerate(['Left','Right']):
                plt.subplot(1,2,cc+1)
                week_data = [b for (a,b) in feats[ch]]
                feat_data = np.array([a for (a,b) in feats[ch]])
                #pdb.set_trace()
                if plot_type == 'scatter':
                    plt.scatter(week_data,feat_data,alpha=0.1)
                    #stats time here
                    weekdistr = {week:[a for (a,b) in feats[ch] if b == week] for week in weeks}
                    outstats = stats.ks_2samp(weekdistr[weeks[0]],weekdistr[weeks[1]])
                    print(outstats)
                    sns.despine(top=True,right=True)
                elif plot_type == 'boxplot':
                    #THIS PART IS CLUGE
                    ser1 = np.array([fd for fd,weeek in zip(feat_data,week_data) if weeek == weeks[0]])
                    ser2 = np.array([fd for fd,weeek in zip(feat_data,week_data) if weeek == weeks[1]])
                    plt.boxplot([ser1,ser2])
                    plt.ylim((-10,-3.5))
                    plt.xticks([1,2],[weeks[0],weeks[1]])
                    print(stats.ttest_ind(ser1,ser2))
                    
                plt.title(ch)
                plt.xlabel('Week')
                plt.ylabel('Power (dB)')

                
            plt.suptitle(feat + ' over weeks; ' + str(pt))
                    
        
        return feats