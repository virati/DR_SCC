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

from DBS_Osc import unity
import scipy.stats as stats

import pdb
import numpy as np

import matplotlib.pyplot as plt

class OBands:
    def __init__(self,BRFrame):
        #Bring in the BR Data Frame
        self.BRFrame = BRFrame
        
    def feat_extract(self):
        big_list = self.BRFrame.file_meta
        for rr in big_list:
            feat_dict = {key:[] for key in dbo.feat_dict.keys()}
            for featname,dofunc in dbo.feat_dict.items():
                datacontainer = {ch: rr['Data'][ch] for ch in rr['Data'].keys()}
                feat_dict[featname] = dofunc['fn'](datacontainer,self.BRFrame.data_basis,dofunc['param'])
            rr.update({'FeatVect':feat_dict})
            
        
    def scatter_state(self,week='all',pt='all',feat='Alpha',circ='',plot=True,plot_type='scatter'):
        #generate our data to visualize
        if week == 'all':
            week = dbo.Phase_List('ephys')
        if pt == 'all':
            pt = dbo.all_pts
        
        fmeta = self.BRFrame.file_meta
        feats = {'Left':0,'Right':0}
        
        if feat == 'fSlope' or feat == 'nFloor':
            dispfunc = unity
        else:
            dispfunc = np.log10
        
        #do day and night here
        if circ != '':
            fdnmeta = [rr for rr in fmeta if rr['Circadian'] == circ]
            #circ_recs = -(len(fdnmeta) - len(fmeta))
            #print(len(fmeta))
            #print(circ_recs)
        else:
            fdnmeta = fmeta
        
        feats['Left'] = [(dispfunc(rr['FeatVect'][feat]['Left']),rr['Phase']) for rr in fdnmeta if rr['Patient'] in pt and rr['Phase'] in week]
        feats['Right'] = [(dispfunc(rr['FeatVect'][feat]['Right']),rr['Phase']) for rr in fdnmeta if rr['Patient'] in pt and rr['Phase'] in week]

        
        #plot the figure
        if plot:
            plt.figure()
            for cc,ch in enumerate(['Left','Right']):
                plt.subplot(1,2,cc+1)
                week_data = [b for (a,b) in feats[ch]]
                feat_data = [a for (a,b) in feats[ch]]
                #pdb.set_trace()
                if plot_type == 'scatter':
                    plt.scatter(week_data,feat_data,alpha=0.1)
                elif plot_type == 'boxplot':
                    plt.boxplot([[fd for fd,week in zip(feat_data,week_data) if week == 'C01'],[fd for fd,week in zip(feat_data,week_data) if week == 'C23']])
                    print(stats.kstest([fd for fd,week in zip(feat_data,week_data) if week == 'C01'],[fd for fd,week in zip(feat_data,week_data) if week == 'C23']))
                plt.title(ch)
                plt.xlabel('Week')
                plt.ylabel('Power (dB)')

                
            plt.suptitle(feat + ' over weeks; ' + str(pt))
                    
        
        return feats