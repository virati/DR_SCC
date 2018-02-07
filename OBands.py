#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:05:07 2018

@author: virati
Traditional Analyses
"""
import sys

sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

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
            
        
    def scatter_state(self,week='all',pt='all',feat='Alpha',plot=True):
        #generate our data to visualize
        if week == 'all':
            week = dbo.Phase_List('therapy')
        if pt == 'all':
            pt = dbo.all_pts
        
        fmeta = self.BRFrame.file_meta
        
        list1 = [(np.log10(rr['FeatVect'][feat]['Left']),rr['Phase']) for rr in fmeta if rr['Patient'] in pt and rr['Phase'] in week]
        list2 = [(np.log10(rr['FeatVect'][feat]['Right']),rr['Phase']) for rr in fmeta if rr['Patient'] in pt and rr['Phase'] in week]
        
        #plot the figure
        if plot:
            plt.figure()
            
            week_data = [b for (a,b) in list1]
            feat_data = [a for (a,b) in list1]
            #pdb.set_trace()
            plt.scatter(week_data,feat_data,alpha=0.1)
            
        sc_states = list1
        
        return sc_states