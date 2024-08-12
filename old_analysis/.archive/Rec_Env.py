#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:57:23 2018

@author: virati
Z-impedance library for the main class that can even do regressions!
"""

import sys

sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo
from DBS_Osc import nestdict


import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import scipy.signal as sig


def gm_ratio(a,b,c):
    return a / (a + b + c)
    
class Z_Frame:
    def __init__(self,pt_list=['901','903','905','906','907','908']):
        df = pd.read_csv('/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Impedances/MDT_Z_Table.csv',header=None)
        
        Z_table = df.as_matrix()
        #self.Z_table = Z_table
        self.pt_list = pt_list
        
        ##
        lefts = Z_table[0::4,0::2]
        rights = Z_table[0::4,1::2]
        #Z table is now loaded in
        self.Z_dict = {'Left':0,'Right':0}
        self.Z_dict['Left'] = lefts
        self.Z_dict['Right'] = rights
        
        #load in the active contacts
        self.load_OnTs()
        
        #compute the recording contacts
        pt_dict = self.pt_dict
        new_dict = {key:(np.array(val['OnT']) + 1, np.array(val['OnT']) - 1) for key,val in pt_dict.items()}
        
        
    def load_OnTs(self):
        pt_dict = nestdict()
        pt_dict['901']['OnT'] = (2,1)
        pt_dict['903']['OnT'] = (2,2)
        pt_dict['905']['OnT'] = (2,1)
        pt_dict['906']['OnT'] = (2,2)
        pt_dict['907']['OnT'] = (1,1)
        pt_dict['908']['OnT'] = (2,1)
        
        self.pt_dict = pt_dict
        
    def plot_Zs(self):
        plt.figure()
        for ss,side in enumerate(['Left','Right']):
            plt.subplot(1,2,ss+1)
            plt.plot(self.Z_dict[side])
            plt.legend(self.pt_list)
            plt.xlim((0,28))
            plt.ylim((0,2000))
        
        
class GMWM:
    def __init__(self):
        #load in the GM/WM/CSF table
        base_dir = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/CT/'
        files = {'Left':'VTRVTA_6AMP_1.csv','Right':'VTRVTA_6AMP_2.csv'}
        table_format = ['Subj', 'Contact', 'GM','WM', 'CSF']
        
        for side in ['Left','Right']:
            GMWMVals = {key: pd.read_csv(base_dir + val,names=table_format) for key,val in files.items()}
        
        self.Vals = GMWMVals
        
        self.compute_ratios()
    
    def make_matrix(self):
        
    
    def compute_ratios(self):
        
        
        for side in ['Left','Right']:
            vals = self.Vals[side]
            
            vals['GM_Ratio'] = gm_ratio(vals['GM'],vals['WM'],vals['CSF'])
    
    def get_ratio(self,pt,side,electrode):
        #go through and plot the GM ratio for all recording electrodes
        for rr in self.Vals[side]:
            print(rr)
        #return [rr['GM_Ratio'] for rr in self.Vals[side] if rr['Subj'] == 'DBS'+pt and rr['Contact'] == electrode]
    
    def dict_to_matrix(self):
        ratios = self.Ratios
        for side in ['Left','Right']:
            pass
            #out_list = [[etrode_val for etrode_val in ]]

anatomy = GMWM()
anatomy.get_ratio('901','Left','E3')
#%%
#impeds = Z_Frame()
#impeds.plot_Zs()
        
    
