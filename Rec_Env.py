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

import pandas as pd
import numpy as np
import scipy.signal as sig

class Z_Frame:
    def __init__(self,pt_list=['901','903','905','906','907','908']):
        df = pd.read_csv('/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Impedances/MDT_Z_Table.csv',header=None)
        
        Z_table = df.as_matrix()
        self.Z_table = Z_table
        
        lefts = big_table[0::4,0::2]
        rights = big_table[0::4,1::2]
        #Z table is now loaded in
        
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
        
class GMWM:
    def __init__(self):
        #load in the GM/WM/CSF table
        pass
    
impeds = Z_Frame()
        
    
