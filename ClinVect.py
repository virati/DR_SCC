#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:27:33 2018

@author: virati
Clinical Vector Class
"""

import json
from collections import defaultdict
import numpy as np
import sys
import pdb

import scipy.signal as sig

sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

import matplotlib.pyplot as plt


class CFrame:
    do_pts = ['901','903','905','906','907','908']
    scale_max = {'HDRS17':40,'MADRS':50,'BDI':60,'GAF':100}
    
    def __init__(self,incl_scales = ['HDRS17','MADRS','BDI','GAF'],norm_scales=False):
        #load in our JSON file
        #Import the data structure needed for the CVect
        ClinVect = json.load(open('/home/virati/Dropbox/ClinVec.json'))
        clin_dict = defaultdict(dict)
        for pp in range(len(ClinVect['HAMDs'])):
            ab = ClinVect['HAMDs'][pp]
            clin_dict[ab['pt']] = defaultdict(dict)
            for phph,phase in enumerate(ClinVect['HAMDs'][pp]['phases']):
                for ss,scale in enumerate(incl_scales):
                    if norm_scales and scale != 'dates':
                        clin_dict[ab['pt']][phase][scale] = ab[scale][phph] / self.scale_max[scale]
                    else:
                        clin_dict[ab['pt']][phase][scale] = ab[scale][phph]
        
        self.do_scales = incl_scales
        self.clin_dict = clin_dict
        
        
        #Setup derived measures
        #THIS IS JUST A COPY PASTE FROM SCALE DYNAMICS, need to merge this in with above so it's all done properly
        DSS_dict = defaultdict(dict)
        for ss,scale in enumerate(do_scales):
            for pp in range(len(ClinVect['HAMDs'])):
                ab = ClinVect['HAMDs'][pp]
                DSS_dict[ab['pt']][scale] = ab[scale]
                if scale == 'HDRS17':
                    big_dict[ab['pt']]['mHDRS'] = sig.medfilt(big_dict[ab['pt']]['HDRS17'],5)
        
        self.derived_measures()
        self.load_stim_changes()
        
    def derived_measures(self):
        self.mHDRS_gen()
        self.dsc_gen()
        
    def DEPRgen_subdim(self,latentdim=1):
        #just do it for all patients
        for pp,pat in enumerate(self.do_pts):
            all_cscores = (' '.join(w) for w in self.clin_dict[pat])
            sC = np.vstack((big_dict[pat]['HDRS17'],big_dict[pat]['HDRS17'],big_dict[pat]['HDRS17'],big_dict[pat]['HDRS17']))

        self.bvect = {'LR': lrvect,'SC': srvect}
    
    def plot_scale(self,scale='HDRS17',pts='all'):
        if pts == 'all':
            pts = dbo.all_pts
        
        plt.figure()
        for patient in pts:
            #pt_tcourse = {rr:self.clin_dict['DBS'+patient][rr][scale] for rr in self.clin_dict['DBS'+patient]}
            pt_tcourse = self.pt_scale_tcourse(patient)
            #now setup the right order
            prop_order = dbo.Phase_List('all')
            ordered_tcourse = [pt_tcourse[phase][scale] for phase in prop_order]
            
            plt.plot(ordered_tcourse)
        plt.title(scale + ' for ' + str(pts))
        
        
    def pt_scale_tcourse(self,pt):
        #return dictionary with all scales, and each element of that dictionary should be a NP vector
        pt_tcourse = {rr:self.clin_dict['DBS'+pt][rr] for rr in self.clin_dict['DBS'+pt]}
        return pt_tcourse
    

    def load_stim_changes(self):
        pass
    
    def mHDRS_gen(self):
        for ss,scale in enumerate(self.do_scales):
            for pat in self.clin_dict.keys():
                self.clin_dict[pat]['mHDRS'] = sig.medfilt(self.clin_dict[pat]['HDRS17'],5)
                
    def dsc_gen(self):
        pass
    
    def c_dict(self):
        clindict = self.clin_dict
        #This will generate a dictionary with each key being a scale, but each value being a matrix for all patients and timepoints
        big_dict = {scale:[[clindict[pt][week][scale] for week in week_ordered] for pt in self.do_pts] for scale in self.do_scales}
        self.scale_dict = big_dict
        
        
    def c_vect(self):
        #each patient will be a dict key
        c_vects = {el:0 for el in self.do_pts}
        for pp,pt in enumerate(self.do_pts):
            #vector with all clinical measures in the thing
            #return will be phase x clinscores
            c_vect[pt] = 0


if __name__=='__main__':
    TestFrame = CFrame()
    TestFrame.c_dict()