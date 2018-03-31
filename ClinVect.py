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

import scipy.stats as stats

import scipy.signal as sig

sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
sys.path.append('/home/virati/Dropbox/projects/libs/robust-pca/')
sys.path.append('/home/virati/Dropbox/projects/')
#import rpcaADMM
import r_pca
import DBS_Osc as dbo

import matplotlib.pyplot as plt

def pca(data,numComps=None):
    m,n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data,rowvar=False)
    evals,evecs = np.linalg.eigh(R)
    idx=np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    
    if numComps is not None:
        evecs = evecs[:,:numComponents]
    
    return np.dot(evecs.T,data.T).T,evals,evecs

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
        #self._rawClinVect = ClinVect
        
        self.do_scales = incl_scales
        self.clin_dict = clin_dict
        
        
        #Setup derived measures
        #THIS IS JUST A COPY PASTE FROM SCALE DYNAMICS, need to merge this in with above so it's all done properly
        DSS_dict = defaultdict(dict)
        for ss,scale in enumerate(incl_scales):
            for pp in range(len(ClinVect['HAMDs'])):
                ab = ClinVect['HAMDs'][pp]
                DSS_dict[ab['pt']][scale] = ab[scale]
                
        self.DSS_dict = DSS_dict
        
        self.derived_measures()
        self.load_stim_changes()
        
    def derived_measures(self):
        self.mHDRS_gen()
        self.dsc_gen()
    
       
    def mHDRS_gen(self):
        print('Generating mHDRS')
        ph_lut = dbo.all_phases        
    
        for pat in self.DSS_dict.keys():
            mhdrs_tser = sig.medfilt(self.DSS_dict[pat]['HDRS17'],5)
            self.DSS_dict[pat]['mHDRS'] = mhdrs_tser
            for phph in range(mhdrs_tser.shape[0]):
                self.clin_dict[pat][ph_lut[phph]]['mHDRS'] = mhdrs_tser[phph]/self.scale_max['HDRS17']
    
    def dsc_gen(self):
        print('Generating DSC Measure')
        allptX = []
        
        #get phase lookup table
        ph_lut = dbo.all_phases
        big_dict = self.DSS_dict
        opt_lam_dict = defaultdict(dict)
        pt_ll = defaultdict(dict)
        
        for pp,pat in enumerate(self.do_pts):
            llscore = np.zeros(50)
            pthd = np.array(big_dict['DBS'+pat]['HDRS17'])/30
            ptgaf = np.array(big_dict['DBS'+pat]['GAF'])/100
            ptmd = np.array(big_dict['DBS'+pat]['MADRS'])[np.arange(0,32,1)]/45
            ptbdi = np.array(big_dict['DBS'+pat]['BDI'])/60
            ptmhd = np.array(big_dict['DBS'+pat]['mHDRS'])/25
            
            sX = np.vstack((ptmhd,ptmd,ptbdi,ptgaf)).T
    
    
            #lump it into a big observation vector AS WELL and do the rPCA on the large one later
            allptX.append(sX)
            
            min_changes = 100
            for ll,lmbda_s in enumerate(np.linspace(0.3,0.5,50)):
                #lmbda = 0.33 did very well here
                RPCA = r_pca.R_pca(sX,lmbda=lmbda_s)
                L,S = RPCA.fit()
                Srcomp, Srevals, Srevecs = pca(S)
                Lrcomp, Lrevals, Lrevecs = pca(L)
                
                #compare sparse component numbers of nonzero
                #derivative is best bet here
                sdiff = np.diff(Srcomp,axis=0)[:,0] #grab just the HDRS sparse deviations
                
                num_changes = np.sum(np.array(sdiff > 0.006).astype(np.int))
                
                exp_probs = 3
                nchange_diff = np.abs(num_changes - exp_probs)
                
                if nchange_diff <= min_changes:
                    opt_sparseness = num_changes
                    min_changes = nchange_diff
                    best_lmbda_s = lmbda_s
                    
                #shift_srcomp = Srcomp - np.median(Srcomp,0)
                #llscore[ll] = num_changes[pp] - len(np.where(np.sum(np.abs(shift_srcomp),1) < 1e-6))
            opt_lam_dict[pat] = {'Deviation': min_changes,'Lambda':best_lmbda_s,'Sparseness':opt_sparseness}
            #print(revals/np.sum(revals))
            
            #We have the "optimal" lambda now
            RPCA = r_pca.R_pca(sX,lmbda=opt_lam_dict[pat]['Lambda'])
            L,S = RPCA.fit()
            Srcomp, Srevals, Srevecs = pca(S)
            Lrcomp, Lrevals, Lrevecs = pca(L)
            
            DSC_scores= -stats.zscore(np.mean(Lrcomp[:,:],axis=1))
            
            for phph in range(DSC_scores.shape[0]):
                #self.clin_dict[pt][ph_lut[phph]]['DSC']= new_scores[phph]
                self.clin_dict['DBS'+pat][ph_lut[phph]]['DSC'] = DSC_scores[phph]/3
        
    
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
            plt.legend(pts)
        plt.title(scale + ' for ' + str(pts))
        
        
    def pt_scale_tcourse(self,pt):
        #return dictionary with all scales, and each element of that dictionary should be a NP vector
        pt_tcourse = {rr:self.clin_dict['DBS'+pt][rr] for rr in self.clin_dict['DBS'+pt]}
        return pt_tcourse
    

    def load_stim_changes(self):
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
    TestFrame.plot_scale(scale='DSC')
    TestFrame.plot_scale(scale='HDRS17')
    #TestFrame.c_dict()