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
import DBSpace as dbo
from DBSpace import nestdict

from DBSpace import unity,displog
import scipy.stats as stats

import pdb
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

#sns.set()

sns.set_context('talk')
sns.set(font_scale=4)
sns.set_style('white')

class naive_readout:
    def __init__(self,feat_frame,ClinFrame):
        self.feat_frame = feat_frame
        
        self.pts = ['901','903','905','906','907','908']
        self.bands = ['Delta','Theta','Alpha','Beta*','Gamma1']
        self.all_feats = ['L-' + band for band in bands] + ['R-' + band for band in bands]
        
        self.ClinFrame = ClinFrame
        
        self.circ = 'day'
    
    def find_pt_extremes(self):
        hdrs_info = nestdict()
        week_labels = ClinFrame.week_labels()
        
        for pt in self.pts:
            pt_hdrs_traj = [a for a in ClinFrame.DSS_dict['DBS'+pt]['HDRS17raw']][8:]
            
            hdrs_info[pt]['max']['index'] = np.argmax(pt_hdrs_traj)
            hdrs_info[pt]['min']['index'] = np.argmin(pt_hdrs_traj)
            hdrs_info[pt]['max']['week'] = week_labels[np.argmax(pt_hdrs_traj)+8]
            hdrs_info[pt]['min']['week'] = week_labels[np.argmin(pt_hdrs_traj)+8]
            
            hdrs_info[pt]['max']['HDRSr'] = pt_hdrs_traj[hdrs_info[pt]['max']['index']]
            hdrs_info[pt]['min']['HDRSr'] = pt_hdrs_traj[hdrs_info[pt]['min']['index']]
            hdrs_info[pt]['traj']['HDRSr'] = pt_hdrs_traj

    def set_comparison_weeks(self,fix=[]):
        #we do this on a per-patient basis
        if fix == []:
            return {pt:{'high':'C01','low':'C24'}}    
        else:
            return {pt:{'high':fix[0],'low':fix[1]}}
    
    
        
    def sig_bands(self,stats='ks',weeks=['C01','C24']):
        self.ks_stats = nestdict()
        self.week_distr = nestdict()
        
        for pt in self.pts:
            for ff in self.bands:
                print('Computing ' + ' ' + ff)
                _,self.ks_stats[pt][ff],self.week_distr[pt][ff] = self.feat_frame.scatter_state(weeks=weeks,pt=pt,feat=ff,circ=self.circ,plot=False,plot_type='scatter',stat=stats)

        self.K_weeks = weeks
        self.K_stat_type = stats
        self.K_stats = np.array([[[ks_stats[pt][band][side][0] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
        self.P_val = np.array([[[ks_stats[pt][band][side][1] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
        
        #Do the change values here too
        self.pre_feat_vals = np.array([[[self.week_distr[pt][band][side][self.K_weeks[0]] for side in ['Left','Right']] for band in self.bands] for pt in self.pts]).reshape(6,-1,order='F')
        self.post_feat_vals = np.array([[[self.week_distr[pt][band][side][self.K_weeks[1]] for side in ['Left','Right']] for band in self.bands] for pt in self.pts]).reshape(6,-1,order='F')

    def band_change(self,patient,band,plot=False):

        pp = pts.index(patient)
        ff = self.all_feats.index(band)
        if plot:
            plt.figure()
            sns.violinplot(y=self.pre_feat_vals[pp,ff])
            sns.violinplot(y=self.post_feat_vals[pp,ff],color='red',alpha=0.3)
    
        return (np.mean(post_feat_vals[pp,ff]) - np.mean(pre_feat_vals[pp,ff]))
    
    def band_change_analysis(self):
        change_grid = np.zeros((len(self.pts),len(self.all_feats)))
        for pp,pt in enumerate(self.pts):
            for ff,freq in enumerate(self.all_feats):
                change_grid[pp,ff] = self.band_change(pt,freq,plot=False)

    def flat_ensemble_change(self,band):
        ff = all_feats.index(band)
        pre_flat_list = [item for sublist in pre_feat_vals[:,ff] for item in sublist]
        post_flat_list = [item for sublist in post_feat_vals[:,ff] for item in sublist]
        
        return pre_flat_list, post_flat_list

    def all_ensemble_change(self,plot=True):
        pre_flat_list = []
        post_flat_list = []
        
        for ff,freq in enumerate(self.all_feats):
            pre_flat_list.append([item for sublist in self.pre_feat_vals[:,ff] for item in sublist])
            post_flat_list.append([item for sublist in self.post_feat_vals[:,ff] for item in sublist])
            
        if plot:
            plt.figure()
            ax = sns.violinplot(data=pre_flat_list,color='blue')
            ax = sns.violinplot(data=post_flat_list,color='red',alpha=0.3)
            
            
            plt.setp(ax.collections,alpha=0.3)
    
    def get_ensemble_change(self,band,plot=True):
        ff = self.all_feats.index(band)
        pre_flat_list = [item for sublist in self.pre_feat_vals[:,ff] for item in sublist]
        post_flat_list = [item for sublist in self.post_feat_vals[:,ff] for item in sublist]
        if plot:
            plt.figure()
            ax = sns.violinplot(x=pre_flat_list)
            ax = sns.violinplot(x=post_flat_list,color='red',alpha=0.3)
            plt.title(band)
            plt.xlim(-10,10)
            plt.setp(ax.collections,alpha=0.3)
        #do some stats here
        stat_check = stats.ks_2samp(pre_flat_list,post_flat_list)
        return {'diff':(np.mean(pre_flat_list) - np.mean(post_flat_list)),'var':(np.var(pre_flat_list) - np.var(post_flat_list)),'ks':stat_check}



class OBands:
    def __init__(self,BRFrame):
        #Bring in the BR Data Frame
        self.BRFrame = BRFrame
        self.do_pts = dbo.all_pts
        
    def poly_subtr(self,inp_psd,polyord=6):
        raise ValueError
        
        pchann = np.poly1d(np.polyfit(self.BRFrame.data_basis,inp_psd,polyord))
        
        return inp_psd - pchann
        
    def feat_extract(self,do_corrections=False):
        big_list = self.BRFrame.file_meta
        #go through ALL files and do the feature extraction
        for rr in big_list:
            feat_dict = {key:[] for key in dbo.feat_dict.keys()}
            for featname,dofunc in dbo.feat_dict.items():
                if do_corrections == False:
                    datacontainer = {ch: rr['Data'][ch] for ch in rr['Data'].keys()}
                    #Do we want to do any preprocessing for the PSDs before we send it to the next round?
                    #Maybe a poly-fit subtraction?
                    feat_dict[featname] = dofunc['fn'](datacontainer,self.BRFrame.data_basis['F'],dofunc['param'])
                else:
                    pre_correction = {ch: rr['Data'][ch] for ch in rr['Data'].keys()}
                    datacontainer,_ = dbo.poly_subtrEEG(pre_correction,self.BRFrame.data_basis['F'])
                    feat_dict[featname] = dofunc['fn'](datacontainer,self.BRFrame.data_basis['F'],dofunc['param'])
                    
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
            
            [plt.plot(fvect,10*np.log10(val[:,ch,:].T),color='black',alpha=0.3,label=weeks[0]) for key,val in week_avg.items() if key == weeks[0]]
            plt.plot(fvect,w0ptavg,color='black',linewidth=3)            
            [plt.plot(fvect,10*np.log10(val[:,ch,:].T),color='green',alpha=0.3,label=weeks[1]) for key,val in week_avg.items() if key == weeks[1]]
            plt.plot(fvect,w1ptavg,color='green',linewidth=3)
            plt.xlim((0,50))
            plt.ylim((-80,0))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')
            plt.title(chann)
            sns.despine(top=True,right=True)
            
            #diffPSDs = pt x 513 x 2chann x 2timepoints
            #to do per-patient baseline subtraction
            
            
        plt.suptitle('Per Patient PSDs for: ' + str(weeks))
        
    def day_vs_nite(self):
        
        
        fdnmeta = self.BRFrame.file_meta
        Circ = {'day':[],'night':[]}
        
        bands = dbo.feat_order
        #feats['Left'] = [((rr['FeatVect'][feat]['Left']),rr['Circadian']) for rr in fdnmeta]
        #feats['Right'] = [((rr['FeatVect'][feat]['Right']),rr['Circadian']) for rr in fdnmeta]
        
        sleeps = ['day','night']
        
        for light in sleeps:
            Circ[light] = [[[(rr['FeatVect'][feat]['Left'],rr['FeatVect'][feat]['Right']) for feat in dbo.feat_order] for rr in fdnmeta if rr['Patient'] == pt and rr['Circadian'] == light] for pt in self.do_pts]
        #night_recs = [[[(rr['FeatVect'][feat]['Left'],rr['FeatVect'][feat]['Right']) for feat in dbo.feat_order] for rr in fdnmeta if rr['Patient'] == pt and rr['Circadian'] == 'night'] for pt in self.do_pts]
        
        #feats['Left'] = [[((rr['FeatVect'][feat]['Left']),rr['Circadian']) for rr in fdnmeta if rr['Patient'] == pt] for pt in dbo.all_pts]
        #feats['Right'] = [[((rr['FeatVect'][feat]['Right']),rr['Circadian']) for rr in fdnmeta if rr['Patient'] == pt] for pt in dbo.all_pts]
        
        #Get a days only list
        #pdb.set_trace()
        pt_day_nite = nestdict()
        for pp,pt in enumerate(self.do_pts):
            day_matr = np.array(Circ['day'][pp]).reshape(-1,10,order='F')
            night_matr = np.array(Circ['night'][pp]).reshape(-1,10,order='F')
            for feat in range(10):
                outstat = stats.ranksums(day_matr[:,feat],night_matr[:,feat])[1]
                outvar = (np.var(day_matr[:,feat]),np.var(night_matr[:,feat]))
                
                pt_day_nite[pt][feat]['Pval'] = outstat
                pt_day_nite[pt][feat]['Var'] = outvar
        
        
        print(pt_day_nite)
        
        #main_stats = np.array([[[pt_day_nite[pt][feat][side][0] for side in ['Left','Right']] for feat in range(10)] for pt in self.do_pts]).reshape(6,-1,order='F')
        P_val = np.array([[pt_day_nite[pt][feat]['Pval'] for feat in range(10)] for pt in self.do_pts])
        dn_var = np.array([[pt_day_nite[pt][feat]['Var'] for feat in range(10)] for pt in self.do_pts])
        
        
        P_val = P_val.reshape(6,-1,order='F')
        #var = P_val.reshape(6,-1,order='F')
        
        
        #plt.subplot(2,1,1)
        #plt.pcolormesh(main_stats);plt.colorbar()
        #plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
        #plt.yticks(np.arange(6)+0.5,pts)
        #plt.subplot(2,1,2)
        #plt.pcolormesh((P_val < (0.05)).astype(np.int));plt.colorbar()
        #P_val[P_val > (0.05/10)] = 1
        
        plt.figure()
        plt.pcolormesh((P_val < (0.05/10)).astype(np.float32),cmap=plt.get_cmap('Set1_r'));plt.colorbar();
        plt.yticks(np.arange(6)+0.5,self.do_pts)
        plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
        plt.title('P-value of Day-Nite Difference')

        plt.figure()
        plt.subplot(211)
        plt.pcolormesh(dn_var[:,:,0]);plt.colorbar();
        
        plt.yticks(np.arange(6)+0.5,self.do_pts)
        plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
        plt.title('Day')
        
        plt.subplot(212)
        plt.pcolormesh(dn_var[:,:,1]);plt.colorbar();
        plt.yticks(np.arange(6)+0.5,self.do_pts)
        plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
        plt.title('Night')
        # outstats = {'Left':0,'Right':0}
        
        # plt.figure()
        # for cc,ch in enumerate(['Left','Right']):
        #     plt.subplot(1,2,cc+1)
        #     sleep_data = [b for (a,b) in feats[ch]]
        #     feat_data = np.array([a for (a,b) in feats[ch]])
            
        #     circdistr = {circ:[a for (a,b) in feats[ch] if b == circ] for circ in sleeps}
        #     #outstats[ch] = stats.ks_2samp(circdistr[sleeps[0]],circdistr[sleeps[1]])
        #     #print(outstats[ch])
            
        #     outstats[ch] = stats.ttest_ind(circdistr[sleeps[0]],circdistr[sleeps[1]])
        #     print('TTest for Day vs Night')
        #     print(outstats[ch])
            
        #     print('Day is normal: ' + str(stats.mstats.normaltest(circdistr[sleeps[0]])))
        #     print('Night is normal: ' + str(stats.mstats.normaltest(circdistr[sleeps[1]])))
            
            
        #     plt.scatter(sleep_data,feat_data,alpha=0.005)
            
    def c_vs_c(self,early,late):
        
        fdnmeta = self.BRFrame.file_meta
        weeks = [early,late]
            
        diff_states = defaultdict(dict)
        for state in weeks:
            diff_states[state] = [[[(rr['FeatVect'][feat]['Left'],rr['FeatVect'][feat]['Right']) for feat in dbo.feat_order] for rr in fdnmeta if rr['Patient'] == pt and rr['Phase'] == state] for pt in self.do_pts]
        #night_recs = [[[(rr['FeatVect'][feat]['Left'],rr['FeatVect'][feat]['Right']) for feat in dbo.feat_order] for rr in fdnmeta if rr['Patient'] == pt and rr['Circadian'] == 'night'] for pt in self.do_pts]
        
        #feats['Left'] = [[((rr['FeatVect'][feat]['Left']),rr['Circadian']) for rr in fdnmeta if rr['Patient'] == pt] for pt in dbo.all_pts]
        #feats['Right'] = [[((rr['FeatVect'][feat]['Right']),rr['Circadian']) for rr in fdnmeta if rr['Patient'] == pt] for pt in dbo.all_pts]
        
        #Get a days only list
        pt_two_states = defaultdict(dict)
        for pp,pt in enumerate(self.do_pts):
            pt_two_states[pt] = stats.ranksums(np.array(pt_two_states[state[0]][pp]),np.array(pt_two_states[state[1]][pp]))
        
        
        print(pt_day_nite)
    
    
    def set_states(self,default=True):
        if default:
            phases[pt]['high'] = 'C01'
            phases[pt]['low'] = 'C24'
        else:
            pass
        weeks = [high,low]
        
        
    def compare_states(self,weeks,pt='all',feat='Alpha',circ='',plot=True,plot_type='scatter',stat='ks'):
        #this assumes we've already populated the 'high' and 'low' values
        #generate our data to visualize
        if pt == 'all':
            pt = dbo.all_pts
            
            
        fmeta = self.BRFrame.file_meta
        feats = {'Left':0,'Right':0}
        
        swap_key = {weeks[0]:'depr',weeks[1]:'notdepr'}
        
        if feat == 'fSlope' or feat == 'nFloor':
            dispfunc = unity
        else:
            dispfunc = unity
        
        #do day and night here
        if circ != '':
            fdnmeta = [rr for rr in fmeta if rr['Circadian'] == circ]
        else:
            fdnmeta = fmeta
        
        feats['Left'] = [(dispfunc(rr['FeatVect'][feat]['Left']),rr['Phase']) for rr in fdnmeta if rr['Patient'] in pt and rr['Phase'] in weeks]
        feats['Right'] = [(dispfunc(rr['FeatVect'][feat]['Right']),rr['Phase']) for rr in fdnmeta if rr['Patient'] in pt and rr['Phase'] in weeks]

        outstats = defaultdict(dict)
        weeks_osc_distr = {'Left':[],'Right':[]}
        
        for cc,ch in enumerate(['Left','Right']):
            weekdistr = {swap_key[week]:[a for (a,b) in feats[ch] if b == week] for week in weeks}
            if stat == 'ks':
                outstats[ch] = stats.ks_2samp(weekdistr[swap_key[weeks[0]]],weekdistr[swap_key[weeks[1]]])
            elif stat == 'ranksum':
                outstats[ch] = stats.ranksums(weekdistr[swap_key[weeks[0]]],weekdistr[swap_key[weeks[1]]])
            elif stat == 't':
                outstats[ch] = stats.ttest_1samp(weekdistr[swap_key[weeks[0]]],weekdistr[swap_key[weeks[1]]])
                
            weeks_osc_distr[ch] = weekdistr
        
        return feats,outstats, weeks_osc_distr
    
    
    def scatter_state(self,weeks='all',pt='all',feat='Alpha',circ='',plot=True,plot_type='scatter',stat='ks'):
        #generate our data to visualize
        if weeks == 'all':
            weeks = dbo.Phase_List('ephys')
        if pt == 'all':
            pt = dbo.all_pts
            
        #Swap key effort here
        if 'C01' in weeks and 'C24' in weeks:
            swap_key = {'C01':'Depr','C24':'NotDepr'}
        else:
            swap_key = {key:key for key in weeks}
        
        fmeta = self.BRFrame.file_meta
        feats = {'Left':0,'Right':0}
        
        if feat == 'fSlope' or feat == 'nFloor':
            dispfunc = unity
        else:
            dispfunc = unity
        
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

        outstats = defaultdict(dict)
        
        weeks_osc_distr = {'Left':[],'Right':[]}
        
        for cc,ch in enumerate(['Left','Right']):
            weekdistr = {week:[a for (a,b) in feats[ch] if b == week] for week in weeks}
            #
            if stat == 'ks':
                outstats[ch] = stats.ks_2samp(weekdistr[weeks[0]],weekdistr[weeks[1]])
            elif stat == 'ranksum':
                outstats[ch] = stats.ranksums(weekdistr[weeks[0]],weekdistr[weeks[1]])
            elif stat == 't':
                outstats[ch] = stats.ttest_1samp(weekdistr[weeks[0]],weekdistr[weeks[1]])
                
            weeks_osc_distr[ch] = weekdistr
            
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
            plt.subplot(1,2,1)
            plt.ylabel('Power (dB)')

            #plt.tight_layout()
            plt.suptitle(feat + ' over weeks; ' + str(pt))
                    
        
        return feats,outstats, weeks_osc_distr