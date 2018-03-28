#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:28:10 2018

@author: virati
BR Data Library Script

The PURPOSE of this library should be to just bring in the BrainRadio data in a format that the DSV can handle
For example: Determining which phase a recording belongs to will NOT be done in this script, that is under the perview of the DSV

"""
import sys
#Need to important DBSpace and DBS_Osc Libraries
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/SigProc/CFC-Testing/Python CFC/')
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo
import pdb

import numpy as np


from collections import defaultdict

import matplotlib.pyplot as plt

import os
import datetime
import glob
import json

#This function is a general function that finds the "nearest" object to the pivot
#Used in this module for finding the nearest datetime
def nearest(items,pivot):
    return min(items,key=lambda x: abs(x - pivot))

#THIS CLASS IS ONLY FOR GENERATING THE FRAME
#The frame is supposed to only go from Raw Rec -> PSDs, this can later be passed to analysis classes
#Split out anything else into separate classes where the object this generates can go in
class BR_Data_Tree():
    
    def __init__(self,do_pts=['901','903','905','906','907','908'],preload=False):
        self.do_pts = do_pts
        self.fs = 422
        
        #Where is the data?
        self.base_data_dir = '/home/virati/MDD_Data'
        
        CVect = json.load(open('/home/virati/Dropbox/ClinVec.json'))['HAMDs']
        clinvect = {pt['pt']: pt for pt in CVect}
        self.ClinVect = clinvect
        
        self.preloadData = preload
        self.data_basis = []
        
        #how many seconds to take from the chronic recordings
        self.sec_end = 10
    
    def full_sequence(self,data_path=''):
        self.build_phase_dict()
        self.list_files()
        self.meta_files()
        self.Load_Data(path=data_path)
        #now go in and remove anything with a bad flag
        self.Remove_BadFlags()
        
        self.Check_GC()
        
        #take out the phases that don't exist, and any other stuff, but so far that's all this does
        self.prune_meta()
        #in case the meta-data isn't properly updated from the loaded in deta
        print('Data Loaded')
        
    def Check_GC(self):
        #do just the key features
        #get the stim-related and gc related measures
        print('Checking for Gain Compression...')
        for rr in self.file_meta:
            gc_measures = ['Stim','SHarm','THarm','fSlope','nFloor']
            gc_results = {key:0 for key in gc_measures}
            for meas in gc_measures:
                dofunc = dbo.feat_dict[meas]
                gc_results[meas] = dofunc['fn'](rr['Data'],self.data_basis,dofunc['param'])
        
            # let's do some logic to find out if GC is happening
            isgc = (gc_results['nFloor']['Left'] < -8 or gc_results['nFloor']['Right'] < -8) and (gc_results['SHarm']['Left'] / gc_results['THarm']['Left'] < 1 or gc_results['SHarm']['Right'] / gc_results['THarm']['Right'] < 1)
            
            #check if stim is on
            isstim = (gc_results['Stim']['Left'] > 0.0001 or gc_results['Stim']['Right'] > 0.0001)
            
            rr.update({'GC_Flag':{'Flag':isgc,'Raw':gc_results,'Stim':isstim}})
        
    def plot_GC_distribution(self):
        gc_plot = [None] * len(self.file_meta)
        
        for rr,rec in enumerate(self.file_meta):
            gc_plot[rr] = {side: rec['GC_Flag']['Raw']['SHarm'][side] / rec['GC_Flag']['Raw']['THarm'][side] for side in ['Left','Right']}
            
        plt.figure()
        plt.plot(gc_plot)
        
    def Remove_BadFlags(self):
        try:
            self.file_meta = [rr for rr in self.file_meta if rr['BadFlag'] != True]
        except:
            pdb.set_trace()
        
    #First, the goal is to literally come up with a big list of all the recordings
    def list_files(self,rootdir='/home/virati/MDD_Data/BR/'):
        file_list = []
        for pt in self.do_pts:
            for filename in glob.iglob(rootdir + pt + '/**/' + '*.txt',recursive=True):
                #Append the full path to a list
                #check the file's STRUCTURE HERE
                
                islogf = filename[-7:] == 'LOG.txt'
                isrealtf = filename[-6:] == 'RT.txt'
                iseepromf = filename[-9:] == 'Table.txt'
                
                if not (islogf or isrealtf or iseepromf):
                    file_list.append(filename)
        self.file_list = file_list
    
    #Extract is referring to taking information from the raw BR files
    def extract_date(self,fname):
        datestr = fname.split('_')
        # example: '/home/virati/MDD' 'Data/BR/908/Session' '2016' '08' '11' 'Thursday/DBS908' '2016' '08' '10' '17' '20' '28' 'MR' '14.txt'
        # important for date is -8 (year) -> month -> day -> hour -> minute -> second -> ... -> recording number
        return datetime.datetime.strptime(datestr[-8] + '/' + datestr[-7] + '/' + datestr[-9],"%m/%d/%Y")
    
    #Get for computing time based off of file information
    def get_time(self,fname):
        datestr = fname.split('_')
        
        day_bound = [datetime.datetime.strptime('10:00',"%H:%M"),datetime.datetime.strptime('21:00',"%H:%M")]
        
        #where is the time?
        in_time = datetime.datetime.strptime(datestr[-6] + ':' + datestr[-5],"%H:%M")
        
        if in_time < day_bound[0] or in_time > day_bound[1]:
            return 'night'
        else:
            return 'day'
        
        
    #This function is tasked with returning the type of recording; right now we just care whether it's CHRONIC
    def extract_rectype(self,fname):
        filesiz = os.path.getsize(fname)
        
        ftype = None
        if filesiz > 1e5 and filesiz < 1e7:
            ftype = 'Chronic'
        elif filesiz > 1e7:
            ftype = 'Dense'
            #need more logic here to check what type of experiment it actually is
        return ftype
    
    #Extract the recording gain for every file
    def extract_gains(self,fname):
        xml_fname = fname.split('.')[0] + '.xml'
        
    def extract_pt(self,fname):
        return fname.split('BR')[1][1:4]
        
    def build_phase_dict(self):
        #this is where the fun is....
        
        #CONVENTION DECIDED: if a recording is taken between week C04 and C05 -> it belongs to C05 since the clinical questionaiires ask about the LAST 7 days
        
        #Extract the whole big thing for a given patient
        #phdate_dict['DBS901'][phase] = date
        phdate_dict = defaultdict(dict)
        
        for pt in self.do_pts:
            alv = self.ClinVect['DBS'+pt]
            phdate_dict[pt] = defaultdict(dict)
            
            for phph,phase in enumerate(alv['phases']):
                phdate_dict[pt][phase] = datetime.datetime.strptime(alv['dates'][phph],"%m/%d/%Y")
        
        self.pd_dict = phdate_dict
    
    def get_date_phase(self,pt,datet):
        # Given a patient and a date, return the PHASE of the study
        # importantly, 
        searchstruct = self.pd_dict[pt]
        
        #find distance between the datetime provided and ALL phases
        dist_to_ph = {key: datet - val for key,val in searchstruct.items()}
        #only keep the phases that are AFTER the datetime provided
        phases_after = {key: val for key,val in dist_to_ph.items() if val <= datetime.timedelta(0)}
        
        if bool(phases_after):
            closest_phase = max(phases_after,key=phases_after.get)
        else:
            closest_phase = None
            
        return closest_phase
    
    def check_empty_phases(self):
        empty_phases = [rr['Filename'] for rr in self.file_meta if rr['Phase'] == None]
        
        if len(empty_phases):
            print('Some Empty Phases!')
        #pdb.set_trace()
        
    def meta_files(self,mode='Chronic'):
        #file_meta = {} * len(self.file_list)
        if not self.preloadData:
            file_meta = [{} for _ in range(len(self.file_list))]
        else:
            file_meta = self.file_meta
        
        for ff,filen in enumerate(self.file_list):
            #we're going to to each and every file now and give it its metadata
            
            
            #Here, we're going to extract the DATE
            file_dateinfo = self.extract_date(filen)
            file_typeinfo = self.extract_rectype(filen)
            #file_gaininfo = self.extract_gains(filen)
            file_ptinfo = self.extract_pt(filen)
            file_phaseinfo = self.get_date_phase(file_ptinfo,file_dateinfo)
            file_dayniteinfo = self.get_time(filen)
            
            if file_typeinfo == mode:
                file_meta[ff].update({'Filename': filen,'Date': file_dateinfo, 'Type': file_typeinfo,'Patient':file_ptinfo,'Phase':file_phaseinfo,'Circadian':file_dayniteinfo,'BadFlag':False})
            else:
                file_meta[ff] = None
                
        #remove all the NONEs since they have nothing to do with the current analysis mode
        file_meta = [x for x in file_meta if x is not None]
        
        self.file_meta = file_meta
        
    def check_meta(self,prob_condit=0):
        for rr in self.file_meta:
            for ch in ['Left','Right']:
                if rr['Data'][ch].all() == 0:
                    print('PROBLEM: ' + str(rr) + ' has a zero PSD in channel ' + ch)
                    
    def prune_meta(self):
        print('Pruning out recordings that have no Phase in main study...')
        #prune out the parts of file_meta that are not in the study
        new_meta = [rr for rr in self.file_meta if rr['Phase'] != None]
        
        self.file_meta = new_meta
            
    def Load_Data(self,domain='F',path=''):
        #this function will return a feature matrix that will be useful for subsequent analysis
        #check if we're consistent
        if self.preloadData == True and path == '':
            raise Exception('INCONSISTENT LOADING: Preload Flag is True but no Path Specified')
        
        if domain == 'F':
            self.data_basis = np.linspace(0,self.fs/2,2**9+1)
        elif domain == 'T':
            self.data_basis = np.linspace(0,self.sec_end)
        
        
            
        if path == '':
            for rr in self.file_meta:
                #load in the file
                print('Loading in ' + rr['Filename'])
                precheck_data = self.load_file(rr['Filename'],domain=domain)
                
                if precheck_data['Left'].all() != 0 and precheck_data['Right'].all() != 0:
                    rr.update({'Data':precheck_data})
                else:
                    rr.update({'BadFlag':True})
                    
            self.preloadData = False
            
        else:
            print('Loading data from...' + path)
            self.file_meta = np.load(path)
            self.preloadData = True

            
    def Save_Frame(self,path = '/tmp/'):
        print('Saving File Metastructure... ' + path)
        
        np.save(path + 'Chronic_Frame.npy',self.file_meta)
            
            
    def load_file(self,fname,load_intv=(0,-1),domain='T'):
        #call the DBSOsc br_load_method
        #should be 1:1 from file_meta to ts_data
        
        #this returns the full recording
        txtdata = dbo.load_br_file(fname)
        
        #take just the last 10 seconds
        sec_end = self.sec_end
        
        #extract channels
        X = {'Left':txtdata[-(422*sec_end):-1,0].reshape(-1,1),'Right':txtdata[-(422*sec_end):-1,2].reshape(-1,1)}
        
        F = defaultdict(dict)
        
        if domain == 'T':
            
            return X
        
        elif domain == 'F':
            #we want to return the FFT, not the timedomain signal
            #This saves a lot of RAM but obviously has its caveats
            #for this, we want to call the DBS_Osc method for doing FFTs
            #The return from gen_psd is a dictionary eg: {'Left':{'F','PSD'},'Right':{'F','PSD'}}
            F = dbo.gen_psd(X)
            
            
                
            return F
    #PLOTTING FUNCTIONS FOR THE BRFRAME
    
    def plot_file_PSD(self,fname=''):
        if fname != '':
            psd_interest = [(rr['Data']['Left'],rr['Data']['Right']) for rr in DataFrame.file_meta if rr['Filename'] == fname]
        
        plt.figure()
        plt.plot(psd_interest)
    
    def plot_PSD(self,pt='901'):
        #generate out F vector
        fvect = np.linspace(0,211,513)
        
        #quick way to plot all of a patient's recording
        #therapy_phases = dbo.all_phases
        psds = {'Left':0,'Right':0}
        for ch in ['Left','Right']:
            psds[ch] = np.array([np.log10(rr['Data'][ch]) for rr in DataFrame.file_meta if rr['Patient'] == '901' and rr['Phase'] in dbo.Phase_List('ephys')]).T
        
        #list2 = np.array([np.log10(rr['Data']['Left']) for rr in DataFrame.file_meta if rr['Patient'] == '901' and rr['Circadian'] == 'night' and rr['Phase'] in dbo.Phase_List('notherapy')]).T
        
        plt.figure()
        plt.subplot(121)
        plt.plot(fvect,psds['Left'],color='r',alpha=0.01)
        plt.subplot(122)
        plt.plot(fvect,psds['Right'],color='b',alpha=0.01)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.legend({'Therapy','NoTherapy'})    
        
        #%%
        [plt.plot(fvect,np.log10(rr['Data']['Left']),alpha=0.1) for rr in DataFrame.file_meta if rr['Patient'] == '901' and rr['Circadian'] == 'night']
        plt.title('Night')
        
        plt.figure()
        [plt.plot(fvect,np.log10(rr['Data']['Left']),alpha=0.1) for rr in DataFrame.file_meta if rr['Patient'] == '901' and rr['Circadian'] == 'day']
        plt.title('Day')
        
        
        #%%
        #test = [rr for rr in DataFrame.file_meta if rr['Patient'] == '902']
        
        
        #%%
        #for rr in DataFrame.file_meta:
        #    plt.plot(rr['TD']['Left'])
            
if __name__ == '__main__':
    #Unit Test
    DataFrame = BR_Data_Tree()
    #DataFrame.list_files()
    #DataFrame.build_phase_dict()
    #DataFrame.meta_files()
    
    #Load in preloaded file
    #DataFrame.full_sequence(data_path='/tmp/Chronic_Frame.npy')
    #DataFrame.full_sequence(data_path='/home/virati/Chronic_Frame.npy')
    DataFrame.full_sequence()
    DataFrame.Save_Frame()
    
    #plot the PSDs for a specific phase