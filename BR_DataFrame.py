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

def nearest(items,pivot):
    return min(items,key=lambda x: abs(x - pivot))

class BR_Data_Tree():
    def __init__(self,do_pts=['901','903','905','906','907','908']):
        self.do_pts = do_pts
        self.fs = 422
        
        CVect = json.load(open('/home/virati/Dropbox/ClinVec.json'))['HAMDs']
        clinvect = {pt['pt']: pt for pt in CVect}
        self.ClinVect = clinvect
    
    def full_sequence(self,data_path=''):
        self.build_phase_dict()
        self.list_files()
        self.meta_files()
        self.Load_Data(path=data_path)
        
        print('Data Loaded')
        
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
        
    def meta_files(self,mode='Chronic'):
        #file_meta = {} * len(self.file_list)
        file_meta = [{} for _ in range(len(self.file_list))]
        
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
                file_meta[ff] = {'Filename': filen,'Date': file_dateinfo, 'Type': file_typeinfo,'Patient':file_ptinfo,'Phase':file_phaseinfo,'Circadian':file_dayniteinfo}
            else:
                file_meta[ff] = None
                
        #remove all the NONEs since they have nothing to do with the current analysis mode
        file_meta = [x for x in file_meta if x is not None]
        
        self.file_meta = file_meta
        
    def update_meta(self):
        for ff in self.file_meta:
            file_dayniteinfo = self.get_time(ff['Filename'])
            ff.update({'Circadian':file_dayniteinfo})
        
    def Load_Data(self,domain='F',path=''):
        #this function will return a feature matrix that will be useful for subsequent analysis
        if path == '':
            for rr in self.file_meta:
                #load in the file
                print('Loading in ' + rr['Filename'])
                rr.update({'Data':self.load_file(rr['Filename'],domain=domain)})
        else:
            print('Loading data from...' + path)
            self.file_meta = np.load(path)
            
    def Save_Frame(self,path = '/tmp/'):
        print('Saving File Metastructure...')
        
        np.save(path + 'Chronic_Frame.npy',self.file_meta)
            
            
    def load_file(self,fname,load_intv=(0,-1),domain='T'):
        #call the DBSOsc br_load_method
        #should be 1:1 from file_meta to ts_data
        
        #this returns the full recording
        txtdata = dbo.load_br_file(fname)
        
        #take just the last 10 seconds
        sec_end = 10
        
        #extract channels
        X = {'Left':txtdata[-(422*sec_end):-1,0],'Right':txtdata[-(422*sec_end):-1,2]}
        
        F = defaultdict(dict)
        
        if domain == 'T':
            return X
        elif domain == 'F':
            #we want to return the FFT, not the timedomain signal
            #This saves a lot of RAM but obviously has its caveats
            
            #for this, we want to call the DBS_Osc method for doing FFTs
            F = dbo.gen_psd(X)
                
            return F
            
#Unit Test
            
DataFrame = BR_Data_Tree()
#DataFrame.list_files()
#DataFrame.build_phase_dict()
#DataFrame.meta_files()

#Load in preloaded file
DataFrame.full_sequence(data_path='/tmp/Chronic_Frame.npy')
DataFrame.update_meta()

#plot the PSDs for a specific phase

#%%

plt.figure()
#quick way to plot all of a patient's recordings
[plt.plot(np.log10(rr['Data']['Left']),alpha=0.1) for rr in DataFrame.file_meta if rr['Patient'] == '903' and rr['Phase'] == 'C01' and rr['Circadian'] == 'night']


plt.figure()
[plt.plot(np.log10(rr['Data']['Left']),alpha=0.1) for rr in DataFrame.file_meta if rr['Patient'] == '903' and rr['Phase'] == 'C01' and rr['Circadian'] == 'day']


#%%
#test = [rr for rr in DataFrame.file_meta if rr['Patient'] == '902']


#%%
#for rr in DataFrame.file_meta:
#    plt.plot(rr['TD']['Left'])

#plt.show()