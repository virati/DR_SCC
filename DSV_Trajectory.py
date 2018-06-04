#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:16:13 2018

@author: virati
"""

import BR_DataFrame as BRDF
#from BR_DataFrame import *
from ClinVect import CFrame
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import itertools
from DSV import DSV, ORegress

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style("white")

from scipy.interpolate import interp1d


ClinFrame = CFrame(norm_scales=True)
#ClinFrame.plot_scale(pts='all',scale='HDRS17')
#ClinFrame.plot_scale(pts=['901'],scale='MADRS')

BRFrame = BRDF.BR_Data_Tree()
BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_April.npy')
#BRFrame.full_sequence(data_path='/tmp/Chronic_Frame_DEF.npy')
BRFrame.check_empty_phases()

analysis = ORegress(BRFrame,ClinFrame)
analysis.O_feat_extract()

all_pts = ['901','903','905','906','907','908']
#%%
def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

#%%

#if we use the ridge regression
lridge = [-0.00583578, -0.00279751,  0.00131825,  0.01770169,  0.01166687]
rridge = [-1.06586005e-02,  2.42700023e-05,  7.31445236e-03,  2.68723035e-03,-3.90440108e-06]

#if we want to just plot one of the bands
#lridge = [0,0,0,0,1]
#rridge = [0,0,0,0,1]


for pt in all_pts:
    #get all feat vects and associated time
    all_obs = [(cc['FeatVect'],cc['Date'],cc['Phase'],ClinFrame.clin_dict['DBS'+pt][cc['Phase']]['HDRS17']) for cc in analysis.YFrame.file_meta if cc['Patient'] == pt and cc['Circadian'] == 'night']
    
    clincolors = np.array([dd for (aa,bb,cc,dd) in all_obs])
    avg_state = np.zeros((28,2,5))
    avg_traj = np.zeros((28,2))
    #lump states into phases
    for phph,phase in enumerate(dbo.Phase_List('ephys')):
        all_state = np.array([[(pp['Delta'][ss],pp['Theta'][ss],pp['Alpha'][ss],pp['Beta*'][ss],pp['Gamma1'][ss]) for ss in ['Left','Right']] for (pp,cc,dd,ee) in all_obs if dd == phase])#.reshape(-1,10,order='C')
        avg_state[phph,:,:] = np.median(all_state,axis=0)
        avg_traj[phph] = (np.dot(avg_state[phph,0,:],lridge),np.dot(avg_state[phph,1,:],rridge))
    
    #BELOW does it for ALL RECORDINGS
    #vec = np.zeros((all_state.shape[0],2))
    #for rr,rec in enumerate(all_state):
    #    vec[rr,:] = (np.dot(rec[0,:],lridge),np.dot(rec[1,:],rridge))
    
    
    vec = avg_traj
    #plt.figure()
    #plt.scatter(vec[:,0],vec[:,1],alpha=0.4,cmap='jet',c=clincolors)
    
    #let's do some smoothing of the trajectory
    interp = True
    if interp:
        orig_len = len(vec[:,0])
        ti = np.linspace(2,orig_len+1, 10 * orig_len)
        
        x = np.concatenate((vec[-3:-1,0],vec[:,0],vec[1:3,0]))
        y = np.concatenate((vec[-3:-1,1],vec[:,1],vec[1:3,1]))
        t = np.arange(len(x))
        xi = interp1d(t,x,kind='cubic')(ti)
        yi = interp1d(t,y,kind='cubic')(ti)
    
        tstart = 5
        tend = -3
    else:
        tstart = 3 
        tend = -1
        #Now let's plot the trajectories
    
    
    plt.scatter(x,y,alpha=0.4,cmap='jet')
    plt.scatter(x[tstart],y[tstart],marker='o',s=400,color='green')
    plt.scatter(x[tend],y[tend],marker='X',s=400,color='red')
    plt.plot(xi,yi,alpha=0.3)
    #plt.hlines(0,-0.3,0.3)
    #plt.vlines(0,-0.3,0.3)
        
    
    #for pts in range(len(vec)-2):
    #    lines = plt.plot(vec[pts:pts+2,0],vec[pts:pts+2,1],alpha=0.3)[0]
    #    for ll in lines:
    #        add_arrow(ll)
    
    #or tt in range(len(vec)-1):
    #    plt.arrow(vec[tt+1,0]-vec[tt,0],vec[tt+1,1]-vec[tt,1],0.005,0.005,alpha=1)
    #plt.suptitle(pt)
    #plt.colorbar()
    #full_vect = [(cc['Left'],cc['Right']) for cc in ban]