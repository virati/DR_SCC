#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:16:13 2018

@author: virati
Plot the 2d trajectory of the DSV signal across all patients
"""

from DBSpace.readout.ClinVect import CFrame
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import itertools
from DBSpace.readout import DSV
from DBSpace.readout.DSV import DSV, ORegress
from DBSpace.readout.BR_DataFrame import BR_Data_Tree

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo

import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style("white")

from scipy.interpolate import interp1d
import pickle


ClinFrame = CFrame(norm_scales=True)
#ClinFrame.plot_scale(pts='all',scale='HDRS17')
#ClinFrame.plot_scale(pts=['901'],scale='MADRS')

#BRFrame = BRDF.BR_Data_Tree()
#BRFrame.full_sequence(data_path='/home/virati/Chronic_Frame_july.npy')
#BRFrame.full_sequence(data_path='/tmp/Chronic_Frame_DEF.npy')
#BRFrame.check_empty_phases()

BRFrame = pickle.load(open('/home/virati/Chronic_Frame.pickle',"rb"))

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
lridge = [ 0.00213977, -0.00344579,  0.00451551,  0.01977083,  0.00433902]
rridge = [-0.01324946, -0.01107167,  0.00518643, -0.01441286,  0.00091638]

#if we want to just plot one of the bands
#lridge = [0,0,0,0,1]
#rridge = [0,0,0,0,1]

traj_plot = plt.figure()
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
        
        x_orig = sig.detrend(vec[:,0],type='linear')
        y_orig = sig.detrend(vec[:,1],type='linear')
        
        x = np.concatenate((x_orig[-3:-1],x_orig,x_orig[1:3]))
        y = np.concatenate((y_orig[-3:-1],y_orig,y_orig[1:3]))
        t = np.arange(len(x))
        xi = interp1d(t,x,kind='cubic')(ti)
        yi = interp1d(t,y,kind='cubic')(ti)
    
        tstart = 5
        tend = -4
    else:
        tstart = 3 
        tend = -1
        #Now let's plot the trajectories
    
    plt.figure(traj_plot.number)
    plt.scatter(x,y,alpha=0.4,cmap='jet')
    plt.scatter(x[tstart],y[tstart],marker='o',s=400,color='blue')
    plt.scatter(x[tend],y[tend],marker='X',s=400,color='red')
    plt.plot(xi,yi,alpha=0.3)
    
    
    plt.figure()
    plt.plot((x_orig+y_orig))
    plt.title(pt)
    

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