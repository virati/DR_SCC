#%%
%reload_ext autoreload
%autoreload 2

from dbspace.readout.BR_DataFrame import BR_Data_Tree
from dbspace.utils.structures import nestdict
from dbspace.readout import OBands
import scipy.stats as stats

import matplotlib.pyplot as plt
plt.close('all')

from dbspace.utils.structures import nestdict
from dbspace.readout import ClinVect

import seaborn as sns
sns.set(font_scale=4)
sns.set_style("white")

import numpy as np
import pickle
from matplotlib.patches import Rectangle, Circle
from numpy import ndenumerate

#%%
## Parameters for the analysis
ks_stats = nestdict()
pts = ['901','903','905','906','907','908']
bands = ['Delta','Theta','Alpha','Beta*','Gamma1']
all_feats = ['L-' + band for band in bands] + ['R-' + band for band in bands]


# for each patient, let's find the highest and lowest HDRS17 value and the week we find it
ClinFrame = ClinVect.CFrame(clinical_metadata_file = "/data/clinical/mayberg_2013/clinical_vectors_all.json", norm_scales=True)
hdrs_info = ClinFrame.min_max_weeks()

#hdrs_info = nestdict()
#week_labels = ClinFrame.week_labels()
#
#for pt in pts:
#    pt_hdrs_traj = [a for a in ClinFrame.DSS_dict['DBS'+pt]['HDRS17raw']][8:]
#    
#    hdrs_info[pt]['max']['index'] = np.argmax(pt_hdrs_traj)
#    hdrs_info[pt]['min']['index'] = np.argmin(pt_hdrs_traj)
#    hdrs_info[pt]['max']['week'] = week_labels[np.argmax(pt_hdrs_traj)+8]
#    hdrs_info[pt]['min']['week'] = week_labels[np.argmin(pt_hdrs_traj)+8]
#    
#    hdrs_info[pt]['max']['HDRSr'] = pt_hdrs_traj[hdrs_info[pt]['max']['index']]
#    hdrs_info[pt]['min']['HDRSr'] = pt_hdrs_traj[hdrs_info[pt]['min']['index']]
#    hdrs_info[pt]['traj']['HDRSr'] = pt_hdrs_traj

frame_to_analyse = 'Chronic_Frame_Dec2022'
BRFrame = pickle.load(open(f"/tmp/{frame_to_analyse}.pickle","rb"))
#do a check to see if any PSDs are entirely zero; bad sign
BRFrame.check_meta()
#%%

#Move forward with traditional oscillatory band analysis
feat_frame = OBands.OBands(BRFrame)
feat_frame.feat_extract(do_corrections=True)


#main_readout = naive_readout(feat_frame,ClinFrame)
#%%
#week_1 = 'C01'
#week_2 = 'C24'

week_distr = nestdict()
sig_stats = nestdict()
for pt in pts:
    for ff in bands:
        # if we want to compare max vs min hdrs
        feats,sig_stats[pt][ff],week_distr[pt][ff] = feat_frame.compare_states([hdrs_info[pt]['max']['week'],hdrs_info[pt]['min']['week']],feat=ff,circ='day',stat='ks')
        # if we want to compare early vs late
        #feats,sig_stats[pt][ff],week_distr[pt][ff] = feat_frame.compare_states(["C01","C24"],pt=pt,feat=ff,circ='day',stat='ks')
    

# Make our big stats 2d grid for all features across all patients
K_stats = np.array([[[sig_stats[pt][band][side][0] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
P_val = np.array([[[sig_stats[pt][band][side][1] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
pre_feat_vals = np.array([[[week_distr[pt][band][side]['depr'] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')
post_feat_vals = np.array([[[week_distr[pt][band][side]['notdepr'] for side in ['Left','Right']] for band in bands] for pt in pts]).reshape(6,-1,order='F')


depr_list = []
notdepr_list = []
aggr_week_distr = nestdict()

for ff in bands:
    aggr_week_distr[ff] = {dz:{side:[item for sublist in [week_distr[pt][ff][side][dz] for pt in pts] for item in sublist] for side in ['Left','Right']} for dz in ['depr','notdepr']}
    
    plt.figure()
    for ss,side in enumerate(['Left','Right']):
        
        print(ff + ' ' + side)
        
        plt.subplot(1,2,ss+1)
        ax = sns.violinplot(y=aggr_week_distr[ff]['depr'][side],alpha=0.2)
        plt.setp(ax.collections,alpha=0.3)
        ax = sns.violinplot(y=aggr_week_distr[ff]['notdepr'][side],color='red',alpha=0.2)
        plt.setp(ax.collections,alpha=0.3)
        print(stats.ks_2samp(np.array(aggr_week_distr[ff]['depr'][side]),np.array(aggr_week_distr[ff]['notdepr'][side])))
        
    plt.suptitle(ff + ' ' + ' min/max HDRS')

for side in ['Left','Right']:
    for ff in bands:
        depr_list.append(aggr_week_distr[ff]['depr'][side])
        notdepr_list.append(aggr_week_distr[ff]['notdepr'][side])
    

depr_list_flat = [item for sublist in depr_list for item in sublist]
notdepr_list_flat = [item for sublist in notdepr_list for item in sublist]

# Plot them all in same plot
plt.figure()
ax = sns.violinplot(data=depr_list,color='blue') 
ax = sns.violinplot(data=notdepr_list,color='red',alpha=0.3)
_ = plt.setp(ax.collections,alpha=0.3)

#%% GRID GETS PLOTTED HERE
def get_distr_change(patient,band,plot=True):
    
    pp = pts.index(patient)
    ff = all_feats.index(band)
    if plot:
        plt.figure()
        sns.violinplot(y=pre_feat_vals[pp,ff])
        sns.violinplot(y=post_feat_vals[pp,ff],color='red',alpha=0.3)

    return (np.mean(post_feat_vals[pp,ff]) - np.mean(pre_feat_vals[pp,ff]))


change_grid = np.zeros((len(pts),len(all_feats)))
for pp,pt in enumerate(pts):
    for ff,freq in enumerate(all_feats):
        change_grid[pp,ff] = get_distr_change(pt,freq,plot=False)
        
plt.figure()
plt.pcolormesh(change_grid,cmap=plt.cm.get_cmap('viridis'))
plt.colorbar()
plt.xticks(np.arange(10)+0.5,bands + bands,rotation=90)
plt.yticks(np.arange(6)+0.5,pts)
plt.title('Band Change - High -> Low HDRS')

ax = plt.axes()

for index,value in ndenumerate(P_val):
    if value < (0.05/10):
        if change_grid[index] > 0:
            usecolor='red'
        else:
            usecolor='blue'
            
        print(index)
        ax.add_patch(Rectangle((index[1], index[0]), 1, 1, fill=False, edgecolor='red', lw=5))
        ax.add_patch(Circle((index[1]+0.5, index[0]+0.5), 0.2, fill=True, facecolor=usecolor, edgecolor='white', lw=2))
plt.show()

