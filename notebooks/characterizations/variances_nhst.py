# %%
%reload_ext autoreload
%autoreload 2

#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

import dbspace as dbo
from dbspace.readout.OBands import OBands

import pickle

import seaborn as sns

import logging
#%%
logger = logging.getLogger()
logger.setLevel(logging.INFO)

sns.set_theme('paper')
sns.set_style("white")
# %%
base_data_dir = "/home/virati/Data/phd_vrt_2013/"
frame_to_analyse = 'Chronic_FrameFeb2026_F'
do_weeks = ["C01","C24"]
correct_for_mismatch_compression = True

base_data_dir = Path(base_data_dir)
frame_to_analyse = Path(frame_to_analyse)

BRFrame = pickle.load(open(f"{base_data_dir / frame_to_analyse}.pickle","rb"))
BRFrame.check_meta()

# %%
#Move forward with traditional oscillatory band analysis
analysis = OBands(BRFrame, ['901','903','905','906','907','908'])
analysis.feat_extract(do_corrections=correct_for_mismatch_compression)

# %%
# for a single patient and week, plot the distribution for a single oscillation

from dbspace.readout.ClinVect import Phase_List
weeks = Phase_List('ephys')
pt_stacks = analysis.patient_stacks(['905'])
pt_stack = pt_stacks['905']

pt_list = ([[recording['Delta']['Left'] for recording in pt_stack[week]] for week in weeks])
from itertools import zip_longest
pt_matrix = np.transpose(list(zip_longest(*pt_list, fillvalue=np.nan)))
pt_mean = []
pt_var = []
for ii in range(28):
    plt.hist(pt_matrix[ii,:], bins=np.linspace(-70,-50,15),alpha=0.1)
    pt_mean.append(np.nanmean(pt_matrix[ii,:]))
    pt_var.append(np.nanstd(pt_matrix[ii,:]))


# %%

pt_mean = np.array(pt_mean)
pt_var = np.array(pt_var)
plt.show()

plt.figure()
plt.plot(pt_mean)
plt.title('Patient Mean')
plt.show()

plt.figure()
plt.plot(pt_var)
plt.title('Patient Var')
plt.show()

# %% [markdown]
# # Cohort-level variance NHST
# Compare within-week variance of oscillatory power between two timepoints
# across all patients and bands.

# %%
from dbspace.utils.structures import nestdict
from numpy import ndenumerate
from matplotlib.patches import Rectangle, Circle

pts = ['901','903','905','906','907','908']
bands = ['Delta','Theta','Alpha','Beta*','Gamma1']
all_feats = ['L-' + band for band in bands] + ['R-' + band for band in bands]
circ = 'day'

# Gather per-week distributions for each patient/band/side
week_distr = nestdict()
ks_stats = nestdict()
for pt in pts:
    for ff in bands:
        _, ks_stats[pt][ff], week_distr[pt][ff] = analysis.scatter_state(
            weeks=do_weeks, pt=pt, feat=ff, circ=circ, plot=False, plot_type='scatter', stat='ks'
        )

pre_feat_vals = np.array(
    [[[week_distr[pt][band][side][do_weeks[0]] for side in ['Left','Right']] for band in bands] for pt in pts],
    dtype=object
).reshape(6, -1, order='F')

post_feat_vals = np.array(
    [[[week_distr[pt][band][side][do_weeks[1]] for side in ['Left','Right']] for band in bands] for pt in pts],
    dtype=object
).reshape(6, -1, order='F')

# %%
# Compute within-week variance for each patient/feature and test with Levene's test
var_pre = np.zeros((len(pts), len(all_feats)))
var_post = np.zeros((len(pts), len(all_feats)))
var_ratio = np.zeros((len(pts), len(all_feats)))
levene_pval = np.zeros((len(pts), len(all_feats)))

for pp in range(len(pts)):
    for ff in range(len(all_feats)):
        pre_obs = np.array(pre_feat_vals[pp, ff], dtype=float)
        post_obs = np.array(post_feat_vals[pp, ff], dtype=float)
        var_pre[pp, ff] = np.var(pre_obs)
        var_post[pp, ff] = np.var(post_obs)
        var_ratio[pp, ff] = var_post[pp, ff] / var_pre[pp, ff] if var_pre[pp, ff] > 0 else np.nan
        _, levene_pval[pp, ff] = stats.levene(pre_obs, post_obs)

# %%
# Variance change grid
var_change = var_post - var_pre

plt.figure()
plt.pcolormesh(var_change, cmap='RdBu_r')
plt.colorbar(label='Variance change (post - pre)')
plt.xticks(np.arange(10) + 0.5, bands + bands, rotation=90)
plt.yticks(np.arange(6) + 0.5, pts)
plt.title(f'Variance Change: {do_weeks}')
plt.show()

# %%
# Variance ratio grid (post / pre)
plt.figure()
plt.pcolormesh(np.log2(var_ratio), cmap='RdBu_r')
plt.colorbar(label='log2(variance ratio)')
plt.xticks(np.arange(10) + 0.5, bands + bands, rotation=90)
plt.yticks(np.arange(6) + 0.5, pts)
plt.title(f'log2(Variance Ratio) {do_weeks[1]}/{do_weeks[0]}')
plt.show()

# %%
# Significance grid with Bonferroni correction (Levene's test)
bonferroni_alpha = 0.05 / 10

ax = plt.axes()
plt.pcolormesh(var_change, cmap='RdBu_r')
plt.colorbar(label='Variance change')
plt.xticks(np.arange(10) + 0.5, bands + bands, rotation=90)
plt.yticks(np.arange(6) + 0.5, pts)
plt.title(f'Variance Change with Significance: {do_weeks}')

for index, value in ndenumerate(levene_pval):
    if value < bonferroni_alpha:
        usecolor = 'red' if var_change[index] > 0 else 'blue'
        ax.add_patch(Rectangle((index[1], index[0]), 1, 1, fill=False, edgecolor='red', lw=5))
        ax.add_patch(Circle((index[1] + 0.5, index[0] + 0.5), 0.2, fill=True, facecolor=usecolor, edgecolor='white', lw=2))

plt.show()

# %%
# P-value significance grid
plt.figure()
plt.pcolormesh((levene_pval < bonferroni_alpha).astype(np.float32), cmap='Set1_r')
plt.colorbar()
plt.yticks(np.arange(6) + 0.5, pts)
plt.xticks(np.arange(10) + 0.5, bands + bands, rotation=90)
plt.title("Significant Variance Difference (Levene's test)")
plt.show()

# %%
# Ensemble (cohort-pooled) variance comparison per feature
ensemble_levene_p = []
for ff in range(len(all_feats)):
    pre_pooled = np.concatenate([np.array(pre_feat_vals[pp, ff], dtype=float) for pp in range(len(pts))])
    post_pooled = np.concatenate([np.array(post_feat_vals[pp, ff], dtype=float) for pp in range(len(pts))])
    _, p = stats.levene(pre_pooled, post_pooled)
    ensemble_levene_p.append(p)

plt.figure()
plt.plot(ensemble_levene_p, 'o-')
plt.hlines(bonferroni_alpha, 0, 10, label='Bonferroni threshold')
plt.hlines(0.05, 0, 10, linestyle='dotted', label='p=0.05')
plt.xticks(range(len(all_feats)), all_feats, rotation=90)
plt.ylabel('p-value')
plt.title("Ensemble Levene's Test P-Values")
plt.legend()
plt.show()

# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")
