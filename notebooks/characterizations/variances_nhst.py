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
correct_for_mismatch_compression = False

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

weeks = dbo.readout.ClinVect.Phase_List('ephys')
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

plt.figure();plt.plot(pt_var);plt.title('Patient Var');plt.show()


