# %%
%reload_ext autoreload
%autoreload 2
from dbspace.utils.dissertation import notebook_setup

print(notebook_setup.DATADIR)

# %% [markdown]
# # Clinical Controller AUC Analysis
# Trains the DR-SCC decoder, then evaluates its clinical utility as a controller:
# - Can the readout detect when stimulation changes are needed?
# - Compares: readout-based, empirical (HDRS), empirical+readout, oracle, and null controllers
# - Reports precision-recall AUC curves and ROC-based classification performance
#
# Based on `scripts/clinical_controller/AUC_analysis.py` and `PR_analysis.py`.

# %%
import pickle
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d
import scipy.stats as stats

import matplotlib.pyplot as plt
from dbspace.readout import ClinVect, decoder
from dbspace.utils.structures import nestdict

# Logging stuff here
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Plotting settings here
plt.rcParams["image.cmap"] = "tab10"
sns.set_context("paper")
sns.set_style("white")

# %%
base_data_dir = "/home/virati/Data/phd_vrt_2013/"
frame_to_analyse = 'Chronic_FrameFeb2026_F'

do_pts = [
    "901",
    "903",
    "905",
    "906",
    "907",
    "908",
]
test_scale = "pHDRS17"

# %%
# Initialize our Clinical Frame and load in our BR Frame
ClinFrame = ClinVect.CStruct(Path(notebook_setup.DATADIR + "/clinical/clinical_vectors_all.json"), stim_change_file=Path(notebook_setup.DATADIR + "/clinical/voltage_changes.mat"))
if test_scale == "mHDRS":
    ClinFrame.gen_mHDRS()
elif test_scale == "DSC":
    ClinFrame.gen_DSC()

frame_to_analyse = 'Chronic_FrameFeb2026_F'
BRFrame = pickle.load(open(Path(notebook_setup.DATADIR) / f"{frame_to_analyse}.pickle","rb"))

# %% [markdown]
# # Train the decoder

# %%
main_readout = decoder.weekly_decoderCV(
    BRFrame=BRFrame,
    ClinFrame=ClinFrame,
    pts=do_pts,
    clin_measure=test_scale,
    algo="ENR",
    alpha=-4,
    shuffle_null=False,
    FeatureSet="main",
    variance="both",
    standardize=True,
)
main_readout.global_plotting = True
main_readout.filter_recs(rec_class="main_study")
main_readout.split_train_set(0.6)

# %%
main_readout.train_setup()
optimal_alpha = main_readout._path_slope_regression()
main_readout.train_model()

# %%
main_readout.test_setup()
main_readout.test_model()

# %% [markdown]
# # Stim-change table
# Which patients had stimulation changes at which phases?

# %%
print(ClinFrame.Stim_Change_Table())

# %% [markdown]
# ---
# # Precision-Recall Controller Analysis
# Binarizes the clinical state by whether a stimulation change occurred.
# Compares multiple "controller" strategies:
# - **readout**: DR-SCC predicted depression score
# - **empirical**: actual HDRS score
# - **empirical+readout**: residual (HDRS - readout)
# - **oracle**: binarized ground truth + noise
# - **null**: coinflip

# %%
threshold_c = decoder.controller_analysis(main_readout, bin_type="stim_changes")
threshold_c.controller_runs()

# %% [markdown]
# ---
# # ROC Classification Analysis
# Binarizes by HDRS threshold (>0.5 = depressed).
# Reports AUC and ROC curves for readout predictions vs null.

# %%
threshold_c.classif_runs()

# %% [markdown]
# ---
# # Ensemble AUC Analysis
# Run multiple train/test splits to build distributions of AUC values,
# then compare controllers with statistical tests.

# %%
n_runs = 8
n_iters_per_run = 5
algo_list = ["HDRS", "CB", "Random"]
color_code = {"HDRS": "red", "CB": "blue", "Random": "green"}

all_auc_curves = []
all_auc_vals = []
all_null_curves = []

for run in range(n_runs):
    # Re-split and retrain for each run
    main_readout.split_train_set(0.6)
    main_readout.train_setup()
    main_readout.train_model()
    main_readout.test_setup()
    main_readout.test_model()

    run_curves = []
    run_auc_vals = []
    run_null_curves = []

    for itr in range(n_iters_per_run):
        # Subsample the test set
        import random
        n_test = len(main_readout.test_set_y)
        n_sample = int(np.ceil(0.8 * n_test))
        indices = random.sample(range(n_test), n_sample)

        test_y = np.array([main_readout.test_set_y[i] for i in indices])
        test_c = np.array([main_readout.test_set_c[i] for i in indices])
        test_pt = np.array([main_readout.test_set_pt[i] for i in indices])
        test_ph = np.array([main_readout.test_set_ph[i] for i in indices])

        predicted_c = main_readout.decode_model.predict(test_y)

        binarized_c = [
            ClinFrame.query_stim_change(pt, ph)
            for pt, ph in zip(test_pt, test_ph)
        ]
        binarized_c = np.array(binarized_c)

        # Skip if no positive class
        if binarized_c.sum() == 0 or binarized_c.sum() == len(binarized_c):
            continue

        coinflip = np.random.choice([0, 1], size=len(test_pt), p=[0.5, 0.5])

        from sklearn.metrics import precision_recall_curve, auc as sk_auc

        iter_curves = {}
        iter_aucs = []

        # HDRS (empirical) as controller
        prec_h, rec_h, _ = precision_recall_curve(binarized_c, test_c.squeeze())
        iter_curves["HDRS"] = (prec_h, rec_h)
        iter_aucs.append(sk_auc(rec_h, prec_h))

        # CB (readout) as controller
        prec_cb, rec_cb, _ = precision_recall_curve(binarized_c, predicted_c)
        iter_curves["CB"] = (prec_cb, rec_cb)
        iter_aucs.append(sk_auc(rec_cb, prec_cb))

        # Random (coinflip) as controller
        prec_r, rec_r, _ = precision_recall_curve(binarized_c, coinflip)
        iter_curves["Random"] = (prec_r, rec_r)
        iter_aucs.append(sk_auc(rec_r, prec_r))

        run_curves.append(iter_curves)
        run_auc_vals.append(iter_aucs)

    all_auc_curves.append(run_curves)
    all_auc_vals.append(run_auc_vals)

print(f"Completed {n_runs} runs x {n_iters_per_run} iterations")

# %% [markdown]
# # Plot individual PR curves (all runs overlaid)

# %%
mean_recall = np.linspace(0, 1, 100)
curve_lib = {key: [] for key in algo_list}

plt.figure()
for run_curves in all_auc_curves:
    for iter_curves in run_curves:
        for algo in algo_list:
            if algo not in iter_curves:
                continue
            prec, rec = iter_curves[algo]
            # Deduplicate recall values (keep last precision at each recall)
            _, unique_idx = np.unique(rec, return_index=True)
            rec, prec = rec[unique_idx], prec[unique_idx]
            interp_func = interp1d(rec, prec, kind="zero", bounds_error=False, fill_value=0)
            interp_prec = interp_func(mean_recall)
            curve_lib[algo].append(interp_prec)
            plt.plot(mean_recall, interp_prec, color=color_code[algo], alpha=0.05)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Individual PR Curves (all runs)")
plt.legend(algo_list)

# %% [markdown]
# # Mean PR curves with shaded error

# %%
plt.figure()
for algo in algo_list:
    if not curve_lib[algo]:
        continue
    algo_res = np.array(curve_lib[algo])
    mean_prec = np.mean(algo_res, axis=0)
    std_prec = np.std(algo_res, axis=0)

    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)

    plt.plot(mean_recall, mean_prec, color=color_code[algo], linewidth=2, label=algo)
    plt.fill_between(mean_recall, prec_lower, prec_upper, color=color_code[algo], alpha=0.1)

plt.plot([0, 1], [0.5, 0.5], color="gray", linestyle="dotted", label="Chance")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Mean PR Curves (+/- 1 std)")
plt.legend()

# %% [markdown]
# # AUC distributions and comparison

# %%
flat_aucs = [auc_vals for run_vals in all_auc_vals for auc_vals in run_vals]
algo_aucs = np.array(flat_aucs)

bins = np.linspace(0, 1, 30)
line_height = 20

plt.figure()
for aa, algo in enumerate(algo_list):
    plt.hist(algo_aucs[:, aa], bins=bins, color=color_code[algo], label=algo, alpha=0.4)
    med = np.median(algo_aucs[:, aa])
    std = np.std(algo_aucs[:, aa])
    plt.vlines(med, 0, line_height + 5 * aa, color=color_code[algo], linewidth=3)
    plt.hlines(line_height + 5 * aa, med - std, med + std, color=color_code[algo], linewidth=3)

plt.xlabel("PR AUC")
plt.ylabel("Count")
plt.title("PR AUC Distributions")
plt.legend()

# %% [markdown]
# # Pairwise statistical tests (KS test)

# %%
print("Pairwise KS tests between controller AUC distributions:\n")
for aa, algo1 in enumerate(algo_list):
    for bb, algo2 in enumerate(algo_list):
        if bb <= aa:
            continue
        ks_stat, ks_p = stats.ks_2samp(algo_aucs[:, aa], algo_aucs[:, bb])
        print(f"  {algo1} vs {algo2}: KS={ks_stat:.4f}, p={ks_p:.4e}")

# %% Summary statistics
print("\nSummary:")
for aa, algo in enumerate(algo_list):
    vals = algo_aucs[:, aa]
    print(f"  {algo:>8s}: median={np.median(vals):.4f}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

# %%
# last successful run
from datetime import date
today = date.today()
print(f"Last Successful Run: {today}")
