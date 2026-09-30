# %% [markdown]
# # Cohort DR-SCC: Day vs Night vs Both
# Every dissertation / bioRxiv analysis uses daytime recordings only: the decoder class hardcodes
# `circ = "day"` (10:00-21:00 by filename timestamp). This notebook repeats the cohort decoders on
# daytime, nighttime, and all recordings.
#
# Two evaluations per recording set:
# 1. **Paper-style** (recording-level 60/40 split, patient-weeks shared between train and test),
#    over 20 random record orders because the split depends on frame order.
#    - ENR: the paper method (weekly mean band power, alpha from the regularization path).
#    - SVR-RBF: weekly mean + variance, standardized, grid search with patient-grouped CV on train.
# 2. **Leave-one-patient-out** (no shared weeks; independent of record order).
#    - ENR (CV alpha): ElasticNetCV (l1_ratio 0.8), alpha by patient-grouped CV on the 5 training patients.
#    - ENR (paper alpha): ElasticNet (l1_ratio 0.8) at the paper-pipeline alpha, 0.021.
#    - SVR-RBF: as in `svr/cohort_dr_scc_svr_lopo.py`.
#
# Run as a script or notebook. Needs DATA_DIRECTORY (via .env / find_dotenv, or the environment).

# %%
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from dbspace.readout import ClinVect, decoder
from dbspace.utils.dissertation import notebook_setup

import logging
logging.getLogger().setLevel(logging.WARNING)

# %%
DATADIR = Path(notebook_setup.DATADIR or os.environ["DATA_DIRECTORY"])
frame_file = DATADIR / "intermed" / "chronic" / "Chronic_FrameFeb2026_F.pickle"
OUT = Path(os.environ.get("DRSCC_OUT", Path(__file__).resolve().parent / "outputs" if "__file__" in globals() else "outputs")) / "circadian"
OUT.mkdir(parents=True, exist_ok=True)

do_pts = ["901", "903", "905", "906", "907", "908"]
test_scale = "pHDRS17"  # nHDRS in the paper
circ_sets = {"day": ("day",), "night": ("night",), "both": ("day", "night")}
n_orders = int(os.environ.get("N_ORDERS", 20))
train_ratio = 0.6
svr_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.01, 0.05, 0.1],
}

# %%
ClinFrame = ClinVect.CStruct(DATADIR / "clinical" / "clinical_vectors_all.json")
BRFrame = pickle.load(open(frame_file, "rb"))
stored_meta = list(BRFrame.file_meta)


def use_order(seed):
    """Shuffle record order with seed, then restore the decoder's import-time global seeds (2011),
    so each order runs exactly as it would in a fresh process."""
    fm = list(stored_meta)
    random.Random(seed).shuffle(fm)
    BRFrame.file_meta = fm
    np.random.seed(2011)
    random.seed(2011)


def n_recordings(circ):
    return sum(1 for r in stored_meta if r["Circadian"] in circ and r["Patient"] in do_pts)


# %% [markdown]
# ## Paper-style evaluation over random record orders

# %%
def enr_paper(circ):
    r = decoder.weekly_decoderCV(
        BRFrame=BRFrame, ClinFrame=ClinFrame, pts=do_pts, clin_measure=test_scale,
        algo="ENR", alpha=-4, shuffle_null=False, FeatureSet="main", variance=False,
    )
    r.global_plotting = False
    r.circ = circ
    r.filter_recs(rec_class="main_study")
    r.split_train_set(train_ratio)
    r.train_setup()
    r._path_slope_regression()
    r.train_model()
    r.test_setup()
    r.test_model()
    X, y = np.asarray(r.test_set_y), np.asarray(r.test_set_c).squeeze()
    pred = np.asarray(r.decode_model.predict(X)).squeeze()
    return {
        "r2_subsample_mean": float(np.mean([a["Score"] for a in r.test_stats])),
        "r2_full": float(r.decode_model.score(X, y)),
        "pearson_full": float(sp_stats.pearsonr(pred, y)[0]),
        "n_test_weeks": int(X.shape[0]),
    }


def svr_paper(circ):
    d = decoder.weekly_decoderCV(
        BRFrame=BRFrame, ClinFrame=ClinFrame, pts=do_pts, clin_measure=test_scale,
        algo="ENR", alpha=-4, shuffle_null=False, FeatureSet="main", variance="both", standardize=True,
    )
    d.global_plotting = False
    d.circ = circ
    d.filter_recs(rec_class="main_study")
    d.split_train_set(train_ratio)
    d.train_setup()
    d.test_setup()
    tr_y, tr_c, tr_pt = d.train_set_y, d.train_set_c.squeeze(), d.train_set_pt
    te_y, te_c = d.test_set_y, d.test_set_c.squeeze()
    splits = list(GroupKFold(n_splits=min(len(np.unique(tr_pt)), 5)).split(tr_y, tr_c, groups=tr_pt))
    search = GridSearchCV(SVR(kernel="rbf"), svr_grid, cv=splits, scoring="r2", n_jobs=-1).fit(tr_y, tr_c)
    pred = search.best_estimator_.predict(te_y)
    return {
        "r2_full": float(search.best_estimator_.score(te_y, te_c)),
        "pearson_full": float(sp_stats.pearsonr(pred, te_c)[0]),
        "cv_r2_patient_grouped": float(search.best_score_),
        "n_test_weeks": int(te_y.shape[0]),
    }


rows = []
for name, circ in circ_sets.items():
    for seed in range(n_orders):
        for model, fn in [("ENR", enr_paper), ("SVR-RBF", svr_paper)]:
            use_order(seed)
            res = fn(circ)
            rows.append({"recordings": name, "model": model, "order_seed": seed, **res})
            print(name, model, seed, {k: round(v, 3) for k, v in res.items()})
paper = pd.DataFrame(rows)
paper.to_csv(OUT / "paper_style_by_order.csv", index=False)

# %% [markdown]
# ## Leave-one-patient-out

# %%
def all_weeks(circ, variance):
    d = decoder.weekly_decoder(
        BRFrame=BRFrame, ClinFrame=ClinFrame, pts=do_pts, clin_measure=test_scale,
        algo="ENR", shuffle_null=False, FeatureSet="main", variance=variance,
    )
    d.global_plotting = False
    d.circ = circ
    d.filter_recs(rec_class="main_study")
    d.train_set = d.active_rec_list
    d.train_setup()
    return d.train_set_y, d.train_set_c.squeeze(), d.train_set_pt


lopo_rows, lopo_pt_rows = [], []
for name, circ in circ_sets.items():
    BRFrame.file_meta = list(stored_meta)
    for model in ["ENR (CV alpha)", "ENR (paper alpha)", "SVR-RBF"]:
        Y, C, P = all_weeks(circ, variance=False if model.startswith("ENR") else "both")
        nonzero = []
        preds, truth = np.zeros_like(C), C
        for tr, te in LeaveOneGroupOut().split(Y, C, groups=P):
            inner = list(GroupKFold(n_splits=min(len(np.unique(P[tr])), 5)).split(Y[tr], C[tr], groups=P[tr]))
            if model.startswith("ENR"):
                if model == "ENR (CV alpha)":
                    m = ElasticNetCV(l1_ratio=0.8, cv=inner, n_alphas=100, max_iter=10000).fit(Y[tr], C[tr])
                else:
                    m = ElasticNet(alpha=0.021, l1_ratio=0.8, max_iter=10000).fit(Y[tr], C[tr])
                nonzero.append(int(np.sum(m.coef_ != 0)))
                preds[te] = m.predict(Y[te])
            else:
                sc = StandardScaler().fit(Y[tr])
                m = GridSearchCV(SVR(kernel="rbf"), svr_grid, cv=inner, scoring="r2", n_jobs=-1).fit(sc.transform(Y[tr]), C[tr])
                preds[te] = m.best_estimator_.predict(sc.transform(Y[te]))
            pt = P[te][0]
            ss_res = np.sum((C[te] - preds[te]) ** 2)
            ss_tot = np.sum((C[te] - C[te].mean()) ** 2)
            lopo_pt_rows.append({"recordings": name, "model": model, "held_out": pt, "r2": 1 - ss_res / ss_tot,
                                 "pearson": sp_stats.pearsonr(preds[te], C[te])[0], "n_weeks": len(te)})
        r, p = sp_stats.pearsonr(preds, truth)
        lopo_rows.append({"recordings": name, "model": model,
                          "pooled_r2": 1 - np.sum((truth - preds) ** 2) / np.sum((truth - truth.mean()) ** 2),
                          "pooled_pearson": r, "pooled_p": p,
                          "mean_patient_r2": np.mean([x["r2"] for x in lopo_pt_rows if x["recordings"] == name and x["model"] == model]),
                          "nonzero_coefs_per_fold": ",".join(map(str, nonzero))})
        print("LOPO", name, model, {k: round(v, 3) for k, v in lopo_rows[-1].items() if isinstance(v, float)})
lopo = pd.DataFrame(lopo_rows)
lopo_pt = pd.DataFrame(lopo_pt_rows)
lopo.to_csv(OUT / "lopo_pooled.csv", index=False)
lopo_pt.to_csv(OUT / "lopo_by_patient.csv", index=False)

# %% [markdown]
# ## Summary

# %%
def fmt(s):
    return f"{s.mean():.3f} ± {s.std():.3f} [{s.min():.3f}, {s.max():.3f}]"


summary = []
for (name, model), g in paper.groupby(["recordings", "model"], sort=False):
    row = {"recordings": name, "model": model, "n_recordings": n_recordings(circ_sets[name]), "orders": len(g),
           "test_r2_full": fmt(g["r2_full"]), "pearson_full": fmt(g["pearson_full"])}
    if model == "ENR":
        row["test_r2_subsample_mean"] = fmt(g["r2_subsample_mean"])
    else:
        row["cv_r2_patient_grouped"] = fmt(g["cv_r2_patient_grouped"])
    for lm in ([m for m in lopo.model.unique() if m.startswith("ENR")] if model == "ENR" else [model]):
        lp = lopo[(lopo.recordings == name) & (lopo.model == lm)].iloc[0]
        tag = "" if lm == model else " " + lm.split("(")[1].rstrip(")").replace(" ", "_")
        row[f"lopo_pooled_r2{tag}"] = round(lp.pooled_r2, 3)
        row[f"lopo_pooled_r{tag}"] = round(lp.pooled_pearson, 3)
    summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "summary.csv", index=False)
print(summary.to_string(index=False))
print(lopo_pt.pivot_table(index=["recordings", "model"], columns="held_out", values="r2").round(2).to_string())
