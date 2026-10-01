# %% [markdown]
# # Amendment W-A3: within-session variance of each oscillation in the stimulation-off sessions
# Usage: python wd_variance.py [classify] [chain] [null]     Run in the DR_SCC env.
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE), str(HERE.parent / "forward_chain"), str(HERE.parent / "cross_patient"),
                                            os.environ.get("PYTHONPATH", "")])
sys.path.insert(0, str(HERE.parent / "forward_chain"))
sys.path.insert(0, str(HERE.parent / "cross_patient"))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import fc
import xp

PTS, TR = ["905", "906", "907", "908"], ["906", "907", "908"]
fc.PTS = PTS
PB = ["delta", "theta", "alpha", "lbeta", "hbeta", "gamma"]
PW = [f"pw_{s}_{b}" for s in "LR" for b in PB]
DR = ["dr_" + c for c in fc.FEATS]
SETS = {"var": [c + "_var" for c in PW], "mean": [c + "_mean" for c in PW], "mean+var": [c + "_mean" for c in PW] + [c + "_var" for c in PW],
        "dr-var": [c + "_var" for c in DR], "dr-mean+var": [c + "_mean" for c in DR] + [c + "_var" for c in DR]}
SICK, STAB = ["C01", "C02", "C03", "C04"], ["C21", "C22", "C23", "C24"]
NEURAL = ["M2_ENR", "M3_ENR+time", "M4_mixed", "M5_SVR", "M6_SVR+time"]
OUT = HERE / "outputs"
I = fc.data_dir() / "intermed"
g = ["pt", "week", "t"]
stages = sys.argv[1:] or ["classify", "chain", "null"]
t0 = time.time()

seg = pd.concat([pd.read_csv(I / "weekly_dense/segments.csv.gz", dtype={"pt": str, "week": str}),
                 pd.read_csv(I / "weekly_dense/segments_drscc.csv.gz")], axis=1)
off = seg[(seg.state == "off") & (seg.n_files_week == 1)].sort_values(["pt", "t", "start_s"])
off = off[off.groupby(["pt", "week"]).start_s.transform("size") >= 6]
off["block"] = off.groupby(["pt", "week"]).cumcount() // 6
blk = off.groupby(g + ["block"])
B = blk[PW + DR].var(ddof=1).add_suffix("_var").join(blk[PW + DR].mean().add_suffix("_mean")).join(blk.size().rename("n")).reset_index()
B = B[B.n == 6].drop(columns="n")
targets = fc.clinical_targets()
B = B.merge(targets[["pt", "week", "T-raw"]].rename(columns={"T-raw": "y"}), on=["pt", "week"])
print(f"blocks {len(B)}, sessions {B.groupby(['pt','week']).ngroups} ({time.time()-t0:.0f}s)")


def fit(tr, cols):
    sc = MinMaxScaler().fit(tr[cols])
    m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[cols]), tr.lab)
    return lambda X: m.predict_proba(sc.transform(X[cols]))[:, 1]


def va(cols, labels=None):
    """VA1-VA3 for one feature set. labels: optional {(pt, week): 0/1} override (for the permutation null)."""
    d = B[B.week.isin(SICK + STAB)].copy()
    d["lab"] = d.week.isin(STAB).astype(int) if labels is None else d.set_index(["pt", "week"]).index.map(labels).values
    rows, scores = [], []
    for p in TR:
        pred = fit(d[d.pt.isin([q for q in TR if q != p])], cols)
        te = d[d.pt == p]
        pr = pred(te)
        ses = te.assign(pr=pr).groupby("week").agg(lab=("lab", "first"), pr=("pr", "mean"))
        rows.append({"analysis": "VA1 LOPO 906-908", "pt": p, "n_sessions": len(ses), "auc_block": roc_auc_score(te.lab, pr),
                     "auc_session": roc_auc_score(ses.lab, ses.pr) if ses.lab.nunique() == 2 else np.nan})
        if labels is None:
            a = B[(B.pt == p) & (B.t >= 4)]
            scores.append(a.assign(pr=pred(a)).groupby(g).agg(score=("pr", "mean"), y=("y", "first")).reset_index())
    if labels is None:
        pred = fit(d[d.pt.isin(TR)], cols)
        te = d[d.pt == "905"]
        pr = pred(te)
        ses = te.assign(pr=pr).groupby("week").agg(lab=("lab", "first"), pr=("pr", "mean"))
        rows.append({"analysis": "VA2 905 held out, calendar labels", "pt": "905", "n_sessions": len(ses),
                     "auc_block": roc_auc_score(te.lab, pr), "auc_session": roc_auc_score(ses.lab, ses.pr)})
        a = B[(B.pt == "905") & (B.t >= 4)]
        s5 = a.assign(pr=pred(a)).groupby(g).agg(score=("pr", "mean"), y=("y", "first")).reset_index()
        rows.append({"analysis": "VA2 905 held out, clinical-state labels (all C weeks)", "pt": "905", "n_sessions": len(s5),
                     "auc_block": roc_auc_score((a.y < 0.5).astype(int), pred(a)), "auc_session": roc_auc_score((s5.y < 0.5).astype(int), s5.score)})
        scores.append(s5)
    return pd.DataFrame(rows), (pd.concat(scores) if scores else None)


if "classify" in stages:
    A, T, PT = [], [], []
    base = B[B.week.isin(SICK + STAB) & B.pt.isin(TR)].groupby(["pt", "week"]).t.first().reset_index()
    base["lab"] = base.week.isin(STAB).astype(int)
    seeds = np.random.default_rng(2026).integers(0, 2**31, 200)

    def perm(seed, cols):
        r = np.random.default_rng(seed)
        lab = {}
        for p in TR:
            b = base[base.pt == p]
            lab.update(dict(zip(zip(b.pt, b.week), r.permutation(b.lab.values))))
        try:
            return va(cols, labels=lab)[0].auc_block.mean()
        except ValueError:
            return np.nan

    for name, cols in SETS.items():
        a, sc = va(cols)
        a["set"] = name
        A.append(a)
        for p, gd in sc.groupby("pt"):
            r1, p1 = sp_stats.spearmanr(gd.score, gd.y)
            r2, p2 = sp_stats.spearmanr(gd.score, gd.t)
            T.append({"set": name, "pt": p, "n_sessions": len(gd), "rho_score_nHDRS": r1, "p_score_nHDRS": p1, "rho_score_week": r2, "p_score_week": p2})
        nullv = np.array(Parallel(n_jobs=8)(delayed(perm)(s, cols) for s in seeds))
        obs = a[a.analysis == "VA1 LOPO 906-908"].auc_block.mean()
        PT.append({"set": name, "obs_auc_block_mean": obs, "null_mean": np.nanmean(nullv), "null_95": np.nanquantile(nullv, 0.95),
                   "p": (np.nansum(nullv >= obs) + 1) / (np.sum(~np.isnan(nullv)) + 1)})
    pd.concat(A).to_csv(OUT / "v_auc.csv", index=False)
    pd.DataFrame(T).to_csv(OUT / "v_time_vs_state.csv", index=False)
    pd.DataFrame(PT).to_csv(OUT / "v_permutation.csv", index=False)
    print(pd.concat(A).pivot_table(index=["analysis", "pt"], columns="set", values="auc_block", sort=False).round(3).to_string())
    print(pd.DataFrame(PT).round(3).to_string(index=False), f"\nclassify done ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- VA4: continuous tracking, forward chaining
CH = {k: v for k, v in SETS.items() if k != "mean"}
allc = sorted({c for v in CH.values() for c in v})
W = B.groupby(g)[allc + ["y"]].median().reset_index()


def tab(name, norm):
    cols = CH[name]
    d = W[g + ["y"] + cols].copy()
    if norm == "baseline":
        for p in PTS:
            b = B[(B.pt == p) & (B.t <= 3)][cols]
            if len(b) > 5:
                d.loc[d.pt == p, cols] = (d.loc[d.pt == p, cols] - b.mean()) / b.std(ddof=1).replace(0, np.nan)
        d = d.fillna(0.0)
    return d.sort_values(["pt", "t"]).reset_index(drop=True), cols


CELLS = [(n_, nm) for n_ in CH for nm in ("raw", "baseline")]
TABS = {c: tab(*c) for c in CELLS}

if "chain" in stages:
    def run(c, m):
        df, cols = TABS[c]
        s = xp.score(fc.e1(df, cols, [m]))
        s["set"], s["norm"] = c
        return s

    S = pd.concat(Parallel(n_jobs=8)(delayed(run)(c, m) for c in CELLS for m in fc.MODELS))
    S.to_csv(OUT / "v_chain_scores.csv", index=False)
    print(S.sort_values("R2", ascending=False)[["set", "norm", "model", "n", "R2", "r", "MAE"]].head(10).round(3).to_string(index=False),
          f"\nchain done ({time.time()-t0:.0f}s)")

if "null" in stages:
    seeds = np.random.default_rng(2026).integers(0, 2**31, size=100)

    def nd(i, seed, c):
        df, cols = TABS[c]
        s = xp.score(fc.e1(fc.circular_shift(df, np.random.default_rng(seed)), cols, ["M0_persistence"] + NEURAL)).set_index("model")
        return [{"draw": i, "set": c[0], "norm": c[1], "model": m, "gain": s.loc[m, "R2"] - s.loc["M0_persistence", "R2"]} for m in NEURAL]

    N = pd.DataFrame([r for part in Parallel(n_jobs=8)(delayed(nd)(i, s, c) for i, s in enumerate(seeds) for c in CELLS) for r in part])
    N.to_csv(OUT / "v_chain_null.csv", index=False)
    print(f"null done ({time.time()-t0:.0f}s)")
