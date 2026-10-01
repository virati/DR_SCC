# %% [markdown]
# # Amendment W-A2: removing the stimulation effect from the at-home recordings
# Usage: python wd_stimcorr.py [diag] [primary] [null]     Run in the DR_SCC env (after wd_extract.py, wd_drscc.py).
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

import fc
import xp

PTS = ["905", "906", "907", "908"]
fc.PTS = PTS                                   # forward chaining and circular shifts over these four patients
DR = ["dr_" + c for c in fc.FEATS]
FO = [f"fo_{s}_{k}" for s in "LR" for k in ["offset", "exponent"] + fc.BANDS]
SETS = {"DR-SCC": DR, "FOOOF": FO}
NEURAL = ["M2_ENR", "M3_ENR+time", "M4_mixed", "M5_SVR", "M6_SVR+time"]
OUT = HERE / "outputs"
I = fc.data_dir() / "intermed"
g = ["pt", "week", "t"]
stages = sys.argv[1:] or ["diag", "primary", "null"]
t0 = time.time()

# ---------------------------------------------------------------- data: session on/off medians, at-home weekly means
seg = pd.concat([pd.read_csv(I / "weekly_dense/segments.csv.gz", dtype={"pt": str, "week": str}),
                 pd.read_csv(I / "weekly_dense/segments_drscc.csv.gz")], axis=1)
n = seg.groupby(["pt", "week", "file", "state"]).size().unstack(fill_value=0).reset_index().rename(columns={"off": "n_off", "on": "n_on"})
seg = seg.merge(n, on=["pt", "week", "file"])
prim = seg[(seg.n_files_week == 1) & (seg.n_off >= 6)]
OFF = prim[prim.state == "off"].groupby(g)[DR + FO].median().reset_index()
ON = prim[(prim.state == "on") & (prim.n_on >= 3)].groupby(g)[DR + FO].median().reset_index()
band = pd.read_csv(I / "forward_chain/recordings_features.csv.gz", dtype={"pt": str, "week": str})
foo = pd.read_csv(I / "forward_chain/recordings_fooof.csv.gz", dtype={"pt": str, "week": str})
HOME = band[(band.circ == "day") & band.pt.isin(PTS)].groupby(g)[fc.FEATS].mean().add_prefix("dr_").reset_index()
fo_src = [f"{s}_{k}" for s in "LR" for k in ["offset", "exponent"] + fc.BANDS]
HF = foo[(foo.circ == "day") & foo.qc_ok.astype(bool) & foo.pt.isin(PTS)].groupby(g)[fo_src].mean().reset_index()
HF.columns = g + FO
HOME = HOME.merge(HF, on=g)
targets = fc.clinical_targets()
Y = targets[targets.pt.isin(PTS)][["pt", "week", "T-raw"]].rename(columns={"T-raw": "y"})
print(f"sessions: off {len(OFF)}, on {len(ON)}; at-home weeks {len(HOME)} ({time.time()-t0:.0f}s)")


def wide(p, cols):
    """Per-patient frame indexed by t with home / off / on blocks."""
    h = HOME[HOME.pt == p].set_index("t")[cols]
    o = OFF[OFF.pt == p].set_index("t")[cols].reindex(h.index)
    a = ON[ON.pt == p].set_index("t")[cols].reindex(h.index)
    return h, o, a


def correct(kind, cols):
    """At-home weekly features with the stimulation effect removed (C weeks only); label-free."""
    out = []
    for p in PTS:
        h, o, a = wide(p, cols)
        c = h.copy()
        both = o.notna().all(1) & a.notna().all(1) & (h.index >= 4)
        tw = h.index[both]
        cw = h.index[h.index >= 4]
        if kind == "K1":
            off = (a - o).loc[tw]
            for col in cols:
                c.loc[cw, col] = h.loc[cw, col] - np.interp(cw, tw, off[col].values)
        elif kind == "K3":
            for w in cw:
                fit = [t for t in tw if t != w]
                for col in cols:
                    b, a0 = np.polyfit(a.loc[fit, col], o.loc[fit, col], 1)
                    c.loc[w, col] = a0 + b * h.loc[w, col]
        elif kind == "K4":
            m, sd = h.loc[cw].mean(), h.loc[cw].std(ddof=1)
            for w in cw:
                fit = [t for t in tw if t != w]
                d = ((a - o).loc[fit].mean() / sd).values
                u = d / np.linalg.norm(d)
                x = ((h.loc[w] - m) / sd).values
                c.loc[w] = (x - (x @ u) * u) * sd.values + m.values
        c = c.reset_index()
        c.insert(0, "pt", p)
        out.append(c)
    C = pd.concat(out).merge(HOME[g], on=["pt", "t"])
    return C[g + cols]


def spear(x, y):
    ok = x.notna() & y.notna()
    return sp_stats.spearmanr(x[ok], y[ok]) if ok.sum() >= 5 else (np.nan, np.nan)


def bh(p):
    p = np.where(np.isnan(p), 1.0, p)
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return q


# ---------------------------------------------------------------- D1 and V1
if "diag" in stages:
    d1, v1 = [], []
    for sname, cols in SETS.items():
        corr = {k: (HOME[g + cols] if k == "K0" else correct(k, cols)) for k in ("K0", "K3", "K4")}
        for p in PTS:
            h, o, a = wide(p, cols)
            cw = h.index >= 4
            for col in cols:
                r1, r2, r3 = spear(a.loc[cw, col], h.loc[cw, col]), spear(o.loc[cw, col], h.loc[cw, col]), spear(a.loc[cw, col], o.loc[cw, col])
                d1.append({"set": sname, "feature": col, "pt": p, "rho_on_vs_home": r1[0], "rho_off_vs_home": r2[0], "rho_on_vs_off": r3[0],
                           "n_on": int((a.loc[cw, col].notna()).sum()), "n_off": int((o.loc[cw, col].notna()).sum())})
                for k, C in corr.items():
                    cc = C[C.pt == p].set_index("t")[col]
                    r = spear(cc.loc[cw], o.loc[cw, col])
                    v1.append({"set": sname, "correction": k, "feature": col, "pt": p, "rho_vs_off": r[0], "p": r[1]})
    D1, V1 = pd.DataFrame(d1), pd.DataFrame(v1)
    V1["q"] = np.nan
    for k, idx in V1.groupby("correction").groups.items():
        V1.loc[idx, "q"] = bh(V1.loc[idx, "p"].values)
    D1.to_csv(OUT / "s_d1_agreement.csv", index=False)
    V1.to_csv(OUT / "s_v1_by_patient.csv", index=False)
    vs = V1.groupby(["set", "correction", "feature"], sort=False).apply(lambda d: pd.Series({
        "rho_mean": d.rho_vs_off.mean(), "n_validated_pts": int(((d.rho_vs_off > 0) & (d.q < 0.05)).sum())})).reset_index()
    vs["validated"] = vs.n_validated_pts >= 3
    vs.to_csv(OUT / "s_v1_summary.csv", index=False)
    print(D1.groupby("set")[["rho_on_vs_home", "rho_off_vs_home", "rho_on_vs_off"]].mean().round(3).to_string())
    print(vs.groupby(["set", "correction"]).agg(rho_mean=("rho_mean", "mean"), validated=("validated", "sum")).round(3).to_string())
    print(f"diag done ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- V2: forward chaining on the session weeks
KEEP = OFF[g]                                                     # same (patient, week) rows for every table


def build(kind, sname, norm):
    cols = SETS[sname]
    C = OFF[g + cols] if kind == "OFF" else (HOME[g + cols] if kind == "K0" else correct(kind, cols))
    if norm == "baseline":
        C = fc.normalize(C.copy(), "baseline")
    return C.merge(KEEP, on=g).merge(Y, on=["pt", "week"]).sort_values(["pt", "t"]).reset_index(drop=True), cols


CELLS = [(k, s, nm) for k in ("K1", "K3", "K4") for s in SETS for nm in ("raw", "baseline")]
REFS = [("K0", s, nm) for s in SETS for nm in ("raw", "baseline")] + [("OFF", s, "raw") for s in SETS]
if "primary" in stages or "null" in stages:
    TABS = {c: build(*c) for c in CELLS + REFS}

if "primary" in stages:
    def run(c, models):
        df, cols = TABS[c]
        s = xp.score(fc.e1(df, cols, models))
        s["correction"], s["set"], s["norm"] = c
        return s

    S = pd.concat(Parallel(n_jobs=8)(delayed(run)(c, [m]) for c in CELLS + REFS for m in fc.MODELS))
    S.to_csv(OUT / "s_v2_scores.csv", index=False)
    show = S[S.model.isin(["M0_persistence"] + NEURAL)].sort_values("R2", ascending=False)
    print(show[["correction", "set", "norm", "model", "n", "R2", "r", "MAE"]].head(14).round(3).to_string(index=False))
    print(f"primary done ({time.time()-t0:.0f}s)")

if "null" in stages:
    seeds = np.random.default_rng(2026).integers(0, 2**31, size=int(os.environ.get("N_NULL", 100)))

    def nd(i, seed, c):
        df, cols = TABS[c]
        s = xp.score(fc.e1(fc.circular_shift(df, np.random.default_rng(seed)), cols, ["M0_persistence"] + NEURAL)).set_index("model")
        return [{"draw": i, "correction": c[0], "set": c[1], "norm": c[2], "model": m, "R2": s.loc[m, "R2"],
                 "gain": s.loc[m, "R2"] - s.loc["M0_persistence", "R2"]} for m in NEURAL]

    N = pd.DataFrame([r for part in Parallel(n_jobs=8)(delayed(nd)(i, s, c) for i, s in enumerate(seeds) for c in CELLS) for r in part])
    N.to_csv(OUT / "s_v2_null.csv", index=False)
    print(f"null done ({time.time()-t0:.0f}s)")
