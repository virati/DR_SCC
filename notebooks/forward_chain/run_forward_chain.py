# %% [markdown]
# # DR-SCC forward chaining: run everything in PROTOCOL.md
# Usage: python run_forward_chain.py [stage ...]   stages: primary exploratory e3 null   (default: all)
# Env: DATA_DIRECTORY (or .env via find_dotenv), JOBS (default 8), N_NULL (default 100).
# Aggregate scores -> ./outputs/ (committed). Per-week predictions -> $DATA_DIRECTORY/intermed/forward_chain/.

# %%
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import fc

JOBS = int(os.environ.get("JOBS", 8))
N_NULL = int(os.environ.get("N_NULL", 100))
stages = sys.argv[1:] or ["primary", "exploratory", "e3", "null"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
PRED = fc.data_dir() / "intermed" / "forward_chain"
PRED.mkdir(parents=True, exist_ok=True)

FSETS = ["F-mean", "F-dist", "F-circ", "F-noGC"]
NORMS = ["baseline", "raw"]
TARGETS = ["T-raw", "T-smooth", "T-DSC"]
PRIMARY = ("F-mean", "baseline", "T-raw")

t0 = time.time()
rec = fc.build_recordings(PRED / "recordings_features.csv.gz")
targets = fc.clinical_targets()
print(f"recordings {len(rec)}  weeks/pt {rec.groupby('pt').week.nunique().to_dict()}  ({time.time()-t0:.0f}s)")


def run_e1(cond, models=fc.MODELS, e2=False):
    df, feats = fc.table(rec, targets, *cond)
    p = fc.e1(df, feats, models, e2=e2)
    p["fset"], p["norm"], p["target"] = cond
    return p


def tag(df, cond):
    df["fset"], df["norm"], df["target"] = cond
    return df


# %% primary: E1 x primary condition, all models
if "primary" in stages:
    parts = Parallel(n_jobs=JOBS)(delayed(run_e1)(PRIMARY, [m]) for m in fc.MODELS)
    P = pd.concat(parts)
    P.to_csv(PRED / "pred_E1_primary.csv", index=False)
    S = fc.score(P)
    S.to_csv(OUT / "E1_primary_scores.csv", index=False)
    print(S.round(3).to_string(index=False), f"\n({time.time()-t0:.0f}s)")

# %% exploratory: E1 and E2 over every condition
if "exploratory" in stages:
    conds = [(f, n, t) for f in FSETS for n in NORMS for t in TARGETS]
    parts = Parallel(n_jobs=JOBS)(delayed(run_e1)(c, [m]) for c in conds for m in fc.MODELS)
    P = pd.concat(parts)
    P.to_csv(PRED / "pred_E1_all.csv.gz", index=False)
    fc.score(P, by=("fset", "norm", "target", "model")).to_csv(OUT / "E1_all_scores.csv", index=False)
    lin = ["M0_persistence", "M1_time", "M2_ENR", "M3_ENR+time"]
    parts = Parallel(n_jobs=JOBS)(delayed(run_e1)(c, lin, True) for c in conds)
    P2 = pd.concat(parts)
    P2.to_csv(PRED / "pred_E2_all.csv.gz", index=False)
    fc.score(P2, by=("fset", "norm", "target", "model")).to_csv(OUT / "E2_all_scores.csv", index=False)
    print(f"exploratory done ({time.time()-t0:.0f}s)")

# %% E3 calibration curve: primary condition and its raw-feature twin (for H3)
if "e3" in stages:
    def run_e3(cond, m):
        df, feats = fc.table(rec, targets, *cond)
        return tag(fc.e3(df, feats, [m]), cond)
    conds = [PRIMARY, ("F-mean", "raw", "T-raw")]
    P3 = pd.concat(Parallel(n_jobs=JOBS)(delayed(run_e3)(c, m) for c in conds for m in fc.MODELS))
    P3.to_csv(PRED / "pred_E3.csv", index=False)
    S3 = fc.score(P3, by=("norm", "N", "model"))
    S3.to_csv(OUT / "E3_scores.csv", index=False)
    print(S3[S3.N == 0].round(3).to_string(index=False), f"\n({time.time()-t0:.0f}s)")

# %% circular-shift null for the primary condition
if "null" in stages:
    df0, feats = fc.table(rec, targets, *PRIMARY)
    seeds = np.random.default_rng(2026).integers(0, 2**31, size=N_NULL)

    def null_draw(i, seed, m):
        d = fc.circular_shift(df0, np.random.default_rng(seed))
        s = fc.score(fc.e1(d, feats, [m]))
        s["draw"] = i
        return s

    N = pd.concat(Parallel(n_jobs=JOBS)(delayed(null_draw)(i, s, m)
                                         for i, s in enumerate(seeds) for m in fc.MODELS if m != "M0_persistence"))
    N.to_csv(OUT / "E1_primary_null.csv", index=False)
    print(f"null done ({time.time()-t0:.0f}s)")
