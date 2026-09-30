# %% [markdown]
# # DR-SCC forward chaining with FOOOF features (PROTOCOL.md, amendment 1)
# Prereq: uv run notebooks/forward_chain/fooof_features.py   (writes recordings_fooof.csv.gz)
# Usage: python run_fooof.py      Env: DATA_DIRECTORY, JOBS (8), N_NULL (100)
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
OUT = HERE / "outputs"
PRED = fc.data_dir() / "intermed" / "forward_chain"
t0 = time.time()
foo = fc.load_fooof()
targets = fc.clinical_targets()
print(f"FOOOF recordings {len(foo)}, QC-excluded {int((~foo.qc_ok.astype(bool)).sum())}")
conds = [("F-fooof", n, t) for n in ["raw", "baseline"] for t in ["T-raw", "T-smooth"]]
PRIMARY = ("F-fooof", "raw", "T-raw")


def run(cond, models, kind):
    df, feats = fc.table(foo, targets, *cond)
    if kind == "E1":
        p = fc.e1(df, feats, models)
    elif kind == "E2":
        p = fc.e1(df, feats, models, e2=True)
    else:
        p = fc.e3(df, feats, models)
    p["fset"], p["norm"], p["target"] = cond
    return p


lin = ["M0_persistence", "M1_time", "M2_ENR", "M3_ENR+time"]
by = ("fset", "norm", "target", "model")
E1 = pd.concat(Parallel(n_jobs=JOBS)(delayed(run)(c, [m], "E1") for c in conds for m in fc.MODELS))
E1.to_csv(PRED / "pred_fooof_E1.csv.gz", index=False)
fc.score(E1, by=by).to_csv(OUT / "fooof_E1_scores.csv", index=False)
E2 = pd.concat(Parallel(n_jobs=JOBS)(delayed(run)(c, lin, "E2") for c in conds))
fc.score(E2, by=by).to_csv(OUT / "fooof_E2_scores.csv", index=False)
E3 = pd.concat(Parallel(n_jobs=JOBS)(delayed(run)(c, [m], "E3") for c in conds[:2] for m in fc.MODELS))
E3.to_csv(PRED / "pred_fooof_E3.csv.gz", index=False)
fc.score(E3, by=("fset", "norm", "N", "model")).to_csv(OUT / "fooof_E3_scores.csv", index=False)
print(fc.score(E1[(E1.norm == "raw") & (E1.target == "T-raw")]).round(3).to_string(index=False), f"({time.time()-t0:.0f}s)")

df0, feats = fc.table(foo, targets, *PRIMARY)
seeds = np.random.default_rng(2026).integers(0, 2**31, size=N_NULL)  # same draws as the F-mean null


def null_draw(i, seed, m):
    s = fc.score(fc.e1(fc.circular_shift(df0, np.random.default_rng(seed)), feats, [m]))
    s["draw"] = i
    return s


N = pd.concat(Parallel(n_jobs=JOBS)(delayed(null_draw)(i, s, m) for i, s in enumerate(seeds)
                                     for m in fc.MODELS if m != "M0_persistence"))
N.to_csv(OUT / "fooof_E1_primary_null.csv", index=False)
print(f"null done ({time.time()-t0:.0f}s)")
