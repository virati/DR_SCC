# %% [markdown]
# # Cross-patient decoding, phase 1: run PROTOCOL.md
# Prereq: uv run notebooks/cross_patient/xp_features.py ; forward_chain feature tables present.
# Usage: python run_xp.py [primary] [null] [calib]      Env: DATA_DIRECTORY, JOBS (8), N_NULL (100)
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE), str(HERE.parent / "forward_chain"), os.environ.get("PYTHONPATH", "")])
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "forward_chain"))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import fc
import xp

JOBS, N_NULL = int(os.environ.get("JOBS", 8)), int(os.environ.get("N_NULL", 100))
stages = sys.argv[1:] or ["primary", "null", "calib"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
PRED = fc.data_dir() / "intermed" / "cross_patient"
FIRST_PICK = (True, "F-riem", "riem", "ENR")
t0 = time.time()
rec = xp.recordings()
targets = fc.clinical_targets()
CELLS = xp.cells()
TABLES = {(mc, fam, al): xp.table(rec, targets, fam, al, mc) for mc, fam, al, _ in CELLS}
print(f"{len(CELLS)} cells, {len(TABLES)} feature tables ({time.time()-t0:.0f}s)")


def run_cell(cell, df=None, N=0):
    mc, fam, al, m = cell
    d, cols = TABLES[(mc, fam, al)]
    d = d if df is None else df
    p, dropped = xp.lopo(d, cols, [m], mc, N=N)
    p["mc"], p["family"], p["align"] = mc, fam, al
    return p, dropped


def label(c):
    return f"{'MC' if c[0] else 'noMC'} | {c[1]} | {c[2]} | {c[3]}"


if "primary" in stages:
    res = Parallel(n_jobs=JOBS)(delayed(run_cell)(c) for c in CELLS)
    P = pd.concat([r[0] for r in res])
    P.to_csv(PRED / "pred_lopo.csv.gz", index=False)
    S = xp.score(P, by=("mc", "family", "align", "model"))
    refs = []
    for mc in (False, True):
        d, cols = TABLES[(mc, "F-band", "none")]
        rp, _ = xp.lopo(d, cols, [], mc, with_refs=True)
        rs = xp.score(rp)
        rs["mc"] = mc
        refs.append(rs)
    R = pd.concat(refs)
    S.to_csv(OUT / "lopo_scores.csv", index=False)
    R.to_csv(OUT / "lopo_reference_scores.csv", index=False)
    drops = {label(c): r[1] for c, r in zip(CELLS, res) if c[0]}
    pd.DataFrame([{"cell": k, "held_out": p, "dropped": ";".join(v)} for k, dd in drops.items() for p, v in dd.items()]
                 ).to_csv(OUT / "mc_screen_dropped.csv", index=False)
    print(R[["mc", "model", "R2", "r", "MAE"]].round(3).to_string(index=False))
    print(S.sort_values("R2", ascending=False)[["mc", "family", "align", "model", "R2", "r", "MAE"]].head(12).round(3).to_string(index=False))
    print(f"primary done ({time.time()-t0:.0f}s)")

if "null" in stages:
    seeds = np.random.default_rng(2026).integers(0, 2**31, size=N_NULL)

    def null_cell(i, seed, c):
        d, _ = TABLES[(c[0], c[1], c[2])]
        p, _ = run_cell(c, df=fc.circular_shift(d, np.random.default_rng(seed)))
        s = xp.score(p)
        return {"draw": i, "cell": label(c), "R2": float(s.R2.iloc[0]), "r": float(s.r.iloc[0])}

    N = pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(null_cell)(i, s, c) for i, s in enumerate(seeds) for c in CELLS))
    N.to_csv(OUT / "lopo_null.csv", index=False)
    print(f"null done ({time.time()-t0:.0f}s)")

if "calib" in stages:
    S = pd.read_csv(OUT / "lopo_scores.csv")
    N = pd.read_csv(OUT / "lopo_null.csv")
    S["cell"] = [label((mc, f, a, m)) for mc, f, a, m in zip(S.mc, S.family, S.align, S.model)]
    dmax = N.groupby("draw").R2.max()
    S["p_selection_corrected"] = [(np.sum(dmax >= r) + 1) / (len(dmax) + 1) for r in S.R2]
    S["p_uncorrected"] = [(np.sum(N[N.cell == c].R2 >= r) + 1) / (N[N.cell == c].R2.notna().sum() + 1)
                          for c, r in zip(S.cell, S.R2)]
    S.to_csv(OUT / "lopo_scores_with_p.csv", index=False)
    best = S.sort_values("R2", ascending=False).iloc[0]
    cell = (bool(best.mc), best.family, best.align, best.model)
    parts = [run_cell(cell, N=n)[0] for n in (0, 4, 8, 12)]
    C = xp.score(pd.concat(parts), by=("N", "model"))
    C["cell"] = label(cell)
    C.to_csv(OUT / "calibration_best_cell.csv", index=False)
    print(C[["N", "R2", "r", "MAE"]].round(3).to_string(index=False), f"\ncalib done ({time.time()-t0:.0f}s)")
