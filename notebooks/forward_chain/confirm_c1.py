"""Amendment 2, C1: confirm the exploratory FOOOF x baseline result on held-out NIGHT recordings.
Everything frozen as in amendment 1; only the recording set changes (F-fooof-night). Criteria (each required,
for M3 and separately for M4): (a) pooled R2 > persistence R2, (b) circular-shift p(R2) < 0.05, (c) pooled r > 0."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import fc

COND = ("F-fooof-night", "baseline", "T-raw")
foo, targets = fc.load_fooof(), fc.clinical_targets()
df0, feats = fc.table(foo, targets, *COND)
print(f"night weeks {len(df0)}, recordings used {int(((foo.circ == 'night') & foo.qc_ok.astype(bool)).sum())}")

obs_pred = pd.concat(Parallel(n_jobs=8)(delayed(fc.e1)(df0, feats, [m]) for m in fc.MODELS))
obs_pred.to_csv(fc.data_dir() / "intermed/forward_chain/pred_confirm_C1.csv", index=False)
obs = fc.score(obs_pred).set_index("model")
obs.to_csv(HERE / "outputs" / "confirm_C1_scores.csv")

seeds = np.random.default_rng(2026).integers(0, 2**31, size=100)


def draw(i, seed, m):
    s = fc.score(fc.e1(fc.circular_shift(df0, np.random.default_rng(seed)), feats, [m]))
    s["draw"] = i
    return s


N = pd.concat(Parallel(n_jobs=8)(delayed(draw)(i, s, m) for i, s in enumerate(seeds)
                                 for m in fc.MODELS if m != "M0_persistence"))
N.to_csv(HERE / "outputs" / "confirm_C1_null.csv", index=False)
w = N.pivot_table(index="draw", columns="model", values="R2")
rows = []
for m in ["M2_ENR", "M3_ENR+time", "M4_mixed", "M5_SVR", "M6_SVR+time"]:
    r2, r = obs.loc[m, "R2"], obs.loc[m, "r"]
    p = (np.sum(w[m].dropna() >= r2) + 1) / (w[m].notna().sum() + 1)
    a, b, c = r2 > obs.loc["M0_persistence", "R2"], p < 0.05, r > 0
    rows.append({"model": m, "R2": r2, "r": r, "persistence_R2": obs.loc["M0_persistence", "R2"], "p_R2": p,
                 "a_beats_persistence": a, "b_null_p_lt_05": b, "c_r_positive": c,
                 "confirmed": bool(a and b and c) if m in ("M3_ENR+time", "M4_mixed") else None})
T = pd.DataFrame(rows)
T.to_csv(HERE / "outputs" / "confirm_C1_tests.csv", index=False)
print(obs.round(3).to_string())
print(T.round(3).to_string(index=False))
