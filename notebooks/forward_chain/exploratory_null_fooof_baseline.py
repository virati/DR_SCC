"""EXPLORATORY (not a pre-registered test): circular-shift null for F-fooof x baseline x T-raw, the best
exploratory FOOOF condition. Same 100 draws as the pre-registered nulls. Reported as exploratory in RESULTS.md."""
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

foo, targets = fc.load_fooof(), fc.clinical_targets()
df0, feats = fc.table(foo, targets, "F-fooof", "baseline", "T-raw")
seeds = np.random.default_rng(2026).integers(0, 2**31, size=100)


def draw(i, seed, m):
    s = fc.score(fc.e1(fc.circular_shift(df0, np.random.default_rng(seed)), feats, [m]))
    s["draw"] = i
    return s


N = pd.concat(Parallel(n_jobs=8)(delayed(draw)(i, s, m) for i, s in enumerate(seeds)
                                 for m in fc.MODELS if m != "M0_persistence"))
N.to_csv(HERE / "outputs" / "fooof_baseline_E1_null_EXPLORATORY.csv", index=False)
obs = pd.read_csv(HERE / "outputs" / "fooof_E1_scores.csv")
obs = obs[(obs.norm == "baseline") & (obs.target == "T-raw")].set_index("model")
w = N.pivot_table(index="draw", columns="model", values="R2")
rows = []
for m in ["M2_ENR", "M3_ENR+time", "M4_mixed", "M5_SVR", "M6_SVR+time"]:
    d = obs.loc[m, "R2"] - obs.loc["M1_time", "R2"]
    nd = (w[m] - w["M1_time"]).dropna()
    rows.append({"model": m, "R2": obs.loc[m, "R2"], "r": obs.loc[m, "r"], "dR2_vs_time": d,
                 "p_dR2": (np.sum(nd >= d) + 1) / (len(nd) + 1),
                 "p_R2": (np.sum(w[m].dropna() >= obs.loc[m, "R2"]) + 1) / (w[m].notna().sum() + 1),
                 "dR2_vs_persistence": obs.loc[m, "R2"] - obs.loc["M0_persistence", "R2"]})
T = pd.DataFrame(rows)
T.to_csv(HERE / "outputs" / "fooof_baseline_E1_tests_EXPLORATORY.csv", index=False)
print(T.round(3).to_string(index=False))
