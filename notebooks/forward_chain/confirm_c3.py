"""Amendment 4, C3: selection-corrected circular-shift null for the exploratory FOOOF lead.
Grid: F-fooof x {raw, baseline} x {T-raw, T-smooth} x {M2..M6}, E1, daytime (exactly amendment 1).
Statistic per cell: R2(model) - R2(M0 persistence). Null: per draw, max gain over all 20 cells, with the
same per-patient shift offsets in every cell. Lead survives if p(max null >= observed max) < 0.05."""
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

NEURAL = ["M2_ENR", "M3_ENR+time", "M4_mixed", "M5_SVR", "M6_SVR+time"]
CONDS = [("F-fooof", n, t) for n in ["raw", "baseline"] for t in ["T-raw", "T-smooth"]]
foo, targets = fc.load_fooof(), fc.clinical_targets()
tables = {c: fc.table(foo, targets, *c) for c in CONDS}

# observed: the amendment-1 daytime runs, recomputed here from the same frozen code for self-containment
def gains(cond, df, feats):
    s = fc.score(fc.e1(df, feats, ["M0_persistence"] + NEURAL)).set_index("model")
    return {(cond[1], cond[2], m): s.loc[m, "R2"] - s.loc["M0_persistence", "R2"] for m in NEURAL}


obs = {}
for d in Parallel(n_jobs=4)(delayed(gains)(c, *tables[c]) for c in CONDS):
    obs.update(d)
obs_s = pd.Series(obs)
obs_max, obs_arg = obs_s.max(), obs_s.idxmax()
print(f"observed max gain {obs_max:.3f} at {obs_arg}")

seeds = np.random.default_rng(2026).integers(0, 2**31, size=100)  # same draws as all earlier nulls


def null_cell(i, seed, cond):
    df, feats = tables[cond]
    g = gains(cond, fc.circular_shift(df, np.random.default_rng(seed)), feats)
    return [{"draw": i, "norm": k[0], "target": k[1], "model": k[2], "gain": v} for k, v in g.items()]


rows = [r for part in Parallel(n_jobs=8)(delayed(null_cell)(i, s, c) for i, s in enumerate(seeds) for c in CONDS)
        for r in part]
N = pd.DataFrame(rows)
N.to_csv(HERE / "outputs" / "confirm_C3_null_cells.csv", index=False)
draw_max = N.groupby("draw").gain.max()
p = (np.sum(draw_max >= obs_max) + 1) / (len(draw_max) + 1)
cells = pd.DataFrame([{"norm": k[0], "target": k[1], "model": k[2], "gain_vs_persistence": v,
                       "p_selection_corrected": (np.sum(draw_max >= v) + 1) / (len(draw_max) + 1)}
                      for k, v in obs.items()]).sort_values("gain_vs_persistence", ascending=False)
cells.to_csv(HERE / "outputs" / "confirm_C3_cells.csv", index=False)
pd.DataFrame([{"observed_max_gain": obs_max, "cell": " / ".join(obs_arg), "p_selection_corrected": p,
               "null_max_mean": draw_max.mean(), "null_max_95pct": draw_max.quantile(0.95),
               "survives": bool(p < 0.05), "n_draws": len(draw_max)}]).to_csv(HERE / "outputs" / "confirm_C3_summary.csv", index=False)
print(f"C3: observed max {obs_max:.3f}; null max mean {draw_max.mean():.3f}, 95th pct {draw_max.quantile(0.95):.3f}; p = {p:.3f}")
print(cells.round(3).head(8).to_string(index=False))
