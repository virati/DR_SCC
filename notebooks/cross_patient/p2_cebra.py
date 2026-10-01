# /// script
# requires-python = ">=3.8,<3.9"
# dependencies = ["cebra==0.4.0", "torch==2.2.2", "numpy==1.24.4", "scipy==1.10.1", "pandas==2.0.3",
#                 "scikit-learn==1.3.2", "joblib==1.3.2", "python-dotenv==1.0.1", "matplotlib==3.7.5", "setuptools<70"]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Cross-patient phase 2, P2-C (EXPLORATORY, amendment P2): CEBRA-Behavior, calibration-free leave-one-patient-out.
    uv run notebooks/cross_patient/p2_cebra.py [primary] [null]      (after p2_export.py)"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed
from scipy import stats as sp_stats
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv(usecwd=True))
DATA = Path(os.environ["DATA_DIRECTORY"]) / "intermed" / "cross_patient" / "p2"
OUT = Path(__file__).resolve().parent / "outputs"
PTS = ["901", "903", "905", "906", "907", "908"]
R = pd.read_csv(DATA / "recordings_aligned.csv.gz", dtype={"pt": str, "week": str})
COLS = json.load(open(DATA / "recording_feature_cols.json"))
PARAMS = dict(model_architecture="offset1-model", output_dimension=3, max_iterations=2000, batch_size=512,
              temperature=1, time_offsets=1, conditional="time_delta", device="cpu", verbose=False)


def fold(fam, p, rec, seed=2026):
    import torch
    import cebra
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cols = COLS[fam]
    d = rec.dropna(subset=cols)
    tr, te = d[d.pt != p], d[(d.pt == p) & (d.t >= 4)]
    model = cebra.CEBRA(**PARAMS).fit(tr[cols].values.astype("float32"), tr.y.values.astype("float32"))
    Etr = model.transform(tr[cols].values.astype("float32"))
    Ete = model.transform(te[cols].values.astype("float32"))
    knn = KNeighborsRegressor(n_neighbors=25).fit(Etr, tr.y.values)
    te = te.assign(pred=knn.predict(Ete))
    wk = te.groupby(["pt", "week", "t"]).agg(y=("y", "first"), pred=("pred", "mean")).reset_index()
    cons = np.nan
    try:
        from cebra.integrations.sklearn.metrics import consistency_score
        embs = [Etr[(tr.pt == q).values] for q in tr.pt.unique()]
        labs = [tr[tr.pt == q].y.values for q in tr.pt.unique()]
        # DEVIATION (descriptive metric only): default 100 label bins fail on sparse weekly labels; 10 bins used
        sc, _, _ = consistency_score(embeddings=embs, labels=labs, between="datasets", num_discretization_bins=10)
        cons = float(np.nanmean(sc))
    except Exception:
        pass
    wk["consistency"] = cons
    return wk


def score(g):
    y, yh = g.y.values, g.pred.values
    row = {"n": len(g), "R2": 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2),
           "r": sp_stats.pearsonr(yh, y)[0] if np.std(yh) > 0 else np.nan, "MAE": np.mean(np.abs(y - yh)),
           "consistency_mean": g.consistency.mean()}
    for p, gg in g.groupby("pt"):
        yy, pp = gg.y.values, gg.pred.values
        row[f"R2_{p}"] = 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
        row[f"r_{p}"] = sp_stats.pearsonr(pp, yy)[0] if np.std(pp) > 0 else np.nan
    return row


def shifted(seed):
    """Standard per-patient offsets; roll each patient's 28-week label series and map back to recordings."""
    rng = np.random.default_rng(seed)
    out = R.copy()
    for p in PTS:
        wk = out[out.pt == p].groupby("t").y.first().sort_index()
        new = dict(zip(wk.index, np.roll(wk.values, rng.integers(1, 28))))
        m = out.pt == p
        out.loc[m, "y"] = out.loc[m, "t"].map(new)
    return out


if __name__ == "__main__":
    stages = sys.argv[1:] or ["primary", "null"]
    JOBS = int(os.environ.get("JOBS", 8))
    t0 = time.time()
    if "primary" in stages:
        parts = Parallel(n_jobs=JOBS)(delayed(fold)(f, p, R) for f in COLS for p in PTS)
        W = pd.concat([w.assign(family=f) for (f, p), w in zip([(f, p) for f in COLS for p in PTS], parts)])
        W.to_csv(DATA / "pred_p2_cebra.csv", index=False)
        S = pd.DataFrame([{"family": f, **score(g)} for f, g in W.groupby("family")])
        S.to_csv(OUT / "p2_cebra_scores.csv", index=False)
        print(S[["family", "R2", "r", "MAE", "consistency_mean"]].round(3).to_string(index=False), f"({time.time()-t0:.0f}s)")
    if "null" in stages:
        seeds = np.random.default_rng(2026).integers(0, 2**31, size=100)[:20]

        def draw(i, seed, f):
            rec = shifted(seed)
            w = pd.concat([fold(f, p, rec) for p in PTS])
            s = score(w)
            return {"draw": i, "family": f, "R2": s["R2"], "r": s["r"]}

        N = pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(draw)(i, s, f) for i, s in enumerate(seeds) for f in COLS))
        N.to_csv(OUT / "p2_cebra_null.csv", index=False)
        print(f"null done ({time.time()-t0:.0f}s)")
