# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==1.26.4", "scipy==1.11.4", "pandas==2.1.4", "scikit-learn==1.4.2", "joblib==1.3.2", "python-dotenv==1.0.1"]
# ///
"""Cohort-size scaling of next-week nHDRS prediction (fc_scaling/PROTOCOL.md): S1-S5 and the neural-shift nulls.
    uv run notebooks/fc_scaling/fcs_run.py      Env: DATA_DIRECTORY (or .env), JOBS, N_NULL_SCALE (50), N_NULL_GAIN (200)
Predictions: $DATA_DIRECTORY/intermed/fc_scaling/*.csv.gz"""
import itertools
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forward_chain"))
import fc  # weekly(), normalize(): the feature tables of extend/forward_chain, unchanged

load_dotenv(find_dotenv(usecwd=True))
DATA = Path(os.environ["DATA_DIRECTORY"])
OUT = DATA / "intermed/fc_scaling"
OUT.mkdir(parents=True, exist_ok=True)
PTS, WEEKS = fc.PTS, fc.WEEKS
TEST_T = list(range(6, 28))                                   # C03-C24
JOBS = int(os.environ.get("JOBS", 14))
N_SCALE, N_GAIN = int(os.environ.get("N_NULL_SCALE", 50)), int(os.environ.get("N_NULL_GAIN", 200))
NEURAL_MODELS = ["M2", "M3", "M3h"]
TIME, HIST = ["t", "sqt"], ["lag1", "lag2"]


def targets():
    clin = {p["pt"][3:]: p for p in json.load(open(DATA / "clinical/clinical_vectors_all.json"))["HAMDs"]}
    rows = []
    for pt in PTS:
        h = dict(zip(clin[pt]["phases"], clin[pt]["HDRS17"]))
        base = np.mean([h[w] for w in ("A04", "A03", "A02", "A01")])
        raw = [h[w] / base for w in WEEKS]
        for i, w in enumerate(WEEKS):
            rows.append(dict(pt=pt, week=w, t=i, **{"T-raw": raw[i], "T-smooth": float(np.mean(raw[max(0, i - 2):i + 1]))}))
    return pd.DataFrame(rows)


def table(rec, fset, norm, target, T):
    W = fc.normalize(fc.weekly(rec, fset), norm)
    feats = [c for c in W.columns if c not in ("pt", "week", "t")]
    df = W.merge(T[["pt", "week", "t", target]].rename(columns={target: "y"}), on=["pt", "week", "t"]).sort_values(["pt", "t"]).reset_index(drop=True)
    df["sqt"] = np.sqrt(df.t)
    g = df.groupby("pt").y
    df["lag1"], df["lag2"] = g.shift(1).fillna(1.0), g.shift(2).fillna(1.0)
    return df, feats


def cols_of(model, feats):
    return {"M1": TIME, "M2": feats, "M3": feats + TIME, "M3h": feats + TIME + HIST, "H3": HIST + TIME}[model]


def enr(train, test, cols, grouped):
    sc = StandardScaler().fit(train[cols].values)
    if grouped and train.pt.nunique() > 1:
        cv = list(GroupKFold(n_splits=min(train.pt.nunique(), 5)).split(train, groups=train.pt))
    else:
        cv = list(TimeSeriesSplit(n_splits=3).split(train))
    m = ElasticNetCV(l1_ratio=0.8, cv=cv, n_alphas=100, max_iter=20000).fit(sc.transform(train[cols].values), train.y.values)
    return m.predict(sc.transform(test[cols].values))


def chain(df, feats, p, cohort, models, test_t=TEST_T):
    """Forward chain for target p with a given cohort (tuple of other patients). One row per test week and model."""
    own, oth = df[df.pt == p], df[df.pt.isin(cohort)]
    rows = []
    for k in test_t:
        test, past = own[own.t == k], own[own.t < k]
        if test.empty:
            continue
        train = pd.concat([oth, past])
        for m in models:
            if m == "H0":
                pr = past.y.values[-1]
            elif m == "H1":
                pr = past.y.mean()
            elif m == "H2":
                w = 0.5 ** np.arange(len(past))[::-1]
                pr = float((past.y.values * w).sum() / w.sum())
            else:
                pr = float(enr(train, test, cols_of(m, feats), grouped=len(cohort) > 0)[0])
            rows.append((p, len(cohort), ",".join(cohort), k, m, test.y.values[0], pr))
    return rows


def shift_neural(df, feats, rng):
    out = df.copy()
    for p in PTS:
        idx = out.index[out.pt == p]
        out.loc[idx, feats] = np.roll(out.loc[idx, feats].values, rng.integers(1, len(idx)), axis=0)
    return out


COLS = ["pt", "k", "cohort", "t", "model", "y", "pred"]


def scaling(df, feats, models, sizes=range(6)):
    jobs = [(p, c) for p in PTS for k in sizes for c in itertools.combinations([q for q in PTS if q != p], k)]
    return pd.DataFrame([r for x in (chain(df, feats, p, c, models) for p, c in jobs) for r in x], columns=COLS)


def one_scaling_job(df, feats, p, c, models):
    return chain(df, feats, p, c, models)


def curve(P, sst):
    """Pooled R² per model and cohort size: each patient's squared error averaged over its subsets of that size."""
    P = P.assign(se=(P.y - P.pred) ** 2)
    e = P.groupby(["model", "k", "pt", "cohort"]).se.sum().groupby(["model", "k", "pt"]).mean().groupby(["model", "k"]).sum()
    return (1 - e / sst).rename("R2").reset_index()


if __name__ == "__main__":
    T = targets()
    foo = pd.read_csv(DATA / "intermed/forward_chain/recordings_fooof.csv.gz", dtype={"pt": str, "week": str, "circ": str})
    rec = pd.read_csv(DATA / "intermed/forward_chain/recordings_features.csv.gz", dtype={"pt": str, "week": str, "circ": str})
    card = pd.read_csv(DATA / "intermed/ctrl_affine/cardiac/home.csv.gz", dtype={"pt": str, "week": str})
    card["flag"] = card.flagged_L.astype(bool) | card.flagged_R.astype(bool)
    card["on"] = (card.line_L > 1) | (card.line_R > 1)
    foo = foo.merge(card[["pt", "file", "flag", "on"]].drop_duplicates(["pt", "file"]), on=["pt", "file"], how="left")
    CONDS = {"primary: FOOOF, baseline-normalised": (foo, "F-fooof", "baseline"), "FOOOF, raw": (foo, "F-fooof", "raw"),
             "band power (10), baseline-normalised": (rec, "F-mean", "baseline")}
    tabs = {name: table(r, fs, nm, "T-raw", T) for name, (r, fs, nm) in CONDS.items()}
    dfp, featp = tabs["primary: FOOOF, baseline-normalised"]
    ytest = dfp[dfp.t.isin(TEST_T)].y.values
    sst = float(((ytest - ytest.mean()) ** 2).sum())
    json.dump(dict(sst=sst, n_test=int(len(ytest)), n_feats={k: len(v[1]) for k, v in tabs.items()}), open(OUT / "meta.json", "w"))
    subsets = [(p, c) for p in PTS for k in range(6) for c in itertools.combinations([q for q in PTS if q != p], k)]

    # ---- S1: scaling curves
    S1 = []
    for name, (df, feats) in tabs.items():
        res = Parallel(n_jobs=JOBS)(delayed(one_scaling_job)(df, feats, p, c, ["M1", "M2", "M3"]) for p, c in subsets)
        S1.append(pd.DataFrame([r for x in res for r in x], columns=COLS).assign(cond=name))
        print("S1 done:", name, flush=True)
    S1 = pd.concat(S1)
    S1.to_csv(OUT / "s1_scaling.csv.gz", index=False)

    # ---- S3: beyond clinical history (k = 5), raw and smoothed targets; S5 uses these predictions
    S3 = []
    for name, (r, fs, nm) in CONDS.items():
        for tgt in ("T-raw", "T-smooth"):
            df, feats = table(r, fs, nm, tgt, T)
            res = Parallel(n_jobs=JOBS)(delayed(chain)(df, feats, p, tuple(q for q in PTS if q != p), ["H0", "H1", "H2", "H3", "M1", "M3", "M3h"]) for p in PTS)
            S3.append(pd.DataFrame([x for rr in res for x in rr], columns=COLS).assign(cond=name, target=tgt))
    S3 = pd.concat(S3)
    S3.to_csv(OUT / "s3_history.csv.gz", index=False)
    print("S3 done", flush=True)

    # ---- S2: calibration curve (cohort of five + own first N weeks; fixed test weeks C17-C24)
    S2 = []
    for name, (df, feats) in tabs.items():
        for N in (0, 2, 4, 6, 8, 12, 16, 20):
            for p in PTS:
                own = df[df.pt == p]
                calib, test = own[own.t < N], own[own.t >= 20]
                train = pd.concat([df[df.pt != p], calib])
                for m in ("M1", "M3"):
                    for yv, pr, tt in zip(test.y.values, enr(train, test, cols_of(m, feats), True), test.t.values):
                        S2.append((name, N, p, tt, m, yv, pr))
                if len(calib):
                    S2 += [(name, N, p, tt, "H0", yv, calib.y.values[-1]) for yv, tt in zip(test.y.values, test.t.values)]
    pd.DataFrame(S2, columns=["cond", "N", "pt", "t", "model", "y", "pred"]).to_csv(OUT / "s2_calibration.csv.gz", index=False)
    print("S2 done", flush=True)

    # ---- S4: robustness (primary features; k = 5)
    S4 = []
    variants = {"heartbeat-flagged recordings dropped": foo[~foo.flag.fillna(False).astype(bool)],
                "stimulation-on recordings only in therapy weeks": foo[(foo.week.str.startswith("B")) | foo.on.fillna(False).astype(bool)]}
    for vn, fr in variants.items():
        df, feats = table(fr, "F-fooof", "baseline", "T-raw", T)
        res = Parallel(n_jobs=JOBS)(delayed(chain)(df, feats, p, tuple(q for q in PTS if q != p), ["M1", "M3", "H3", "M3h"]) for p in PTS)
        S4.append(pd.DataFrame([x for rr in res for x in rr], columns=COLS).assign(variant=vn))
    pd.concat(S4).to_csv(OUT / "s4_robustness.csv.gz", index=False)
    print("S4 done", flush=True)

    # ---- nulls (neural-shift), primary and sensitivity conditions
    def gain_draw(name, seed):
        df, feats = tabs[name]
        d = shift_neural(df, feats, np.random.default_rng(seed))
        rows = [r for p in PTS for r in chain(d, feats, p, tuple(q for q in PTS if q != p), ["M2", "M3", "M3h"])]
        P = pd.DataFrame(rows, columns=COLS)
        return {"cond": name, "draw": seed, **{m: 1 - float(((g.y - g.pred) ** 2).sum()) / sst for m, g in P.groupby("model")}}

    G = pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(gain_draw)(name, s) for name in tabs for s in range(N_GAIN)))
    G.to_csv(OUT / "null_gain_k5.csv", index=False)
    print("gain null done", flush=True)

    def scale_draw(seed):
        df, feats = dfp, featp
        d = shift_neural(df, feats, np.random.default_rng(10_000 + seed))
        rows = [r for p, c in subsets for r in chain(d, feats, p, c, ["M3"])]
        return curve(pd.DataFrame(rows, columns=COLS), sst).assign(draw=seed)

    Ns = pd.concat(Parallel(n_jobs=JOBS)(delayed(scale_draw)(s) for s in range(N_SCALE)))
    Ns.to_csv(OUT / "null_scaling_primary.csv", index=False)
    print("scaling null done", flush=True)
