"""Cross-patient decoding, phase 1 (PROTOCOL.md): weekly features, label-free alignment, mismatch-compression
screen, models (ENR, SVR-RBF, anchor regression, ICP), leave-one-patient-out and calibration evaluation."""
import itertools
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats as sp_stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forward_chain"))
import fc  # noqa: E402  (shared: targets, fit helpers, scoring)

from sklearn.linear_model import ElasticNetCV, LinearRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

warnings.filterwarnings("ignore")
PTS, WEEKS, BANDS = fc.PTS, fc.WEEKS, fc.BANDS
FAMILIES = ["F-band", "F-fooof-per", "F-rel", "F-asym", "F-riem"]
MODELS = ["ENR", "SVR", "Anchor", "ICP"]
ANCHOR_GAMMA = 5.0


# ---------------------------------------------------------------- recording table
def recordings():
    D = fc.data_dir() / "intermed"
    xp = pd.read_csv(D / "cross_patient/recordings_xp.csv.gz", dtype={"pt": str, "week": str})
    band = pd.read_csv(D / "forward_chain/recordings_features.csv.gz", dtype={"pt": str, "week": str})
    foo = pd.read_csv(D / "forward_chain/recordings_fooof.csv.gz", dtype={"pt": str, "week": str})
    assert len(xp) == len(band) == len(foo) and (xp.file.values == foo.file.values).all()
    assert (xp.pt.values == band.pt.values).all() and (xp.week.values == band.week.values).all()
    per = foo[[f"{s}_{b}" for s in "LR" for b in BANDS]].copy()
    per.columns = [f"per_{c}" for c in per.columns]
    out = pd.concat([xp, band[fc.FEATS].add_prefix("band_"), per, foo[["qc_ok"]]], axis=1)
    return out[out.circ == "day"].reset_index(drop=True)


def mc_recording_mask(rec):
    """Protocol step 1: drop recordings whose GCr (either side) is in the pooled daytime top decile."""
    qL, qR = rec.GCr_L.quantile(0.9), rec.GCr_R.quantile(0.9)
    return ~((rec.GCr_L > qL) | (rec.GCr_R > qR))


def stim_voltage():
    M = sio.loadmat(fc.data_dir() / "clinical/voltage_changes.mat")["StimMatrix"]
    return {pt: {w: float(M[i, 4 + t]) for t, w in enumerate(WEEKS)} for i, pt in enumerate(PTS)}


# ---------------------------------------------------------------- SPD helpers (2x2)
def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)) @ V.T


def spd_log(C):
    return _eig_fn(C, np.log)


def spd_exp(C):
    return _eig_fn(C, np.exp)


def spd_isqrt(C):
    return _eig_fn(C, lambda w: 1 / np.sqrt(w))


def riem_mean(Cs, iters=50):
    M = np.mean(Cs, axis=0)
    for _ in range(iters):
        Mi, Ms = spd_isqrt(M), _eig_fn(M, np.sqrt)
        T = np.mean([spd_log(Mi @ C @ Mi) for C in Cs], axis=0)
        M = Ms @ spd_exp(T) @ Ms
        if np.linalg.norm(T) < 1e-10:
            break
    return M


def tri(L):
    return [L[0, 0], L[1, 1], np.sqrt(2) * L[0, 1]]  # LL, RR, LR (isometric upper triangle)


# ---------------------------------------------------------------- weekly features
def weekly(rec, family, align):
    g = ["pt", "week", "t"]
    base = rec.groupby(g)
    W = base[["GCr_L", "GCr_R"]].mean().reset_index()
    if family in ("F-band", "F-rel", "F-asym"):
        B = base[[f"band_{c}" for c in fc.FEATS]].mean().reset_index()
        cols = [f"band_{c}" for c in fc.FEATS]
        if family == "F-rel" or family == "F-asym":
            for s in "LR":
                sc = [f"band_{s}{b}" for b in BANDS]
                B[sc] = B[sc].sub(B[sc].mean(axis=1), axis=0)
            if family == "F-asym":
                for b in BANDS:
                    B[f"asym_{b}"] = B[f"band_L{b}"] - B[f"band_R{b}"]
                cols = [f"asym_{b}" for b in BANDS]
        F = B[g + cols]
    elif family == "F-fooof-per":
        cols = [f"per_{s}_{b}" for s in "LR" for b in BANDS]
        F = rec[rec.qc_ok.astype(bool)].groupby(g)[cols].mean().reset_index()
    elif family == "F-riem":
        rows = []
        for key, grp in base:
            row = dict(zip(g, key))
            for b in BANDS:
                row[b] = np.array([[grp[f"{b}_LL"].mean(), grp[f"{b}_LR"].mean()],
                                   [grp[f"{b}_LR"].mean(), grp[f"{b}_RR"].mean()]])
            rows.append(row)
        M = pd.DataFrame(rows)
        feats = []
        for pt, idx in M.groupby("pt").groups.items():
            sub = M.loc[idx]
            for b in BANDS:
                Cs = np.stack(sub[b].values)
                if align == "riem":
                    ref = riem_mean(Cs[sub.t.values <= 3])
                    Ri = spd_isqrt(ref)
                    Ls = [spd_log(Ri @ C @ Ri) for C in Cs]
                    disp = np.mean([np.linalg.norm(Ls[i]) for i in np.where(sub.t.values <= 3)[0]]) or 1.0
                    Ls = [L / disp for L in Ls]
                else:
                    Ls = [spd_log(C) for C in Cs]
                for i, L in zip(idx, Ls):
                    for name, v in zip(("LL", "RR", "LR"), tri(L)):
                        M.loc[i, f"riem_{b}_{name}"] = v
        cols = [f"riem_{b}_{n}" for b in BANDS for n in ("LL", "RR", "LR")]
        F = M[g + cols]
    else:
        raise ValueError(family)
    F = W.merge(F, on=g, how="inner")
    if align == "z":
        for pt, idx in F.groupby("pt").groups.items():
            b = F.loc[idx][F.loc[idx, "t"] <= 3][cols]
            F.loc[idx, cols] = (F.loc[idx, cols] - b.mean()) / b.std(ddof=1).replace(0, np.nan)
        F[cols] = F[cols].fillna(0.0)
    return F, cols


def table(rec, targets, family, align, mc):
    r = rec[mc_recording_mask(rec)] if mc else rec
    F, cols = weekly(r, family, align)
    sv = stim_voltage()
    F["stimV"] = [sv[p][w] for p, w in zip(F.pt, F.week)]
    T = targets[["pt", "week", "t", "T-raw"]].rename(columns={"T-raw": "y"})
    return F.merge(T, on=["pt", "week", "t"]).sort_values(["pt", "t"]).reset_index(drop=True), cols


# ---------------------------------------------------------------- mismatch-compression feature screen
def _side(c):
    if c.endswith(("_LL",)) or c.startswith(("band_L", "per_L")):
        return "GCr_L"
    if c.endswith(("_RR",)) or c.startswith(("band_R", "per_R")):
        return "GCr_R"
    return "both"


def mc_screen(train, cols, patients):
    """Drop features whose within-patient Spearman rho with weekly GCr has BH-FDR p < 0.05 in >= 3 patients."""
    pv, keys = [], []
    for c in cols:
        for p in patients:
            d = train[train.pt == p]
            gcr = d[["GCr_L", "GCr_R"]].mean(axis=1) if _side(c) == "both" else d[_side(c)]
            rho, pval = sp_stats.spearmanr(d[c], gcr)
            pv.append(1.0 if np.isnan(pval) else pval)
            keys.append((c, p))
    pv = np.asarray(pv)
    order = np.argsort(pv)
    m = len(pv)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((pv[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    hits = pd.Series([k[0] for k, qq in zip(keys, q) if qq < 0.05]).value_counts()
    drop = set(hits[hits >= 3].index)
    return [c for c in cols if c not in drop], sorted(drop)


# ---------------------------------------------------------------- models
def _cv(train):
    return list(GroupKFold(n_splits=min(train.pt.nunique(), 5)).split(train, groups=train.pt))


def fit_predict(model, train, test, cols):
    if not cols:
        return np.full(len(test), train.y.mean())
    sc = StandardScaler().fit(train[cols].values)
    Xtr, Xte, y = sc.transform(train[cols].values), sc.transform(test[cols].values), train.y.values
    if model == "ENR":
        return ElasticNetCV(l1_ratio=0.8, cv=_cv(train), n_alphas=100, max_iter=20000).fit(Xtr, y).predict(Xte)
    if model == "SVR":
        return GridSearchCV(SVR(kernel="rbf"), fc.SVR_GRID, cv=_cv(train), scoring="neg_mean_squared_error",
                            n_jobs=1).fit(Xtr, y).best_estimator_.predict(Xte)
    if model == "Anchor":
        A = np.column_stack([pd.get_dummies(train.pt).values.astype(float), train[["GCr_L", "GCr_R", "stimV"]].values,
                             np.ones(len(train))])
        P = A @ np.linalg.pinv(A)
        W = np.eye(len(train)) + (np.sqrt(ANCHOR_GAMMA) - 1) * P
        xm, ym = Xtr.mean(0), y.mean()
        m = ElasticNetCV(l1_ratio=0.8, cv=_cv(train), n_alphas=100, max_iter=20000, fit_intercept=False)
        m.fit(W @ (Xtr - xm), W @ (y - ym))
        return ym + (Xte - xm) @ m.coef_
    if model == "ICP":
        enr = ElasticNetCV(l1_ratio=0.8, cv=_cv(train), n_alphas=100, max_iter=20000).fit(Xtr, y)
        top = list(np.argsort(-np.abs(enr.coef_))[:6])
        groups = train.pt.values
        accepted = []
        for k in range(len(top) + 1):
            for S in itertools.combinations(top, k):
                S = list(S)
                pred = LinearRegression().fit(Xtr[:, S], y).predict(Xtr[:, S]) if S else np.full(len(y), y.mean())
                res = y - pred
                parts = [res[groups == p] for p in np.unique(groups)]
                p_mean = sp_stats.kruskal(*parts).pvalue
                p_var = sp_stats.levene(*parts).pvalue
                if min(2 * min(p_mean, p_var), 1.0) > 0.05:
                    accepted.append(set(S))
        S = sorted(set.intersection(*accepted)) if accepted else []
        if not S:
            return np.full(len(test), y.mean())
        return LinearRegression().fit(Xtr[:, S], y).predict(Xte[:, S])
    raise ValueError(model)


def references(train, test):
    time_tr = np.column_stack([train.t, np.sqrt(train.t)])
    time_te = np.column_stack([test.t, np.sqrt(test.t)])
    sc = StandardScaler().fit(time_tr)
    tpred = ElasticNetCV(l1_ratio=0.8, cv=_cv(train), n_alphas=100, max_iter=20000).fit(
        sc.transform(time_tr), train.y).predict(sc.transform(time_te))
    return {"REF_train_mean": np.full(len(test), train.y.mean()), "REF_time": tpred}


# ---------------------------------------------------------------- evaluation
def lopo(df, cols, models, mc, N=0, with_refs=False):
    rows, dropped = [], {}
    for p in PTS:
        own = df[df.pt == p]
        test = own[own.t >= max(4, N)]
        train = pd.concat([df[df.pt != p], own[own.t < N]])
        use = cols
        if mc:
            use, dropped[p] = mc_screen(df[df.pt != p], cols, [q for q in PTS if q != p])
        preds = {m: fit_predict(m, train, test, use) for m in models}
        if with_refs:
            preds.update(references(train, test))
        for m, pr in preds.items():
            for (_, r), v in zip(test.iterrows(), pr):
                rows.append({"N": N, "pt": p, "week": r.week, "t": r.t, "model": m, "y": r.y, "pred": v})
    return pd.DataFrame(rows), dropped


def score(pred, by=("model",)):
    out = []
    for key, g in pred.groupby(list(by)):
        y, yh = g.y.values, g.pred.values
        key = key if isinstance(key, tuple) else (key,)
        row = {**dict(zip(by, key)), "n": len(g), "R2": 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2),
               "r": sp_stats.pearsonr(yh, y)[0] if np.std(yh) > 0 else np.nan, "MAE": np.mean(np.abs(y - yh))}
        for p, gg in g.groupby("pt"):
            yy, pp = gg.y.values, gg.pred.values
            row[f"R2_{p}"] = 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
            row[f"r_{p}"] = sp_stats.pearsonr(pp, yy)[0] if np.std(pp) > 0 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def cells():
    out = []
    for mc in (False, True):
        for fam in FAMILIES:
            for al in (["none", "z", "riem"] if fam == "F-riem" else ["none", "z"]):
                for m in MODELS:
                    out.append((mc, fam, al, m))
    return out
