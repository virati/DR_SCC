"""Forward-chaining DR-SCC analysis (see PROTOCOL.md). Feature building, models, evaluations, null."""
import hashlib
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

PTS = ["901", "903", "905", "906", "907", "908"]
WEEKS = ["B0" + str(i) for i in range(1, 5)] + ["C%02d" % i for i in range(1, 25)]
T_OF = {w: i for i, w in enumerate(WEEKS)}
C_WEEKS = WEEKS[4:]
BANDS = ["Delta", "Theta", "Alpha", "Beta*", "Gamma1"]
FEATS = ["L" + b for b in BANDS] + ["R" + b for b in BANDS]
FRAME_MD5 = "f40adf2f4c6c2988ccf29023a4959477"
CLIN_MD5 = "a29431b316cf8d3630c0a0894c32593c"
SVR_GRID = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.05]}
MODELS = ["M0_persistence", "M1_time", "M2_ENR", "M3_ENR+time", "M4_mixed", "M5_SVR", "M6_SVR+time"]


def data_dir():
    from dbspace.utils.dissertation import notebook_setup
    return Path(notebook_setup.DATADIR or os.environ["DATA_DIRECTORY"])


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- per-recording features
def load_fooof():
    """FOOOF per-recording table from fooof_features.py (amendment 1)."""
    return pd.read_csv(data_dir() / "intermed/forward_chain/recordings_fooof.csv.gz",
                       dtype={"pt": str, "week": str, "circ": str})


def build_recordings(cache):
    """One row per recording: patient, week, t, circadian, GC flag, 10 paper features."""
    if Path(cache).exists():
        return pd.read_csv(cache, dtype={"pt": str, "week": str})
    from dbspace.readout import ClinVect, decoder
    D = data_dir()
    frame, clin = D / "intermed/chronic/Chronic_FrameFeb2026_F.pickle", D / "clinical/clinical_vectors_all.json"
    assert md5(frame) == FRAME_MD5 and md5(clin) == CLIN_MD5, "input data md5 mismatch"
    CF = ClinVect.CStruct(clin)
    BR = pickle.load(open(frame, "rb"))
    dec = decoder.weekly_decoder(BRFrame=BR, ClinFrame=CF, pts=PTS, clin_measure="pHDRS17", algo="ENR",
                                 shuffle_null=False, FeatureSet="main", variance=False)
    recs = [r for r in BR.file_meta if r["Patient"] in PTS and r["Phase"] in T_OF]
    X, _ = dec.calculate_states_in_set(recs)
    df = pd.DataFrame(X, columns=dec.feat_labels)
    df.columns = FEATS
    df.insert(0, "pt", [r["Patient"] for r in recs])
    df.insert(1, "week", [r["Phase"] for r in recs])
    df.insert(2, "t", [T_OF[r["Phase"]] for r in recs])
    df.insert(3, "circ", [r["Circadian"] for r in recs])
    df.insert(4, "gc", [bool(r["GC_Flag"]["Flag"]) for r in recs])
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def clinical_targets():
    """pt x week table: nHDRS (pHDRS17), causal 3-week trailing mean, DSC."""
    from dbspace.readout import ClinVect
    CF = ClinVect.CStruct(data_dir() / "clinical/clinical_vectors_all.json")
    CF.gen_DSC()
    rows = []
    for pt in PTS:
        raw = [CF.get_depression_measure("DBS" + pt, "pHDRS17", w) for w in WEEKS]
        smooth = [np.mean(raw[max(0, i - 2): i + 1]) for i in range(len(raw))]
        dsc = [CF.get_depression_measure("DBS" + pt, "DSC", w) for w in WEEKS]
        for i, w in enumerate(WEEKS):
            rows.append({"pt": pt, "week": w, "t": i, "T-raw": raw[i], "T-smooth": smooth[i], "T-DSC": dsc[i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- weekly feature sets
def weekly(rec, fset):
    g = ["pt", "week", "t"]
    day = rec[rec.circ == "day"]
    if fset == "F-fooof":  # amendment 1: rec is the FOOOF table (recordings_fooof.csv.gz)
        cols = [c for c in rec.columns if c[:2] in ("L_", "R_") and not c.endswith("_r2")]
        W = day[day.qc_ok.astype(bool)].groupby(g)[cols].mean()
        return W.reset_index()
    if fset == "F-mean":
        W = day.groupby(g)[FEATS].mean()
    elif fset == "F-noGC":
        W = day[~day.gc].groupby(g)[FEATS].mean()
    elif fset == "F-dist":
        grp = day.groupby(g)[FEATS]
        q = grp.quantile(0.75) - grp.quantile(0.25)
        W = pd.concat([grp.mean().add_prefix("mean_"), grp.median().add_prefix("med_"),
                       q.add_prefix("iqr_"), grp.var().add_prefix("var_")], axis=1)
    elif fset == "F-circ":
        dm = day.groupby(g)[FEATS].mean()
        nm = rec[rec.circ == "night"].groupby(g)[FEATS].mean().reindex(dm.index)
        diff = (dm - nm).fillna(0.0)  # weeks without night recordings: no contrast
        W = pd.concat([dm.add_prefix("day_"), diff.add_prefix("dmn_")], axis=1)
    else:
        raise ValueError(fset)
    return W.reset_index()


def normalize(W, how):
    if how == "raw":
        return W
    cols = [c for c in W.columns if c not in ("pt", "week", "t")]
    out = W.copy()
    for pt, idx in W.groupby("pt").groups.items():
        base = W.loc[idx][W.loc[idx, "t"] <= 3][cols]  # B01-B04
        mu, sd = base.mean(), base.std(ddof=1).replace(0, np.nan)
        if how == "center":  # POST-HOC variant (not in PROTOCOL.md): subtract B-week mean, no scaling
            out.loc[idx, cols] = W.loc[idx, cols] - mu
        else:
            out.loc[idx, cols] = (W.loc[idx, cols] - mu) / sd
    return out.fillna(0.0)


def table(rec, targets, fset, norm, target):
    W = normalize(weekly(rec, fset), norm)
    T = targets[["pt", "week", "t", target]].rename(columns={target: "y"})
    df = W.merge(T, on=["pt", "week", "t"], how="inner").sort_values(["pt", "t"]).reset_index(drop=True)
    feats = [c for c in W.columns if c not in ("pt", "week", "t")]
    return df, feats


# ---------------------------------------------------------------- models
def _time(df):
    return np.column_stack([df.t.values, np.sqrt(df.t.values)])


def _inner(train, grouped):
    if grouped and train.pt.nunique() > 1:
        return list(GroupKFold(n_splits=min(train.pt.nunique(), 5)).split(train, groups=train.pt))
    return list(TimeSeriesSplit(n_splits=3).split(train))


def fit_predict(model, train, test, feats, grouped=True):
    """Return predictions for test rows. train/test include pt, t, y and feature columns."""
    if model == "M0_persistence":
        return None  # handled by caller (needs the patient's own history)
    if model == "M1_time":
        Xtr, Xte = _time(train), _time(test)
    elif model in ("M2_ENR", "M5_SVR"):
        Xtr, Xte = train[feats].values, test[feats].values
    else:  # +time and mixed
        Xtr = np.column_stack([train[feats].values, _time(train)])
        Xte = np.column_stack([test[feats].values, _time(test)])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    y = train.y.values
    cv = _inner(train, grouped)
    if model in ("M1_time", "M2_ENR", "M3_ENR+time"):
        m = ElasticNetCV(l1_ratio=0.8, cv=cv, n_alphas=100, max_iter=20000).fit(Xtr, y)
        return m.predict(Xte)
    if model in ("M5_SVR", "M6_SVR+time"):
        m = GridSearchCV(SVR(kernel="rbf"), SVR_GRID, cv=cv, scoring="neg_mean_squared_error", n_jobs=1).fit(Xtr, y)
        return m.best_estimator_.predict(Xte)
    if model == "M4_mixed":
        import statsmodels.api as sm
        # neural features + t fixed (drop sqrt t column to keep the protocol's "features + t")
        Xtr_m, Xte_m = sm.add_constant(Xtr[:, :-1], has_constant="add"), sm.add_constant(Xte[:, :-1], has_constant="add")
        r = None
        for method in ("bfgs", "powell", "nm"):  # lbfgs can "converge" to a degenerate fit with llf = inf
            try:
                cand = sm.MixedLM(y, Xtr_m, groups=train.pt.values).fit(reml=True, method=method, disp=False)
            except Exception:
                continue
            if np.isfinite(cand.llf):
                r = cand
                break
        if r is None:
            return np.full(len(test), np.nan)
        fe = Xte_m @ np.asarray(r.fe_params)
        try:
            reff = r.random_effects
        except ValueError:  # random-intercept variance estimated as 0: model is fixed-effects only
            reff = {}
        re = np.array([float(np.asarray(reff.get(p, [0.0]))[0]) for p in test.pt.values])
        return fe + re
    raise ValueError(model)


# ---------------------------------------------------------------- evaluations
def e1(df, feats, models, e2=False):
    """Forward chaining. E1: others' weeks + target's past weeks. E2: target's past weeks only."""
    rows = []
    for p in PTS:
        own = df[df.pt == p]
        others = df[df.pt != p]
        for w in C_WEEKS:
            k = T_OF[w]
            test = own[own.t == k]
            if test.empty:
                continue
            past = own[own.t < k]
            if e2 and len(past) < 6:
                continue
            train = past if e2 else pd.concat([others, past])
            for m in models:
                if m == "M0_persistence":
                    pred = past.y.values[-1] if len(past) else np.nan
                elif e2 and m in ("M4_mixed", "M5_SVR", "M6_SVR+time"):
                    continue
                else:
                    pred = fit_predict(m, train, test, feats, grouped=not e2)[0]
                rows.append({"pt": p, "week": w, "t": k, "model": m, "y": test.y.values[0], "pred": pred})
    return pd.DataFrame(rows)


def e3(df, feats, models, Ns=(0, 2, 4, 8, 12, 16)):
    rows = []
    for N in Ns:
        for p in PTS:
            own = df[df.pt == p]
            calib = own[own.t < N]
            test = own[(own.t >= max(N, 4))]
            train = pd.concat([df[df.pt != p], calib])
            for m in models:
                if m == "M0_persistence":
                    if calib.empty:
                        continue
                    pred = np.full(len(test), calib.y.values[-1])
                else:
                    pred = fit_predict(m, train, test, feats, grouped=True)
                for (_, r), pr in zip(test.iterrows(), pred):
                    rows.append({"N": N, "pt": p, "week": r.week, "t": r.t, "model": m, "y": r.y, "pred": pr})
    return pd.DataFrame(rows)


def score(pred, by=("model",)):
    out = []
    for key, g in pred.dropna(subset=["pred"]).groupby(list(by)):
        y, yh = g.y.values, g.pred.values
        per_pt = [sp_stats.pearsonr(gg.pred, gg.y)[0] for _, gg in g.groupby("pt") if len(gg) > 2 and gg.pred.std() > 0]
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(by, key)), "n": len(g),
                    "R2": 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2),
                    "r": sp_stats.pearsonr(yh, y)[0] if np.std(yh) > 0 else np.nan,
                    "MAE": np.mean(np.abs(y - yh)),
                    "mean_pt_r": np.nanmean(per_pt) if per_pt else np.nan})
    return pd.DataFrame(out)


def circular_shift(df, rng):
    """Shift each patient's 28-week target series by an independent offset in 1..27."""
    out = df.copy()
    for p in PTS:
        idx = out.index[out.pt == p]
        s = rng.integers(1, 28)
        order = out.loc[idx].sort_values("t").index
        out.loc[order, "y"] = np.roll(out.loc[order, "y"].values, s)
    return out
