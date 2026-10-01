# /// script
# requires-python = ">=3.8,<3.9"
# dependencies = ["numpy==1.24.4", "scipy==1.10.1", "pandas==2.0.3", "scikit-learn==1.3.2", "pot==0.9.1", "joblib==1.3.2", "python-dotenv==1.0.1"]
# ///
"""Cross-patient phase 2 (amendment P2): P2-A metric pullback, P2-B stitched LDS, P2-D Gromov-Wasserstein transport.
Calibration-free leave-one-patient-out + 100-draw circular-shift null.
    uv run notebooks/cross_patient/p2.py [primary] [null]      (after p2_export.py)"""
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
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv(usecwd=True))
DATA = Path(os.environ["DATA_DIRECTORY"]) / "intermed" / "cross_patient" / "p2"
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PTS = ["901", "903", "905", "906", "907", "908"]
INPUTS = ["F-band", "F-fooof-per", "F-rel", "F-asym", "F-riem"]
METHODS = ["P2A_pullback", "P2B_stitchLDS", "P2D_GW"]
CLIN = ["c_HDRS", "c_BDI", "c_GAF"]
K = 2
KEPT = json.load(open(DATA / "mc_kept_features_per_fold.json"))
TABLES = {f: pd.read_csv(DATA / f"weekly_{f}.csv", dtype={"pt": str, "week": str}).sort_values(["pt", "t"]).reset_index(drop=True)
          for f in INPUTS}


# ------------------------------------------------------------------ P2-A: shared metric pullback
def _pairs(X, C, groups):
    dX, dC = [], []
    for g in np.unique(groups):
        i, j = np.triu_indices((groups == g).sum(), 1)
        Xg, Cg = X[groups == g], C[groups == g]
        dX.append(Xg[i] - Xg[j])
        dC.append(np.linalg.norm(Cg[i] - Cg[j], axis=1))
    return np.vstack(dX), np.concatenate(dC)


def pullback_fit(X, C, groups, seed=2026):
    dX, dC = _pairs(X, C, groups)
    d = X.shape[1]

    def f(w):
        W = w.reshape(K, d)
        P = dX @ W.T
        n = np.linalg.norm(P, axis=1) + 1e-12
        r = n - dC
        g = 2 * ((r / n)[:, None] * P).T @ dX
        return np.sum(r ** 2), g.ravel()

    rng = np.random.default_rng(seed)
    inits = [PCA(K).fit(X).components_.ravel()] + [rng.normal(0, 0.1, K * d) for _ in range(4)]
    best = min((minimize(f, w0, jac=True, method="L-BFGS-B") for w0 in inits), key=lambda r: r.fun)
    return best.x.reshape(K, d)


def p2a(train, test, cols):
    sc = StandardScaler().fit(train[cols])
    Xtr, Xte = sc.transform(train[cols]), sc.transform(test[cols])
    W = pullback_fit(Xtr, train[CLIN].values, train.pt.values)
    Ztr, Zte = Xtr @ W.T, Xte @ W.T
    cv = list(GroupKFold(n_splits=train.pt.nunique()).split(Ztr, groups=train.pt))
    m = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}, cv=cv, scoring="neg_mean_squared_error").fit(Ztr, train.y)
    return m.predict(Zte), None


# ------------------------------------------------------------------ P2-B: stitched LDS (EM, Kalman/RTS)
def smooth(O, F, Q, mu0, V0, C, off, Rd):
    T, k = len(O), F.shape[0]
    mp, Vp, mf, Vf = np.zeros((T, k)), np.zeros((T, k, k)), np.zeros((T, k)), np.zeros((T, k, k))
    R = np.diag(Rd)
    for t in range(T):
        mp[t], Vp[t] = (mu0, V0) if t == 0 else (F @ mf[t - 1], F @ Vf[t - 1] @ F.T + Q)
        S = C @ Vp[t] @ C.T + R
        Kg = Vp[t] @ C.T @ np.linalg.inv(S)
        mf[t] = mp[t] + Kg @ (O[t] - C @ mp[t] - off)
        Vf[t] = (np.eye(k) - Kg @ C) @ Vp[t]
    ms, Vs, Vl = mf.copy(), Vf.copy(), np.zeros((T - 1, k, k))
    for t in range(T - 2, -1, -1):
        J = Vf[t] @ F.T @ np.linalg.inv(Vp[t + 1])
        ms[t] = mf[t] + J @ (ms[t + 1] - mp[t + 1])
        Vs[t] = Vf[t] + J @ (Vs[t + 1] - Vp[t + 1]) @ J.T
        Vl[t] = Vs[t + 1] @ J.T  # cov(z_{t+1}, z_t)
    return ms, Vs, Vl, mf


def _emis_update(O, ms, Vs):
    """Regress observations on [z, 1] using smoothed moments -> (C, off, Rdiag)."""
    T = len(O)
    Ez1 = np.column_stack([ms, np.ones(T)])
    Szz = sum(np.block([[Vs[t] + np.outer(ms[t], ms[t]), ms[t][:, None]], [ms[t][None, :], np.ones((1, 1))]])
              for t in range(T))
    B = (O.T @ Ez1) @ np.linalg.inv(Szz)
    Rd = np.maximum(np.mean(O ** 2, 0) - np.diag(B @ (Ez1.T @ O)) / T, 1e-4)  # diag of E[(o - B[z;1])(o)^T]
    return B[:, :-1], B[:, -1], Rd


def lds_train(seqs_x, seqs_y, iters=200):
    """seqs_x: list of (T, d) per patient; seqs_y: list of (T,). Shared F, Q, mu0, V0, c, d, s2; per-patient A, b, R."""
    allx = np.vstack(seqs_x)
    pca = PCA(K).fit(allx)
    F, Q, mu0, V0 = 0.9 * np.eye(K), 0.1 * np.eye(K), np.zeros(K), np.eye(K)
    A = [pca.components_.T.copy() for _ in seqs_x]
    b = [x.mean(0) for x in seqs_x]
    Rd = [np.maximum(x.var(0) * 0.5, 1e-3) for x in seqs_x]
    zs = np.vstack([pca.transform(x) for x in seqs_x])
    lr = np.linalg.lstsq(np.column_stack([zs, np.ones(len(zs))]), np.concatenate(seqs_y), rcond=None)[0]
    c, dd, s2 = lr[:K], lr[K], np.var(np.concatenate(seqs_y)) * 0.5
    for _ in range(iters):
        stats = []
        for p, (x, y) in enumerate(zip(seqs_x, seqs_y)):
            O = np.column_stack([x, y])
            C = np.vstack([A[p], c[None, :]])
            off = np.concatenate([b[p], [dd]])
            ms, Vs, Vl, _ = smooth(O, F, Q, mu0, V0, C, off, np.concatenate([Rd[p], [s2]]))
            stats.append((x, y, ms, Vs, Vl))
            A[p], b[p], Rd[p] = _emis_update(x, ms, Vs)
        # shared readout c, d, s2 from all patients
        Ys = np.concatenate([s[1] for s in stats])[:, None]
        Ms = np.vstack([s[2] for s in stats])
        Vss = np.concatenate([s[3] for s in stats])
        cy, dy, sy = _emis_update(Ys, Ms, Vss)
        c, dd, s2 = cy[0], dy[0], sy[0]
        # dynamics
        S11 = sum(s[3][t] + np.outer(s[2][t], s[2][t]) for s in stats for t in range(len(s[2]) - 1))
        S21 = sum(s[4][t] + np.outer(s[2][t + 1], s[2][t]) for s in stats for t in range(len(s[2]) - 1))
        S22 = sum(s[3][t + 1] + np.outer(s[2][t + 1], s[2][t + 1]) for s in stats for t in range(len(s[2]) - 1))
        n = sum(len(s[2]) - 1 for s in stats)
        F = S21 @ np.linalg.inv(S11)
        Q = (S22 - F @ S21.T) / n
        Q = (Q + Q.T) / 2 + 1e-6 * np.eye(K)
        mu0 = np.mean([s[2][0] for s in stats], 0)
        V0 = np.mean([s[3][0] + np.outer(s[2][0] - mu0, s[2][0] - mu0) for s in stats], 0) + 1e-6 * np.eye(K)
    return dict(F=F, Q=Q, mu0=mu0, V0=V0, c=c, d=dd, A=np.mean(A, 0), b=np.mean(b, 0), R=np.mean(Rd, 0))


def lds_heldout(x, P, iters=200):
    A, b, Rd = P["A"].copy(), P["b"].copy(), P["R"].copy()
    for _ in range(iters):
        ms, Vs, _, mf = smooth(x, P["F"], P["Q"], P["mu0"], P["V0"], A, b, Rd)
        A, b, Rd = _emis_update(x, ms, Vs)
    ms, Vs, _, mf = smooth(x, P["F"], P["Q"], P["mu0"], P["V0"], A, b, Rd)
    return ms @ P["c"] + P["d"], mf @ P["c"] + P["d"]


def p2b(train, test_full, cols, test_mask):
    sc = StandardScaler().fit(train[cols])
    sx = [sc.transform(train[train.pt == p][cols]) for p in train.pt.unique()]
    sy = [train[train.pt == p].y.values for p in train.pt.unique()]
    P = lds_train(sx, sy)
    smooth_pred, filt_pred = lds_heldout(sc.transform(test_full[cols]), P)
    return smooth_pred[test_mask], filt_pred[test_mask]


# ------------------------------------------------------------------ P2-D: Gromov-Wasserstein label transport
def p2d(train, test_full, cols, test_mask):
    import ot
    sc = StandardScaler().fit(train[cols])
    Xp = sc.transform(test_full[cols])
    Dp = ot.dist(Xp, Xp, metric="euclidean")
    Dp /= Dp.max()
    preds = []
    for q in train.pt.unique():
        tq = train[train.pt == q]
        Xq = sc.transform(tq[cols])
        Dq = ot.dist(Xq, Xq, metric="euclidean")
        Dq /= Dq.max()
        T = ot.gromov.gromov_wasserstein(Dp, Dq, ot.unif(len(Xp)), ot.unif(len(Xq)), loss_fun="square_loss")
        preds.append((T @ tq.y.values) / T.sum(1))
    return np.mean(preds, 0)[test_mask], None


# ------------------------------------------------------------------ LOPO
def lopo_cell(fam, method, df):
    rows = []
    for p in PTS:
        cols = KEPT[fam][p]
        own = df[df.pt == p].sort_values("t")
        train = df[df.pt != p]
        mask = (own.t >= 4).values
        if method == "P2A_pullback":
            pred, extra = p2a(train, own[mask], cols)
        elif method == "P2B_stitchLDS":
            pred, extra = p2b(train, own, cols, mask)
        else:
            pred, extra = p2d(train, own, cols, mask)
        test = own[mask]
        for i, (_, r) in enumerate(test.iterrows()):
            rows.append({"pt": p, "week": r.week, "t": r.t, "y": r.y, "pred": pred[i],
                         "pred_causal": extra[i] if extra is not None else np.nan})
    out = pd.DataFrame(rows)
    out["family"], out["method"] = fam, method
    return out


def score(g, col="pred"):
    y, yh = g.y.values, g[col].values
    row = {"n": len(g), "R2": 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2),
           "r": sp_stats.pearsonr(yh, y)[0] if np.std(yh) > 0 else np.nan, "MAE": np.mean(np.abs(y - yh))}
    for p, gg in g.groupby("pt"):
        yy, pp = gg.y.values, gg[col].values
        row[f"R2_{p}"] = 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
        row[f"r_{p}"] = sp_stats.pearsonr(pp, yy)[0] if np.std(pp) > 0 else np.nan
    return row


def shift(df, seed):
    """Same per-patient offsets as fc.circular_shift (rng.integers(1, 28) per patient, PTS order), applied to y + clinical."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for p in PTS:
        idx = out[out.pt == p].sort_values("t").index
        s = rng.integers(1, 28)
        for c in ["y"] + CLIN:
            out.loc[idx, c] = np.roll(out.loc[idx, c].values, s)
    return out


if __name__ == "__main__":
    stages = sys.argv[1:] or ["primary", "null"]
    JOBS = int(os.environ.get("JOBS", 8))
    cells = [(f, m) for f in INPUTS for m in METHODS]
    t0 = time.time()
    if "primary" in stages:
        P = pd.concat(Parallel(n_jobs=JOBS)(delayed(lopo_cell)(f, m, TABLES[f]) for f, m in cells))
        P.to_csv(DATA / "pred_p2_lopo.csv.gz", index=False)
        S = pd.DataFrame([{"family": f, "method": m, **score(g)} for (f, m), g in P.groupby(["family", "method"])])
        Sc = pd.DataFrame([{"family": f, "method": m, **score(g, "pred_causal")} for (f, m), g in
                           P[P.method == "P2B_stitchLDS"].groupby(["family", "method"])])
        S.to_csv(OUT / "p2_lopo_scores.csv", index=False)
        Sc.to_csv(OUT / "p2_lds_causal_scores.csv", index=False)
        print(S[["family", "method", "R2", "r", "MAE"]].sort_values("R2", ascending=False).round(3).to_string(index=False),
              f"\n({time.time()-t0:.0f}s)")
    if "null" in stages:
        seeds = np.random.default_rng(2026).integers(0, 2**31, size=100)

        def draw(i, seed, f, m):
            s = score(lopo_cell(f, m, shift(TABLES[f], seed)))
            return {"draw": i, "family": f, "method": m, "R2": s["R2"], "r": s["r"]}

        N = pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(draw)(i, s, f, m) for i, s in enumerate(seeds) for f, m in cells))
        N.to_csv(OUT / "p2_lopo_null.csv", index=False)
        print(f"null done ({time.time()-t0:.0f}s)")
