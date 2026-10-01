# %% [markdown]
# # Cross-patient amendment P3: FOOOF-based families (incl. the forward-chaining winner) in calibration-free LOPO
# Usage: python p3.py [primary] [null] [calib]    Env: DATA_DIRECTORY, JOBS (8), N_NULL (100). Run in the DR_SCC env.
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

BANDS = list(fc.BANDS)
BAND_HZ = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 14), "Beta*": (14, 20), "Gamma1": (35, 50)}  # as xp_features / fooof_features
FAMILIES = {"F-fooof-full": ["none", "z"], "F-fooof-aper": ["none", "z"], "F-fooof-rel": ["none", "z"],
            "F-fooof-asym": ["none", "z"], "F-fooof-riem": ["riem"]}
MODELS = ["ENR", "ENR+time", "SVR", "SVR+time", "Anchor", "ICP"]
X8 = (False, "F-fooof-full", "z", "ENR+time")
OUT = HERE / "outputs"
PRED = fc.data_dir() / "intermed" / "cross_patient"


def side(c):
    if c.startswith(("L_", "per_L", "band_L")) or c.endswith("_LL"):
        return "GCr_L"
    if c.startswith(("R_", "per_R", "band_R")) or c.endswith("_RR"):
        return "GCr_R"
    return "both"


xp._side = side  # the screen matches offset/exponent and flattened auto-terms to their side's GCr


def recordings():
    D = fc.data_dir() / "intermed"
    x = pd.read_csv(D / "cross_patient/recordings_xp.csv.gz", dtype={"pt": str, "week": str})
    foo = pd.read_csv(D / "forward_chain/recordings_fooof.csv.gz", dtype={"pt": str, "week": str})
    assert (x.file.values == foo.file.values).all()
    keep = [f"{s}_{k}" for s in "LR" for k in ["offset", "exponent"] + BANDS]
    r = pd.concat([x, foo[keep + ["qc_ok"]]], axis=1)
    r = r[(r.circ == "day") & r.qc_ok.astype(bool)].reset_index(drop=True)
    # FOOOF-flattened co-spectra: C_flat = D^-1/2 C D^-1/2, D = diag(mean aperiodic power over the band's bins)
    f = np.linspace(0, 211, 513)
    for b, (lo, hi) in BAND_HZ.items():
        fb = f[(f >= lo) & (f < hi)]
        ap = {s: np.array([np.mean(10 ** (o - e * np.log10(fb))) for o, e in zip(r[f"{s}_offset"], r[f"{s}_exponent"])])
              for s in "LR"}
        r[f"fl_{b}_LL"] = r[f"{b}_LL"] / ap["L"]
        r[f"fl_{b}_RR"] = r[f"{b}_RR"] / ap["R"]
        r[f"fl_{b}_LR"] = r[f"{b}_LR"] / np.sqrt(ap["L"] * ap["R"])
    return r


def zalign(F, cols):
    for p, idx in F.groupby("pt").groups.items():
        b = F.loc[idx][F.loc[idx, "t"] <= 3][cols]
        F.loc[idx, cols] = (F.loc[idx, cols] - b.mean()) / b.std(ddof=1).replace(0, np.nan)
    F[cols] = F[cols].fillna(0.0)
    return F


def weekly(rec, fam, align):
    g = ["pt", "week", "t"]
    base = rec.groupby(g)
    W = base[["GCr_L", "GCr_R"]].mean().reset_index()
    per = {s: [f"{s}_{b}" for b in BANDS] for s in "LR"}
    if fam == "F-fooof-full":
        cols = [f"{s}_{k}" for s in "LR" for k in ["offset", "exponent"] + BANDS]
        F = base[cols].mean().reset_index()
    elif fam == "F-fooof-aper":
        cols = [f"{s}_{k}" for s in "LR" for k in ["offset", "exponent"]]
        F = base[cols].mean().reset_index()
    elif fam in ("F-fooof-rel", "F-fooof-asym"):
        F = base[per["L"] + per["R"]].mean().reset_index()
        for s in "LR":
            F[per[s]] = F[per[s]].sub(F[per[s]].mean(axis=1), axis=0)
        cols = per["L"] + per["R"]
        if fam == "F-fooof-asym":
            for b in BANDS:
                F[f"asym_{b}"] = F[f"L_{b}"] - F[f"R_{b}"]
            cols = [f"asym_{b}" for b in BANDS]
            F = F[g + cols]
    elif fam == "F-fooof-riem":
        rows = []
        for key, grp in base:
            row = dict(zip(g, key))
            for b in BANDS:
                row[b] = np.array([[grp[f"fl_{b}_LL"].mean(), grp[f"fl_{b}_LR"].mean()],
                                   [grp[f"fl_{b}_LR"].mean(), grp[f"fl_{b}_RR"].mean()]])
            rows.append(row)
        M = pd.DataFrame(rows)
        for p, idx in M.groupby("pt").groups.items():
            sub = M.loc[idx]
            for b in BANDS:
                Cs = np.stack(sub[b].values)
                ref = xp.riem_mean(Cs[sub.t.values <= 3])
                Ri = xp.spd_isqrt(ref)
                Ls = [xp.spd_log(Ri @ C @ Ri) for C in Cs]
                disp = np.mean([np.linalg.norm(Ls[i]) for i in np.where(sub.t.values <= 3)[0]]) or 1.0
                for i, L in zip(idx, Ls):
                    for name, v in zip(("LL", "RR", "LR"), xp.tri(L / disp)):
                        M.loc[i, f"friem_{b}_{name}"] = v
        cols = [f"friem_{b}_{n}" for b in BANDS for n in ("LL", "RR", "LR")]
        F = M[g + cols]
    F = W.merge(F, on=g, how="inner")
    if align == "z":
        F = zalign(F, cols)
    return F, cols


def table(rec, targets, fam, align, mc):
    r = rec[xp.mc_recording_mask(rec)] if mc else rec
    F, cols = weekly(r, fam, align)
    sv = xp.stim_voltage()
    F["stimV"] = [sv[p][w] for p, w in zip(F.pt, F.week)]
    F["sqrt_t"] = np.sqrt(F.t)
    T = targets[["pt", "week", "t", "T-raw"]].rename(columns={"T-raw": "y"})
    return F.merge(T, on=["pt", "week", "t"]).sort_values(["pt", "t"]).reset_index(drop=True), cols


def lopo(df, cols, models, mc, N=0):
    rows, dropped = [], {}
    for p in fc.PTS:
        own = df[df.pt == p]
        test, train = own[own.t >= max(4, N)], pd.concat([df[df.pt != p], own[own.t < N]])
        use = cols
        if mc:
            use, dropped[p] = xp.mc_screen(df[df.pt != p], cols, [q for q in fc.PTS if q != p])
        for m in models:
            base, timed = m.split("+")[0], m.endswith("+time")
            pr = xp.fit_predict(base, train, test, use + (["t", "sqrt_t"] if timed else []))
            for (_, r), v in zip(test.iterrows(), pr):
                rows.append({"N": N, "pt": p, "week": r.week, "t": r.t, "model": m, "y": r.y, "pred": v})
    return pd.DataFrame(rows), dropped


def label(c):
    return f"{'MC' if c[0] else 'noMC'} | {c[1]} | {c[2]} | {c[3]}"


if __name__ == "__main__":
    JOBS, N_NULL = int(os.environ.get("JOBS", 8)), int(os.environ.get("N_NULL", 100))
    stages = sys.argv[1:] or ["primary", "null", "calib"]
    t0 = time.time()
    rec, targets = recordings(), fc.clinical_targets()
    CELLS = [(mc, fam, al, m) for mc in (False, True) for fam, als in FAMILIES.items() for al in als for m in MODELS]
    TABLES = {(mc, fam, al): table(rec, targets, fam, al, mc) for mc, fam, al, _ in CELLS}
    print(f"{len(CELLS)} cells, {len(TABLES)} tables, {len(rec)} QC-ok daytime recordings ({time.time()-t0:.0f}s)")

    def run(c, df=None, N=0):
        d, cols = TABLES[c[:3]]
        p, dropped = lopo(d if df is None else df, cols, [c[3]], c[0], N=N)
        p["mc"], p["family"], p["align"] = c[0], c[1], c[2]
        return p, dropped

    if "primary" in stages:
        res = Parallel(n_jobs=JOBS)(delayed(run)(c) for c in CELLS)
        P = pd.concat([r[0] for r in res])
        P.to_csv(PRED / "pred_p3_lopo.csv.gz", index=False)
        S = xp.score(P, by=("mc", "family", "align", "model"))
        S.to_csv(OUT / "p3_lopo_scores.csv", index=False)
        pd.DataFrame([{"cell": label(c), "held_out": p, "dropped": ";".join(v)} for c, r in zip(CELLS, res) if c[0]
                      for p, v in r[1].items()]).to_csv(OUT / "p3_mc_screen_dropped.csv", index=False)
        print(S.sort_values("R2", ascending=False)[["mc", "family", "align", "model", "R2", "r", "MAE"]].head(12).round(3).to_string(index=False))
        print(f"primary done ({time.time()-t0:.0f}s)")
    if "null" in stages:
        seeds = np.random.default_rng(2026).integers(0, 2**31, size=N_NULL)

        def nd(i, seed, c):
            p, _ = run(c, df=fc.circular_shift(TABLES[c[:3]][0], np.random.default_rng(seed)))
            s = xp.score(p)
            return {"draw": i, "cell": label(c), "R2": float(s.R2.iloc[0]), "r": float(s.r.iloc[0])}

        pd.DataFrame(Parallel(n_jobs=JOBS)(delayed(nd)(i, s, c) for i, s in enumerate(seeds) for c in CELLS)
                     ).to_csv(OUT / "p3_lopo_null.csv", index=False)
        print(f"null done ({time.time()-t0:.0f}s)")
    if "calib" in stages:
        S = pd.read_csv(OUT / "p3_lopo_scores.csv")
        best = S.sort_values("R2", ascending=False).iloc[0]
        cells = [X8, (bool(best["mc"]), best["family"], best["align"], best["model"])]
        C = pd.concat([xp.score(run(c, N=n)[0], by=("N", "model")).assign(cell=label(c)) for c in cells for n in (0, 4, 8, 12)])
        C.to_csv(OUT / "p3_calibration.csv", index=False)
        print(C[["cell", "N", "R2", "r", "MAE"]].round(3).to_string(index=False), f"\ncalib done ({time.time()-t0:.0f}s)")
