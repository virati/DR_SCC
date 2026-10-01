# %% [markdown]
# # Weekly in-clinic sessions: (a) replication, (b) checks, (c) extension  -- PROTOCOL.md
# Usage: python wd_analysis.py [a] [b] [b4] [c]     Run in the DR_SCC env after wd_extract.py and wd_drscc.py.
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE), str(HERE.parent / "forward_chain"), str(HERE.parent / "cross_patient"),
                                            os.environ.get("PYTHONPATH", "")])
sys.path.insert(0, str(HERE.parent / "forward_chain"))
sys.path.insert(0, str(HERE.parent / "cross_patient"))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

import fc

PTS = ["905", "906", "907", "908"]
PB = ["delta", "theta", "alpha", "lbeta", "hbeta", "gamma"]
PAPER = [f"pw_{s}_{b}" for s in "LR" for b in PB] + [f"coh_{b}" for b in PB] + ["pac_L", "pac_R"]
NUIS = ["nz_L_100_120", "nz_R_100_120", "nz_L_58_62", "nz_R_58_62"]
DR = ["dr_" + c for c in fc.FEATS]
FO = [f"fo_{s}_{k}" for s in "LR" for k in ["offset", "exponent"] + fc.BANDS]
BLOCKS = {"B": ["B01", "B02", "B03", "B04"], "C01-04": ["C01", "C02", "C03", "C04"], "C05-08": ["C05", "C06", "C07", "C08"],
          "C13-16": ["C13", "C14", "C15", "C16"], "C21-24": ["C21", "C22", "C23", "C24"]}
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
D = fc.data_dir() / "intermed" / "weekly_dense"
stages = sys.argv[1:] or ["a", "b", "b4", "c"]
t0 = time.time()

seg = pd.concat([pd.read_csv(D / "segments.csv.gz", dtype={"pt": str, "week": str}), pd.read_csv(D / "segments_drscc.csv.gz")], axis=1)
targets = fc.clinical_targets()
seg = seg.merge(targets[["pt", "week", "T-raw"]].rename(columns={"T-raw": "y"}), on=["pt", "week"], how="left")
off_all = seg[seg.state == "off"]
n_off = off_all.groupby(["pt", "week", "file"]).size().rename("n_off").reset_index()
seg = seg.merge(n_off, on=["pt", "week", "file"], how="left")
PRIM = seg[(seg.n_files_week == 1) & (seg.n_off >= 6)]           # primary: single-file weeks, >= 6 clean off segments
OFF = PRIM[PRIM.state == "off"].reset_index(drop=True)
print(f"primary off segments {len(OFF)}, sessions {OFF.groupby(['pt','week']).ngroups} ({time.time()-t0:.0f}s)")


def make(kind):
    if kind == "MLP":
        return MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3, max_iter=2000, random_state=2026)
    return LogisticRegression(C=1.0, max_iter=5000)


def contrast(data, neg, pos, feats, kind, labels=None):
    """Leave-one-patient-out AUROC for block `pos` (1) vs block `neg` (0). Returns one row per held-out patient."""
    d = data[data.week.isin(BLOCKS[neg] + BLOCKS[pos])].copy()
    d["lab"] = d.week.isin(BLOCKS[pos]).astype(int) if labels is None else d.set_index(["pt", "week"]).index.map(labels).values
    rows = []
    for p in PTS:
        tr, te = d[d.pt != p], d[d.pt == p]
        if te.lab.nunique() < 2 or tr.lab.nunique() < 2:
            rows.append({"held_out": p, "auc_segment": np.nan, "auc_session": np.nan})
            continue
        sc = MinMaxScaler().fit(tr[feats])
        m = make(kind).fit(sc.transform(tr[feats]), tr.lab)
        pr = m.predict_proba(sc.transform(te[feats]))[:, 1]
        ses = te.assign(pr=pr).groupby("week").agg(lab=("lab", "first"), pr=("pr", "mean"))
        rows.append({"held_out": p, "auc_segment": roc_auc_score(te.lab, pr),
                     "auc_session": roc_auc_score(ses.lab, ses.pr) if ses.lab.nunique() == 2 else np.nan})
    return pd.DataFrame(rows)


def hdrs_gap(neg, pos):
    w = targets[targets.pt.isin(PTS)]
    return float(w[w.week.isin(BLOCKS[pos])]["T-raw"].mean() - w[w.week.isin(BLOCKS[neg])]["T-raw"].mean())


# ------------------------------------------------------------------ (a) + (b) B1-B3
if "a" in stages or "b" in stages:
    res = []
    for name, neg, pos, feats in [("sick vs stable (paper)", "C01-04", "C21-24", PAPER),
                                  ("B01-04 vs C01-04 (both sick)", "B", "C01-04", PAPER),
                                  ("C13-16 vs C21-24 (both late)", "C13-16", "C21-24", PAPER),
                                  ("C01-04 vs C05-08 (both early)", "C01-04", "C05-08", PAPER),
                                  ("sick vs stable, nuisance features only", "C01-04", "C21-24", NUIS)]:
        for kind in ("MLP", "LR"):
            r = contrast(OFF, neg, pos, feats, kind)
            r["contrast"], r["model"], r["nHDRS_gap"] = name, kind, hdrs_gap(neg, pos)
            res.append(r)
    A = pd.concat(res)
    A.to_csv(OUT / "a_b_contrasts.csv", index=False)
    print(A.groupby(["contrast", "model"], sort=False).agg(seg=("auc_segment", "mean"), ses=("auc_session", "mean"),
                                                             gap=("nHDRS_gap", "first")).round(3).to_string())
    # B1: single features, direction learned on training patients
    d = OFF[OFF.week.isin(BLOCKS["C01-04"] + BLOCKS["C21-24"])].assign(lab=lambda x: x.week.isin(BLOCKS["C21-24"]).astype(int))
    sf = []
    for ft in PAPER:
        aucs = []
        for p in PTS:
            tr, te = d[d.pt != p], d[d.pt == p]
            sgn = np.sign(tr[tr.lab == 1][ft].mean() - tr[tr.lab == 0][ft].mean()) or 1.0
            aucs.append(roc_auc_score(te.lab, sgn * te[ft]))
        sf.append({"feature": ft, "auc_segment_mean": np.mean(aucs), **{f"auc_{p}": a for p, a in zip(PTS, aucs)}})
    pd.DataFrame(sf).sort_values("auc_segment_mean", ascending=False).to_csv(OUT / "b1_single_features.csv", index=False)
    # B3: session-level permutation null (labels permuted within patient), logistic regression
    obs = contrast(OFF, "C01-04", "C21-24", PAPER, "LR")
    base = d.groupby(["pt", "week"]).lab.first()
    rng = np.random.default_rng(2026)

    def perm(seed):
        r = np.random.default_rng(seed)
        lab = base.copy()
        for p in PTS:
            lab.loc[p] = r.permutation(lab.loc[p].values)
        c = contrast(OFF, "C01-04", "C21-24", PAPER, "LR", labels=lab.to_dict())
        return c.auc_segment.mean(), c.auc_session.mean()

    N = np.array(Parallel(n_jobs=8)(delayed(perm)(s) for s in rng.integers(0, 2**31, 200)))
    pd.DataFrame({"null_auc_segment": N[:, 0], "null_auc_session": N[:, 1]}).to_csv(OUT / "b3_permutation_null.csv", index=False)
    pd.DataFrame([{"obs_auc_segment": obs.auc_segment.mean(), "obs_auc_session": obs.auc_session.mean(),
                   "p_segment": (np.sum(N[:, 0] >= obs.auc_segment.mean()) + 1) / 201,
                   "p_session": (np.nansum(N[:, 1] >= obs.auc_session.mean()) + 1) / 201}]).to_csv(OUT / "b3_permutation_test.csv", index=False)
    print(f"a/b done ({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------ session-level tables
W_OFF = OFF.groupby(["pt", "week", "t"])[PAPER + DR + FO + ["y"]].median().reset_index()


def zbase(tab, cols, segs):
    """z-score against the patient's B-week stimulation-off SEGMENTS (mean, sd); B-week sessions are few."""
    out = tab.copy()
    for p in PTS:
        b = segs[(segs.pt == p) & (segs.t <= 3)][cols]
        if len(b) > 5:
            out.loc[out.pt == p, cols] = (tab.loc[tab.pt == p, cols] - b.mean()) / b.std(ddof=1).replace(0, np.nan)
    return out.fillna(0.0)


def fchain(tab, cols, models=fc.MODELS):
    df = tab[["pt", "week", "t", "y"] + cols].dropna().sort_values(["pt", "t"]).reset_index(drop=True)
    old = fc.PTS
    fc.PTS = [p for p in old if p in set(df.pt)]
    try:
        return fc.e1(df, cols, models), df
    finally:
        fc.PTS = old


def shift(df, seed):
    old = fc.PTS
    fc.PTS = [p for p in old if p in set(df.pt)]
    try:
        return fc.circular_shift(df, np.random.default_rng(seed))
    finally:
        fc.PTS = old


# ------------------------------------------------------------------ (b) B4: continuous tracking, forward chaining
if "b4" in stages:
    cells = {("raw",): W_OFF, ("baseline",): zbase(W_OFF, PAPER, OFF)}
    S, tabs = [], {}
    for (norm,), tab in cells.items():
        p, df = fchain(tab, PAPER)
        tabs[norm] = df
        s = fc.score(p)
        s["norm"] = norm
        S.append(s)
    S = pd.concat(S)
    S.to_csv(OUT / "b4_forward_chain_scores.csv", index=False)
    seeds = np.random.default_rng(2026).integers(0, 2**31, size=100)
    neural = [m for m in fc.MODELS if m not in ("M0_persistence", "M1_time")]

    def nd(i, seed, norm):
        d = shift(tabs[norm], seed)
        old = fc.PTS
        fc.PTS = [p for p in old if p in set(d.pt)]
        try:
            s = fc.score(fc.e1(d, PAPER, ["M0_persistence"] + neural)).set_index("model")
        finally:
            fc.PTS = old
        return [{"draw": i, "norm": norm, "model": m, "R2": s.loc[m, "R2"], "gain": s.loc[m, "R2"] - s.loc["M0_persistence", "R2"]} for m in neural]

    N = pd.DataFrame([r for part in Parallel(n_jobs=8)(delayed(nd)(i, s, n) for i, s in enumerate(seeds) for n in tabs) for r in part])
    N.to_csv(OUT / "b4_null.csv", index=False)
    print(S[["norm", "model", "n", "R2", "r", "MAE"]].round(3).to_string(index=False), f"\nb4 done ({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------ (c) extension
if "c" in stages:
    I = fc.data_dir() / "intermed"
    band = pd.read_csv(I / "forward_chain/recordings_features.csv.gz", dtype={"pt": str, "week": str})
    foo = pd.read_csv(I / "forward_chain/recordings_fooof.csv.gz", dtype={"pt": str, "week": str})
    xpt = pd.read_csv(I / "cross_patient/recordings_xp.csv.gz", dtype={"pt": str, "week": str})
    g = ["pt", "week", "t"]
    CH = band[band.circ == "day"].groupby(g)[fc.FEATS].mean().add_prefix("dr_").reset_index()                  # at-home, stim-on in C weeks
    fo_cols = [f"{s}_{k}" for s in "LR" for k in ["offset", "exponent"] + fc.BANDS]
    CHF = foo[(foo.circ == "day") & foo.qc_ok.astype(bool)].groupby(g)[fo_cols].mean().reset_index()
    CHF.columns = g + FO
    GCR = xpt[xpt.circ == "day"].groupby(g)[["GCr_L", "GCr_R"]].mean().reset_index()
    # C1: clean-reference agreement (C weeks only: at-home recordings there are stimulation-on)
    rows = []
    for cols, chron in ((DR, CH), (FO, CHF)):
        m = W_OFF.merge(chron, on=g, suffixes=("_off", "_home")).merge(GCR, on=g)
        m = m[m.t >= 4]
        for c in cols:
            for p in PTS:
                d = m[m.pt == p]
                rho, pv = sp_stats.spearmanr(d[c + "_off"], d[c + "_home"])
                side = "GCr_L" if ("_L" in c or c.startswith("dr_L")) else "GCr_R"
                rg, pg = sp_stats.spearmanr(d[c + "_home"], d[side])
                rows.append({"feature": c, "pt": p, "n_weeks": len(d), "rho_off_vs_home": rho, "p": pv, "rho_home_vs_GCr": rg, "p_GCr": pg})
    C1 = pd.DataFrame(rows)
    for col, q in (("p", "q"), ("p_GCr", "q_GCr")):
        pv = C1[col].fillna(1).values
        o = np.argsort(pv)
        qq = np.empty(len(pv))
        qq[o] = np.minimum.accumulate((pv[o] * len(pv) / (np.arange(len(pv)) + 1))[::-1])[::-1]
        C1[q] = qq
    C1.to_csv(OUT / "c1_agreement_by_patient.csv", index=False)
    summ = C1.groupby("feature", sort=False).apply(lambda d: pd.Series({
        "rho_mean": d.rho_off_vs_home.mean(), "rho_min": d.rho_off_vs_home.min(), "rho_max": d.rho_off_vs_home.max(),
        "n_validated_pts": int(((d.rho_off_vs_home > 0) & (d.q < 0.05)).sum()),
        "validated": bool(((d.rho_off_vs_home > 0) & (d.q < 0.05)).sum() >= 3),
        "rho_GCr_mean": d.rho_home_vs_GCr.mean(), "n_GCr_sig_pts": int((d.q_GCr < 0.05).sum())})).reset_index()
    summ.to_csv(OUT / "c1_agreement_summary.csv", index=False)
    # C2: stimulation-on minus stimulation-off within the same session
    ON = PRIM[PRIM.state == "on"].groupby(g)[DR + FO].median().reset_index()
    both = W_OFF.merge(ON, on=g, suffixes=("_off", "_on"))
    sv = {p: {w: v for w, v in d.items()} for p, d in __import__("xp").stim_voltage().items()}
    both["stimV"] = [sv[p][w] for p, w in zip(both.pt, both.week)]
    c2 = []
    for c in DR + FO:
        for p in PTS:
            d = both[(both.pt == p) & (both.t >= 4)]
            if len(d) < 4:
                continue
            diff = d[c + "_on"] - d[c + "_off"]
            c2.append({"feature": c, "pt": p, "n_sessions": len(d), "off_mean": d[c + "_off"].mean(), "on_mean": d[c + "_on"].mean(),
                       "on_minus_off": diff.mean(), "off_week_sd": d[c + "_off"].std(ddof=1),
                       "effect_in_week_sd": diff.mean() / d[c + "_off"].std(ddof=1),
                       "rho_diff_vs_stimV": sp_stats.spearmanr(diff, d.stimV)[0] if d.stimV.nunique() > 1 else np.nan})
    pd.DataFrame(c2).to_csv(OUT / "c2_stim_on_minus_off.csv", index=False)
    # C3: same four patients and weeks, forward chaining: off-session vs at-home DR-SCC features
    M = W_OFF[g + ["y"] + DR].merge(CH, on=g, suffixes=("_off", "_home"))
    offc, homec = [c + "_off" for c in DR], [c + "_home" for c in DR]
    c3 = []
    for name, cols in (("stim-off session", offc), ("at-home chronic", homec), ("both", offc + homec)):
        p, _ = fchain(M, cols)
        s = fc.score(p)
        s["features"] = name
        c3.append(s)
    # chronic-trained ENR applied unchanged to stimulation-off features of the test week
    rows = []
    for p in PTS:
        own = M[M.pt == p].sort_values("t")
        for _, te in own[own.t >= 4].iterrows():
            tr = pd.concat([M[M.pt != p], own[own.t < te.t]])
            trd = tr[g + ["y"]].assign(**{c: tr[c + "_home"] for c in DR})
            ted = pd.DataFrame([{**{k: te[k] for k in g + ["y"]}, **{c: te[c + "_off"] for c in DR}}])
            pr = fc.fit_predict("M2_ENR", trd, ted, DR)[0]
            rows.append({"pt": p, "week": te.week, "t": te.t, "model": "M2_ENR", "y": te.y, "pred": pr})
    s = fc.score(pd.DataFrame(rows))
    s["features"] = "chronic-trained ENR -> stim-off features"
    c3.append(s)
    pd.concat(c3).to_csv(OUT / "c3_transfer_scores.csv", index=False)
    # C4: co-coherence in the chronic recordings, all six patients, forward chaining
    x = xpt[xpt.circ == "day"].copy()
    for b in fc.BANDS:
        x[f"cc_{b}"] = x[f"{b}_LR"] ** 2 / (x[f"{b}_LL"] * x[f"{b}_RR"])
    CC = x.groupby(g)[[f"cc_{b}" for b in fc.BANDS]].mean().reset_index()
    T6 = band[band.circ == "day"].groupby(g)[fc.FEATS].mean().reset_index().merge(CC, on=g).merge(
        targets[["pt", "week", "T-raw"]].rename(columns={"T-raw": "y"}), on=["pt", "week"])
    c4 = []
    for name, cols in (("band power (10)", fc.FEATS), ("band power + co-coherence (15)", fc.FEATS + [f"cc_{b}" for b in fc.BANDS]),
                       ("co-coherence only (5)", [f"cc_{b}" for b in fc.BANDS])):
        p, _ = fchain(T6, cols)
        s = fc.score(p)
        s["features"] = name
        c4.append(s)
    pd.concat(c4).to_csv(OUT / "c4_coherence_scores.csv", index=False)
    print(summ.round(2).to_string(index=False), f"\nc done ({time.time()-t0:.0f}s)")
