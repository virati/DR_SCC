# /// script
# requires-python = ">=3.8,<3.9"
# dependencies = ["numpy==1.24.4", "scipy==1.10.1", "pandas==2.0.3", "scikit-learn==1.3.2", "joblib==1.3.2", "python-dotenv==1.0.1"]
# ///
"""Left-right phase offset: PH1 description, PH2 contrasts, PH3 relation to depression (PROTOCOL.md).
    uv run notebooks/phase_offset/phase_analysis.py      (after phase_extract.py)"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv(usecwd=True))
DATA = Path(os.environ["DATA_DIRECTORY"])
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(2026)
WEEKS = ["B0" + str(i) for i in range(1, 5)] + ["C%02d" % i for i in range(1, 25)]
UNC, COR = ["delta", "theta", "alpha", "beta", "gamma"], ["delta", "theta", "alpha", "bstar", "gamma1"]
ALLB = ["delta", "theta", "alpha", "beta", "bstar", "gamma", "gamma1"]
deg = lambda z: float(np.degrees(np.angle(z)))

U = pd.read_csv(DATA / "intermed/phase_offset/units_phase.csv.gz", dtype={"pt": str, "week": str})
U["stim_on"] = (U.line_L > 1) | (U.line_R > 1)
home = U[U.source == "home"]
qL, qR = home.GCr_L.quantile(0.9), home.GCr_R.quantile(0.9)
U["gc_excl"] = (U.source == "home") & ((U.GCr_L > qL) | (U.GCr_R > qR))
U["cond"] = np.where(U.source == "home", U.circ + "/" + np.where(U.stim_on, "on", "off"), "session/" + U.state.fillna(""))
for b in ALLB:
    U[b] = U[f"{b}_re"] + 1j * U[f"{b}_im"]
    U[b + "_s"] = U[f"{b}_sre"] + 1j * U[f"{b}_sim"]
pd.crosstab(U[U.source == "home"].week.str[0], [U[U.source == "home"].circ, U[U.source == "home"].stim_on]).to_csv(OUT / "home_stim_state_by_phase.csv")
clin = json.load(open(DATA / "clinical/clinical_vectors_all.json"))["HAMDs"]
Y = pd.DataFrame([{"pt": p["pt"][3:], "week": w, "y": h / np.mean(p["HDRS17"][:4])} for p in clin for w, h in zip(p["phases"], p["HDRS17"])])


def weekly(corr):
    """One row per source/pt/week/cond/band: mean coherency, mean surrogate coherency, n."""
    d = U[~U.gc_excl] if corr == "corrected" else U
    bands = COR if corr == "corrected" else UNC
    rows = []
    for (src, pt, week, t, cond), g in d.groupby(["source", "pt", "week", "t", "cond"]):
        for b in bands:
            rows.append({"corr": corr, "source": src, "pt": pt, "week": week, "t": t, "cond": cond, "band": b,
                         "c": g[b].mean(), "cs": g[b + "_s"].mean(), "n": len(g)})
    W = pd.DataFrame(rows)
    W["mag"], W["smag"] = np.abs(W.c), np.abs(W.cs)
    thr = W.groupby(["pt", "cond", "band"]).smag.quantile(0.95).rename("thr").reset_index()
    W = W.merge(thr, on=["pt", "cond", "band"])
    W["reliable"] = W.mag > W.thr
    return W


W = pd.concat([weekly("uncorrected"), weekly("corrected")], ignore_index=True)
W = W[W.n >= 3]
W.assign(phase_deg=np.degrees(np.angle(W.c)), c_re=W.c.values.real, c_im=W.c.values.imag).drop(columns=["c", "cs"]).to_csv(
    DATA / "intermed/phase_offset/weekly_phase.csv", index=False)

# ---------------------------------------------------------------- PH1
ph1 = []
for (corr, pt, cond, band), g in W.groupby(["corr", "pt", "cond", "band"], sort=False):
    r = g[g.reliable]
    z = np.exp(1j * np.angle(r.c.values)) if len(r) else np.array([np.nan])
    ph1.append({"corr": corr, "pt": pt, "cond": cond, "band": band, "n_weeks": len(g), "frac_reliable": g.reliable.mean(),
                "phase_deg": deg(z.mean()) if len(r) else np.nan, "resultant_len": float(np.abs(z.mean())) if len(r) else np.nan,
                "coherency_mag_median": g.mag.median(), "imag_coh_median": float(np.median(g.c.values.imag)),
                "surrogate_mag_median": g.smag.median()})
PH1 = pd.DataFrame(ph1)
PH1.to_csv(OUT / "ph1_description.csv", index=False)


# ---------------------------------------------------------------- PH2
def paired(a, b):
    """a, b: weekly frames (same pt/band) -> paired by week on reliable weeks; sign-flip permutation on mean sin(d)."""
    m = a[a.reliable].merge(b[b.reliable], on="week", suffixes=("_a", "_b"))
    if len(m) < 5:
        return None
    d = np.angle(m.c_b.values * np.conj(m.c_a.values))
    obs = np.mean(np.sin(d))
    null = np.array([np.mean(np.sin(d) * RNG.choice([-1, 1], len(d))) for _ in range(2000)])
    return {"n_weeks": len(m), "delta_deg": deg(np.exp(1j * d).mean()), "mean_sin": obs, "p": (np.sum(np.abs(null) >= abs(obs)) + 1) / 2001}


def unpaired(a, b):
    a, b = a[a.reliable], b[b.reliable]
    if len(a) < 3 or len(b) < 5:
        return None
    za, zb = np.exp(1j * np.angle(a.c.values)), np.exp(1j * np.angle(b.c.values))
    stat = lambda x, y: np.angle(y.mean() * np.conj(x.mean()))
    obs = stat(za, zb)
    allz, na = np.concatenate([za, zb]), len(za)
    null = []
    for _ in range(2000):
        p = RNG.permutation(len(allz))
        null.append(stat(allz[p[:na]], allz[p[na:]]))
    return {"n_weeks": len(a) + len(b), "delta_deg": float(np.degrees(obs)), "mean_sin": float(np.sin(obs)),
            "p": (np.sum(np.abs(null) >= abs(obs)) + 1) / 2001}


def sub(corr, pt, cond, band):
    return W[(W["corr"] == corr) & (W.pt == pt) & (W.cond == cond) & (W.band == band)]


ph2 = []
for pt in sorted(W.pt.unique()):
    for corr, bands in (("uncorrected", UNC), ("corrected", COR)):
        for b in bands:
            for st in ("on", "off"):
                r = paired(sub(corr, pt, f"day/{st}", b), sub(corr, pt, f"night/{st}", b))
                if r:
                    ph2.append({"contrast": "i night-day (home)", "detail": f"stim {st}", "corr": corr, "pt": pt, "band": b, **r})
            r = paired(sub(corr, pt, "session/off", b), sub(corr, pt, "session/on", b))
            if r:
                ph2.append({"contrast": "ii on-off (session)", "detail": "", "corr": corr, "pt": pt, "band": b, **r})
            for circ in ("day", "night"):
                r = unpaired(sub(corr, pt, f"{circ}/off", b), sub(corr, pt, f"{circ}/on", b))
                if r:
                    ph2.append({"contrast": "iii on-off (home, C vs B weeks)", "detail": circ, "corr": corr, "pt": pt, "band": b, **r})
    for cond in sorted(W.cond.unique()):
        for bc, bu in (("bstar", "beta"), ("gamma1", "gamma")):
            r = paired(sub("uncorrected", pt, cond, bu), sub("corrected", pt, cond, bc))
            if r:
                ph2.append({"contrast": "iv corrected-uncorrected", "detail": cond, "corr": f"{bc} vs {bu}", "pt": pt, "band": bc, **r})
        if not cond.startswith("session"):
            for b in ("delta", "theta", "alpha"):
                r = paired(sub("uncorrected", pt, cond, b), sub("corrected", pt, cond, b))
                if r:
                    ph2.append({"contrast": "iv corrected-uncorrected", "detail": cond, "corr": "GCr screen", "pt": pt, "band": b, **r})
PH2 = pd.DataFrame(ph2)
PH2["q"] = np.nan
for k, idx in PH2.groupby("contrast").groups.items():
    p = PH2.loc[idx, "p"].values
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    PH2.loc[idx, "q"] = q
PH2.to_csv(OUT / "ph2_contrasts.csv", index=False)
cons = []
for (c, d, corr, b), g in PH2.groupby(["contrast", "detail", "corr", "band"], sort=False):
    sig = g[g.q < 0.05]
    pos, neg = int((sig.mean_sin > 0).sum()), int((sig.mean_sin < 0).sum())
    need = 3 if c.startswith("ii ") or "session" in d else 4
    cons.append({"contrast": c, "detail": d, "corr": corr, "band": b, "n_patients": len(g), "n_sig_pos": pos, "n_sig_neg": neg,
                 "median_delta_deg": g.delta_deg.median(), "consistent": max(pos, neg) >= need})
CONS = pd.DataFrame(cons)
CONS.to_csv(OUT / "ph2_consistency.csv", index=False)

# ---------------------------------------------------------------- PH3: relation to depression
ph3 = []
for cond in ("session/off", "day/on"):
    for pt in sorted(W.pt.unique()):
        for b in COR:
            g = sub("corrected", pt, cond, b).merge(Y, on=["pt", "week"]).sort_values("t")
            g = g[g.t >= 4]
            if len(g) < 8:
                continue
            X = np.column_stack([np.ones(len(g)), np.cos(np.angle(g.c.values)), np.sin(np.angle(g.c.values))])
            r2 = lambda y: 1 - np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2) / np.sum((y - y.mean()) ** 2)
            obs = r2(g.y.values)
            null = [r2(np.roll(g.y.values, s)) for s in RNG.integers(1, len(g), 200)]
            ph3.append({"cond": cond, "pt": pt, "band": b, "n_weeks": len(g), "R2": obs, "null_R2_mean": np.mean(null),
                        "p": (np.sum(np.array(null) >= obs) + 1) / 201, "coherency_mag_median": g.mag.median()})
PH3 = pd.DataFrame(ph3)
o = np.argsort(PH3.p.values)
q = np.empty(len(PH3))
q[o] = np.minimum.accumulate((PH3.p.values[o] * len(PH3) / (np.arange(len(PH3)) + 1))[::-1])[::-1]
PH3["q"] = q
PH3.to_csv(OUT / "ph3_phase_vs_nhdrs.csv", index=False)

# classification on stimulation-off session segments (design of weekly_dense W-A1)
S = U[(U.source == "session") & (U.state == "off")].copy()
PHF = []
for b in COR:
    S[f"cos_{b}"], S[f"sin_{b}"], S[f"mag_{b}"] = np.cos(np.angle(S[b])), np.sin(np.angle(S[b])), np.abs(S[b])
    PHF += [f"cos_{b}", f"sin_{b}", f"mag_{b}"]
seg = pd.read_csv(DATA / "intermed/weekly_dense/segments.csv.gz", dtype={"pt": str, "week": str})
PW = [c for c in seg.columns if c.startswith("pw_")]
S = S.merge(seg[seg.state == "off"][["file", "start_s"] + PW], on=["file", "start_s"]).merge(Y, on=["pt", "week"])
TR, SICK, STAB = ["906", "907", "908"], ["C01", "C02", "C03", "C04"], ["C21", "C22", "C23", "C24"]


def classify(cols, labels=None):
    d = S[S.week.isin(SICK + STAB)].copy()
    d["lab"] = d.week.isin(STAB).astype(int) if labels is None else d.set_index(["pt", "week"]).index.map(labels).values

    def fit(tr):
        sc = MinMaxScaler().fit(tr[cols])
        m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(tr[cols]), tr.lab)
        return lambda X: m.predict_proba(sc.transform(X[cols]))[:, 1]

    rows = []
    for p in TR:
        pred = fit(d[d.pt.isin([q_ for q_ in TR if q_ != p])])
        te = d[d.pt == p]
        pr = pred(te)
        ses = te.assign(pr=pr).groupby("week").agg(lab=("lab", "first"), pr=("pr", "mean"))
        rows.append({"analysis": "LOPO 906-908", "pt": p, "auc_segment": roc_auc_score(te.lab, pr),
                     "auc_session": roc_auc_score(ses.lab, ses.pr) if ses.lab.nunique() == 2 else np.nan})
    if labels is None:
        pred = fit(d[d.pt.isin(TR)])
        a = S[(S.pt == "905") & (S.t >= 4)]
        s5 = a.assign(pr=pred(a)).groupby("week").agg(y=("y", "first"), pr=("pr", "mean"))
        rows.append({"analysis": "905 held out, clinical-state labels", "pt": "905", "auc_segment": roc_auc_score((a.y < 0.5).astype(int), pred(a)),
                     "auc_session": roc_auc_score((s5.y < 0.5).astype(int), s5.pr)})
    return pd.DataFrame(rows)


base = S[S.week.isin(SICK + STAB) & S.pt.isin(TR)].groupby(["pt", "week"]).size().reset_index()[["pt", "week"]]
base["lab"] = base.week.isin(STAB).astype(int)
cl, pt_rows = [], []
for name, cols in (("phase (15)", PHF), ("power (12)", PW), ("power + phase (27)", PW + PHF)):
    c = classify(cols)
    c["features"] = name
    cl.append(c)
    obs = c[c.analysis == "LOPO 906-908"].auc_segment.mean()
    null = []
    for _ in range(200):
        lab = {}
        for p in TR:
            b_ = base[base.pt == p]
            lab.update(dict(zip(zip(b_.pt, b_.week), RNG.permutation(b_.lab.values))))
        try:
            null.append(classify(cols, labels=lab).auc_segment.mean())
        except ValueError:
            pass
    pt_rows.append({"features": name, "obs_auc_segment_mean": obs, "null_mean": np.mean(null), "null_95": np.quantile(null, 0.95),
                    "p": (np.sum(np.array(null) >= obs) + 1) / (len(null) + 1)})
pd.concat(cl).to_csv(OUT / "ph3_classification.csv", index=False)
pd.DataFrame(pt_rows).to_csv(OUT / "ph3_classification_permutation.csv", index=False)

print("stim state by phase (home):\n", pd.read_csv(OUT / "home_stim_state_by_phase.csv").to_string())
print(CONS[CONS.consistent].to_string(index=False) if CONS.consistent.any() else "no consistent contrast")
print(pd.DataFrame(pt_rows).round(3).to_string(index=False))
