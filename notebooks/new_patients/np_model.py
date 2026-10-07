# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==1.26.4", "scipy==1.11.4", "pandas==2.1.4", "scikit-learn==1.4.2", "joblib==1.3.2"]
# ///
"""Frozen models, expected accuracy, resemblance checks and sealed predictions (PROTOCOL.md P1-P4, amendment N-A1).
    DATA_DIRECTORY=... uv run notebooks/new_patients/np_model.py        Outputs: outputs/*.csv, RESULTS.md"""
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")
DATA = Path(os.environ["DATA_DIRECTORY"])
IN = DATA / "intermed/new_patients"
HERE = Path(__file__).resolve().parent
O = HERE / "outputs"
O.mkdir(exist_ok=True)
OLD, NEW = ["901", "903", "905", "906", "907", "908"], ["910", "912", "913"]
MAIN = {"910": "E1-E3;E8-E10", "912": "E1-E3;E9-E11", "913": "E0-E2;E8-E10"}
rng = np.random.default_rng(0)
f2 = lambda x, d=2: "—" if x is None or pd.isna(x) else f"{x:.{d}f}"

# ---- data
H = pd.read_csv(IN / "home.csv.gz", dtype={"pt": str}, low_memory=False)
H = H[H.err.isna()].reset_index(drop=True) if "err" in H else H
H["on"] = (H.line_L > 1) | (H.line_R > 1)
H["time"] = pd.to_datetime(H.time)
clin = {p["pt"][3:]: p for p in json.load(open(DATA / "clinical/clinical_vectors_all.json"))["HAMDs"]}
nh = {}
for pt, c in clin.items():
    h = np.array(c["HDRS17"], float)
    base = np.nanmean([v for v, ph in zip(h, c["phases"]) if ph.startswith("A")])
    nh.update({(pt, ph): v / base for v, ph in zip(h, c["phases"])})
H["y"] = [nh.get((p, w), np.nan) for p, w in zip(H.pt, H.week)]
TR = H[H.pt.isin(OLD) & H.y.notna()].reset_index(drop=True)
NW = H[H.pt.isin(NEW)].reset_index(drop=True)
mont = NW.groupby(["pt", "montage"]).size().rename("recordings").reset_index()
NW = NW[NW.montage == NW.pt.map(MAIN)].reset_index(drop=True)
cols = lambda *pre: [c for c in H.columns if c.startswith(pre)]
BLOCKS = {"band power": cols("bp_"), "coherence and phase": cols("coh_", "cos_", "sin_"), "coupling": cols("pac_"), "FOOOF": cols("fo_"), "catch22": cols("c22_")}
SETS = {"M1": BLOCKS["band power"], "M2": BLOCKS["band power"] + BLOCKS["coherence and phase"] + BLOCKS["coupling"] + BLOCKS["FOOOF"], "M3": BLOCKS["catch22"]}
NAMES = {"M1": "M1, DR-SCC-like", "M2": "M2, full set", "M3": "M3, catch22"}
STATES = {"all": lambda d: np.ones(len(d), bool), "off": lambda d: ~d.on.values, "on": lambda d: d.on.values}


def enr(X, y, g):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      ElasticNetCV(l1_ratio=0.5, n_alphas=30, cv=list(LeaveOneGroupOut().split(X, y, g)), max_iter=5000))
    return m.fit(X, y)


def met(y, p, const):
    r = pearsonr(y, p)[0] if np.std(p) > 0 and len(y) > 2 else np.nan
    return dict(r2=r2_score(y, p), r=r, mae=np.mean(np.abs(y - p)), r2_const=r2_score(y, np.full(len(y), const)), mae_const=np.mean(np.abs(y - const)))


def lopo(mk, st, held):
    d = TR[STATES[st](TR)]
    tr, te = d[d.pt != held], d[d.pt == held]
    if len(te) < 20:
        return []
    m = enr(tr[SETS[mk]].values, tr.y.values, tr.pt.values)
    p = m.predict(te[SETS[mk]].values)
    c = tr.y.mean()
    wk = pd.DataFrame({"w": te.week.values, "y": te.y.values, "p": p}).groupby("w").mean()
    return [dict(model=mk, state=st, held=held, level="recording", n=len(te), **met(te.y.values, p, c)),
            dict(model=mk, state=st, held=held, level="week", n=len(wk), **met(wk.y.values, wk.p.values, c))]


# ---- P1
jobs = [(mk, st, h) for mk in SETS for st in STATES for h in OLD]
P1 = pd.DataFrame([r for rr in Parallel(n_jobs=12)(delayed(lopo)(*j) for j in jobs) for r in rr])
P1.to_csv(O / "p1_expected_accuracy.csv", index=False)

# ---- frozen models and P3 predictions
frozen, coef = {}, []
for mk in SETS:
    for st in STATES:
        d = TR[STATES[st](TR)]
        frozen[(mk, st)] = (enr(d[SETS[mk]].values, d.y.values, d.pt.values), d)
        e = frozen[(mk, st)][0][-1]
        coef += [dict(model=mk, state=st, feature=c, coef=v, alpha=e.alpha_) for c, v in zip(SETS[mk], e.coef_)]
pd.DataFrame(coef).to_csv(O / "frozen_coefficients.csv", index=False)
PR = NW[["pt", "file", "time", "montage", "on", "circ"]].copy()
PR["day"] = (NW.time - NW.groupby("pt").time.transform("min")).dt.days
PR["week"] = PR.day // 7 + 1
for mk in SETS:
    PR[mk] = frozen[(mk, "all")][0].predict(NW[SETS[mk]].values)
    PR[mk + "_matched"] = np.where(NW.on.values, frozen[(mk, "on")][0].predict(NW[SETS[mk]].values), frozen[(mk, "off")][0].predict(NW[SETS[mk]].values))
PR.to_csv(O / "predictions_home.csv", index=False)
trp = {mk: frozen[(mk, "all")][0].predict(TR[SETS[mk]].values) for mk in SETS}

# ---- M4
SG = pd.read_csv(DATA / "intermed/weekly_dense/segments.csv.gz", dtype={"pt": str, "week": str})
PAPER = [c for c in SG.columns if c.startswith(("pw_", "coh_")) or c in ("pac_L", "pac_R")]
SICK, STAB = ["C01", "C02", "C03", "C04"], ["C21", "C22", "C23", "C24"]
t4 = SG[(SG.state == "off") & SG.pt.isin(["906", "907", "908"]) & SG.week.isin(SICK + STAB)]
m4 = make_pipeline(MinMaxScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(t4[PAPER].values, t4.week.isin(SICK).astype(int).values)
SN = pd.read_csv(IN / "sessions_new.csv.gz", dtype={"pt": str})
SF = pd.read_csv(IN / "session_files.csv", dtype={"pt": str})
SN = SN.merge(SF[["file", "montage"]], on="file")
SN["time"] = pd.to_datetime(SN.time)
sn_mont = SF.groupby(["pt", "montage"]).size().rename("files").reset_index()
SNo = SN[(SN.state == "off") & (SN.montage == SN.pt.map(MAIN))].copy()
SNo["p_sick"] = m4.predict_proba(SNo[PAPER].values)[:, 1]
first = NW.groupby("pt").time.min()
SS = SNo.groupby(["pt", "file"]).agg(time=("time", "first"), segments=("p_sick", "size"), p_sick=("p_sick", "mean")).reset_index()
SS["week"] = [(t - first[p]).days // 7 + 1 for t, p in zip(SS.time, SS.pt)]
SS.to_csv(O / "predictions_sessions.csv", index=False)
tr4 = m4.predict_proba(t4[PAPER].values)[:, 1]

# ---- P2
def resemble(A, B, feats):
    """A: target recordings, B: reference pool. Share of A's values outside B's 1-99 % range, and A-vs-B AUROC."""
    a, b = A[feats].values, B[feats].values
    if len(a) < 20 or len(b) < 50:
        return np.nan, np.nan
    if len(b) > 4000:
        b = b[rng.choice(len(b), 4000, replace=False)]
    lo, hi = np.nanpercentile(b, 1, axis=0), np.nanpercentile(b, 99, axis=0)
    out = np.nanmean((a < lo) | (a > hi))
    X, y = np.vstack([a, b]), np.r_[np.ones(len(a)), np.zeros(len(b))]
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return out, roc_auc_score(y, p)


p2 = []
for st in ("off", "on"):
    T = TR[STATES[st](TR)]
    for bk, feats in BLOCKS.items():
        for pt in OLD:
            o, a = resemble(T[T.pt == pt], T[T.pt != pt], feats)
            p2.append(dict(kind="original", pt=pt, state=st, block=bk, n=int((T.pt == pt).sum()), outside=o, auc=a))
        for pt in NEW:
            A = NW[(NW.pt == pt) & STATES[st](NW)]
            o, a = resemble(A, T, feats)
            p2.append(dict(kind="new", pt=pt, state=st, block=bk, n=len(A), outside=o, auc=a))
So = SG[SG.state == "off"]
for pt in ["905", "906", "907", "908"]:
    o, a = resemble(So[So.pt == pt], So[So.pt.isin(["906", "907", "908"]) & (So.pt != pt)], PAPER)
    p2.append(dict(kind="original", pt=pt, state="off", block="in-clinic (20)", n=int((So.pt == pt).sum()), outside=o, auc=a))
for pt in ["910", "913"]:
    o, a = resemble(SNo[SNo.pt == pt], So[So.pt.isin(["906", "907", "908"])], PAPER)
    p2.append(dict(kind="new", pt=pt, state="off", block="in-clinic (20)", n=int((SNo.pt == pt).sum()), outside=o, auc=a))
P2 = pd.DataFrame(p2)
P2.to_csv(O / "p2_resemblance.csv", index=False)

# ---- P4 (910)
def shift_p(a, b, two=False):
    obs = spearmanr(a, b).correlation
    nul = np.array([spearmanr(a, np.roll(b, s)).correlation for s in range(1, len(a))])
    return obs, ((1 + (np.abs(nul) >= abs(obs)).sum()) if two else (1 + (nul >= obs).sum())) / (1 + len(nul))


W = PR[PR.pt == "910"].groupby("week").agg(n=("M1", "size"), on=("on", "mean"), **{k: (k, "mean") for mk in SETS for k in (mk, mk + "_matched")})
W4 = SS[SS.pt == "910"].groupby("week").p_sick.mean()
W = W.join(W4.rename("M4"))
W.to_csv(O / "weekly_910.csv")
Wk = W[W.n >= 10]
p4 = []
for a, b in [("M1", "M2"), ("M1", "M3"), ("M2", "M3")]:
    r, p = shift_p(Wk[a].values, Wk[b].values)
    p4.append(dict(a=NAMES[a], b=NAMES[b], weeks=len(Wk), rho=r, p=p))
for a in SETS:
    d = Wk.dropna(subset=["M4"])
    r, p = shift_p(d[a].values, d.M4.values)
    p4.append(dict(a=NAMES[a], b="M4, in-clinic marker", weeks=len(d), rho=r, p=p))
for a in [*SETS, "M4"]:
    d = Wk.dropna(subset=[a])
    r, p = shift_p(d[a].values, d.index.values.astype(float), two=True)
    p4.append(dict(a=NAMES.get(a, "M4, in-clinic marker"), b="week number", weeks=len(d), rho=r, p=p))
P4 = pd.DataFrame(p4)
P4.to_csv(O / "p4_agreement_910.csv", index=False)

# ---- RESULTS.md
sha = lambda n: hashlib.sha256((O / n).read_bytes()).hexdigest()
ylo, yhi = TR.y.min(), TR.y.max()
L = ["# Pure inference on DBS910, DBS912 and DBS913: results", "", "<!-- ai-start -->",
     "Generated by `np_model.py`. Protocol and amendment N-A1: `PROTOCOL.md`, both committed before any model was fitted.", "",
     "**There are no depression scores for these three patients on this machine. Nothing below says whether a prediction is right.**", "",
     "## Sealed predictions", "", "| File | SHA-256 |", "|---|---|",
     f"| `outputs/predictions_home.csv` | `{sha('predictions_home.csv')}` |", f"| `outputs/predictions_sessions.csv` | `{sha('predictions_sessions.csv')}` |", "",
     f"Training set: {len(TR)} at-home recordings from six patients; nHDRS in training runs from {f2(ylo)} to {f2(yhi)} (mean {f2(TR.y.mean())}).", "",
     "## What was recorded", "", "| Patient | sensing contacts | at-home recordings | used |", "|---|---|---|---|"]
L += [f"| {r.pt} | {r.montage} | {r.recordings} | {'yes' if r.montage == MAIN[r.pt] else 'no'} |" for r in mont.itertuples()]
L += ["", "| Patient | recordings used | first | last | stimulation on | day | night-time hours (22–06) |", "|---|---|---|---|---|---|---|"]
for pt in NEW:
    d = PR[PR.pt == pt]
    nt = ((d.time.dt.hour >= 22) | (d.time.dt.hour < 6)).mean()
    L.append(f"| {pt} | {len(d)} | {d.time.min().date()} | {d.time.max().date()} | {f2(d.on.mean())} | {f2(1 - nt)} | {f2(nt)} |")
L += ["", "In-clinic files: " + "; ".join(f"{r.pt} {r.montage}: {r.files}" for r in sn_mont.itertuples()) + ". Stimulation-off segments used: "
      + ", ".join(f"{pt}: {int((SNo.pt == pt).sum())} from {SS[SS.pt == pt].file.nunique()} files" for pt in ["910", "913"]) + ".", "",
      "## P1: accuracy to expect in a patient the model has never seen", "",
      "Each original patient left out in turn. R² is against that patient's true nHDRS; \"constant\" predicts the training mean for everyone. Week level (weekly mean prediction).", ""]
for st, lab in (("all", "Models trained on all recordings (primary)"), ("off", "Stimulation-off recordings only"), ("on", "Stimulation-on recordings only")):
    g = P1[(P1.state == st) & (P1.level == "week")]
    L += [f"**{lab}**", "", "| Model | measure | " + " | ".join(OLD) + " | median |", "|---|---|" + "---|" * (len(OLD) + 1)]
    for mk in SETS:
        q = g[g.model == mk].set_index("held")
        for k, lb in (("r2", "R²"), ("r", "r"), ("mae", "mean abs. error")):
            L.append(f"| {NAMES[mk]} | {lb} | " + " | ".join(f2(q[k].get(p)) for p in OLD) + f" | {f2(q[k].median())} |")
    q = g[g.model == "M1"].set_index("held")
    L += ["| constant | R² | " + " | ".join(f2(q.r2_const.get(p)) for p in OLD) + f" | {f2(q.r2_const.median())} |",
          "| constant | mean abs. error | " + " | ".join(f2(q.mae_const.get(p)) for p in OLD) + f" | {f2(q.mae_const.median())} |", ""]
g = P1[(P1.state == "all") & (P1.level == "recording")]
L += ["Recording level, primary models: " + "; ".join(f"{NAMES[mk]} median R² {f2(g[g.model == mk].r2.median())}, median r {f2(g[g.model == mk].r.median())}" for mk in SETS) + ".", "",
      "M4 (quoted from `notebooks/weekly_dense/RESULTS.md`, not recomputed): leave-one-patient-out among 906–908 AUROC 0.97–1.0; 905 against clinical state 0.86 at segment level.", "",
      "## P2: do the new recordings resemble the training recordings?", "",
      "Within stimulation state. \"Outside\" = share of feature values outside the reference 1st–99th percentile range (2% expected by construction). AUROC = how easily the patient is told from the reference pool (0.5 = can't be; 1.0 = trivially). The original patients, each against the other five, give the reference range.", ""]
for st in ("off", "on"):
    g = P2[(P2.state == st) & (P2.block != "in-clinic (20)")]
    L += [f"**Stimulation {st}, at home**", "", "| Block | original patients: outside (range) | original patients: AUROC (range) | " + " | ".join(f"{p}: outside / AUROC (n)" for p in NEW) + " |", "|---|---|---|" + "---|" * len(NEW)]
    for bk in BLOCKS:
        q = g[g.block == bk]
        o = q[q.kind == "original"].dropna(subset=["auc"])
        cells = []
        for p in NEW:
            r = q[(q.pt == p)].iloc[0]
            cells.append("too few" if pd.isna(r.auc) else f"{f2(r.outside)} / {f2(r.auc)} ({int(r.n)})")
        L.append(f"| {bk} | {f2(o.outside.min())}–{f2(o.outside.max())} | {f2(o.auc.min())}–{f2(o.auc.max())} | " + " | ".join(cells) + " |")
    L.append("")
g = P2[P2.block == "in-clinic (20)"]
L += ["**In-clinic stimulation-off segments (20 features), against 906–908**", "", "| Patient | kind | segments | outside | AUROC |", "|---|---|---|---|---|"]
L += [f"| {r.pt} | {r.kind} | {int(r.n)} | {f2(r.outside)} | {f2(r.auc)} |" for r in g.itertuples()]
L += ["", "## P3: the predictions", "", "Predicted nHDRS (1.0 = the patient's own pre-implant level). Median, with the 10th–90th percentile in brackets.", "",
      "| Patient | state | recordings | " + " | ".join(NAMES[mk] for mk in SETS) + " | " + " | ".join(mk + " state-matched" for mk in SETS) + " |", "|---|---|---|" + "---|" * (2 * len(SETS))]
q3 = lambda v: f"{f2(np.median(v))} ({f2(np.percentile(v, 10))} to {f2(np.percentile(v, 90))})"
for pt in NEW:
    for st in ("off", "on"):
        d = PR[(PR.pt == pt) & (PR.on == (st == "on"))]
        if len(d) < 5:
            continue
        L.append(f"| {pt} | {st} | {len(d)} | " + " | ".join(q3(d[mk].values) for mk in SETS) + " | " + " | ".join(q3(d[mk + '_matched'].values) for mk in SETS) + " |")
L.append("| training patients | all | " + str(len(TR)) + " | " + " | ".join(q3(trp[mk]) for mk in SETS) + " | | | |")
L += ["", "Share of predictions outside the nHDRS range seen in training: " + "; ".join(
    f"{pt}: " + ", ".join(f"{mk} {f2(((PR[PR.pt == pt][mk] < ylo) | (PR[PR.pt == pt][mk] > yhi)).mean())}" for mk in SETS) for pt in NEW) + ".", "",
      "**910 over time** (weeks from the first recording; weeks with at least 10 recordings)", "",
      "| Week | recordings | stimulation on | M1 | M2 | M3 | M4: probability of \"sick\" |", "|---|---|---|---|---|---|---|"]
L += [f"| {int(w)} | {int(r.n)} | {f2(r.on)} | {f2(r.M1)} | {f2(r.M2)} | {f2(r.M3)} | {f2(r.M4)} |" for w, r in Wk.iterrows()]
L += ["", "**In-clinic marker (M4), per session file**", "", f"Training segments for reference: mean probability of \"sick\" {f2(tr4[t4.week.isin(SICK).values].mean())} in the sick weeks and {f2(tr4[~t4.week.isin(SICK).values].mean())} in the stable weeks.", ""]
for pt in ["910", "913"]:
    d = SS[SS.pt == pt]
    L.append(f"- {pt}: {len(d)} files, {d.time.min().date()} to {d.time.max().date()}; probability of \"sick\" median {f2(d.p_sick.median())}, range {f2(d.p_sick.min())} to {f2(d.p_sick.max())}; "
             f"share of files above 0.5: {f2((d.p_sick > 0.5).mean())}.")
L += ["", "## P4: do the models agree with each other in 910?", "", "Spearman correlation across weeks; p from circular week shifts (one-sided for agreement, two-sided for the trend with week).", "",
      "| | against | weeks | ρ | p |", "|---|---|---|---|---|"]
L += [f"| {r.a} | {r.b} | {r.weeks} | {f2(r.rho)} | {f2(r.p, 3)} |" for r in P4.itertuples()]
L += ["", "<!-- ai-end -->", ""]
(HERE / "RESULTS.md").write_text("\n".join(L))
print("\n".join(L[5:]))
