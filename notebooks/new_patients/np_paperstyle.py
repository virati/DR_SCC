# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==1.26.4", "scipy==1.11.4", "pandas==2.1.4", "scikit-learn==1.4.2", "joblib==1.3.2"]
# ///
"""Amendment N-A2 (post hoc): the paper-style decoder M1p, frozen and applied.   DATA_DIRECTORY=... uv run notebooks/new_patients/np_paperstyle.py"""
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA = Path(os.environ["DATA_DIRECTORY"])
HERE = Path(__file__).resolve().parent
O = HERE / "outputs"
OLD, NEW = ["901", "903", "905", "906", "907", "908"], ["910", "912", "913"]
MAIN = {"910": "E1-E3;E8-E10", "912": "E1-E3;E9-E11", "913": "E0-E2;E8-E10"}
f2 = lambda x, d=2: "—" if x is None or pd.isna(x) else f"{x:.{d}f}"
H = pd.read_csv(DATA / "intermed/new_patients/home.csv.gz", dtype={"pt": str}, low_memory=False)
H = H[H.err.isna()].reset_index(drop=True)
H["on"] = (H.line_L > 1) | (H.line_R > 1)
clin = {p["pt"][3:]: p for p in json.load(open(DATA / "clinical/clinical_vectors_all.json"))["HAMDs"]}
nh = {}
for pt, c in clin.items():
    h = np.array(c["HDRS17"], float)
    nh.update({(pt, ph): v / np.nanmean([x for x, q in zip(h, c["phases"]) if q.startswith("A")]) for v, ph in zip(h, c["phases"])})
H["y"] = [nh.get((p, w), np.nan) for p, w in zip(H.pt, H.week)]
TR = H[H.pt.isin(OLD) & H.y.notna()].reset_index(drop=True)
BP = [c for c in H.columns if c.startswith("bp_")]
fit = lambda X, y, s=0: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=0.5, n_alphas=30, cv=KFold(5, shuffle=True, random_state=s), max_iter=5000)).fit(X, y)


def split(s):
    a, b = train_test_split(np.arange(len(TR)), test_size=0.4, random_state=s)
    m = fit(TR[BP].values[a], TR.y.values[a], s)
    p = m.predict(TR[BP].values[b])
    out = dict(seed=s, pt="all", r2=r2_score(TR.y.values[b], p), r=pearsonr(TR.y.values[b], p)[0])
    return [out] + [dict(seed=s, pt=q, r2=r2_score(TR.y.values[b][TR.pt.values[b] == q], p[TR.pt.values[b] == q]), r=pearsonr(TR.y.values[b][TR.pt.values[b] == q], p[TR.pt.values[b] == q])[0]) for q in OLD]


def lopo(h):
    tr, te = TR[TR.pt != h], TR[TR.pt == h]
    p = fit(tr[BP].values, tr.y.values).predict(te[BP].values)
    c = tr.y.mean()
    wk = pd.DataFrame({"w": te.week.values, "y": te.y.values, "p": p}).groupby("w").mean()
    mk = lambda y, q: dict(r2=r2_score(y, q), r=pearsonr(y, q)[0], mae=np.mean(np.abs(y - q)), r2_const=r2_score(y, np.full(len(y), c)), mae_const=np.mean(np.abs(y - c)))
    return [dict(held=h, level="week", **mk(wk.y.values, wk.p.values)), dict(held=h, level="recording", **mk(te.y.values, p))]


W = pd.DataFrame([r for rr in Parallel(n_jobs=10)(delayed(split)(s) for s in range(20)) for r in rr])
LO = pd.DataFrame([r for rr in Parallel(n_jobs=6)(delayed(lopo)(h) for h in OLD) for r in rr])
W.to_csv(O / "a2_within_cohort.csv", index=False); LO.to_csv(O / "a2_unseen.csv", index=False)
m = fit(TR[BP].values, TR.y.values)
en = m[-1]
PR = pd.read_csv(O / "predictions_home.csv", dtype={"pt": str})
NW = H[H.pt.isin(NEW)]
NW = NW[NW.montage == NW.pt.map(MAIN)].reset_index(drop=True)
assert (NW.file.values == PR.file.values).all()
PR["M1p"] = m.predict(NW[BP].values)
PR[["pt", "file", "time", "on", "week", "M1p"]].to_csv(O / "predictions_home_paperstyle.csv", index=False)
trp = m.predict(TR[BP].values)
ylo, yhi = TR.y.min(), TR.y.max()


def shift_p(a, b, two=True):
    obs = spearmanr(a, b).correlation
    nul = np.array([spearmanr(a, np.roll(b, s)).correlation for s in range(1, len(a))])
    return obs, (1 + (np.abs(nul) >= abs(obs)).sum()) / (1 + len(nul))


W9 = pd.read_csv(O / "weekly_910.csv").set_index("week")
W9["M1p"] = PR[PR.pt == "910"].groupby("week").M1p.mean()
W9 = W9[W9.n >= 10]
rows = []
for lab, d in (("all weeks", W9), ("stimulation-on weeks only (on in > 90% of recordings)", W9[W9.on > 0.9])):
    for a, b, bl in (("M1p", "on", "share of recordings with stimulation on"), ("M1p", "M4", "M4, in-clinic marker"), ("M1p", None, "week number"), ("M2", "on", "share of recordings with stimulation on (M2)"), ("M2", "M4", "M4, in-clinic marker (M2)"), ("M2", None, "week number (M2)")):
        q = d.dropna(subset=[a] + ([b] if b else []))
        if b == "on" and q.on.nunique() < 3:
            continue
        r, p = shift_p(q[a].values, q[b].values if b else q.index.values.astype(float))
        rows.append(dict(weeks_set=lab, against=bl, weeks=len(q), rho=r, p=p))
P4 = pd.DataFrame(rows); P4.to_csv(O / "a2_910_agreement.csv", index=False)
q3 = lambda v: f"{f2(np.median(v))} ({f2(np.percentile(v, 10))} to {f2(np.percentile(v, 90))})"
sha = hashlib.sha256((O / "predictions_home_paperstyle.csv").read_bytes()).hexdigest()
wa = W[W.pt == "all"]
L = ["# New patients: the paper-style decoder (amendment N-A2, post hoc)", "", "<!-- ai-start -->",
     "Generated by `np_paperstyle.py`. Added after the pre-registered models M1 and M3 turned out to be constants. Everything here is post hoc.", "",
     "**No depression scores exist for the new patients on this machine. Nothing below says whether a prediction is right.**", "",
     f"Sealed: `outputs/predictions_home_paperstyle.csv`, SHA-256 `{sha}`.", "",
     "## The frozen model", "", f"Penalty {en.alpha_:.4f}; intercept {f2(en.intercept_)}. Coefficients per standard deviation of each feature:", "",
     "| Feature | coefficient |", "|---|---|"]
L += [f"| {c[3]} {c[4:].replace('*', chr(92) + '*')} | {v:+.3f} |" for c, v in sorted(zip(BP, en.coef_), key=lambda t: -abs(t[1]))]
L += ["", "## How good is it where the paper tested it, and where it is being used here?", "",
      f"**Within the cohort** (60/40 split of recordings, average of 20 random splits): R² {f2(wa.r2.mean(), 3)} (range {f2(wa.r2.min(), 3)} to {f2(wa.r2.max(), 3)}), r {f2(wa.r.mean(), 3)}. The published value is R² 0.137.", "",
      "Per patient within those test sets (mean over splits): " + "; ".join(f"{p}: R² {f2(W[W.pt == p].r2.mean())}, r {f2(W[W.pt == p].r.mean())}" for p in OLD) + ".", "",
      "**In a patient left out entirely** (the situation of 910, 912 and 913):", "", "| Level | measure | " + " | ".join(OLD) + " | median |", "|---|---|" + "---|" * 7]
for lv in ("week", "recording"):
    g = LO[LO.level == lv].set_index("held")
    for k, lb in (("r2", "R²"), ("r2_const", "R² of the constant"), ("r", "r"), ("mae", "mean abs. error"), ("mae_const", "mean abs. error of the constant")):
        L.append(f"| {lv} | {lb} | " + " | ".join(f2(g[k].get(p)) for p in OLD) + f" | {f2(g[k].median())} |")
L += ["", "## Predictions for the new patients", "", "Predicted nHDRS; median (10th to 90th percentile).", "", "| Patient | state | recordings | M1p |", "|---|---|---|---|"]
for pt in NEW:
    for st in (False, True):
        d = PR[(PR.pt == pt) & (PR.on == st)]
        if len(d) >= 5:
            L.append(f"| {pt} | {'on' if st else 'off'} | {len(d)} | {q3(d.M1p.values)} |")
L += [f"| training patients, stimulation off | off | {int((~TR.on).sum())} | {q3(trp[~TR.on.values])} |", f"| training patients, stimulation on | on | {int(TR.on.sum())} | {q3(trp[TR.on.values])} |",
      f"| training patients, true nHDRS, stimulation off | | | {q3(TR.y.values[~TR.on.values])} |", f"| training patients, true nHDRS, stimulation on | | | {q3(TR.y.values[TR.on.values])} |", "",
      "Share outside the nHDRS range seen in training: " + ", ".join(f"{pt} {f2(((PR[PR.pt == pt].M1p < ylo) | (PR[PR.pt == pt].M1p > yhi)).mean())}" for pt in NEW) + ".", "",
      "**910 by week**", "", "| Week | recordings | stimulation on | M1p | M2 | M4: probability of \"sick\" |", "|---|---|---|---|---|---|"]
L += [f"| {int(w)} | {int(r.n)} | {f2(r.on)} | {f2(r.M1p)} | {f2(r.M2)} | {f2(r.M4)} |" for w, r in W9.iterrows()]
L += ["", "## 910: what do the weekly predictions follow?", "", "Spearman correlation across weeks; two-sided p from circular week shifts.", "", "| Weeks | M1p (or M2 where marked) against | weeks | ρ | p |", "|---|---|---|---|---|"]
L += [f"| {r.weeks_set} | {r.against} | {r.weeks} | {f2(r.rho)} | {f2(r.p, 3)} |" for r in P4.itertuples()]
L += ["", "<!-- ai-end -->", ""]
(HERE / "RESULTS_paperstyle.md").write_text("\n".join(L))
print("\n".join(L[8:]))
