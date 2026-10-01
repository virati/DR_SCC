"""Phase 2 (amendment P2) input export, run in the DR_SCC env (needs dbspace for clinical scales):
weekly feature tables (MC screen on, label-free alignment per family), per-fold MC feature drops, clinical vectors
(pHDRS17, pBDI, pGAF), and recording-level z-aligned features for CEBRA. -> $DATA_DIRECTORY/intermed/cross_patient/p2/"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "forward_chain"))
import numpy as np
import pandas as pd

import fc
import xp

INPUTS = {"F-band": "z", "F-fooof-per": "z", "F-rel": "z", "F-asym": "z", "F-riem": "riem"}
OUT = fc.data_dir() / "intermed" / "cross_patient" / "p2"
OUT.mkdir(parents=True, exist_ok=True)

rec = xp.recordings()
targets = fc.clinical_targets()
from dbspace.readout import ClinVect  # noqa: E402

CF = ClinVect.CStruct(fc.data_dir() / "clinical/clinical_vectors_all.json")
clin = pd.DataFrame([{"pt": p, "week": w, "c_HDRS": CF.get_depression_measure("DBS" + p, "pHDRS17", w),
                      "c_BDI": CF.get_depression_measure("DBS" + p, "pBDI", w),
                      "c_GAF": CF.get_depression_measure("DBS" + p, "pGAF", w)} for p in fc.PTS for w in fc.WEEKS])

drops = {}
for fam, al in INPUTS.items():
    df, cols = xp.table(rec, targets, fam, al, mc=True)
    df = df.merge(clin, on=["pt", "week"])
    df.to_csv(OUT / f"weekly_{fam}.csv", index=False)
    drops[fam] = {p: xp.mc_screen(df[df.pt != p], cols, [q for q in fc.PTS if q != p])[0] for p in fc.PTS}
    drops[fam]["_all_cols"] = cols
json.dump(drops, open(OUT / "mc_kept_features_per_fold.json", "w"), indent=1)

# recording-level, for CEBRA (exploratory): MC decile exclusion, then per-patient z against B-week recordings
r = rec[xp.mc_recording_mask(rec)].copy()
band = [f"band_{c}" for c in fc.FEATS]
per = [f"per_{s}_{b}" for s in "LR" for b in fc.BANDS]
rel = r[band].copy()
for s in "LR":
    sc = [f"band_{s}{b}" for b in fc.BANDS]
    rel[sc] = rel[sc].sub(rel[sc].mean(axis=1), axis=0)
asym = pd.DataFrame({f"asym_{b}": rel[f"band_L{b}"] - rel[f"band_R{b}"] for b in fc.BANDS})
R = pd.concat([r[["pt", "week", "t", "file", "qc_ok"]].reset_index(drop=True), r[band].reset_index(drop=True),
               r[per].reset_index(drop=True), rel.add_prefix("rel_").reset_index(drop=True), asym.reset_index(drop=True)], axis=1)
feat_cols = {"F-band": band, "F-fooof-per": per, "F-rel": ["rel_" + c for c in band], "F-asym": list(asym.columns)}
for fam, cols in feat_cols.items():
    for p, idx in R.groupby("pt").groups.items():
        b = R.loc[idx][R.loc[idx, "t"] <= 3][cols]
        R.loc[idx, cols] = (R.loc[idx, cols] - b.mean()) / b.std(ddof=1).replace(0, np.nan)
R = R.merge(targets[["pt", "week", "T-raw"]].rename(columns={"T-raw": "y"}), on=["pt", "week"])
R.to_csv(OUT / "recordings_aligned.csv.gz", index=False)
json.dump(feat_cols, open(OUT / "recording_feature_cols.json", "w"), indent=1)
print({k: len(v) for k, v in feat_cols.items()}, len(R), "recordings")
