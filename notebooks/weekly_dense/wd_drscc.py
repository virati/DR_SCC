"""Apply the DR-SCC paper feature pipeline (dbspace e014944: 5th-order polynomial subtraction + band medians) to the
per-segment Welch PSDs from wd_extract.py, identically to the chronic recordings. Run in the DR_SCC env.
-> $DATA_DIRECTORY/intermed/weekly_dense/segments_drscc.csv.gz (same row order as segments.csv.gz)"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forward_chain"))
import numpy as np
import pandas as pd

import fc
from dbspace.readout import ClinVect, decoder

D = fc.data_dir() / "intermed" / "weekly_dense"
seg = pd.read_csv(D / "segments.csv.gz", dtype={"pt": str, "week": str})
psd = np.load(D / "segments_psd.npz")["psd"]
assert len(seg) == len(psd)
CF = ClinVect.CStruct(fc.data_dir() / "clinical/clinical_vectors_all.json")
BR = types.SimpleNamespace(data_basis={"F": np.linspace(0, 211, 513)}, file_meta=[])
dec = decoder.weekly_decoder(BRFrame=BR, ClinFrame=CF, pts=["905", "906", "907", "908"], clin_measure="pHDRS17",
                             algo="ENR", shuffle_null=False, FeatureSet="main", variance=False)
recs = [{"Data": {"Left": psd[i, 0], "Right": psd[i, 1]}, "Patient": p, "Phase": w} for i, (p, w) in enumerate(zip(seg.pt, seg.week))]
X, _ = dec.calculate_states_in_set(recs)
out = pd.DataFrame(X, columns=["dr_" + c for c in fc.FEATS])
out.to_csv(D / "segments_drscc.csv.gz", index=False)
print(out.shape, out.describe().loc[["mean", "std"]].round(2).to_string())
