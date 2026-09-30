# /// script
# requires-python = ">=3.8,<3.9"
# dependencies = ["fooof==1.1.0", "numpy==1.24.4", "scipy==1.10.1", "pandas==2.0.3", "joblib==1.3.2", "python-dotenv==1.0.1", "matplotlib==3.7.5"]
# ///
"""FOOOF features for every recording (PROTOCOL.md, amendment 1). Self-contained:
    uv run notebooks/forward_chain/fooof_features.py
Reads the plain frame export (numpy-only pickle), writes
$DATA_DIRECTORY/intermed/forward_chain/recordings_fooof.csv.gz
"""
import hashlib
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv(usecwd=True))
DATA = Path(os.environ["DATA_DIRECTORY"])
FRAME = DATA / "intermed/chronic/Chronic_FrameFeb2026_plain.pkl"
FRAME_MD5 = "07e0f3a2c88b4b323567aa88d84089f8"
OUT = DATA / "intermed/forward_chain/recordings_fooof.csv.gz"

PTS = ["901", "903", "905", "906", "907", "908"]
WEEKS = ["B0" + str(i) for i in range(1, 5)] + ["C%02d" % i for i in range(1, 25)]
T_OF = {w: i for i, w in enumerate(WEEKS)}
BANDS = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 14), "Beta*": (14, 20), "Gamma1": (35, 50)}
FIT_RANGE = [1, 55]
SETTINGS = dict(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1, aperiodic_mode="fixed", verbose=False)
R2_MIN = 0.5


def fit_one(freqs, psd):
    from fooof import FOOOF
    band = (freqs >= FIT_RANGE[0]) & (freqs <= FIT_RANGE[1])
    if not np.all(np.isfinite(psd[band])) or np.any(psd[band] <= 0):
        return None
    fm = FOOOF(**SETTINGS)
    try:
        fm.fit(freqs, psd, FIT_RANGE)
    except Exception:
        return None
    if not fm.has_model:
        return None
    out = {"offset": fm.aperiodic_params_[0], "exponent": fm.aperiodic_params_[1], "r2": fm.r_squared_}
    for name, (lo, hi) in BANDS.items():
        sel = (fm.freqs >= lo) & (fm.freqs <= hi)
        out[name] = float(np.mean(fm._spectrum_flat[sel]))
    return out


def fit_chunk(freqs, chunk):
    rows = []
    for r in chunk:
        row = {"pt": r["Patient"], "week": r["Phase"], "t": T_OF[r["Phase"]], "circ": r["Circadian"],
               "gc": bool(r["GC_Flag"]["Flag"]), "file": os.path.basename(r["Filename"])}
        ok = True
        for side, ch in (("L", "Left"), ("R", "Right")):
            res = fit_one(freqs, np.asarray(r["Data"][ch], dtype=float))
            if res is None:
                ok = False
                res = {k: np.nan for k in ["offset", "exponent", "r2", *BANDS]}
            ok = ok and res["r2"] >= R2_MIN
            row.update({f"{side}_{k}": v for k, v in res.items()})
        row["qc_ok"] = ok
        rows.append(row)
    return rows


if __name__ == "__main__":
    blob = FRAME.read_bytes()
    assert hashlib.md5(blob).hexdigest() == FRAME_MD5, "plain frame md5 mismatch"
    frame = pickle.loads(blob)
    del blob
    freqs = np.asarray(frame["data_basis"]["F"], dtype=float)
    recs = [r for r in frame["file_meta"] if r["Patient"] in PTS and r["Phase"] in T_OF]
    jobs = int(os.environ.get("JOBS", 8))
    chunks = [recs[i:i + 200] for i in range(0, len(recs), 200)]
    rows = [x for part in Parallel(n_jobs=jobs)(delayed(fit_chunk)(freqs, c) for c in chunks) for x in part]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} recordings, {int((~df.qc_ok).sum())} excluded by QC "
          f"(R2 < {R2_MIN} or bad PSD); median R2 L {df.L_r2.median():.3f} R {df.R_r2.median():.3f}", file=sys.stderr)
