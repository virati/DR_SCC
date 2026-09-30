# /// script
# requires-python = ">=3.8,<3.9"
# dependencies = ["numpy==1.24.4", "scipy==1.10.1", "pandas==2.0.3", "joblib==1.3.2", "python-dotenv==1.0.1"]
# ///
"""Per-recording confound and cross-spectral features (cross-patient PROTOCOL.md, steps 1-2). Self-contained:
    uv run notebooks/cross_patient/xp_features.py
- GCr per channel = log10 P(66 Hz) - log10 P(64 Hz), from the frame PSD (nearest bins)            [Ch 4, eq 4.17]
- per band, the real 2x2 L/R co-spectral matrix (Welch CSD with the frame's settings) from the raw .txt,
  last 10 s of channels 0 (Left) and 2 (Right), averaged over the band's bins
Rows follow the frame's file_meta order filtered to the 6 patients and B01-C24, the same order as the
forward-chaining feature tables, and carry the file name for checking.
Output: $DATA_DIRECTORY/intermed/cross_patient/recordings_xp.csv.gz
"""
import hashlib
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sig
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv(usecwd=True))
DATA = Path(os.environ["DATA_DIRECTORY"])
FRAME = DATA / "intermed/chronic/Chronic_FrameFeb2026_plain.pkl"
FRAME_MD5 = "07e0f3a2c88b4b323567aa88d84089f8"
OUT = DATA / "intermed/cross_patient/recordings_xp.csv.gz"

PTS = ["901", "903", "905", "906", "907", "908"]
WEEKS = ["B0" + str(i) for i in range(1, 5)] + ["C%02d" % i for i in range(1, 25)]
T_OF = {w: i for i, w in enumerate(WEEKS)}
BANDS = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 14), "Beta*": (14, 20), "Gamma1": (35, 50)}
FS, SEC = 422, 10
WELCH = dict(fs=FS, window="blackmanharris", nperseg=512, noverlap=128, nfft=1024)


def local_path(fname):
    """Frame filenames are absolute paths on the machine that built it; re-root them under DATA."""
    p = Path(fname)
    if p.exists():
        return p
    parts = p.parts
    if "lfp" in parts:
        return DATA / "neural" / "lfp" / Path(*parts[parts.index("lfp") + 1:])
    return p


def one(rec, freqs):
    row = {"pt": rec["Patient"], "week": rec["Phase"], "t": T_OF[rec["Phase"]], "circ": rec["Circadian"],
           "file": os.path.basename(rec["Filename"])}
    i64, i66 = np.argmin(np.abs(freqs - 64)), np.argmin(np.abs(freqs - 66))
    for side, ch in (("L", "Left"), ("R", "Right")):
        psd = np.asarray(rec["Data"][ch], dtype=float)
        row[f"GCr_{side}"] = (np.log10(psd[i66]) - np.log10(psd[i64])) if psd[i66] > 0 and psd[i64] > 0 else np.nan
    try:
        raw = pd.read_csv(local_path(rec["Filename"]), sep=",", header=None).values
        x, y = raw[-(FS * SEC):-1, 0].astype(float), raw[-(FS * SEC):-1, 2].astype(float)
        f, pxx = sig.welch(x, **WELCH)
        _, pyy = sig.welch(y, **WELCH)
        _, pxy = sig.csd(x, y, **WELCH)
        for b, (lo, hi) in BANDS.items():
            sel = (f >= lo) & (f < hi)
            row[f"{b}_LL"], row[f"{b}_RR"], row[f"{b}_LR"] = pxx[sel].mean(), pyy[sel].mean(), np.real(pxy[sel]).mean()
        row["xspec_ok"] = True
    except Exception as e:
        row["xspec_ok"] = False
        row["xspec_err"] = f"{type(e).__name__}: {e}"[:120]
    return row


if __name__ == "__main__":
    blob = FRAME.read_bytes()
    assert hashlib.md5(blob).hexdigest() == FRAME_MD5, "plain frame md5 mismatch"
    frame = pickle.loads(blob)
    del blob
    freqs = np.asarray(frame["data_basis"]["F"], dtype=float)
    recs = [r for r in frame["file_meta"] if r["Patient"] in PTS and r["Phase"] in T_OF]
    rows = Parallel(n_jobs=int(os.environ.get("JOBS", 8)), batch_size=64)(delayed(one)(r, freqs) for r in recs)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} recordings; cross-spectra ok {int(df.xspec_ok.sum())}; "
          f"GCr L median {df.GCr_L.median():.3f} R {df.GCr_R.median():.3f}", file=sys.stderr)
