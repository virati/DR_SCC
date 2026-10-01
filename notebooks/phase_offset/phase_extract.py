# /// script
# requires-python = ">=3.8,<3.9"
# dependencies = ["numpy==1.24.4", "scipy==1.10.1", "pandas==2.0.3", "joblib==1.3.2", "python-dotenv==1.0.1"]
# ///
"""Left-right band coherency (complex) for every at-home 10-s recording and every weekly-session 10-s segment
(phase_offset PROTOCOL.md, "Quantities").   uv run notebooks/phase_offset/phase_extract.py
Output: $DATA_DIRECTORY/intermed/phase_offset/units_phase.csv.gz"""
import glob
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
OUT = DATA / "intermed/phase_offset/units_phase.csv.gz"
PTS = ["901", "903", "905", "906", "907", "908"]
WEEKS = ["B0" + str(i) for i in range(1, 5)] + ["C%02d" % i for i in range(1, 25)]
T_OF = {w: i for i, w in enumerate(WEEKS)}
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 14), "beta": (14, 30), "bstar": (14, 20), "gamma": (30, 50), "gamma1": (35, 50)}
FS, N = 422, 4219
WELCH = dict(fs=FS, window="blackmanharris", nperseg=512, noverlap=128, nfft=1024)


def local_path(fname):
    p = Path(fname)
    if p.exists():
        return p
    parts = p.parts
    return DATA / "neural" / "lfp" / Path(*parts[parts.index("lfp") + 1:]) if "lfp" in parts else p


def unit(x, y):
    f, pxx = sig.welch(x, **WELCH)
    _, pyy = sig.welch(y, **WELCH)
    _, pxy = sig.csd(x, y, **WELCH)                      # conj(X) * Y: angle > 0 means right (y) leads left (x)
    _, pxs = sig.csd(x, np.roll(y, 5 * FS), **WELCH)     # surrogate: same spectra, phase locking destroyed
    row = {}
    for side, p in (("L", pxx), ("R", pyy)):
        line = p[(f >= 129) & (f <= 131)].mean()
        nb = p[((f >= 120) & (f <= 127)) | ((f >= 133) & (f <= 140))].mean()
        row[f"line_{side}"] = np.log10(line / nb)
        row[f"GCr_{side}"] = np.log10(p[np.argmin(np.abs(f - 66))]) - np.log10(p[np.argmin(np.abs(f - 64))])
    for b, (lo, hi) in BANDS.items():
        s = (f >= lo) & (f < hi)
        den = np.sqrt(pxx[s].mean() * pyy[s].mean())
        c, cs = pxy[s].mean() / den, pxs[s].mean() / den
        row[f"{b}_re"], row[f"{b}_im"], row[f"{b}_sre"], row[f"{b}_sim"] = c.real, c.imag, cs.real, cs.imag
    return row


def home(rec):
    base = {"source": "home", "pt": rec["Patient"], "week": rec["Phase"], "t": T_OF[rec["Phase"]], "circ": rec["Circadian"],
            "state": "", "file": os.path.basename(rec["Filename"])}
    try:
        X = pd.read_csv(local_path(rec["Filename"]), sep=",", header=None).values
        return {**base, **unit(X[-(FS * 10):-1, 0].astype(float), X[-(FS * 10):-1, 2].astype(float))}
    except Exception as e:
        return {**base, "err": f"{type(e).__name__}: {e}"[:100]}


def session(path, segs):
    X = pd.read_csv(path, header=None).values.astype(float)
    rows = []
    for r in segs.itertuples():
        a = int(round(r.start_s * FS))
        rows.append({"source": "session", "pt": r.pt, "week": r.week, "t": r.t, "circ": "day", "state": r.state, "file": r.file,
                     **unit(X[a:a + N, 0], X[a:a + N, 2])})
    return rows


if __name__ == "__main__":
    blob = FRAME.read_bytes()
    assert hashlib.md5(blob).hexdigest() == FRAME_MD5
    frame = pickle.loads(blob)
    del blob
    recs = [r for r in frame["file_meta"] if r["Patient"] in PTS and r["Phase"] in T_OF]
    J = int(os.environ.get("JOBS", 8))
    H = Parallel(n_jobs=J, batch_size=64)(delayed(home)(r) for r in recs)
    seg = pd.read_csv(DATA / "intermed/weekly_dense/segments.csv.gz", dtype={"pt": str, "week": str})
    n_off = seg[seg.state == "off"].groupby(["pt", "week", "file"]).size().rename("n_off").reset_index()
    seg = seg.merge(n_off, on=["pt", "week", "file"])
    seg = seg[(seg.n_files_week == 1) & (seg.n_off >= 6)]            # primary sessions, as weekly_dense
    paths = {os.path.basename(p): p for pt in seg.pt.unique() for p in glob.glob(str(DATA / "neural/lfp" / pt / "**" / "*__MR_*.txt"), recursive=True)}
    S = Parallel(n_jobs=J)(delayed(session)(paths[fn], g[["pt", "week", "t", "state", "file", "start_s"]]) for fn, g in seg.groupby("file"))
    df = pd.DataFrame(H + [r for part in S for r in part])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"home {len(H)} (errors {int(df.get('err', pd.Series(dtype=object)).notna().sum())}), session segments {len(df) - len(H)}", file=sys.stderr)
