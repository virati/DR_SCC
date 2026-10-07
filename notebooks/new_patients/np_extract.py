# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==1.26.4", "scipy==1.11.4", "pandas==2.1.4", "joblib==1.3.2", "python-dotenv==1.0.1",
#                 "pycatch22==0.4.5", "fooof==1.1.0", "matplotlib==3.8.4"]
# ///
"""Features for the at-home recordings of all nine patients and the in-clinic files of the new ones (PROTOCOL.md, Features).
    DATA_DIRECTORY=... uv run notebooks/new_patients/np_extract.py
Outputs in $DATA_DIRECTORY/intermed/new_patients/: home.csv.gz, sessions_new.csv.gz, session_files.csv"""
import glob
import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pycatch22
import scipy.signal as sig
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "weekly_dense"))
import wd_extract as WD                                       # in-clinic functions, unchanged

DATA = Path(os.environ["DATA_DIRECTORY"])
OUT = DATA / "intermed/new_patients"
FS, N = 422, 4220
OLD, NEW = ["901", "903", "905", "906", "907", "908"], ["910", "912", "913"]
WELCH = WD.WELCH
FW = np.linspace(0, 211, 513)
BP = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta*": (13, 20), "Gamma1": (35, 60)}   # dbspace e014944 FEAT_DICT
bpf = lambda lo, hi: sig.butter(3, [lo, hi], btype="band", fs=FS, output="sos")
SOS = {"Theta": bpf(4, 8), "Alpha": bpf(8, 13), "Gamma1": bpf(35, 60), "dPh": bpf(1.5, 3.0), "dAm": bpf(20, 35)}
LP = sig.butter(4, 50, btype="low", fs=FS, output="sos")
STAMP = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})__MR")


def band_power(psd):
    lp = 10 * np.log10(psd)
    r = 10 ** ((lp - np.poly1d(np.polyfit(FW, lp, 5))(FW)) / 10)
    return [10 * np.log10(np.median(r[(FW > lo) & (FW < hi)])) for lo, hi in BP.values()]


def device(psd):
    line = np.log10(psd[(FW >= 129) & (FW <= 131)].mean() / psd[((FW >= 120) & (FW <= 127)) | ((FW >= 133) & (FW <= 140))].mean())
    return line, np.log10(psd[np.argmin(np.abs(FW - 66))]) - np.log10(psd[np.argmin(np.abs(FW - 64))])


def tort(ph, am, nbins=18):
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    m = np.array([am[(ph >= edges[k]) & (ph < edges[k + 1])].mean() for k in range(nbins)])
    p = m / m.sum()
    return float((np.log(nbins) + np.sum(p * np.log(p + 1e-12))) / np.log(nbins))


def montage(path):
    try:
        t = open(path[:-4] + ".xml", encoding="utf-8-sig", errors="ignore").read()
        ch = re.findall(r"<Channel(\d)>.*?<ChannelType>(\w+)</ChannelType>.*?<PlusInput>(\w+)</PlusInput>.*?<MinusInput>(\w+)</MinusInput>", t, re.S)
        return ";".join(f"{p}-{m}" for _, ty, p, m in ch if ty == "TD")
    except Exception:
        return ""


def home(pt, week, circ, path):
    g = STAMP.search(path).groups()
    base = {"pt": pt, "week": week, "circ": circ, "file": os.path.basename(path), "time": "%s-%s-%s %s:%s:%s" % g}
    try:
        X = pd.read_csv(path, sep=",", header=None).values
        if len(X) < N:
            return {**base, "err": "short"}
        x, y = X[-N:, 0].astype(float), X[-N:, 2].astype(float)
        row = dict(base, montage=montage(path) if pt in NEW else "")
        P = {}
        for side, v in (("L", x), ("R", y)):
            f, p = sig.welch(v[:-1], **WELCH)
            P[side] = p
            row.update({f"bp_{side}{b}": val for b, val in zip(BP, band_power(p))})
            row[f"line_{side}"], row[f"gcr_{side}"] = device(p)
            ff = WD.fooof_feats(f, p)
            ok = ff is not None and ff["r2"] >= 0.5
            for k in ["offset", "exponent", *WD.FB]:
                row[f"fo_{side}_{k}"] = ff[k] if ok else np.nan
        _, pxy = sig.csd(x[:-1], y[:-1], **WELCH)
        for b, (lo, hi) in BP.items():
            s = (FW >= lo) & (FW < hi)
            c = pxy[s].mean() / np.sqrt(P["L"][s].mean() * P["R"][s].mean())
            row[f"coh_{b}"], row[f"cos_{b}"], row[f"sin_{b}"] = abs(c), np.cos(np.angle(c)), np.sin(np.angle(c))
        an = {(side, b): sig.hilbert(sig.sosfiltfilt(SOS[b], v)) for side, v in (("L", x), ("R", y)) for b in SOS}
        for pb in ("Theta", "Alpha"):
            for ps, as_ in (("L", "L"), ("R", "R"), ("L", "R"), ("R", "L")):
                row[f"pac_{pb}{ps}_Gamma1{as_}"] = tort(np.angle(an[(ps, pb)]), np.abs(an[(as_, "Gamma1")]))
        for side in "LR":
            row[f"pac_delta_{side}"] = tort(np.angle(an[(side, "dPh")]), np.abs(an[(side, "dAm")]))
        for side, v in (("L", x), ("R", y)):
            r = pycatch22.catch22_all(list(sig.sosfiltfilt(LP, v - v.mean())))
            row.update({f"c22_{side}_{n}": val for n, val in zip(r["names"], r["values"])})
        return row
    except Exception as e:
        return {**base, "err": f"{type(e).__name__}: {e}"[:120]}


def session(pt, path):
    g = STAMP.search(path).groups()
    rows, _, info = WD.process(pt, "", -1, path, 1)
    t = "%s-%s-%s %s:%s:%s" % g
    return [dict(r, time=t) for r in rows], dict(info, time=t, montage=montage(path))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pickle.load(open(DATA / "intermed/chronic/Chronic_FrameFeb2026_plain.pkl", "rb"))
    jobs = [(r["Patient"], r["Phase"], r["Circadian"], r["Filename"]) for r in frame["file_meta"] if r["Patient"] in OLD]
    del frame
    dense = []
    for pt in NEW:
        for f in sorted(glob.glob(str(OUT / "raw" / f"DBS{pt}" / "**" / "*__MR_*.txt"), recursive=True)):
            sz = os.path.getsize(f)
            if sz > 1e7:
                dense.append((pt, f))
            elif sz < 1e6:
                jobs.append((pt, "", "", f))
    J = int(os.environ.get("JOBS", 14))
    H = pd.DataFrame(Parallel(n_jobs=J, batch_size=32)(delayed(home)(*j) for j in jobs))
    H.to_csv(OUT / "home.csv.gz", index=False)
    res = Parallel(n_jobs=min(J, 8))(delayed(session)(*d) for d in dense)
    pd.DataFrame([r for rr, _ in res for r in rr]).to_csv(OUT / "sessions_new.csv.gz", index=False)
    pd.DataFrame([i for _, i in res]).to_csv(OUT / "session_files.csv", index=False)
    print("home", H.shape, "errors", H.get("err", pd.Series(dtype=object)).value_counts().to_dict(), H.groupby("pt").size().to_dict(),
          "| session segments", sum(len(rr) for rr, _ in res), "from", len(dense), "files", file=sys.stderr)
