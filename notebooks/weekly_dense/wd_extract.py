# /// script
# requires-python = ">=3.8,<3.9"
# dependencies = ["numpy==1.24.4", "scipy==1.10.1", "pandas==2.0.3", "joblib==1.3.2", "python-dotenv==1.0.1",
#                 "fooof==1.1.0", "matplotlib==3.7.5"]
# ///
"""Weekly in-clinic sessions: stimulation on/off segmentation and per-segment features (PROTOCOL.md, preprocessing).
    uv run notebooks/weekly_dense/wd_extract.py
Outputs in $DATA_DIRECTORY/intermed/weekly_dense/: segments.csv.gz (one row per clean 10-s segment) and
segments_psd.npz (Welch PSDs per segment, for the DR-SCC paper pipeline in wd_drscc.py)."""
import datetime
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sig
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed
from scipy.signal.windows import dpss

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv(usecwd=True))
DATA = Path(os.environ["DATA_DIRECTORY"])
OUT = DATA / "intermed" / "weekly_dense"
PTS = ["905", "906", "907", "908"]
WEEKS = ["B0" + str(i) for i in range(1, 5)] + ["C%02d" % i for i in range(1, 25)]
FS, SEG = 422, 10
PB = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "lbeta": (13, 20), "hbeta": (20, 30), "gamma": (30, 40)}
FB = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 14), "Beta*": (14, 20), "Gamma1": (35, 50)}
WELCH = dict(fs=FS, window="blackmanharris", nperseg=512, noverlap=128, nfft=1024)
TAPERS = dpss(FS * SEG, 4, 7)
SOS_PH = sig.butter(3, [1.5, 3.0], btype="band", fs=FS, output="sos")
SOS_AM = sig.butter(3, [20, 35], btype="band", fs=FS, output="sos")


def stim_windows(X, win=2):
    on = None
    for ch in (0, 2):
        f, t, S = sig.spectrogram(X[:, ch], fs=FS, nperseg=FS * win, noverlap=0)
        line = S[(f >= 129) & (f <= 131)].mean(0)
        nb = S[((f >= 120) & (f <= 127)) | ((f >= 133) & (f <= 140))].mean(0)
        m = sig.medfilt((np.log10(line / nb) > 1.0).astype(float), 5) > 0.5
        on = m if on is None else (on | m)
    return on, win


def runs(on, win, n):
    """Sample ranges for each state with the protocol's guards."""
    out, start = [], 0
    for i in range(1, len(on) + 1):
        if i == len(on) or on[i] != on[start]:
            a, b = start * win * FS, min(i * win * FS, n)
            state = "on" if on[start] else "off"
            prev_on = start > 0 and on[start - 1]
            lead = 10 if (state == "off" and prev_on) else (4 if start > 0 else 2)
            trail = 4 if i < len(on) else 0
            a, b = a + lead * FS, b - trail * FS
            if b - a >= SEG * FS:
                out.append((state, a, b))
            start = i
    return out


def tort_mi(x, nbins=18):
    ph = np.angle(sig.hilbert(sig.sosfiltfilt(SOS_PH, x)))
    am = np.abs(sig.hilbert(sig.sosfiltfilt(SOS_AM, x)))
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    m = np.array([am[(ph >= edges[k]) & (ph < edges[k + 1])].mean() for k in range(nbins)])
    p = m / m.sum()
    return float((np.log(nbins) + np.sum(p * np.log(p + 1e-12))) / np.log(nbins))


def fooof_feats(f, psd):
    from fooof import FOOOF
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1, aperiodic_mode="fixed", verbose=False)
    try:
        fm.fit(f, psd, [1, 55])
    except Exception:
        return None
    if not fm.has_model:
        return None
    out = {"offset": fm.aperiodic_params_[0], "exponent": fm.aperiodic_params_[1], "r2": fm.r_squared_}
    for b, (lo, hi) in FB.items():
        out[b] = float(np.mean(fm._spectrum_flat[(fm.freqs >= lo) & (fm.freqs <= hi)]))
    return out


def seg_features(x, y):
    row = {}
    freqs = np.fft.rfftfreq(FS * SEG, 1 / FS)
    Xk, Yk = np.fft.rfft(TAPERS * (x - x.mean())), np.fft.rfft(TAPERS * (y - y.mean()))
    Sxx, Syy, Sxy = (np.abs(Xk) ** 2).mean(0) / FS, (np.abs(Yk) ** 2).mean(0) / FS, (Xk * np.conj(Yk)).mean(0) / FS
    coh = np.abs(Sxy) ** 2 / (Sxx * Syy)
    for b, (lo, hi) in PB.items():
        sel = (freqs >= lo) & (freqs < hi)
        row[f"pw_L_{b}"], row[f"pw_R_{b}"] = np.log10(Sxx[sel].mean()), np.log10(Syy[sel].mean())
        row[f"coh_{b}"] = coh[sel].mean()
    row["pac_L"], row["pac_R"] = tort_mi(x), tort_mi(y)
    for side, s, S in (("L", x, Sxx), ("R", y, Syy)):
        row[f"nz_{side}_100_120"] = np.log10(S[(freqs >= 100) & (freqs < 120)].mean())
        row[f"nz_{side}_58_62"] = np.log10(S[(freqs >= 58) & (freqs < 62)].mean())
    psds = []
    ok = True
    for side, s in (("L", x), ("R", y)):
        f, p = sig.welch(s[:-1], **WELCH)  # 4219 samples, as the chronic frame
        psds.append(p)
        ff = fooof_feats(f, p)
        ok = ok and ff is not None and ff["r2"] >= 0.5
        for k, v in (ff or {k: np.nan for k in ["offset", "exponent", "r2", *FB]}).items():
            row[f"fo_{side}_{k}"] = v
    row["fooof_ok"] = ok
    return row, np.stack(psds)


def process(pt, week, t, f, n_files):
    X = pd.read_csv(f, header=None).values.astype(float)
    on, win = stim_windows(X)
    rows, psds = [], []
    rr = runs(on, win, len(X))
    for state in ("off", "on"):
        segs = [(a + k * SEG * FS) for st, a, b in rr if st == state for k in range((b - a) // (SEG * FS))]
        if not segs:
            continue
        allx = np.concatenate([X[a:a + SEG * FS, 0] for a in segs]), np.concatenate([X[a:a + SEG * FS, 2] for a in segs])
        thr = [10 * 1.4826 * np.median(np.abs(v - np.median(v))) for v in allx]
        med = [np.median(v) for v in allx]
        for i, a in enumerate(segs):
            x, y = X[a:a + SEG * FS, 0], X[a:a + SEG * FS, 2]
            if np.max(np.abs(x - med[0])) > thr[0] or np.max(np.abs(y - med[1])) > thr[1]:
                continue
            feats, p = seg_features(x, y)
            rows.append({"pt": pt, "week": week, "t": t, "file": os.path.basename(f), "n_files_week": n_files,
                         "state": state, "start_s": a / FS, **feats})
            psds.append(p)
    return rows, psds, {"pt": pt, "week": week, "file": os.path.basename(f), "minutes": len(X) / FS / 60,
                        "on_s": int(on.sum() * win), "off_s": int((~on).sum() * win)}


if __name__ == "__main__":
    clin = {p["pt"][3:]: p for p in json.load(open(DATA / "clinical/clinical_vectors_all.json"))["HAMDs"]}
    jobs = []
    for pt in PTS:
        files = []
        for f in glob.glob(str(DATA / "neural/lfp" / pt / "**" / "*__MR_*.txt"), recursive=True):
            if os.path.getsize(f) > 1e7:
                p = os.path.basename(f).split("_")
                files.append((datetime.date(int(p[-9]), int(p[-8]), int(p[-7])), f))
        for t, w in enumerate(WEEKS):
            d = datetime.datetime.strptime(clin[pt]["dates"][clin[pt]["phases"].index(w)], "%m/%d/%Y").date()
            hit = sorted(f for dt, f in files if abs((dt - d).days) <= 1)
            jobs += [(pt, w, t, f, len(hit)) for f in hit]
    res = Parallel(n_jobs=int(os.environ.get("JOBS", 8)))(delayed(process)(*j) for j in jobs)
    rows = [r for rr, _, _ in res for r in rr]
    psd = np.stack([p for _, pp, _ in res for p in pp])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "segments.csv.gz", index=False)
    np.savez_compressed(OUT / "segments_psd.npz", psd=psd.astype(np.float64))
    pd.DataFrame([s for _, _, s in res]).to_csv(OUT / "sessions.csv", index=False)
    df = pd.DataFrame(rows)
    print(f"{len(jobs)} files, {len(df)} clean segments (off {int((df.state=='off').sum())}, on {int((df.state=='on').sum())}); "
          f"FOOOF ok {df.fooof_ok.mean():.3f}", file=sys.stderr)
