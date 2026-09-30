"""POST-HOC (not pre-registered): baseline centering without scaling. See RESULTS.md, post-hoc section.
Pre-registered baseline z-scoring divides by sigma from 4 B-weeks; stimulation-on weeks then reach |z| ~ 95.
This checks whether subtracting the B-week mean alone changes the E1 and E3 N=0 picture."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(HERE))
import pandas as pd
from joblib import Parallel, delayed
import fc
rec = fc.build_recordings(fc.data_dir() / "intermed/forward_chain/recordings_features.csv.gz")
targets = fc.clinical_targets()
cond = ("F-mean", "center", "T-raw")
df, feats = fc.table(rec, targets, *cond)
e1 = pd.concat(Parallel(n_jobs=8)(delayed(fc.e1)(df, feats, [m]) for m in fc.MODELS))
e3 = pd.concat(Parallel(n_jobs=8)(delayed(fc.e3)(df, feats, [m], (0,)) for m in fc.MODELS if m != "M0_persistence"))
s1, s3 = fc.score(e1), fc.score(e3)
s1.to_csv(HERE / "outputs" / "posthoc_center_E1.csv", index=False)
s3.to_csv(HERE / "outputs" / "posthoc_center_E3_N0.csv", index=False)
print("E1\n", s1.round(3).to_string(index=False)); print("E3 N=0\n", s3.round(3).to_string(index=False))
