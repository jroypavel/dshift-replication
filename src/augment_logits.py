#!/usr/bin/env python3
"""Add Platt-recalibrated logits (and the five-ratio Altman logit) to existing per-seed prediction pickles,
so that the classic benchmarks get the same validation-year recalibration as the ML models.
Logits are deterministic, so the same predictions are written into every seed file."""
import argparse, glob, os, pickle
import numpy as np
import common as C
from run_experiments import prep, logit_models

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--panel", default="../results/panel.parquet")
ap.add_argument("--label", default="y_perf")
ap.add_argument("--include-financials", action="store_true")
ap.add_argument("--cs-rank", action="store_true")
ap.add_argument("--train-end", type=int, default=2017)
a = ap.parse_args()
df, _ = C.load_panel(a.panel, exclude_financials=not a.include_financials, label=a.label, cs_rank=a.cs_rank)
tr, va, te = C.split_fixed(df, a.train_end, a.train_end + 1, a.train_end + 2)
Xtr, Xva, Xte = prep(tr, va, te)
ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
# sanity: the stored test index must match
ti = __import__("pandas").read_parquet(os.path.join(a.res, "fixed", "test_index.parquet"))
assert len(ti) == len(te) and (ti["permno"].to_numpy() == te["permno"].to_numpy()).all()
mods = logit_models(Xtr, ytr, Xva, yva)
new = {}
for name, m in mods.items():
    po, pv = m.predict(Xte), m.predict(Xva)
    z = np.zeros_like(po["p"]); zv = np.zeros_like(pv["p"])
    new[name] = dict(p=po["p"].astype(np.float32), u_epi=z.astype(np.float32), u_logit=z.astype(np.float32),
                     u_tot=po["u_tot"].astype(np.float32), p_val=pv["p"].astype(np.float32),
                     u_val=zv.astype(np.float32), u_val_logit=zv.astype(np.float32))
files = sorted(glob.glob(os.path.join(a.res, "fixed", "seed_*.pkl")))
for f in files:
    d = pickle.load(open(f, "rb"))
    for name, v in new.items():
        if name in d:  # existing deterministic logits must coincide
            assert np.allclose(d[name]["p"], v["p"], atol=1e-6), name
        else:
            d[name] = v
    pickle.dump(d, open(f, "wb"))
print(a.res, "augmented", len(files), "files with", [k for k in new])
