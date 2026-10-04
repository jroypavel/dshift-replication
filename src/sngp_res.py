#!/usr/bin/env python3
"""Residual SNGP (closer to Liu et al. 2020): tuned on the 2018 validation year with 8 distinct random
configurations, then fitted with 10 seeds; predictions saved in the same format as run_experiments.py."""
import argparse, json, os, pickle, itertools
import numpy as np
import common as C, nn_jax as N
from run_experiments import prep, ALL_IDX, log

GRID = dict(norm_c=[0.5, 0.95, 2.0], scale=[0.5, 1.0, 2.0], ridge=[0.1, 1.0, 10.0], width=[64, 128], n_blocks=[2, 3])
ap = argparse.ArgumentParser()
ap.add_argument("--panel", default="../results/panel.parquet")
ap.add_argument("--out", default="../results/sngp_res")
ap.add_argument("--trials", type=int, default=8)
ap.add_argument("--seeds", type=int, default=10)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "fixed"), exist_ok=True)
df, _ = C.load_panel(a.panel)
tr, va, te = C.split_fixed(df)
Xtr, Xva, Xte = prep(tr, va, te)
ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
Xa, Xv, Xt = Xtr[:, ALL_IDX], Xva[:, ALL_IDX], Xte[:, ALL_IDX]
cfgp = os.path.join(a.out, "config.json")
if not os.path.exists(cfgp):
    allc = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    rng = np.random.default_rng(0)
    cands = [allc[i] for i in rng.choice(len(allc), a.trials, replace=False)]
    res = []
    for c in cands:
        m = N.fit_sngp_res(Xa, ytr, Xv, yva, seed=0, **c); res.append((float(m.val_logloss), c)); log("  sngp_res", c, f"{m.val_logloss:.5f}")
    res.sort(key=lambda r: r[0])
    json.dump({"best": res[0][1], "best_val": res[0][0], "all": res}, open(cfgp, "w"), indent=1)
best = json.load(open(cfgp))["best"]; log("best", best)
for s in range(a.seeds):
    f = os.path.join(a.out, "fixed", f"seed_{s}.pkl")
    if os.path.exists(f):
        continue
    m = N.fit_sngp_res(Xa, ytr, Xv, yva, seed=s, **best)
    po, pv = m.predict(Xt), m.predict(Xv)
    out = {"sngp_res": dict(p=po["p"].astype(np.float32), u_epi=po["u_epi"].astype(np.float32), u_logit=po["u_logit"].astype(np.float32),
                            u_tot=po["u_tot"].astype(np.float32), p_val=pv["p"].astype(np.float32), u_val=pv["u_epi"].astype(np.float32),
                            u_val_logit=pv["u_logit"].astype(np.float32))}
    pickle.dump(out, open(f, "wb")); log("seed", s, "done")
te[["permno", "t", "y"]].to_parquet(os.path.join(a.out, "fixed", "test_index.parquet"))
log("ALL DONE")
