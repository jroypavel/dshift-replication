#!/usr/bin/env python3
"""Importance-weighted logits (Shimodaira 2000) for each test year: a LightGBM domain classifier separates
training firm-years (2011-2017) from the firms of forecast year T (covariates are known at the forecast date);
out-of-fold odds give density-ratio weights for the training sample (clipped at the 99th percentile and
normalised to mean one); weighted logits are refitted and optionally Platt-recalibrated on 2018."""
import os, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import common as C, baselines as B
from run_experiments import prep, ALL_IDX, CHS_IDX
from nn_jax import _entropy

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--features", default="all", choices=["all", "acct"]); a = ap.parse_args()
MARKET = ["nimta", "tlmta", "exret", "sigma", "rsize", "cashmta", "mb", "price", "me_tl", "ret12"]
from run_experiments import UNION
DOM_IDX = ALL_IDX if a.features == "all" else [i for i in ALL_IDX if UNION[i] not in MARKET]
suffix = "_iw" if a.features == "all" else "_iwacct"
out_dir = "../results/iw/fixed" if a.features == "all" else "../results/iwacct/fixed"; os.makedirs(out_dir, exist_ok=True)
df, _ = C.load_panel("../results/panel.parquet")
tr, va, te = C.split_fixed(df)
Xtr, Xva, Xte = prep(tr, va, te)
ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
yrs = te.t.to_numpy()
preds = {k: np.zeros(len(te)) for k in (f"logit_chs{suffix}", f"logit_chs{suffix}_platt", f"logit_all{suffix}", f"logit_all{suffix}_platt")}
pv = {k: np.zeros(len(va)) for k in preds}
ess = {}
for T in sorted(np.unique(yrs)):
    m = yrs == T
    X = np.vstack([Xtr[:, DOM_IDX], Xte[m][:, DOM_IDX]]); d = np.r_[np.zeros(len(Xtr)), np.ones(m.sum())]
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=50, verbose=-1, n_jobs=2)
    q = cross_val_predict(clf, X, d, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1][: len(Xtr)]
    w = (q / (1 - q)) * (len(Xtr) / m.sum())
    w = np.minimum(w, np.quantile(w, 0.99)); w = w / w.mean()
    ess[int(T)] = float(w.sum() ** 2 / (w ** 2).sum())
    for name, idx in ((f"logit_chs{suffix}", CHS_IDX), (f"logit_all{suffix}", ALL_IDX)):
        est = LogisticRegression(C=1e4, max_iter=3000).fit(Xtr[:, idx], ytr, sample_weight=w)
        mdl = B.SKModel(est, idx)
        preds[name][m] = mdl.predict(Xte[m])["p"]
        rc = B.Recal(mdl, "platt", Xva, yva)
        preds[name + "_platt"][m] = rc.predict(Xte[m])["p"]
        if T == 2019:
            pv[name] = mdl.predict(Xva)["p"]; pv[name + "_platt"] = rc.predict(Xva)["p"]
    print(T, "effective sample size of weighted training set:", round(ess[int(T)]), "of", len(Xtr), flush=True)
z = np.zeros(len(te), np.float32); zv = np.zeros(len(va), np.float32)
outp = {k: dict(p=v.astype(np.float32), u_epi=z, u_logit=z, u_tot=_entropy(v).astype(np.float32), p_val=pv[k].astype(np.float32),
                u_val=zv, u_val_logit=zv) for k, v in preds.items()}
pickle.dump(outp, open(os.path.join(out_dir, "seed_0.pkl"), "wb"))
te[["permno", "t", "y"]].to_parquet(os.path.join(out_dir, "test_index.parquet"))
va[["permno", "t", "y"]].to_parquet(os.path.join(out_dir, "val_index.parquet"))
pd.Series(ess).to_csv(os.path.join(os.path.dirname(out_dir), "ess_by_year.csv"))
print("saved")
