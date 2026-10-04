#!/usr/bin/env python3
"""Extra paired comparisons (firm-clustered bootstrap, 1,000 draws, same construction as evaluate_r2)."""
import numpy as np, pandas as pd
import evaluate_r2 as E
from names_r2 import BASES_PLATT
R = "../results"
idx, vidx, P = E.load_dir(f"{R}/main_r2", ["fixed"])
P = E.merge_extra(P, idx, [f"{R}/iwacct", f"{R}/iw"])
P = E.add_platt(P, idx, ["mlp", "vids", "vids_authors"])
y = idx["y"].to_numpy().astype(float); firm = idx["permno"].to_numpy()
rng = np.random.default_rng(1)
uniq, inv = np.unique(firm, return_inverse=True)
members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
draws = [np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))]) for _ in range(1000)]
def perobs(name):
    p = np.clip(P[name]["p"], 1e-6, 1 - 1e-6); return ((y[None, :] - p) ** 2).mean(0)
rows = []
for A, B in (("logit_all_iwacct_platt", "logit_all_platt"), ("logit_all_iw_platt", "logit_all_platt"), ("vids", "mlp"),
             ("vids_authors", "mlp"), ("vids_authors", "vids")):
    dl = perobs(A) - perobs(B); bs = np.array([dl[ix].mean() for ix in draws])
    rows.append(dict(A=A, B=B, diff=dl.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), p=min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
d = pd.DataFrame(rows); d.to_csv(f"{R}/main_r2/eval_r2/extra_sig.csv", index=False)
d[["diff", "lo", "hi"]] *= 1e3; print(d.round(4).to_string())
