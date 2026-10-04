#!/usr/bin/env python3
"""Number of distinct hyper-parameter configurations among the 8 random draws per model (appendix table of selected settings).
Replays the seeded draws of run_experiments.tune() (default 8 trials, generator seed 0), so no model fitting is needed.
Run from src_r2/:  python3 tuning_distinct.py   ->  ../results/diagnostics_v2/tuning_distinct.csv"""
import json, os
import numpy as np
import pandas as pd

# Mirrors GRIDS and sample_grid() in ../src/run_experiments.py (copied so that this script needs only numpy and pandas).
GRIDS = {
    "lgbm": dict(num_leaves=[4, 8, 16, 31], learning_rate=[0.02, 0.05], min_child_samples=[20, 50, 100],
                 reg_lambda=[1.0, 10.0], colsample_bytree=[0.6, 0.8, 1.0]),
    "rf": dict(min_samples_leaf=[5, 10, 20, 50], max_features=["sqrt", 0.3]),
    "mlp": dict(hidden=[(32, 32), (64, 64), (128, 64)], rep=[8, 16, 32], lr=[1e-3, 3e-3], wd=[1e-5, 1e-4, 1e-3]),
    "mcd": dict(drop_p=[0.1, 0.2, 0.3]),
    "sngp": dict(norm_c=[1.0, 2.0, 4.0], scale=[0.5, 1.0, 2.0], ridge=[0.1, 1.0, 10.0]),
    "llla": dict(prior_prec=[0.1, 1.0, 10.0, 100.0]),
    "vids": dict(m=[50, 200, 500], lam=[1e-5, 1e-4, 5e-4, 1e-3, 5e-3], tau=[0.0, 0.001, 0.01], lr=[1e-3, 3e-3],
                 sigma_init=[0.003, 0.01, 0.03]),
}


def sample_grid(rng, grid, trials):
    return [{k: v[rng.integers(len(v))] for k, v in grid.items()} for _ in range(trials)]


TRIALS = 8
rng = np.random.default_rng(0)
rows = []
for name in ("lgbm", "rf", "mlp", "mcd", "sngp", "llla", "vids"):       # same order as tune()
    cfgs = sample_grid(rng, GRIDS[name], TRIALS)
    rows.append((name, len({json.dumps(c, sort_keys=True, default=str) for c in cfgs})))
os.makedirs("../results/diagnostics_v2", exist_ok=True)
pd.DataFrame(rows, columns=["model", "distinct"]).to_csv("../results/diagnostics_v2/tuning_distinct.csv", index=False)
print(rows)
