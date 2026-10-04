#!/usr/bin/env python3
"""Port validation: the authors' synthetic binary-classification experiment (Logistic.ipynb; paper Sec. 4.1.2 and
Appendix Table 1) re-run with our JAX port, using the notebook's settings. x ~ Beta(1/2,1/2), rho(x)=sigmoid(-5+10x),
training points with x in (0.3, 0.7) removed; embedding g = one linear layer to 8 dimensions trained with a bias-free
head (BCE, Adam 1e-3, 100 epochs, batch 32); VIDS with J=30, m=20, n=500, K=30, tau=0.001, lambda=0.005, Adam 1e-3,
posterior network 512-256-128-64-32 trained from a random initialisation, no gradient clipping, 1000 posterior draws."""
import numpy as np, pandas as pd
import sys
import nn_jax as N
KERAS = len(sys.argv) > 1 and sys.argv[1] == "keras"

def sig(z): return 1 / (1 + np.exp(-z))
rows = []
for rep in range(10):
    rng = np.random.default_rng(100 + rep)
    X = rng.beta(0.5, 0.5, 500).reshape(-1, 1).astype(np.float32)
    Xn = rng.beta(0.5, 0.5, 500).reshape(-1, 1).astype(np.float32)
    keep = ((X < 0.3) | (X > 0.7)).ravel(); X = X[keep]
    rho, rhon = sig(-5 + 10 * X.ravel()), sig(-5 + 10 * Xn.ravel())
    Y = rng.binomial(1, rho).astype(np.float32); Yn = rng.binomial(1, rhon).astype(np.float32)
    mlp = N.fit_mlp(X, Y, X, Y, seed=rep, hidden=(), rep=8, lr=1e-3, wd=0.0, epochs=100, batch=32, patience=1000)
    vm = N.fit_vids(mlp, X, Y, X, Y, seed=rep, J=30, m=20, n=500, K=30, tau=0.001, lam=0.005, lr=1e-3,
                    widths=(512, 256, 128, 64, 32), init_from_pretrained=False, sigma_init=0.69, n_samples=1000, keras_init=KERAS,
                    checkpoint="none", clip=None)
    for name, out in (("MLP", mlp.predict(Xn)), ("VIDS", vm.predict(Xn))):
        p = out["p"]
        # calibration error against the true probabilities, 10 equal-width bins of the forecast
        bins = np.clip((p * 10).astype(int), 0, 9)
        ace = sum((bins == b).mean() * abs(p[bins == b].mean() - rhon[bins == b].mean()) for b in range(10) if (bins == b).any())
        gap = (Xn.ravel() > 0.3) & (Xn.ravel() < 0.7)
        u = out["u_epi"]
        rows.append(dict(rep=rep, model=name, acc=float(((p > 0.5) == Yn).mean()), ace_binned=float(ace),
                         mae_true=float(np.abs(p - rhon).mean()),
                         nll=float(-np.mean(Yn * np.log(np.clip(p, 1e-8, 1)) + (1 - Yn) * np.log(np.clip(1 - p, 1e-8, 1)))),
                         sd_gap=float(u[gap].mean()), sd_edges=float(u[~gap].mean())))
    print(rows[-2], rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(f"../results/vids_audit/toy_logistic{'_keras' if KERAS else ''}.csv", index=False)
print(df.groupby("model").agg(["mean", "std"]).round(4).T.to_string())
