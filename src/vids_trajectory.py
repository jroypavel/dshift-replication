#!/usr/bin/env python3
"""Audit of the VIDS port: validation (2018) log loss and test (2019-23) Brier along the training path,
for the pretrained-head initialisation and for the authors' from-scratch initialisation."""
import json, math, sys, time
from functools import partial
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
import common as C, nn_jax as N, metrics as M
from run_experiments import prep, ALL_IDX

df, _ = C.load_panel("../results/panel.parquet"); tr, va, te = C.split_fixed(df)
Xtr, Xva, Xte = prep(tr, va, te); ytr, yva, yte = tr.y.to_numpy(), va.y.to_numpy(), te.y.to_numpy()
Xa, Xv, Xt = Xtr[:, ALL_IDX], Xva[:, ALL_IDX], Xte[:, ALL_IDX]
cfg = json.load(open("../results/main_v2/best_configs.json")); cfg["mlp"]["hidden"] = tuple(cfg["mlp"]["hidden"])
seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
CHECK = [0, 10, 25, 50, 100, 150, 300, 500, 1000]

SETTINGS = {
    "tuned_pre":     dict(J=16, m=200, n=1000, lam=1e-4, tau=0.01, lr=3e-3, sigma_init=0.03, pre=True,  widths=(128, 64, 32)),
    "authors_pre":   dict(J=10, m=20, n=1000, lam=5e-3, tau=1e-3, lr=1e-3, sigma_init=0.03, pre=True,  widths=(128, 64, 32)),
    "authors_scr":   dict(J=10, m=20, n=1000, lam=5e-3, tau=1e-3, lr=1e-3, sigma_init=0.69, pre=False, widths=(256, 128, 64, 32)),
    "tuned_scr":     dict(J=16, m=200, n=1000, lam=1e-4, tau=0.01, lr=3e-3, sigma_init=0.03, pre=False, widths=(128, 64, 32)),
}

rows = []
for s in seeds:
    mlp = N.fit_mlp(Xa, ytr, Xv, yva, seed=s, epochs=150, **cfg["mlp"])
    pm = mlp.predict(Xt)["p"]; vm = mlp.predict(Xv)["p"]
    rows.append(dict(seed=s, setting="mlp", K=0, val=M.logloss(yva, vm), brier=M.brier(yte, pm), eo=pm.sum() / yte.sum()))
    G = jnp.asarray(mlp.embed(Xa)); y = jnp.asarray(ytr, dtype=jnp.float32); Nn, rep = G.shape
    for name, st in SETTINGS.items():
        t0 = time.time()
        key = jax.random.PRNGKey(10_000 + s)
        hp = N.init_mlp(key, [2 * rep, *st["widths"], 2 * rep], last_scale=0.1)
        Wl, bl = hp[-1]
        mu0 = jnp.asarray(mlp.w) if st["pre"] else jnp.zeros(rep)
        raw0 = jnp.full((rep,), math.log(math.expm1(st["sigma_init"])))
        if st["pre"]:
            Wl = jnp.zeros_like(Wl)
        hp[-1] = (Wl, jnp.concatenate([mu0, raw0]))
        opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(st["lr"])); ost = opt.init(hp)
        batched = jax.vmap(partial(N._vids_split_loss, adaptive=True), in_axes=(None, 0, 0, 0, 0, 0, None, None))
        J, m, n, lam, tau = st["J"], st["m"], st["n"], st["lam"], st["tau"]

        @jax.jit
        def step(hp, ost, itr, ite, eps):
            def loss(h):
                Ls = batched(h, G[itr], y[itr], G[ite], y[ite], eps, lam, rep)
                return jnp.mean(Ls) + tau * jnp.var(Ls)
            l, gr = jax.value_and_grad(loss)(hp)
            up, ost = opt.update(gr, ost, hp)
            return optax.apply_updates(hp, up), ost, l

        rng = np.random.default_rng(s)
        for k in range(max(CHECK) + 1):
            if k in CHECK:
                mod = N.VIDSModel(mlp, hp, G, rep, n_samples=100, seed=s)
                pv, pt = mod.predict(Xv)["p"], mod.predict(Xt)["p"]
                rows.append(dict(seed=s, setting=name, K=k, val=M.logloss(yva, pv), brier=M.brier(yte, pt), eo=pt.sum() / yte.sum(),
                                 corr_mlp=float(np.corrcoef(pt, pm)[0, 1])))
            if k == max(CHECK):
                break
            itr = rng.integers(0, Nn, (J, n)); ite = rng.integers(0, Nn, (J, m))
            eps = rng.standard_normal((J, m, rep)).astype(np.float32)
            hp, ost, l = step(hp, ost, itr, ite, eps)
        print(f"seed {s} {name} done in {time.time()-t0:.0f}s", flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"../results/vids_audit/trajectory_{'_'.join(map(str, seeds))}.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(5).to_string())
