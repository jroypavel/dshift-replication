#!/usr/bin/env python3
"""Final VIDS runs for the revised paper (revision 2).

Main specification = the authors' algorithm (Slavutsky & Blei 2025, released code): the posterior network is trained from
a random Keras-default initialisation for K steps and the final state is used (no checkpoint selection). Architecture of
the posterior network follows the authors' binary real-data notebook (CelebA): widths 32/16/8/4 x embedding dimension,
J=10 environments, pseudo-training sets of n=1000. The remaining hyper-parameters are tuned on the validation year with the
same budget as every other model (8 random draws, validation log loss, base MLP of seed 0).

Usage:
  python3 vids_r2.py tune   --design fixed      --mlp-cfg ../results/main_v2/best_configs.json --out ../results/vids_r2_cfg_fixed.json [--ext 40]
  python3 vids_r2.py refit  --src ../results/main_v3 --dst ../results/main_r2 --cfg ../results/vids_r2_cfg_fixed.json --mode full
"""
import argparse, glob, json, os, pickle, shutil, time
import numpy as np, pandas as pd
import common as C, nn_jax as N
from run_experiments import prep, ALL_IDX, sample_grid, log

GRID = dict(K=[50, 150, 500, 1000], lam=[1e-4, 1e-3, 5e-3, 1e-2], tau=[0.0, 1e-3, 1e-2], m=[20, 200], lr=[1e-3, 3e-3])
FIXED = dict(J=10, n=1000, widths=(256, 128, 64, 32), keras_init=True, init_from_pretrained=False, checkpoint="none")
AUTHORS = dict(K=50, lam=5e-3, tau=1e-3, m=20, lr=1e-3)        # CelebA notebook (binary classification on real data)


def windows_for(a, df):
    if a.design == "fixed":
        return [("fixed", *C.split_fixed(df, a.train_end, a.train_end + 1, a.train_end + 2))]
    return [(f"T{T}", df[df.t <= T - 2], df[df.t == T - 1], df[df.t == T]) for T in range(2019, int(df.t.max()) + 1)]


def load(a):
    df, _ = C.load_panel(a.panel, exclude_financials=not a.include_financials, label=a.label, cs_rank=a.cs_rank)
    return df


def mlp_cfg(path):
    cfg = json.load(open(path)); cfg["mlp"]["hidden"] = tuple(cfg["mlp"]["hidden"]); return cfg["mlp"]


def fit(base, Xa, ytr, Xv, yva, seed, hp, **over):
    kw = {**FIXED, **hp, **over}
    return N.fit_vids(base, Xa, ytr, Xv, yva, seed=seed, **kw)


def tune(a):
    df = load(a); (w, tr, va, te), = windows_for(a, df)
    Xtr, Xva, _ = prep(tr, va, te); ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
    Xa, Xv = Xtr[:, ALL_IDX], Xva[:, ALL_IDX]
    base = N.fit_mlp(Xa, ytr, Xv, yva, seed=0, epochs=150, **mlp_cfg(a.mlp_cfg))
    out = {"grid": {k: list(v) for k, v in GRID.items()}, "fixed": {k: v for k, v in FIXED.items()}, "authors": AUTHORS,
           "base_mlp_val": float(base.val_logloss)}
    for tag, trials, rs in (("main", 8, 0), ("ext", a.ext, 1)):
        if trials == 0:
            continue
        res = []
        for c in sample_grid(np.random.default_rng(rs), GRID, trials):
            t0 = time.time(); m = fit(base, Xa, ytr, Xv, yva, 0, c)
            res.append((float(m.val_logloss), c)); log(f"  {tag} {c} -> val {m.val_logloss:.5f} ({time.time()-t0:.0f}s)")
        res.sort(key=lambda r: r[0] if np.isfinite(r[0]) else 9e9)
        out[tag] = {"best": res[0][1], "best_val": res[0][0], "all": res,
                    "distinct": len({json.dumps(c, sort_keys=True) for _, c in res})}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    log("tuned:", out["main"]["best"], "ext:", out.get("ext", {}).get("best"))


def pack(mod, XT, XV):
    po, pv = mod.predict(XT), mod.predict(XV)
    f = lambda v: np.asarray(v).astype(np.float32)
    return dict(p=f(po["p"]), u_epi=f(po["u_epi"]), u_logit=f(po["u_logit"]), u_tot=f(po["u_tot"]),
                p_val=f(pv["p"]), u_val=f(pv["u_epi"]), u_val_logit=f(pv["u_logit"]))


def refit(a):
    cfgj = json.load(open(a.cfg)); hp = cfgj["main"]["best"]; ext = cfgj.get("ext", {}).get("best")
    mc = mlp_cfg(a.mlp_cfg)
    df = load(a)
    diag = []
    for wname, tr, va, te in windows_for(a, df):
        sd, dd = os.path.join(a.src, wname), os.path.join(a.dst, wname); os.makedirs(dd, exist_ok=True)
        for fn in ("test_index.parquet", "val_index.parquet"):
            if os.path.exists(os.path.join(sd, fn)):
                shutil.copy(os.path.join(sd, fn), os.path.join(dd, fn))
        Xtr, Xva, Xte = prep(tr, va, te); ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
        Xa, Xv, Xt = Xtr[:, ALL_IDX], Xva[:, ALL_IDX], Xte[:, ALL_IDX]
        files = sorted(glob.glob(os.path.join(sd, "seed_*.pkl")), key=lambda f: int(os.path.basename(f)[5:-4]))
        for f in files[:a.seeds]:
            s = int(os.path.basename(f)[5:-4]); out_f = os.path.join(dd, os.path.basename(f))
            if os.path.exists(out_f):
                continue
            t0 = time.time()
            d = {k: v for k, v in pickle.load(open(f, "rb")).items() if not k.startswith("vids") and not k.startswith("_")}
            mlp = N.fit_mlp(Xa, ytr, Xv, yva, seed=s, epochs=150, **mc)
            assert np.allclose(mlp.predict(Xt)["p"], d["mlp"]["p"], atol=1e-5), f"base MLP not reproduced ({wname} seed {s})"
            specs = {"vids": ({}, hp)}
            if a.mode == "full":
                specs["vids_authors"] = ({}, AUTHORS)
                specs["vids_pre"] = (dict(init_from_pretrained=True, keras_init=False, sigma_init=0.03), hp)
                for tag, lam in (("1e4", 1e-4), ("1e3", 1e-3), ("5e3", 5e-3), ("1e2", 1e-2), ("5e2", 5e-2)):
                    if lam != hp["lam"]:
                        specs[f"vids_lam{tag}"] = ({}, {**hp, "lam": lam})
                specs["vids_fixedprior"] = (dict(adaptive=False), hp)
                specs["vids_tau0" if hp["tau"] != 0 else "vids_tau1e2"] = ({}, {**hp, "tau": 0.0 if hp["tau"] != 0 else 1e-2})
                specs["vids_malt"] = ({}, {**hp, "m": 20 if hp["m"] != 20 else 200})
                if ext is not None:
                    specs["vids_ext"] = ({}, ext)
            for name, (over, h) in specs.items():
                m = fit(mlp, Xa, ytr, Xv, yva, s, h, **over)
                d[name] = pack(m, Xt, Xv)
                diag.append(dict(window=wname, seed=s, model=name, val_start=float(m.val_start), val_final=float(m.val_logloss),
                                 finite=bool(np.isfinite(m.val_logloss))))
            d["_vids_port"] = "r2"
            pickle.dump(d, open(out_f, "wb"))
            log(f"{a.dst} {wname} seed {s}: {len(specs)} VIDS fits ({time.time()-t0:.0f}s)")
    p = os.path.join(a.dst, "vids_r2_diag.csv")
    pd.DataFrame(diag).to_csv(p, mode="a", index=False, header=not os.path.exists(p))
    log("ALL DONE", a.dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "refit"])
    ap.add_argument("--panel", default="../results/panel.parquet")
    ap.add_argument("--design", default="fixed", choices=["fixed", "rolling"])
    ap.add_argument("--train-end", type=int, default=2017)
    ap.add_argument("--label", default="y_perf")
    ap.add_argument("--include-financials", action="store_true")
    ap.add_argument("--cs-rank", action="store_true")
    ap.add_argument("--mlp-cfg", default="../results/main_v2/best_configs.json")
    ap.add_argument("--out"); ap.add_argument("--ext", type=int, default=0)
    ap.add_argument("--src"); ap.add_argument("--dst"); ap.add_argument("--cfg")
    ap.add_argument("--mode", default="lean", choices=["full", "lean"])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    tune(a) if a.cmd == "tune" else refit(a)
