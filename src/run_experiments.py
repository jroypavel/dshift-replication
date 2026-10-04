#!/usr/bin/env python3
"""Run all models on the fixed (train<=2017 / val=2018 / test>=2019) or rolling design.
Outputs per-seed prediction pickles + best_configs.json into --out."""
import argparse, json, os, pickle, time, sys
import numpy as np
import pandas as pd
import common as C
import baselines as B
import nn_jax as N

UNION = list(dict.fromkeys(C.FEATURE_SETS["all"] + ["z_altman"]))
IDX = {c: i for i, c in enumerate(UNION)}
ALL_IDX = [IDX[c] for c in C.FEATURE_SETS["all"]]
OHL_IDX = [IDX[c] for c in C.FEATURE_SETS["ohlson"]]
CHS_IDX = [IDX[c] for c in C.FEATURE_SETS["chs"]]
ALT_IDX = [IDX["z_altman"]]
ALT5_IDX = [IDX[c] for c in ["wcta", "re_ta", "ebit_ta", "me_tl", "sales_ta"]]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def prep(tr, va, te):
    pp = C.Preprocessor(UNION).fit(tr)
    return pp.transform(tr), pp.transform(va), pp.transform(te)


# ------------------------------------------------------------------ hyper-parameter search (validation log-loss)
def sample_grid(rng, grid, trials):
    out = []
    for _ in range(trials):
        out.append({k: v[rng.integers(len(v))] for k, v in grid.items()})
    return out


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


def tune(Xtr, ytr, Xva, yva, trials, quick):
    rng = np.random.default_rng(0)
    best = {}
    Xa, Xv = Xtr[:, ALL_IDX], Xva[:, ALL_IDX]

    def search(name, fit):
        cfgs = sample_grid(rng, GRIDS[name], trials)
        res = []
        for c in cfgs:
            try:
                m = fit(c); res.append((m.val_logloss, c))
                log(f"  tune {name} {c} -> val logloss {m.val_logloss:.5f}")
            except Exception as e:
                log(f"  tune {name} {c} FAILED {e}")
        res.sort(key=lambda r: r[0])
        best[name] = res[0][1]; best[name + "_val"] = res[0][0]
        return best[name]

    ep = 8 if quick else 150
    search("lgbm", lambda c: B.fit_lgbm(Xa, ytr, Xv, yva, seed=0, **c))
    search("rf", lambda c: B.fit_rf(Xa, ytr, Xv, yva, seed=0, n_estimators=100 if quick else 400, **c))
    search("mlp", lambda c: N.fit_mlp(Xa, ytr, Xv, yva, seed=0, epochs=ep, **c))
    base = N.fit_mlp(Xa, ytr, Xv, yva, seed=0, epochs=ep, **best["mlp"])
    search("mcd", lambda c: N.fit_mlp(Xa, ytr, Xv, yva, seed=0, epochs=ep, **{**best["mlp"], **c}))
    search("sngp", lambda c: N.fit_sngp(Xa, ytr, Xv, yva, seed=0, epochs=ep, **c))
    search("llla", lambda c: N.fit_llla(base, Xa, Xv, yva, seed=0, **c))
    K = 60 if quick else 1000
    search("vids", lambda c: N.fit_vids(base, Xa, ytr, Xv, yva, seed=0, K=K, **c))
    return best


# ------------------------------------------------------------------ fit everything for one seed
def logit_models(Xtr, ytr, Xva, yva):
    """Classic logits, each also Platt-recalibrated on the validation year (same treatment as the ML models)."""
    out = {}
    for name, idx in (("logit_altman", ALT_IDX), ("logit_altman5", ALT5_IDX), ("logit_ohlson", OHL_IDX),
                      ("logit_chs", CHS_IDX), ("logit_all", ALL_IDX)):
        m = B.fit_logit(Xtr, ytr, Xva, yva, idx)
        out[name] = m
        out[name + "_platt"] = B.Recal(m, "platt", Xva, yva)
    return out


def fit_all(seed, Xtr, ytr, Xva, yva, Xte, cfg, quick, core=False):
    Xa, Xv, Xt = Xtr[:, ALL_IDX], Xva[:, ALL_IDX], Xte[:, ALL_IDX]
    ep = 8 if quick else 150
    K = 60 if quick else 1000
    preds = {}

    def rec(name, model, full=False):
        XT, XV = (Xte, Xva) if full else (Xt, Xv)
        po, pv = model.predict(XT), model.predict(XV)
        zero = np.zeros_like(po["p"])
        preds[name] = dict(p=po["p"].astype(np.float32), u_epi=po["u_epi"].astype(np.float32),
                           u_logit=po.get("u_logit", zero).astype(np.float32),
                           u_tot=po["u_tot"].astype(np.float32), p_val=pv["p"].astype(np.float32),
                           u_val=pv["u_epi"].astype(np.float32),
                           u_val_logit=pv.get("u_logit", np.zeros_like(pv["p"])).astype(np.float32))
        log(f"    seed {seed} {name}: done")

    # finance classics (deterministic in the data, so identical across seeds), raw and Platt-recalibrated
    for name, mdl in logit_models(Xtr, ytr, Xva, yva).items():
        rec(name, mdl, full=True)
    # tree ensembles
    rf = B.fit_rf(Xa, ytr, Xv, yva, seed=seed, n_estimators=100 if quick else 400, **cfg["rf"]); rec("rf", rf)
    rec("rf_platt", B.Recal(rf, "platt", Xv, yva))
    lg = B.fit_lgbm(Xa, ytr, Xv, yva, seed=seed, **cfg["lgbm"]); rec("lgbm", lg)
    rec("lgbm_platt", B.Recal(lg, "platt", Xv, yva)); rec("lgbm_iso", B.Recal(lg, "isotonic", Xv, yva))
    # neural nets
    mlp = N.fit_mlp(Xa, ytr, Xv, yva, seed=seed, epochs=ep, **cfg["mlp"]); rec("mlp", mlp)
    rec("mlp_ts", B.Recal(mlp, "temp", Xv, yva))
    ens = N.fit_ensemble(Xa, ytr, Xv, yva, seed=seed, n_members=5, epochs=ep, **cfg["mlp"]); rec("ens5", ens)
    rec("ens5_ts", B.Recal(ens, "temp", Xv, yva))
    mcd = N.fit_mlp(Xa, ytr, Xv, yva, seed=seed, epochs=ep, **{**cfg["mlp"], **cfg["mcd"]}); rec("mcd", mcd)
    rec("sngp", N.fit_sngp(Xa, ytr, Xv, yva, seed=seed, epochs=ep, **cfg["sngp"]))
    rec("llla", N.fit_llla(mlp, Xa, Xv, yva, seed=seed, **cfg["llla"]))
    v = cfg["vids"]
    vm = N.fit_vids(mlp, Xa, ytr, Xv, yva, seed=seed, K=K, **v); rec("vids", vm)
    rec("vids_ts", B.Recal(vm, "temp", Xv, yva))
    if core:
        return preds
    # the authors' settings for binary classification on real data (CelebA notebook): J=10, m=20, n=1000, K=50,
    # tau=0.001, lambda=0.005, lr=1e-3; posterior mean initialised at the pretrained head as in our main variant
    rec("vids_authors", N.fit_vids(mlp, Xa, ytr, Xv, yva, seed=seed, J=10, m=20, n=1000, K=50 if not quick else 10,
                                   tau=0.001, lam=0.005, lr=1e-3, sigma_init=v["sigma_init"]))
    # sensitivity to the KL (prior) weight, other settings as tuned
    for lam_tag, lam_val in (("1e3", 1e-3), ("5e3", 5e-3), ("1e2", 1e-2), ("5e2", 5e-2)):
        rec(f"vids_lam{lam_tag}", N.fit_vids(mlp, Xa, ytr, Xv, yva, seed=seed, K=K, **{**v, "lam": lam_val}))
    # ablations (same tuned values unless stated)
    rec("vids_strict", N.fit_vids(mlp, Xa, ytr, Xv, yva, seed=seed, K=K, init_from_pretrained=False, **v))
    rec("vids_fixedprior", N.fit_vids(mlp, Xa, ytr, Xv, yva, seed=seed, K=K, adaptive=False, **v))
    rec("vids_tau0", N.fit_vids(mlp, Xa, ytr, Xv, yva, seed=seed, K=K, **{**v, "tau": 0.0}))
    for mm in (50, 500):
        rec(f"vids_m{mm}", N.fit_vids(mlp, Xa, ytr, Xv, yva, seed=seed, K=K, **{**v, "m": mm}))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="../results/panel.parquet")
    ap.add_argument("--out", default="../results/main")
    ap.add_argument("--design", default="fixed", choices=["fixed", "rolling"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--label", default="y_perf")
    ap.add_argument("--include-financials", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cs-rank", action="store_true", help="cross-sectionally rank market-based features within year")
    ap.add_argument("--tuned", default=None, help="reuse best_configs.json from this path")
    ap.add_argument("--train-end", type=int, default=2017)
    ap.add_argument("--core", action="store_true", help="skip VIDS ablations/sensitivity variants")
    ap.add_argument("--first-test", type=int, default=2019, help="first test year of the rolling design")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, info = C.load_panel(a.panel, exclude_financials=not a.include_financials, label=a.label, cs_rank=a.cs_rank)
    log("sample:", info, "| positives:", int(df.y.sum()))

    if a.design == "fixed":
        windows = [("fixed", *C.split_fixed(df, a.train_end, a.train_end + 1, a.train_end + 2))]
    else:
        windows = []
        for T in range(a.first_test, int(df.t.max()) + 1):
            windows.append((f"T{T}", df[df.t <= T - 2], df[df.t == T - 1], df[df.t == T]))

    cfg_path = a.tuned or os.path.join(a.out, "best_configs.json")
    for wname, tr, va, te in windows:
        wdir = os.path.join(a.out, wname); os.makedirs(wdir, exist_ok=True)
        log(f"== window {wname}: train {len(tr)} ({int(tr.y.sum())} pos) val {len(va)} ({int(va.y.sum())}) test {len(te)} ({int(te.y.sum())})")
        Xtr, Xva, Xte = prep(tr, va, te)
        ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
        te[["permno", "t", "y"]].to_parquet(os.path.join(wdir, "test_index.parquet"))
        va[["permno", "t", "y"]].to_parquet(os.path.join(wdir, "val_index.parquet"))
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path)); log("using tuned configs from", cfg_path)
        else:
            log("tuning on validation year...")
            cfg = tune(Xtr, ytr, Xva, yva, 2 if a.quick else a.trials, a.quick)
            json.dump(cfg, open(cfg_path, "w"), default=str, indent=1)
        # json turns tuples to lists
        if isinstance(cfg["mlp"].get("hidden"), list):
            cfg["mlp"]["hidden"] = tuple(cfg["mlp"]["hidden"])
        for s in range(a.seeds):
            f = os.path.join(wdir, f"seed_{s}.pkl")
            if os.path.exists(f):
                continue
            t0 = time.time()
            p = fit_all(s, Xtr, ytr, Xva, yva, Xte, cfg, a.quick, core=a.core)
            pickle.dump(p, open(f, "wb"))
            log(f"  seed {s} finished in {time.time()-t0:.0f}s")
    log("ALL DONE")


if __name__ == "__main__":
    main()
