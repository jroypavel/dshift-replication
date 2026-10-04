#!/usr/bin/env python3
"""Evaluation for the revised paper (revision 2).

Reads per-seed prediction pickles from --res (one or more windows) plus optional --extra result directories that share
the test index, adds post-hoc recalibration (Platt for every model, temperature for the networks that lack it), the
explicit shift adjustments (lagged-rate intercepts, EM prior shift), and writes CSVs to <res>/eval_r2/:

  metrics_by_seed.csv / metrics_pooled.csv   pooled test-period metrics per seed, and mean/SD across seeds
  metrics_by_year.csv                        per model and year: E/O (seed-averaged forecasts), 95% band of E/O for a
                                             calibrated forecast (simulated Poisson-binomial), AUC and Brier (seed means)
  significance.csv                           firm-clustered bootstrap of loss/AUC differences, Holm-adjusted p-values
  year_diffs.csv                             per-year Brier differences for the same pairs
  uncertainty.csv / uncertainty_by_year.csv  what epistemic uncertainty measures (see uncertainty_tests)
  risk_coverage.csv                          Brier of the retained set after removal by risk, by uncertainty, by residual
"""
import argparse, glob, os, pickle
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from patsy import dmatrix
import metrics as M
from names_r2 import BASES_PLATT, BASES_TS, POSTHOC, SIG_REFS, UNC

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ loading
def load_dir(res, windows):
    idx, vidx, P = [], {}, {}
    for w in windows:
        wd = os.path.join(res, w)
        ti = pd.read_parquet(os.path.join(wd, "test_index.parquet")); ti["window"] = w; idx.append(ti)
        vp = os.path.join(wd, "val_index.parquet")
        vidx[w] = pd.read_parquet(vp) if os.path.exists(vp) else None
        files = sorted(glob.glob(os.path.join(wd, "seed_*.pkl")), key=lambda f: int(os.path.basename(f)[5:-4]))
        seeds = [pickle.load(open(f, "rb")) for f in files]
        names = {n for n in set(seeds[0]).intersection(*[set(s) for s in seeds]) if not n.startswith("_")}
        for name in names:
            d = P.setdefault(name, {"p": [], "ue": [], "ul": [], "pv": {}, "uv": {}, "uvl": {}, "yv": {}})
            g = lambda k, s: s[name].get(k, np.zeros_like(s[name]["p"]))
            d["p"].append(np.stack([s[name]["p"] for s in seeds]))
            d["ue"].append(np.stack([s[name]["u_epi"] for s in seeds]))
            d["ul"].append(np.stack([g("u_logit", s) for s in seeds]))
            d["pv"][w] = np.stack([s[name]["p_val"] for s in seeds])
            d["uv"][w] = np.stack([s[name]["u_val"] for s in seeds])
            d["uvl"][w] = np.stack([s[name].get("u_val_logit", np.zeros_like(s[name]["p_val"])) for s in seeds])
            d["yv"][w] = vidx[w]["y"].to_numpy().astype(float) if vidx[w] is not None else None
    for name in list(P):
        d = P[name]
        if len(d["p"]) != len(windows):
            del P[name]; continue
        for k in ("p", "ue", "ul"):
            d[k] = np.concatenate(d[k], axis=1).astype(np.float64)
    return pd.concat(idx, ignore_index=True), vidx, P


def merge_extra(P, idx, extra_dirs):
    S = next(iter(P.values()))["p"].shape[0]
    for ed in extra_dirs:
        ei, _, EP = load_dir(ed, ["fixed"])
        assert len(ei) == len(idx) and (ei["permno"].to_numpy() == idx["permno"].to_numpy()).all(), ed
        for name, d in EP.items():
            if name in P:
                continue
            if d["p"].shape[0] != S:                     # deterministic single-seed extras: broadcast
                reps = int(np.ceil(S / d["p"].shape[0]))
                for k in ("p", "ue", "ul"):
                    d[k] = np.tile(d[k], (reps, 1))[:S]
                for k in ("pv", "uv", "uvl"):
                    d[k] = {w: np.tile(v, (reps, 1))[:S] for w, v in d[k].items()}
            d["yv"] = {w: (v if v is not None else P[next(iter(P))]["yv"][w]) for w, v in d["yv"].items()}
            P[name] = d
    return P


# ------------------------------------------------------------------ post-hoc recalibration on the validation year
def _copy(d, p):
    return {"p": p, "ue": d["ue"], "ul": d["ul"], "pv": d["pv"], "uv": d["uv"], "uvl": d["uvl"], "yv": d["yv"]}


def add_platt(P, idx, bases):
    """Platt scaling (logistic regression of the outcome on the forecast's logit) fitted per seed and window on the
    validation year; same estimator as baselines.Recal('platt')."""
    win = idx["window"].to_numpy()
    for b in bases:
        if b not in P or b + "_platt" in P:
            continue
        p = P[b]["p"]; out = np.empty_like(p); pv_out = {}
        for w in np.unique(win):
            m = win == w; pvw = np.empty_like(P[b]["pv"][w])
            for s in range(p.shape[0]):
                zv = logit(P[b]["pv"][w][s]); yv = P[b]["yv"][w]
                est = LogisticRegression(C=1e6, max_iter=1000).fit(zv[:, None], yv)
                out[s, m] = est.predict_proba(logit(p[s, m])[:, None])[:, 1]
                pvw[s] = est.predict_proba(zv[:, None])[:, 1]
            pv_out[w] = pvw
        d = _copy(P[b], out); d["pv"] = pv_out
        P[b + "_platt"] = d
    return P


def add_ts(P, idx, bases):
    win = idx["window"].to_numpy()
    Ts = np.exp(np.linspace(-1.5, 1.5, 121))
    for b in bases:
        if b not in P or b + "_ts" in P:
            continue
        p = P[b]["p"]; out = np.empty_like(p)
        for w in np.unique(win):
            m = win == w
            for s in range(p.shape[0]):
                zv = logit(P[b]["pv"][w][s]); yv = P[b]["yv"][w]
                T = Ts[int(np.argmin([M.logloss(yv, sig(zv / T)) for T in Ts]))]
                out[s, m] = sig(logit(p[s, m]) / T)
        P[b + "_ts"] = _copy(P[b], out)
    return P


# ------------------------------------------------------------------ explicit shift adjustments (post hoc)
def add_posthoc(P, idx, rates):
    """Lagged-rate intercepts: shift each forecast year's logits so that the mean forecast equals the failure rate realised
    in the previous forecast year (known by the forecast date), or the mean of the previous three years. EM prior shift
    (Saerens et al. 2002): estimate the forecast year's failure rate from its unlabelled forecasts under pure label shift."""
    yr = idx["t"].to_numpy(); win = idx["window"].to_numpy()
    for b in POSTHOC:
        if b not in P:
            continue
        p = np.clip(P[b]["p"], EPS, 1 - EPS)
        lag, lag3, em = np.empty_like(p), np.empty_like(p), np.empty_like(p)
        for s in range(p.shape[0]):
            for t in np.unique(yr):
                m = yr == t; z = logit(p[s, m])
                for arr, target in ((lag, rates[int(t) - 1]), (lag3, np.mean([rates[int(t) - k] for k in (1, 2, 3)]))):
                    c = optimize.brentq(lambda c: sig(z + c).mean() - target, -15, 15)
                    arr[s, m] = sig(z + c)
                w = win[m][0]
                pi_s = float(np.clip(P[b]["pv"][w][s].mean(), EPS, 1 - EPS)); pi_t = pi_s; q = p[s, m]
                for _ in range(1000):
                    a1, a0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
                    q = a1 * p[s, m] / (a1 * p[s, m] + a0 * (1 - p[s, m]))
                    new = float(q.mean())
                    if abs(new - pi_t) < 1e-12:
                        break
                    pi_t = new
                em[s, m] = q
        for suf, arr in (("_lag", lag), ("_lag3", lag3), ("_em", em)):
            d = _copy(P[b], arr); d["ue"] = np.zeros_like(arr); d["ul"] = np.zeros_like(arr)
            P[b + suf] = d
    return P


# ------------------------------------------------------------------ metrics
def citl_intercept(y, p):
    try:
        r = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit(p)).fit()
        return float(r.params[0])
    except Exception:
        return np.nan


def corp(y, p):
    """CORP decomposition of the Brier score (Dimitriadis, Gneiting & Jordan 2021): BS = MCB - DSC + UNC."""
    pc = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y).predict(p)
    bs, bsc = M.brier(y, p), M.brier(y, pc)
    unc = y.mean() * (1 - y.mean())
    return bs - bsc, unc - bsc, unc


def topdecile(y, p, yr):
    hit = 0
    for t in np.unique(yr):
        m = yr == t; k = int(np.ceil(0.1 * m.sum()))
        hit += y[m][np.argsort(-p[m])[:k]].sum()
    return hit / max(y.sum(), 1)


def mal_eo(y, p, yr):
    out = []
    for t in np.unique(yr):
        m = yr == t; O, E = y[m].sum(), p[m].sum()
        if O > 0:
            out.append(abs(np.log(E / O)))
    return float(np.mean(out))


def pooled_metrics(y, p, yr):
    a, b = M.calib_slope_intercept(y, p)
    mcb, dsc, unc = corp(y, p)
    bs = M.brier(y, p)
    return dict(brier=bs, bss=1 - bs / unc, mcb=mcb, dsc=dsc, unc=unc, logloss=M.logloss(y, p), ece=M.ece(y, p),
                eo=float(p.sum() / y.sum()), citl_int=citl_intercept(y, p), cal_slope=b, cal_int=a,
                auc=M.auc(y, p), ap=M.ap(y, p), top10=topdecile(y, p, yr), mal_eo=mal_eo(y, p, yr))


def calibrated_band(p, n_sim=20000, seed=0):
    """Central 95% range of E/O if outcomes were drawn from the forecasts themselves (Poisson-binomial)."""
    rng = np.random.default_rng(seed)
    O = np.concatenate([(rng.random((1000, len(p))) < p[None, :]).sum(1) for _ in range(n_sim // 1000)])
    O = np.maximum(O, 1)
    E = p.sum()
    return float(E / np.quantile(O, 0.975)), float(E / np.quantile(O, 0.025))


# ------------------------------------------------------------------ uncertainty
def spline_resid(lp, lu, df=5):
    X = np.asarray(dmatrix(f"cr(x, df={df})", {"x": lp}, return_type="dataframe"))
    beta = np.linalg.lstsq(X, lu, rcond=None)[0]
    fit = X @ beta
    r2 = 1 - np.var(lu - fit) / np.var(lu)
    return lu - fit, float(r2)


def matched_removal(y, p, r, frac=0.10, n_bins=20):
    """Within each of n_bins quantile bins of p, drop the frac with the highest residual uncertainty r.
    Returns Brier(retained) - Brier(all); random removal within bins leaves its expectation unchanged."""
    b = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False).to_numpy()
    keep = np.ones(len(y), bool)
    for k in range(n_bins):
        ix = np.flatnonzero(b == k); n = int(round(frac * len(ix)))
        keep[ix[np.argsort(-r[ix])[:n]]] = False
    return M.brier(y[keep], p[keep]) - M.brier(y, p)


def removal_brier(y, p, score, frac):
    k = int(round(frac * len(y)))
    keep = np.ones(len(y), bool); keep[np.argsort(-score)[:k]] = False
    return M.brier(y[keep], p[keep])


def info_test(y, p, r, groups):
    """Does residual uncertainty predict outcomes beyond the forecast?  y ~ a + b*logit(p) + c*z + d*z*logit(p),
    z = standardised r; firm-clustered Wald test of c = d = 0."""
    lp = logit(p); z = (r - r.mean()) / (r.std() + 1e-12)
    X = sm.add_constant(np.column_stack([lp, z, lp * z]))
    try:
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
        R = np.zeros((2, 4)); R[0, 2] = 1; R[1, 3] = 1
        w = res.wald_test(R, scalar=True)
        return dict(c_z=float(res.params[2]), p_z=float(res.pvalues[2]), d_int=float(res.params[3]), p_int=float(res.pvalues[3]),
                    wald=float(w.statistic), p_joint=float(w.pvalue))
    except Exception as e:
        return dict(c_z=np.nan, p_z=np.nan, d_int=np.nan, p_int=np.nan, wald=np.nan, p_joint=np.nan)


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--windows", nargs="+", default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--panel-path", default="../results/panel.parquet")
    ap.add_argument("--label", default="y_perf")
    ap.add_argument("--with-financials", action="store_true")
    a = ap.parse_args()
    windows = a.windows or sorted(d for d in os.listdir(a.res) if os.path.isdir(os.path.join(a.res, d)) and not d.startswith("eval"))
    out = os.path.join(a.res, "eval_r2"); os.makedirs(out, exist_ok=True)
    idx, vidx, P = load_dir(a.res, windows)
    P = merge_extra(P, idx, a.extra)
    P = add_platt(P, idx, BASES_PLATT)
    P = add_ts(P, idx, BASES_TS)
    pr = pd.read_parquet(a.panel_path)
    if not a.with_financials:
        pr = pr[~pr["financial"]]
    import common as C
    pr = pr.dropna(subset=C.REQUIRED)
    rates = pr.groupby("t")[a.label].mean().to_dict()
    P = add_posthoc(P, idx, rates)
    y = idx["y"].to_numpy().astype(float); yr = idx["t"].to_numpy(); firm = idx["permno"].to_numpy()
    print(f"test obs {len(y)}, failures {int(y.sum())}, models {len(P)}, windows {windows}", flush=True)

    rows = []
    for name, d in P.items():
        for s in range(d["p"].shape[0]):
            r = pooled_metrics(y, np.clip(d["p"][s], EPS, 1 - EPS), yr); r.update(model=name, seed=s); rows.append(r)
    bs = pd.DataFrame(rows); bs.to_csv(os.path.join(out, "metrics_by_seed.csv"), index=False)
    cols = [c for c in bs.columns if c not in ("model", "seed")]
    agg = bs.groupby("model")[cols].agg(["mean", "std"]); agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["n_seeds"] = bs.groupby("model").size()
    agg.to_csv(os.path.join(out, "metrics_pooled.csv"))

    rows = []
    for name, d in P.items():
        pm = d["p"].mean(0)
        for t in np.unique(yr):
            m = yr == t; O, E = y[m].sum(), pm[m].sum()
            lo, hi = calibrated_band(pm[m]) if name in ("logit_all_platt", "vids_platt", "ens5_platt", "logit_chs_platt") else (np.nan, np.nan)
            per = [(M.auc(y[m], d["p"][s][m]), M.brier(y[m], d["p"][s][m])) for s in range(d["p"].shape[0])]
            rows.append(dict(model=name, t=int(t), n=int(m.sum()), failures=int(O), expected=float(E),
                             eo=float(E / O) if O else np.nan, band_lo=lo, band_hi=hi,
                             auc=float(np.mean([q[0] for q in per])), brier=float(np.mean([q[1] for q in per]))))
    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics_by_year.csv"), index=False)
    if a.light:
        print("light evaluation saved to", out); return

    rng = np.random.default_rng(1)
    uniq, inv = np.unique(firm, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    draws = [np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))]) for _ in range(a.n_boot)]

    def perobs(name, kind):
        p = np.clip(P[name]["p"], EPS, 1 - EPS)
        if kind == "brier":
            return ((y[None, :] - p) ** 2).mean(0)
        return (-(y[None, :] * np.log(p) + (1 - y[None, :]) * np.log(1 - p))).mean(0)

    rows, yrows = [], []
    for A in ("vids_platt", "logit_all_platt"):
        refs = [r for r in SIG_REFS + ["logit_chs"] if r in P and r != A]
        for kind in ("brier", "logloss", "auc"):
            fam = []
            for Bm in refs:
                if kind == "auc":
                    pa, pb = P[A]["p"].mean(0), P[Bm]["p"].mean(0)
                    d0 = M.auc(y, pa) - M.auc(y, pb)
                    bsamp = np.array([M.auc(y[ix], pa[ix]) - M.auc(y[ix], pb[ix]) for ix in draws])
                else:
                    dl = perobs(A, kind) - perobs(Bm, kind); d0 = float(dl.mean())
                    bsamp = np.array([dl[ix].mean() for ix in draws])
                    if kind == "brier":
                        for t in np.unique(yr):
                            yrows.append(dict(A=A, B=Bm, t=int(t), diff=float(dl[yr == t].mean())))
                pval = float(min(1.0, 2 * min((bsamp <= 0).mean(), (bsamp >= 0).mean())))
                fam.append(dict(metric=kind, A=A, B=Bm, diff=float(d0), lo=float(np.quantile(bsamp, .025)),
                                hi=float(np.quantile(bsamp, .975)), p=pval))
            for f, h in zip(fam, holm([f["p"] for f in fam])):
                f["p_holm"] = float(h); rows.append(f)
    pd.DataFrame(rows).to_csv(os.path.join(out, "significance.csv"), index=False)
    pd.DataFrame(yrows).to_csv(os.path.join(out, "year_diffs.csv"), index=False)

    rows, urows, rc = [], [], []
    for name in [n for n in UNC if n in P]:
        d = P[name]
        pbar = np.clip(d["p"].mean(0), EPS, 1 - EPS)
        for scale, U, UV in (("prob", d["ue"], d["uv"]), ("logit", d["ul"], d["uvl"])):
            ubar = U.mean(0)
            if not np.all(ubar > 0):
                continue
            r, r2 = spline_resid(logit(pbar), np.log(ubar))
            row = dict(model=name, scale=scale, spearman=float(stats.spearmanr(ubar, pbar)[0]), r2_spline=r2,
                       brier_all=M.brier(y, pbar), drop_u10=removal_brier(y, pbar, ubar, 0.10),
                       drop_p10=removal_brier(y, pbar, pbar, 0.10), matched10=matched_removal(y, pbar, r))
            gb = [matched_removal(y[ix], pbar[ix], r[ix]) for ix in draws[:400]]
            row["matched10_lo"], row["matched10_hi"] = float(np.quantile(gb, .025)), float(np.quantile(gb, .975))
            row.update(info_test(y, pbar, r, firm))
            rows.append(row)
            uval = np.mean([v.mean() for v in UV.values()])
            for t in np.unique(yr):
                m = yr == t
                urows.append(dict(model=name, scale=scale, t=int(t), rel_val=float(ubar[m].mean() / uval),
                                  rel_val_resid=float(np.exp(r[m].mean())), mean_p=float(pbar[m].mean()), rate=float(y[m].mean())))
            if scale == "prob" and name in ("vids", "ens5"):
                for cov in np.linspace(0.5, 1.0, 11):
                    frac = 1 - cov
                    rc.append(dict(model=name, coverage=cov, by_u=removal_brier(y, pbar, ubar, frac),
                                   by_p=removal_brier(y, pbar, pbar, frac),
                                   matched=(matched_removal(y, pbar, r, frac) + M.brier(y, pbar)) if frac > 0 else M.brier(y, pbar)))
    U = pd.DataFrame(rows)
    if len(U):
        U["p_joint_holm"] = holm(U["p_joint"].fillna(1).to_numpy())
    U.to_csv(os.path.join(out, "uncertainty.csv"), index=False)
    pd.DataFrame(urows).to_csv(os.path.join(out, "uncertainty_by_year.csv"), index=False)
    pd.DataFrame(rc).to_csv(os.path.join(out, "risk_coverage.csv"), index=False)
    print("saved to", out)


if __name__ == "__main__":
    main()
