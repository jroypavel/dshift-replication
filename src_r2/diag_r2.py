#!/usr/bin/env python3
"""Shift diagnostics for the revised paper (revision 2). Writes to ../results/diag_r2/.

1. domain_auc.csv  classifier two-sample tests (Lopez-Paz & Oquab 2017): for each year t, a LightGBM classifier
   separates the firms of year t from those of the training years 2011-2017 (excluding t itself when t is a training
   year, which gives the within-training benchmark). Out-of-fold AUC from 5-fold cross-fitting with folds GROUPED BY
   FIRM, so that the same firm never appears in both the fitting and the scoring fold. Three predictor sets:
   all 24 predictors; the 14 accounting predictors; all 24 replaced by within-year percentile ranks (missing kept missing).
2. year_intercepts.csv / intercept_tests.csv  pooled logit of failure on the CHS-type predictors (preprocessed with
   moments of 2011-2017) with year fixed effects and common slopes; contrasts of the yearly intercepts with years as units.
3. codes.csv  delisting codes of failures by period; sic283.csv  pharmaceutical/biotech shares.
4. descriptives.csv  predictor summaries by period and failure status.
"""
import os
import numpy as np, pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import common as C

OUT = "../results/diag_r2"; os.makedirs(OUT, exist_ok=True)
ALL = C.FEATURE_SETS["all"]
MARKET = ["nimta", "tlmta", "exret", "sigma", "rsize", "cashmta", "mb", "price", "me_tl", "ret12"]
ACCT = [c for c in ALL if c not in MARKET]
df, _ = C.load_panel("../results/panel.parquet")
TRAIN = list(range(2011, 2018))


def cv_auc(X, d, groups, seed=0):
    oof = np.zeros(len(d))
    for trn, tst in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, d, groups):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=50,
                                 verbose=-1, n_jobs=1, random_state=seed)
        clf.fit(X[trn], d[trn]); oof[tst] = clf.predict_proba(X[tst])[:, 1]
    return roc_auc_score(d, oof)


def domain(variants=("all", "accounting", "ranked")):
    ranked = df.copy()
    rng = np.random.default_rng(0)
    ranked["_u"] = rng.random(len(ranked))
    for c in ALL:
        if c in C.BINARY:                                            # binary indicators are left as they are
            continue
        # within-year percentile ranks with random tie-breaking (ties, e.g. the $15 price cap, would otherwise create
        # year-specific mass points); missing values stay missing
        o = ranked.sort_values([c, "_u"])
        r = o.groupby("t").cumcount() + 1
        n = o.groupby("t")[c].transform("count")
        ranked[c] = (r / n).where(o[c].notna()).reindex(ranked.index)
    rows = []
    for t in range(2011, 2024):
        ref = df[df.t.isin([s for s in TRAIN if s != t])]
        tgt = df[df.t == t]
        for tag, data, cols in [v for v in (("all", df, ALL), ("accounting", df, ACCT), ("ranked", ranked, ALL)) if v[0] in variants]:
            r = data.loc[ref.index]; g = data.loc[tgt.index]
            X = np.vstack([r[cols].to_numpy(float), g[cols].to_numpy(float)])
            d = np.r_[np.zeros(len(r)), np.ones(len(g))]
            groups = np.r_[r["permno"].to_numpy(), g["permno"].to_numpy()]
            rows.append(dict(t=t, variant=tag, auc=cv_auc(X, d, groups), n_target=len(g), n_ref=len(r)))
            print(rows[-1], flush=True)
    f = os.path.join(OUT, "domain_auc.csv")
    old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["t", "variant"])
    old = old[~old["variant"].isin(variants)]
    pd.concat([old, pd.DataFrame(rows)]).sort_values(["variant", "t"]).to_csv(f, index=False)


def intercepts():
    cols = C.FEATURE_SETS["chs"]
    tr = df[df.t <= 2017]
    pp = C.Preprocessor(cols).fit(tr)
    d = df[df.t != 2018].copy()
    X = pp.transform(d)
    years = sorted(d.t.unique())
    D = np.column_stack([(d.t.to_numpy() == t).astype(float) for t in years])
    res = sm.GLM(d.y.to_numpy(), np.column_stack([D, X]), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d.permno.to_numpy()})
    b = res.params[:len(years)]; V = res.cov_params()[:len(years), :len(years)]
    yi = pd.DataFrame(dict(t=years, intercept=b, se_firm=np.sqrt(np.diag(V)),
                           n=[int((d.t == t).sum()) for t in years], failures=[int(d.y[d.t == t].sum()) for t in years]))
    yi.to_csv(os.path.join(OUT, "year_intercepts.csv"), index=False)
    slopes = pd.Series(res.params[len(years):], index=cols); slopes.to_csv(os.path.join(OUT, "common_slopes.csv"))
    post = (yi.t >= 2019).to_numpy().astype(float); tt = (yi.t - 2011).to_numpy().astype(float)
    rows = []
    # A: mean(2019-23) - mean(2011-17), years as units
    XA = sm.add_constant(post)
    for cov in ("nonrobust", "HC1"):
        r = sm.OLS(yi.intercept.to_numpy(), XA).fit(cov_type=cov)
        rows.append(dict(contrast="post minus pre (levels)", estimate=r.params[1], se=r.bse[1], p=r.pvalues[1], se_type="years, " + cov))
    w = np.r_[-np.where(post == 0, 1 / (post == 0).sum(), 0)] + np.where(post == 1, 1 / post.sum(), 0)
    rows.append(dict(contrast="post minus pre (levels)", estimate=float(w @ b), se=float(np.sqrt(w @ V @ w)), p=np.nan, se_type="firm-clustered"))
    # B: deviation of 2019-23 from the 2011-17 linear trend
    XB = np.column_stack([np.ones_like(tt), tt, post, post * (tt - tt[post == 1].mean())])
    for cov in ("nonrobust", "HC1"):
        r = sm.OLS(yi.intercept.to_numpy(), XB).fit(cov_type=cov)
        rows.append(dict(contrast="post deviation from pre trend", estimate=r.params[2], se=r.bse[2], p=r.pvalues[2], se_type="years, " + cov))
        if cov == "nonrobust":
            rows.append(dict(contrast="pre-period trend per year", estimate=r.params[1], se=r.bse[1], p=r.pvalues[1], se_type="years, nonrobust"))
    rows.append(dict(contrast="SD of yearly intercepts 2011-17", estimate=float(yi.intercept[post == 0].std(ddof=1)), se=np.nan, p=np.nan, se_type=""))
    rows.append(dict(contrast="SD of yearly intercepts 2019-23", estimate=float(yi.intercept[post == 1].std(ddof=1)), se=np.nan, p=np.nan, se_type=""))
    rows.append(dict(contrast="mean firm-clustered SE of a yearly intercept", estimate=float(yi.se_firm.mean()), se=np.nan, p=np.nan, se_type=""))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "intercept_tests.csv"), index=False)
    print(yi.round(3).to_string()); print(pd.DataFrame(rows).round(4).to_string())


def codes():
    dl = pd.read_parquet("../wrds_data/crsp_msedelist.parquet")
    dl["dlstcd"] = pd.to_numeric(dl["dlstcd"]); dl["dlstdt"] = pd.to_datetime(dl["dlstdt"])
    f = df[df.y == 1][["permno", "t", "sic"] if "sic" in df.columns else ["permno", "t"]].copy()
    lo = pd.to_datetime(f.t.astype(str) + "-06-30"); hi = pd.to_datetime((f.t + 1).astype(str) + "-06-30")
    m = f.merge(dl[["permno", "dlstdt", "dlstcd"]], on="permno", how="left")
    m = m[(m.dlstdt > pd.to_datetime(m.t.astype(str) + "-06-30")) & (m.dlstdt <= pd.to_datetime((m.t + 1).astype(str) + "-06-30"))]
    m = m[(m.dlstcd == 500) | ((m.dlstcd >= 520) & (m.dlstcd <= 584))].drop_duplicates(["permno", "t"])
    assert len(m) == int(df.y.sum()), (len(m), int(df.y.sum()))
    def grp(c):
        if c == 500: return "500"
        if 520 <= c <= 551: return "520-551"
        if c == 552: return "552"
        if c in (560, 561): return "560-561"
        if 570 <= c <= 573: return "570-573"
        if c == 574: return "574"
        if 580 <= c <= 583: return "580-583"
        if c == 584: return "584"
        return "other"
    m["group"] = m.dlstcd.map(grp)
    m["period"] = np.where(m.t <= 2017, "2011-2017", np.where(m.t == 2018, "2018", "2019-2023"))
    tab = m.groupby(["group", "period"]).size().unstack(fill_value=0)
    tab.to_csv(os.path.join(OUT, "codes.csv")); print(tab)
    m.to_csv(os.path.join(OUT, "failure_codes_firmlevel.csv"), index=False)


def descriptives():
    d = df.copy(); d["period"] = np.where(d.t <= 2017, "train", np.where(d.t == 2018, "val", "test"))
    rows = []
    for c in ALL:
        for per in ("train", "test"):
            for yv in (0, 1):
                x = d[(d.period == per) & (d.y == yv)][c]
                stat = x.mean() if c in C.BINARY else x.median()
                rows.append(dict(var=c, period=per, failed=yv, stat=float(stat), kind="mean" if c in C.BINARY else "median",
                                 missing=float(x.isna().mean())))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "descriptives.csv"), index=False)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["intercepts", "codes", "descriptives", "domain"]
    for w in what:
        if w == "domain_ranked":
            domain(("ranked",))
        else:
            globals()[w]()
