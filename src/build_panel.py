#!/usr/bin/env python3
"""
Build the firm-year distress panel from the WRDS extracts.

Unit of observation: firm i at forecast date June 30 of year t (t = 2011..END-1).
Label: 1 if firm i is delisted for performance reasons between Jul 1 of t (exclusive of
       June 30) and Jun 30 of t+1  (CRSP delisting codes 500 and 520-584; Campbell,
       Hilscher & Szilagyi 2008 convention).  Alternative labels are also stored.
Timing / look-ahead rules:
  * Fundamentals: latest fiscal year ending in (Dec 31 of t-2, Dec 31 of t-1], i.e. a lag of
    at least 6 months before the June 30 forecast date.
  * Market data: month-end values up to and including June of year t.
  * Only firms with >= 6 monthly returns in the 12 months before the forecast date.
Sample: CRSP common stocks (shrcd 10/11) on NYSE/AMEX/NASDAQ (exchcd 1/2/3) when msenames
        is available; financial firms (SIC 6000-6999) excluded in the main sample.
"""
import argparse, os, sys
import numpy as np
import pandas as pd


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../wrds_data")
    ap.add_argument("--out", default="../results/panel.parquet")
    ap.add_argument("--first-year", type=int, default=2011)
    ap.add_argument("--last-year", type=int, default=None, help="last forecast year; default = last year with a complete 12m label window")
    a = ap.parse_args()
    D = a.data_dir

    def rd(name, optional=False):
        p = os.path.join(D, name + ".parquet")
        if not os.path.exists(p):
            if optional:
                log(f"[!] optional file missing: {name}")
                return None
            sys.exit(f"missing required file {p}")
        return pd.read_parquet(p)

    msf = rd("crsp_msf")
    names = rd("crsp_msenames", optional=True)
    dl = rd("crsp_msedelist")
    fa = rd("comp_funda")
    ccm = rd("crsp_ccm_link")
    ff = rd("ff_factors_monthly")
    dsf = rd("crsp_dsf_monthly", optional=True)

    # ------------------------------------------------------------------ CRSP monthly
    msf = msf.copy()
    for c in ["ret", "retx", "prc", "vol", "shrout", "cfacpr", "cfacshr"]:
        msf[c] = pd.to_numeric(msf[c], errors="coerce").astype("float64")
    msf["permno"] = msf["permno"].astype("int64")
    msf["date"] = pd.to_datetime(msf["date"])
    msf["ym"] = msf["date"].dt.to_period("M")
    msf["me"] = msf["prc"].abs() * msf["shrout"] / 1000.0          # $ millions
    last_msf = msf["date"].max()
    log(f"CRSP monthly: {msf.date.min().date()} .. {last_msf.date()}  rows={len(msf):,}")

    if names is not None:
        nm = names.copy()
        nm["permno"] = nm["permno"].astype("int64")
        for c in ["shrcd", "exchcd", "siccd"]:
            nm[c] = pd.to_numeric(nm[c], errors="coerce").astype("float64")
        nm["namedt"] = pd.to_datetime(nm["namedt"]); nm["nameendt"] = pd.to_datetime(nm["nameendt"])
        nm = nm.sort_values("namedt")
        msf = msf.sort_values("date")
        msf = pd.merge_asof(msf, nm[["permno", "namedt", "nameendt", "shrcd", "exchcd", "siccd"]],
                            left_on="date", right_on="namedt", by="permno", direction="backward")
        ok = (msf["date"] <= msf["nameendt"]) & msf["shrcd"].isin([10, 11]) & msf["exchcd"].isin([1, 2, 3])
        n0 = msf.permno.nunique()
        msf = msf[ok].copy()
        log(f"share-code/exchange filter: permnos {n0:,} -> {msf.permno.nunique():,}")
    else:
        msf["siccd"] = np.nan

    # market return (decimal) from FF file: date is first of month
    ff = ff.copy(); ff["ym"] = pd.to_datetime(ff["date"]).dt.to_period("M")
    ff["mkt"] = ff["mktrf"] + ff["rf"]
    msf = msf.merge(ff[["ym", "mkt", "rf"]], on="ym", how="left")
    msf = msf.sort_values(["permno", "date"]).reset_index(drop=True)
    msf["lret"] = np.log1p(msf["ret"])
    msf["lexret"] = msf["lret"] - np.log1p(msf["mkt"])

    # ------------------------------------------------------------------ forecast dates
    # trailing 12-month stats ending in June of year t, per permno
    msf["ym_i"] = msf["ym"].astype("int64")  # months since 1970-01
    g = msf.groupby("permno", sort=False)

    def roll(col, fn, win=12, minp=6):
        return getattr(g[col].rolling(win, min_periods=minp), fn)().reset_index(level=0, drop=True)

    # rolling on observation count is fine only if months are contiguous: reindex check
    msf["gap"] = g["ym_i"].diff().fillna(1)
    contiguous_ok = (msf["gap"] <= 3).mean()
    log(f"share of rows with month-gap <=3: {contiguous_ok:.4f}")
    msf["ret12"] = roll("lret", "sum") if False else np.nan
    msf["sigma12"] = roll("ret", "std")
    msf["exret12"] = roll("lexret", "mean")
    msf["lret_sum12"] = roll("lret", "sum")
    msf["nret12"] = roll("ret", "count")
    msf["ret12"] = np.expm1(msf["lret_sum12"])

    jun = msf[msf["date"].dt.month == 6].copy()
    jun["t"] = jun["date"].dt.year
    jun["tot_me"] = jun.groupby("t")["me"].transform("sum")
    jun["rsize"] = np.log(jun["me"] / jun["tot_me"])
    jun["price"] = np.log(np.minimum(jun["prc"].abs(), 15.0))
    jun = jun[(jun["nret12"] >= 6) & (jun["me"] > 0)]

    if dsf is not None:
        d = dsf.copy(); d["mdate"] = pd.to_datetime(d["mdate"]); d["ym"] = d["mdate"].dt.to_period("M")
        d = d.sort_values(["permno", "ym"])
        d["sd_ret3"] = d.groupby("permno")["sd_ret"].transform(lambda s: s.rolling(3, min_periods=2).mean())
        d["amihud12"] = d.groupby("permno")["amihud"].transform(lambda s: s.rolling(12, min_periods=6).mean())
        jun = jun.merge(d[["permno", "ym", "sd_ret3", "amihud12"]], on=["permno", "ym"], how="left")
        jun["sigma_d"] = jun["sd_ret3"] * np.sqrt(252)

    # ------------------------------------------------------------------ Compustat link
    fa = fa.copy()
    for c in fa.columns:
        if c not in ("gvkey", "datadate"):
            fa[c] = pd.to_numeric(fa[c], errors="coerce").astype("float64")
    fa["datadate"] = pd.to_datetime(fa["datadate"])
    lk = ccm.copy()
    lk["linkdt"] = pd.to_datetime(lk["linkdt"]); lk["linkenddt"] = pd.to_datetime(lk["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    fa = fa.merge(lk[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey")
    fa = fa[(fa.datadate >= fa.linkdt) & (fa.datadate <= fa.linkenddt)].copy()
    fa["permno"] = fa["permno"].astype("int64")
    fa = fa.sort_values(["permno", "datadate"]).drop_duplicates(["permno", "datadate"], keep="last")

    for c in ["re", "ebit", "oiadp", "dp", "sic", "txditc", "pstk", "xint", "che", "act", "lct", "ni", "sale", "oibdp", "dltt", "dlc"]:
        if c not in fa.columns:
            fa[c] = np.nan
    # prior-year values for growth / change terms
    fa = fa.sort_values(["permno", "datadate"])
    prev = fa.groupby("permno")[["ni", "sale", "at"]].shift(1)
    prev_dt = fa.groupby("permno")["datadate"].shift(1)
    ok_prev = (fa["datadate"] - prev_dt).dt.days.between(300, 430)
    fa["ni_prev"] = np.where(ok_prev, prev["ni"], np.nan)
    fa["sale_prev"] = np.where(ok_prev, prev["sale"], np.nan)
    fa["at_prev"] = np.where(ok_prev, prev["at"], np.nan)

    # ------------------------------------------------------------------ attach fundamentals to forecast dates
    jun["permno"] = jun["permno"].astype("int64")
    jun = jun.sort_values("date")
    fa_s = fa.sort_values("datadate")
    # choose latest datadate <= Dec 31 of t-1 and > Dec 31 of t-2
    jun["cut"] = pd.to_datetime((jun["t"] - 1).astype(str) + "-12-31")
    panel = pd.merge_asof(jun.sort_values("cut"), fa_s, left_on="cut", right_on="datadate", by="permno", direction="backward")
    panel = panel[panel["datadate"] > (panel["cut"] - pd.DateOffset(years=1))].copy()
    log(f"forecast-date obs with fundamentals: {len(panel):,}")

    # ------------------------------------------------------------------ features
    P = panel
    P = P[(P["at"] > 0) & (P["lt"] > 0)].copy()
    ME = P["me"]
    BE = P["seq"].fillna(P["ceq"]) + P["txditc"].fillna(0) - P["pstk"].fillna(0)
    P["be"] = BE
    MTA = P["lt"] + ME
    ffo = P["ib"] + P["dp"].fillna(0)
    ebit = P["ebit"].fillna(P["oiadp"])
    wc = P["act"] - P["lct"]
    feat = pd.DataFrame(index=P.index)
    # --- CHS / Shumway-type (market + accounting)
    feat["nimta"] = P["ni"] / MTA
    feat["tlmta"] = P["lt"] / MTA
    feat["exret"] = P["exret12"]
    feat["sigma"] = P["sigma12"]
    feat["rsize"] = P["rsize"]
    feat["cashmta"] = P["che"] / MTA
    feat["mb"] = np.where(BE > 0, ME / BE, np.nan)
    feat["price"] = P["price"]
    # --- Ohlson-type (accounting)
    feat["size"] = np.log(P["at"])
    feat["tlta"] = P["lt"] / P["at"]
    feat["wcta"] = wc / P["at"]
    feat["clca"] = np.where(P["act"] > 0, P["lct"] / P["act"], np.nan)
    feat["oeneg"] = (P["lt"] > P["at"]).astype(float)
    feat["nita"] = P["ni"] / P["at"]
    feat["futl"] = ffo / P["lt"]
    feat["intwo"] = ((P["ni"] < 0) & (P["ni_prev"] < 0)).astype(float).where(P["ni_prev"].notna())
    feat["chin"] = (P["ni"] - P["ni_prev"]) / (P["ni"].abs() + P["ni_prev"].abs())
    # --- Altman-type
    feat["re_ta"] = P["re"] / P["at"]
    feat["ebit_ta"] = ebit / P["at"]
    feat["me_tl"] = ME / P["lt"]
    feat["sales_ta"] = P["sale"] / P["at"]
    # --- other
    feat["sgrowth"] = np.log(P["sale"] / P["sale_prev"]).replace([np.inf, -np.inf], np.nan)
    feat["ret12"] = P["ret12"]
    feat["bm_neg"] = (BE <= 0).astype(float)
    if "sigma_d" in P:
        feat["sigma_d"] = P["sigma_d"]; feat["amihud"] = P["amihud12"]
    feat["z_altman"] = (1.2 * feat["wcta"] + 1.4 * feat["re_ta"] + 3.3 * feat["ebit_ta"]
                        + 0.6 * feat["me_tl"] + 1.0 * feat["sales_ta"])
    feat = feat.replace([np.inf, -np.inf], np.nan)

    keep = ["permno", "gvkey", "t", "date", "datadate", "me"]
    out = pd.concat([P[keep], feat], axis=1)
    sic = pd.to_numeric(P["sic"], errors="coerce").fillna(pd.to_numeric(P.get("siccd"), errors="coerce"))
    out["sic"] = sic
    out["financial"] = sic.between(6000, 6999)
    out["utility"] = sic.between(4900, 4999)

    # ------------------------------------------------------------------ labels
    dl = dl.copy(); dl["dlstdt"] = pd.to_datetime(dl["dlstdt"])
    dl["permno"] = dl["permno"].astype("int64"); dl["dlstcd"] = pd.to_numeric(dl["dlstcd"], errors="coerce")
    dl["perf"] = (dl["dlstcd"] == 500) | dl["dlstcd"].between(520, 584)
    dl["liq"] = dl["dlstcd"].between(400, 499)
    dl["bk"] = dl["dlstcd"] == 574
    dl["any_fail"] = dl["perf"] | dl["liq"]
    out["start"] = pd.to_datetime(out["t"].astype(str) + "-06-30")
    out["end"] = pd.to_datetime((out["t"] + 1).astype(str) + "-06-30")
    ev = out[["permno", "start", "end"]].reset_index().merge(dl, on="permno", how="left")
    ev = ev[(ev["dlstdt"] > ev["start"]) & (ev["dlstdt"] <= ev["end"])]
    for lab in ["perf", "liq", "bk", "any_fail"]:
        idx = ev.loc[ev[lab], "index"].unique()
        out["y_" + lab] = out.index.isin(idx).astype(int)
    out["y"] = out["y_perf"]

    # last forecast year with a complete label window
    cutoff_year = (last_msf.year if last_msf.month == 12 else last_msf.year - 1) - 1
    last_year = a.last_year or cutoff_year
    out = out[(out["t"] >= a.first_year) & (out["t"] <= last_year)].copy()
    log(f"forecast years {a.first_year}..{last_year} (complete label windows)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_parquet(a.out, index=False)
    s = out.groupby("t").agg(n=("y", "size"), fail=("y", "sum"), rate=("y", "mean"))
    log(s.to_string())
    log(f"saved {a.out}  rows={len(out):,}")


if __name__ == "__main__":
    main()
