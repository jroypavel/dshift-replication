"""Shared data utilities: feature sets, sample filters, temporal splits, preprocessing."""
import numpy as np
import pandas as pd

FEATURE_SETS = {
    "altman_z": ["z_altman"],
    "ohlson": ["size", "tlta", "wcta", "clca", "oeneg", "nita", "futl", "intwo", "chin"],
    "chs": ["nimta", "tlmta", "exret", "sigma", "rsize", "cashmta", "mb", "price"],
    "all": ["nimta", "tlmta", "exret", "sigma", "rsize", "cashmta", "mb", "price",
            "size", "tlta", "wcta", "clca", "oeneg", "nita", "futl", "intwo", "chin",
            "re_ta", "ebit_ta", "me_tl", "sales_ta", "sgrowth", "ret12", "bm_neg"],
}
REQUIRED = ["nimta", "tlmta", "exret", "sigma", "rsize", "cashmta", "price", "size", "tlta",
            "wcta", "nita", "futl", "re_ta", "ebit_ta", "me_tl", "sales_ta", "ret12"]
IMPUTE = ["mb", "clca", "intwo", "chin", "sgrowth"]     # imputed with training median
BINARY = {"oeneg", "intwo", "bm_neg"}


XS_RANK_COLS = ["exret", "sigma", "ret12", "price", "rsize", "mb", "sigma_d", "amihud"]


def load_panel(path, exclude_financials=True, label="y_perf", cs_rank=False):
    """cs_rank=True replaces market-based features by their within-year cross-sectional percentile
    (removes market-wide level shifts such as the 2021 rally or 2022 sell-off)."""
    df = pd.read_parquet(path)
    n0 = len(df)
    if exclude_financials:
        df = df[~df["financial"]]
    df = df.dropna(subset=REQUIRED).copy()
    if cs_rank:
        for c in XS_RANK_COLS:
            if c in df.columns:
                df[c] = df.groupby("t")[c].rank(pct=True)
    df["y"] = df[label].astype(int)
    df["sample_note"] = ""
    info = {"n_raw": n0, "n_final": len(df)}
    return df.reset_index(drop=True), info


def split_fixed(df, train_end=2017, val_year=2018, test_start=2019):
    tr = df[df.t <= train_end]
    va = df[df.t == val_year]
    te = df[df.t >= test_start]
    return tr, va, te


class Preprocessor:
    """Winsorise at train 1/99 percentiles, impute with train median, standardise with train moments.
    Fitted on the training split only (no look-ahead)."""

    def __init__(self, cols, lo=0.01, hi=0.99):
        self.cols, self.lo, self.hi = list(cols), lo, hi

    def fit(self, df):
        X = df[self.cols]
        self.q_lo = X.quantile(self.lo)
        self.q_hi = X.quantile(self.hi)
        Xw = X.clip(self.q_lo, self.q_hi, axis=1)
        self.med = Xw.median()
        Xi = Xw.fillna(self.med)
        self.mu = Xi.mean()
        self.sd = Xi.std().replace(0, 1.0)
        for c in self.cols:
            if c in BINARY:
                self.mu[c], self.sd[c] = 0.0, 1.0
        return self

    def transform(self, df):
        X = df[self.cols].clip(self.q_lo, self.q_hi, axis=1).fillna(self.med)
        return ((X - self.mu) / self.sd).to_numpy(dtype=np.float32)
