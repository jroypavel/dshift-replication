#!/usr/bin/env python3
"""LaTeX tables and number macros for the revised paper (revision 2), generated from the result files only."""
import json, os, pickle, glob, re
import numpy as np, pandas as pd
import common as C
from names_r2 import NAMES

R = "../results"; OUT = "../paper_r2/generated"; os.makedirs(OUT, exist_ok=True)
EV = f"{R}/main_r2/eval_r2"; DG = f"{R}/diag_r2"
MAC = {}
WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def key2mac(k):
    k = k.replace("ens5", "ens").replace("altman5", "altmanratios")
    parts = re.split(r"[_\-]", k)
    s = "".join(p[:1].upper() + p[1:] for p in parts)
    return "".join(WORDS.get(ch, ch) for ch in s)


def put(name, val):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    if isinstance(val, str) and re.fullmatch(r"-\d[\d.,{}]*", val):
        val = "\\ensuremath{-}" + val[1:]          # typographic minus in text and math mode
    assert name not in MAC or MAC[name] == val, (name, MAC[name], val)
    MAC[name] = val


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def b2(x): return f"{100 * x:.3f}"          # Brier x 10^2
def l2(x): return f"{100 * x:.2f}"          # log loss x 10^2
def pct(x, nd=0): return f"{100 * x:.{nd}f}"
def num(n): return f"{int(n):,}".replace(",", "{,}")
def pv(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def esc(s): return s.replace("_", "\\_").replace("%", "\\%")


def table(fn, caption, label, colspec, header, body, size="\\small", notes=None, resize=False):
    body = [re.sub(r"(?<=[\s\[(&])-(?=\d)", lambda _: "\\ensuremath{-}", r) for r in body]
    L = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
         ("\\resizebox{\\linewidth}{!}{" if resize else "") + f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}" + ("}" if resize else "")]
    if notes:
        L += ["\\par\\smallskip", f"\\parbox{{\\linewidth}}{{\\footnotesize {notes}}}"]
    L += ["\\end{table}"]
    open(os.path.join(OUT, fn), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ data
panel, info = C.load_panel(f"{R}/panel.parquet")
raw = pd.read_parquet(f"{R}/panel.parquet")
pooled = pd.read_csv(f"{EV}/metrics_pooled.csv", index_col=0)
byyear = pd.read_csv(f"{EV}/metrics_by_year.csv")
sig = pd.read_csv(f"{EV}/significance.csv")
unc = pd.read_csv(f"{EV}/uncertainty.csv")
uby = pd.read_csv(f"{EV}/uncertainty_by_year.csv")
roll = pd.read_csv(f"{R}/rolling_r2/eval_r2/metrics_pooled.csv", index_col=0)
rollyr = pd.read_csv(f"{R}/rolling_r2/eval_r2/metrics_by_year.csv")
ROB = {"rolling": roll}
for tag, d in (("csrank", "rob_csrank_r2"), ("anyfail", "rob_anyfail_r2"), ("fin", "rob_fin_r2"), ("tfifteen", "rob_train2015_r2")):
    ROB[tag] = pd.read_csv(f"{R}/{d}/eval_r2/metrics_pooled.csv", index_col=0)
ROBYR = {tag: pd.read_csv(f"{R}/{d}/eval_r2/metrics_by_year.csv") for tag, d in
         (("csrank", "rob_csrank_r2"), ("anyfail", "rob_anyfail_r2"), ("fin", "rob_fin_r2"), ("tfifteen", "rob_train2015_r2"))}
dom = pd.read_csv(f"{DG}/domain_auc.csv")
yi = pd.read_csv(f"{DG}/year_intercepts.csv")
itests = pd.read_csv(f"{DG}/intercept_tests.csv")
codes = pd.read_csv(f"{DG}/codes.csv", index_col=0)
fcodes = pd.read_csv(f"{DG}/failure_codes_firmlevel.csv")
desc = pd.read_csv(f"{DG}/descriptives.csv")
cfg_v = json.load(open(f"{R}/vids_r2_cfg_fixed.json"))
cfg_v15 = json.load(open(f"{R}/vids_r2_cfg_t2015.json"))
cfg = json.load(open(f"{R}/main_v2/best_configs.json"))
cfg_s = json.load(open(f"{R}/sngp_res/config.json"))
vdiag = pd.read_csv(f"{R}/main_r2/vids_r2_diag.csv")
toy = pd.read_csv(f"{R}/vids_audit/toy_logistic_keras.csv")


def m(model, col):
    return float(pooled.loc[model, col + "_mean"])


# ------------------------------------------------------------------ sample, label, counts
tr, va, te = C.split_fixed(panel)
put("nRaw", num(info["n_raw"])); put("nExFin", num((~raw["financial"]).sum())); put("nFinal", num(len(panel)))
put("nFirms", num(panel["permno"].nunique()))
put("nTrain", num(len(tr))); put("nVal", num(len(va))); put("nTest", num(len(te)))
put("posAll", num(panel.y.sum())); put("posTrain", num(tr.y.sum())); put("posVal", num(va.y.sum())); put("posTest", num(te.y.sum()))
put("rateTrain", f"{100 * tr.y.mean():.2f}"); put("rateTest", f"{100 * te.y.mean():.2f}"); put("rateVal", f"{100 * va.y.mean():.2f}")
yr_rate = panel.groupby("t").y.mean()
put("rateMin", f"{100 * yr_rate.min():.1f}"); put("rateMinYear", str(int(yr_rate.idxmin())))
put("rateMax", f"{100 * yr_rate.max():.1f}"); put("rateMaxYear", str(int(yr_rate.idxmax())))
for t in range(2019, 2024):
    put(f"fail{key2mac(str(t))}", str(int(panel[panel.t == t].y.sum())))

# delisting codes
def code_share(g, per):
    return codes.loc[g, per] / codes[per].sum() if g in codes.index else 0.0
put("codeEightyFourTrain", pct(code_share("584", "2011-2017"))); put("codeEightyFourTest", pct(code_share("584", "2019-2023")))
put("codeBankTrain", pct(code_share("574", "2011-2017"))); put("codeBankTest", pct(code_share("574", "2019-2023")))
pc_tr = code_share("552", "2011-2017") + code_share("560-561", "2011-2017")
pc_te = code_share("552", "2019-2023") + code_share("560-561", "2019-2023")
put("codePriceCapTrain", pct(pc_tr)); put("codePriceCapTest", pct(pc_te))
put("codeCompanyTrain", pct(code_share("570-573", "2011-2017"))); put("codeCompanyTest", pct(code_share("570-573", "2019-2023")))
# SIC 283
fc = fcodes.copy(); fc["sic"] = pd.to_numeric(fc["sic"], errors="coerce")
fc["bio"] = (fc["sic"] // 10 == 283)
late = fc[fc.t >= 2021]; early = fc[fc.t <= 2017]
put("bioFailLate", pct(late.bio.mean())); put("bioFailTrain", pct(early.bio.mean()))
put("bioCodeEightyFourLate", pct(late[late.dlstcd == 584].bio.mean()))
ps = panel.copy(); ps["bio"] = (pd.to_numeric(ps["sic"], errors="coerce") // 10 == 283)
put("bioSampleLate", pct(ps[ps.t >= 2021].bio.mean())); put("bioSampleTrain", pct(ps[ps.t <= 2017].bio.mean()))
put("bioRateLate", f"{100 * ps[(ps.t >= 2021) & ps.bio].y.mean():.1f}"); put("nonBioRateLate", f"{100 * ps[(ps.t >= 2021) & ~ps.bio].y.mean():.1f}")
# excluded codes (> 584) and liquidations, from the delisting file
dl = pd.read_parquet("../wrds_data/crsp_msedelist.parquet"); dl["dlstcd"] = pd.to_numeric(dl["dlstcd"]); dl["dlstdt"] = pd.to_datetime(dl["dlstdt"])
pk = panel[["permno", "t"]].merge(dl[["permno", "dlstdt", "dlstcd"]], on="permno")
pk = pk[(pk.dlstdt > pd.to_datetime(pk.t.astype(str) + "-06-30")) & (pk.dlstdt <= pd.to_datetime((pk.t + 1).astype(str) + "-06-30"))]
put("codeAboveTrain", str(int(((pk.dlstcd > 584) & (pk.t <= 2017)).sum()))); put("codeAboveTest", str(int(((pk.dlstcd > 584) & (pk.t >= 2019)).sum())))
put("liqTrain", str(int(((pk.dlstcd // 100 == 4) & (pk.t <= 2017)).sum()))); put("liqTest", str(int(((pk.dlstcd // 100 == 4) & (pk.t >= 2019)).sum())))

# sample table
rows = []
f584 = fcodes.groupby("t").apply(lambda g: (g.dlstcd == 584).mean())
for t, g in panel.groupby("t"):
    rows.append(f"{t} & {num(len(g))} & {int(g.y.sum())} & {100 * g.y.mean():.2f} & {num(round(g['me'].median()))} & {g['nita'].median():.3f} & "
                f"{100 * f584.get(t, 0):.0f} & {100 * ps[ps.t == t].bio.mean():.1f} \\\\")
    if t in (2017, 2018):
        rows.append("\\midrule")
table("tab_sample.tex", "Sample by forecast year. A forecast year $t$ runs from 1 July of $t$ to 30 June of $t+1$; failure is a performance delisting "
      "(CRSP codes 500 and 520--584) within that window.", "tab:sample", "rrrrrrrr",
      "Year & Firms & Failures & Rate (\\%) & \\shortstack{Median market\\\\equity (\\$m)} & \\shortstack{Median\\\\NI/TA} & "
      "\\shortstack{Code 584 share\\\\of failures (\\%)} & \\shortstack{SIC 283\\\\firms (\\%)}", rows, size="\\footnotesize",
      notes="Training years 2011--2017, validation year 2018, test years 2019--2023. Code 584: ``does not meet exchange's financial guidelines "
            "for continued listing''. SIC 283: drugs (pharmaceutical and biotechnology firms).")

# ------------------------------------------------------------------ domain classifier
pv_ = dom.pivot(index="t", columns="variant", values="auc")
trn = pv_.loc[2011:2017]
for v, nm in (("all", "All"), ("accounting", "Acct"), ("ranked", "Rank")):
    put(f"dom{nm}TrMin", f2(trn[v].min())); put(f"dom{nm}TrMax", f2(trn[v].max()))
    put(f"dom{nm}LateMin", f2(pv_.loc[2021:2023, v].min())); put(f"dom{nm}LateMax", f2(pv_.loc[2021:2023, v].max()))
    put(f"dom{nm}EarlyMin", f2(pv_.loc[2018:2020, v].min())); put(f"dom{nm}EarlyMax", f2(pv_.loc[2018:2020, v].max()))
rows = []
for t, r in pv_.iterrows():
    rows.append(f"{t} & {r['all']:.2f} & {r['accounting']:.2f} & {r['ranked']:.2f} \\\\")
    if t in (2017, 2018):
        rows.append("\\midrule")
table("tab_domain.tex", "Classifier two-sample tests: out-of-fold AUC of a classifier that separates the firms of each year from those of the "
      "training years 2011--2017 (for a training year, from the other six training years).", "tab:domain", "lccc",
      "Year & All 24 predictors & 14 accounting predictors & All 24, within-year percentiles", rows, size="\\small",
      notes="LightGBM classifier; five-fold cross-fitting with folds grouped by firm, so that no firm is in both the fitting and the scoring "
            "fold. An AUC of 0.5 means the year cannot be told apart from the reference years. Percentiles break ties at random and leave "
            "missing values missing; the three binary predictors are not transformed.")

# ------------------------------------------------------------------ conditional failure risk (yearly intercepts)
def it(contrast, se_type):
    r = itests[(itests.contrast == contrast) & (itests.se_type == se_type)].iloc[0]; return r
a = it("post minus pre (levels)", "years, nonrobust")
put("icptShift", f2(a.estimate)); put("icptShiftSE", f2(a.se)); put("icptShiftP", f"{a.p:.2f}")
put("icptOddsPct", pct(1 - np.exp(a.estimate)))
af = it("post minus pre (levels)", "firm-clustered"); put("icptShiftSEfirm", f2(af.se))
b = it("post deviation from pre trend", "years, nonrobust")
put("icptTrendDev", f2(b.estimate)); put("icptTrendDevSE", f2(b.se)); put("icptTrendDevP", f"{b.p:.2f}")
tr_ = it("pre-period trend per year", "years, nonrobust"); put("icptTrend", f2(tr_.estimate)); put("icptTrendSE", f2(tr_.se))
sd_pre = itests[itests.contrast == "SD of yearly intercepts 2011-17"].estimate.iloc[0]
put("icptSDpre", f2(sd_pre)); put("icptSEmean", f2(itests[itests.contrast == "mean firm-clustered SE of a yearly intercept"].estimate.iloc[0]))
rows = []
for _, r in yi.iterrows():
    rows.append(f"{int(r.t)} & {r.intercept:.2f} & {r.se_firm:.2f} & {int(r.failures)} \\\\")
    if int(r.t) == 2017:
        rows.append("\\midrule")
table("tab_icpt.tex", "Failure risk at given covariates: year intercepts of a pooled logit of failure on the eight CHS-type predictors with "
      "common slopes (predictors standardised with 2011--2017 moments).", "tab:icpt", "lrrr",
      "Year & Intercept & Firm-clustered SE & Failures", rows, size="\\small",
      notes=f"Mean intercept 2019--2023 minus mean 2011--2017: {a.estimate:.2f} (SE {a.se:.2f} with years as observations, $p={a.p:.2f}$; "
            f"firm-clustered SE {af.se:.2f}). Linear trend over 2011--2017: {tr_.estimate:.2f} per year (SE {tr_.se:.2f}). Mean deviation of "
            f"2019--2023 from that trend: {b.estimate:+.2f} (SE {b.se:.2f}, $p={b.p:.2f}$).")

# ------------------------------------------------------------------ main table
MAINROWS = [("Finance benchmarks (pooled logits)", ["logit_altman", "logit_altman5", "logit_ohlson", "logit_chs", "logit_all"]),
            ("Machine learning", ["rf", "lgbm", "mlp"]),
            ("Uncertainty-aware networks", ["ens5", "mcd", "sngp_res", "llla", "vids"])]
rows = []
for g, ms in MAINROWS:
    rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{g}}}}} \\\\")
    for k in ms:
        kp = k + "_platt"
        rows.append(f"{NAMES[k]} & {b2(m(k, 'brier'))} & {b2(m(kp, 'brier'))} & {l2(m(kp, 'logloss'))} & {f2(m(k, 'eo'))} & "
                    f"{f2(m(kp, 'eo'))} & {f2(m(kp, 'cal_slope'))} & {f3(m(k, 'auc'))} & {f3(m(k, 'ap'))} \\\\")
table("tab_main.tex", "Out-of-time accuracy, fixed design (test years 2019--2023; 14{,}470 firm-years, 286 failures). Every model is shown as "
      "estimated (raw) and after Platt recalibration on the 2018 validation year.", "tab:main", "lrrrrrrrr",
      "& \\multicolumn{2}{c}{Brier $\\times10^{2}$} & Log loss & \\multicolumn{2}{c}{E/O} & Slope & AUC & AP \\\\ "
      "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6} Model & raw & Platt & Platt & raw & Platt & Platt & & ", rows, size="\\footnotesize",
      notes="Seed means over ten random seeds for the networks and tree ensembles (the logits are deterministic); standard deviations are in "
            "Table~\\ref{tab:sd}. Log loss $\\times10^{2}$. E/O: expected over observed failures. Slope: calibration slope. AUC and AP (average "
            "precision) are unchanged by Platt scaling, which is monotone. The Brier score of a constant forecast equal to the test-period "
            f"failure rate is {b2(pooled['unc_mean'].iloc[0])}$\\times10^{{-2}}$.")
for k in ["logit_altman", "logit_altman5", "logit_ohlson", "logit_chs", "logit_all", "rf", "lgbm", "mlp", "ens5", "mcd", "sngp_res", "llla", "vids",
          "sngp", "lgbm_iso", "mlp_ts", "ens5_ts", "mcd_ts", "sngp_res_ts", "llla_ts", "vids_ts"]:
    for kk in (k, k + "_platt"):
        if kk in pooled.index:
            M_ = key2mac(kk)
            put(f"brier{M_}", b2(m(kk, "brier"))); put(f"ll{M_}", l2(m(kk, "logloss"))); put(f"eo{M_}", f2(m(kk, "eo")))
            put(f"auc{M_}", f3(m(kk, "auc"))); put(f"ap{M_}", f3(m(kk, "ap"))); put(f"slope{M_}", f2(m(kk, "cal_slope")))
            put(f"bss{M_}", pct(m(kk, "bss"), 1)); put(f"mal{M_}", f2(m(kk, "mal_eo"))); put(f"topten{M_}", pct(m(kk, "top10")))
            put(f"sdbrier{M_}", f"{100 * float(pooled.loc[kk, 'brier_std']):.3f}")
put("brierConst", b2(pooled["unc_mean"].iloc[0]))
flex = ["rf", "lgbm", "mlp", "ens5", "mcd", "sngp_res", "llla", "vids", "logit_chs", "logit_all"]
put("aucFlexMin", f3(min(m(k, "auc") for k in flex))); put("aucFlexMax", f3(max(m(k, "auc") for k in flex)))
plat = [k + "_platt" for k in ["logit_chs", "logit_all", "rf", "lgbm", "mlp", "ens5", "mcd", "sngp_res", "llla", "vids"]]
put("eoPlattMin", f2(min(m(k, "eo") for k in plat))); put("eoPlattMax", f2(max(m(k, "eo") for k in plat)))
nets = ["mlp_platt", "ens5_platt", "mcd_platt", "sngp_res_platt", "llla_platt", "vids_platt"]
put("brierNetPlattMin", b2(min(m(k, "brier") for k in nets))); put("brierNetPlattMax", b2(max(m(k, "brier") for k in nets)))
put("topTenMin", pct(min(m(k, "top10") for k in flex))); put("topTenMax", pct(max(m(k, "top10") for k in flex)))
put("malPlattMin", f2(min(m(k, "mal_eo") for k in plat))); put("malPlattMax", f2(max(m(k, "mal_eo") for k in plat)))
slopes = [m(k, "cal_slope") for k in plat]
put("slopePlattMin", f2(min(slopes))); put("slopePlattMax", f2(max(slopes)))

# SD table and recalibration comparison (appendix)
rows = []
for g, ms in MAINROWS[1:]:
    for k in ms:
        r = [k, k + "_platt"] + ([k + "_ts"] if k + "_ts" in pooled.index else [])
        cells = [f"{b2(m(x, 'brier'))} ({100 * pooled.loc[x, 'brier_std']:.3f})" for x in (k, k + "_platt")]
        ts = f"{b2(m(k + '_ts', 'brier'))} ({100 * pooled.loc[k + '_ts', 'brier_std']:.3f})" if k + "_ts" in pooled.index else "--"
        eots = f2(m(k + "_ts", "eo")) if k + "_ts" in pooled.index else "--"
        rows.append(f"{NAMES[k]} & {cells[0]} & {ts} & {cells[1]} & {f2(m(k, 'eo'))} & {eots} & {f2(m(k + '_platt', 'eo'))} & "
                    f"{f3(m(k, 'auc'))} ({pooled.loc[k, 'auc_std']:.3f}) \\\\")
table("tab_sd.tex", "Seed variability and the choice of recalibration (fixed design): mean (standard deviation across ten seeds).", "tab:sd",
      "lrrrrrrr", "& \\multicolumn{3}{c}{Brier $\\times10^{2}$} & \\multicolumn{3}{c}{E/O} & \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7} "
      "Model & raw & temperature & Platt & raw & temp. & Platt & AUC", rows, size="\\footnotesize",
      notes="Temperature scaling divides the logit by one constant fitted on 2018; Platt scaling also shifts it. For rare events the shift "
            "matters: temperature scaling cannot lower the level of the forecasts without also compressing their spread.")

# ------------------------------------------------------------------ significance
def sigrow(A, B, kind):
    r = sig[(sig.A == A) & (sig.B == B) & (sig.metric == kind)].iloc[0]; return r
rows = []
for A, label in (("vids_platt", "A. VIDS + Platt minus the comparator"), ("logit_all_platt", "B. Logit, all 24 predictors + Platt minus the comparator")):
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{label}}}}} \\\\")
    for B in ["logit_chs", "logit_chs_platt", "logit_all_platt", "rf_platt", "lgbm_platt", "mlp_platt", "ens5_platt", "mcd_platt", "sngp_res_platt",
              "llla_platt", "vids_platt"]:
        if B == A:
            continue
        rb, rl = sigrow(A, B, "brier"), sigrow(A, B, "logloss")
        rows.append(f"{NAMES[B]} & {1e3 * rb['diff']:.2f} [{1e3 * rb.lo:.2f}, {1e3 * rb.hi:.2f}] & {pv(rb.p_holm)} & "
                    f"{1e3 * rl['diff']:.2f} [{1e3 * rl.lo:.2f}, {1e3 * rl.hi:.2f}] & {pv(rl.p_holm)} \\\\")
        for kind, rr in (("B", rb), ("L", rl)):
            nm = f"sig{key2mac(A)}{kind}{key2mac(B)}"
            put(nm, f"{1e3 * rr['diff']:.2f}"); put(nm + "Lo", f"{1e3 * rr.lo:.2f}"); put(nm + "Hi", f"{1e3 * rr.hi:.2f}"); put(nm + "P", pv(rr.p_holm))
table("tab_sig.tex", "Paired comparisons of loss, fixed design: difference in mean loss $\\times10^{3}$ (negative = first model better), 95\\% "
      "firm-clustered bootstrap interval and Holm-adjusted $p$-value.", "tab:sig", "lrrrr",
      "Comparator & Brier difference & $p_{\\text{Holm}}$ & Log-loss difference & $p_{\\text{Holm}}$", rows, size="\\footnotesize",
      notes="1{,}000 bootstrap draws of firms (all years of a firm are kept together); per-observation losses are averaged over seeds before "
            "differencing. Holm adjustment within each panel and loss. The bootstrap does not capture shocks common to all firms in a year "
            "(Table~\\ref{tab:yeardiff}).")
aucs = sig[sig.metric == "auc"]
put("aucSigMinP", pv(aucs.p_holm.min()))

# year differences (appendix)
yd = pd.read_csv(f"{EV}/year_diffs.csv")
rows = []
for A, B in (("logit_all_platt", "ens5_platt"), ("logit_all_platt", "vids_platt"), ("logit_all_platt", "lgbm_platt"), ("vids_platt", "mlp_platt"),
             ("vids_platt", "ens5_platt")):
    d = yd[(yd.A == A) & (yd.B == B)].set_index("t")["diff"] if ((yd.A == A) & (yd.B == B)).any() else -yd[(yd.A == B) & (yd.B == A)].set_index("t")["diff"]
    rows.append(f"{NAMES[A]} vs {NAMES[B].replace(' + Platt', '')} & " + " & ".join(f"{1e3 * d.loc[t]:.2f}" for t in range(2019, 2024)) + " \\\\")
    if (A, B) == ("logit_all_platt", "ens5_platt"):
        put("ydLogitEnsTwenty", f"{1e3 * d.loc[2020]:.2f}")
table("tab_yeardiff.tex", "Brier-score differences by test year ($\\times10^{3}$; negative = first model better), fixed design, Platt-recalibrated "
      "models.", "tab:yeardiff", "lrrrrr", "Pair & 2019 & 2020 & 2021 & 2022 & 2023", rows, size="\\footnotesize")

# ------------------------------------------------------------------ calibration by year
BY = ["logit_chs_platt", "logit_all_platt", "rf_platt", "lgbm_platt", "mlp_platt", "ens5_platt", "mcd_platt", "sngp_res_platt", "llla_platt", "vids_platt"]
rows = []
yrs = list(range(2019, 2024))
cnt = byyear[byyear.model == "logit_all_platt"].set_index("t")
rows.append("Failures / firm-years & " + " & ".join(f"{int(cnt.loc[t, 'failures'])}/{num(cnt.loc[t, 'n'])}" for t in yrs) + " & \\\\")
rows.append("95\\% range if calibrated & " + " & ".join(f"{cnt.loc[t, 'band_lo']:.2f}--{cnt.loc[t, 'band_hi']:.2f}" for t in yrs) + " & \\\\")
rows.append("\\midrule")
for k in BY:
    d = byyear[byyear.model == k].set_index("t")
    rows.append(f"{NAMES[k]} & " + " & ".join(f2(d.loc[t, "eo"]) for t in yrs) + f" & {f2(m(k, 'mal_eo'))} \\\\")
    for t in yrs:
        put(f"eoYr{key2mac(k)}{key2mac(str(t))}", f2(d.loc[t, "eo"]))
table("tab_byyear.tex", "Calibration in the large by test year, fixed design: expected over observed failures (E/O) of the Platt-recalibrated models.",
      "tab:byyear", "lrrrrrr", "Model & 2019 & 2020 & 2021 & 2022 & 2023 & Mean $|\\log \\mathrm{E/O}|$", rows, size="\\footnotesize",
      notes="E/O from seed-averaged forecasts. The 95\\% range is the central range of E/O that sampling variation alone would produce if the "
            "outcomes were drawn from the all-variable logit's own forecasts. Mean $|\\log\\mathrm{E/O}|$ is averaged over seeds.")
for t in yrs:
    put(f"bandLo{key2mac(str(t))}", f2(cnt.loc[t, "band_lo"])); put(f"bandHi{key2mac(str(t))}", f2(cnt.loc[t, "band_hi"]))
e20 = [byyear[(byyear.model == k) & (byyear.t == 2020)].eo.iloc[0] for k in BY]
put("eoTwentyMin", f2(min(e20))); put("eoTwentyMax", f2(max(e20)))
e22 = [byyear[(byyear.model == k) & (byyear.t == 2022)].eo.iloc[0] for k in BY]
put("eoTwentyTwoMin", f2(min(e22))); put("eoTwentyTwoMax", f2(max(e22)))
e23 = [byyear[(byyear.model == k) & (byyear.t == 2023)].eo.iloc[0] for k in BY]
put("eoTwentyThreeMin", f2(min(e23))); put("eoTwentyThreeMax", f2(max(e23)))
e21 = [byyear[(byyear.model == k) & (byyear.t == 2021)].eo.iloc[0] for k in BY]
put("eoTwentyOneMin", f2(min(e21))); put("eoTwentyOneMax", f2(max(e21)))
e19 = [byyear[(byyear.model == k) & (byyear.t == 2019)].eo.iloc[0] for k in BY]
put("eoNineteenMin", f2(min(e19))); put("eoNineteenMax", f2(max(e19)))
# mean predicted rate vs training rate (why everything over-predicts)
pm = byyear[byyear.model == "logit_all"].expected.sum() / len(te)
put("predRateLogitAll", f"{100 * pm:.2f}")

# ------------------------------------------------------------------ shift corrections and annual re-estimation
rows = ["\\multicolumn{5}{l}{\\emph{A. Fixed design (estimated on 2011--2017, recalibrated on 2018)}} \\\\"]
SH = [("logit_all_platt", "Logit, all 24 + Platt (reference)"),
      ("logit_all_iwacct_platt", "\\quad importance-weighted, accounting-variable classifier"),
      ("logit_all_iw_platt", "\\quad importance-weighted, all-variable classifier"),
      ("logit_all_platt_lag", "\\quad intercept set to last year's failure rate"),
      ("logit_all_platt_lag3", "\\quad intercept set to the last three years' failure rate"),
      ("logit_all_platt_em", "\\quad EM prior-shift correction"),
      ("ens5_platt", "Deep ensemble + Platt (reference)"),
      ("ens5_platt_lag", "\\quad intercept set to last year's failure rate"),
      ("ens5_platt_em", "\\quad EM prior-shift correction"),
      ("vids_platt", "VIDS + Platt (reference)"),
      ("vids_platt_lag", "\\quad intercept set to last year's failure rate"),
      ("vids_platt_em", "\\quad EM prior-shift correction")]
for k, lab in SH:
    rows.append(f"{lab} & {b2(m(k, 'brier'))} & {f2(m(k, 'eo'))} & {f2(m(k, 'mal_eo'))} & {f3(m(k, 'auc'))} \\\\")
    M_ = key2mac(k); put(f"brier{M_}", b2(m(k, "brier"))); put(f"eo{M_}", f2(m(k, "eo"))); put(f"mal{M_}", f2(m(k, "mal_eo")))
rows.append("\\midrule")
rows.append("\\multicolumn{5}{l}{\\emph{B. Rolling design (re-estimated each year on 2011 to $T-2$, recalibrated on $T-1$; three seeds)}} \\\\")
RO = [("logit_all", "Logit, all 24 (raw)"), ("logit_all_platt", "Logit, all 24 + Platt"), ("rf_platt", "Random forest + Platt"), ("lgbm_platt", "LightGBM + Platt"),
      ("mlp_platt", "MLP + Platt"), ("ens5", "Deep ensemble (raw)"), ("ens5_platt", "Deep ensemble + Platt"), ("vids", "VIDS (raw)"), ("vids_platt", "VIDS + Platt")]
for k, lab in RO:
    r = roll.loc[k]
    rows.append(f"{lab} & {b2(r['brier_mean'])} & {f2(r['eo_mean'])} & {f2(r['mal_eo_mean'])} & {f3(r['auc_mean'])} \\\\")
    M_ = key2mac(k); put(f"rollBrier{M_}", b2(r["brier_mean"])); put(f"rollEo{M_}", f2(r["eo_mean"])); put(f"rollMal{M_}", f2(r["mal_eo_mean"]))
table("tab_shift.tex", "Explicit shift corrections and annual re-estimation (test years 2019--2023).", "tab:shift", "lrrrr",
      "Model & Brier $\\times10^{2}$ & E/O & Mean $|\\log\\mathrm{E/O}|$ & AUC", rows, size="\\footnotesize",
      notes="Importance weights are odds from a cross-fitted classifier that separates the training firm-years from the firms of the forecast "
            "year, clipped at the 99th percentile. Rate-based intercepts use only failure rates realised by the forecast date. The EM "
            "correction re-estimates the failure rate of each forecast year from its unlabelled forecasts under a pure label-shift model.")
ess = pd.read_csv(f"{R}/iwacct/ess_by_year.csv", index_col=0).iloc[:, 0]
put("essAcctMin", num(ess.min())); put("essAcctMinYear", str(int(ess.idxmin())))
ess2 = pd.read_csv(f"{R}/iw/ess_by_year.csv", index_col=0).iloc[:, 0]
put("essAllMin", num(ess2.min())); put("essAllMinYear", str(int(ess2.idxmin())))
put("rollMalMin", f2(min(roll.loc[k, "mal_eo_mean"] for k in ["logit_all_platt", "rf_platt", "lgbm_platt", "mlp_platt", "ens5_platt", "vids_platt"])))
put("rollMalMax", f2(max(roll.loc[k, "mal_eo_mean"] for k in ["logit_all_platt", "rf_platt", "lgbm_platt", "mlp_platt", "ens5_platt", "vids_platt"])))
ry = rollyr[rollyr.model == "logit_all_platt"].set_index("t")
for t in yrs:
    put(f"rollEoLogitAllPlattYr{key2mac(str(t))}", f2(ry.loc[t, "eo"]))
ry = rollyr[rollyr.model == "ens5_platt"].set_index("t")
for t in yrs:
    put(f"rollEoEnsPlattYr{key2mac(str(t))}", f2(ry.loc[t, "eo"]))
best_roll = roll.loc[[k for k in roll.index if k.endswith("_platt") or k in ("logit_all", "vids", "rf", "lgbm", "mlp", "ens5", "mcd", "llla")], "brier_mean"].idxmin()
put("rollBestName", NAMES.get(best_roll, best_roll))

# ------------------------------------------------------------------ VIDS table
def valll(model):
    """2018 log loss of a model, seed mean, from the stored validation forecasts."""
    files = sorted(glob.glob(f"{R}/main_r2/fixed/seed_*.pkl"))
    yv = pd.read_parquet(f"{R}/main_r2/fixed/val_index.parquet")["y"].to_numpy()
    out = []
    for f in files:
        d = pickle.load(open(f, "rb"))
        p = np.clip(d[model]["p_val"], 1e-6, 1 - 1e-6)
        out.append(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    return float(np.mean(out))
VR = [("mlp", "MLP (the network VIDS builds on)"), ("vids", "VIDS, tuned on 2018 (main)"), ("vids_ext", "VIDS, 40-draw search"),
      ("vids_authors", "VIDS, authors' settings"),
      ("vids_lam1e3", "VIDS, $\\lambda=10^{-3}$"), ("vids_lam5e3", "VIDS, $\\lambda=5\\times10^{-3}$"), ("vids_lam1e2", "VIDS, $\\lambda=10^{-2}$"),
      ("vids_lam5e2", "VIDS, $\\lambda=5\\times10^{-2}$"),
      ("vids_tau1e2", "VIDS, $\\tau=0.01$"), ("vids_fixedprior", "VIDS, prior without pseudo-test term"), ("vids_malt", "VIDS, $m=200$"),
      ("vids_pre", "VIDS, started at the MLP head")]
rows = []
for k, lab in VR:
    vl = valll(k)
    rows.append(f"{lab} & {100 * vl:.2f} & {b2(m(k, 'brier'))} & {f2(m(k, 'eo'))} & {b2(m(k + '_platt', 'brier'))} & {f3(m(k, 'auc'))} \\\\")
    M_ = key2mac(k); put(f"val{M_}", f"{100 * vl:.2f}")
    put(f"brier{M_}", b2(m(k, "brier"))); put(f"eo{M_}", f2(m(k, "eo"))); put(f"auc{M_}", f3(m(k, "auc")))
    if k + "_platt" in pooled.index:
        put(f"brier{key2mac(k + '_platt')}", b2(m(k + "_platt", "brier")))
    if k == "vids_ext":
        rows.append("\\midrule")
    if k == "vids_lam5e2":
        pass
table("tab_vids.tex", "VIDS: tuning, the authors' settings, the weight of the prior and ablations (fixed design, ten seeds).", "tab:vids", "lrrrrr",
      "Specification & \\shortstack{2018 log loss\\\\$\\times10^{2}$} & \\shortstack{Brier $\\times10^{2}$\\\\raw} & \\shortstack{E/O\\\\raw} & "
      "\\shortstack{Brier $\\times10^{2}$\\\\+ Platt} & AUC", rows, size="\\footnotesize",
      notes=f"Main specification: the authors' algorithm with the hyper-parameters selected on 2018 (K={cfg_v['main']['best']['K']} steps, "
            f"$\\lambda=10^{{-4}}$, $\\tau={cfg_v['main']['best']['tau']}$, $m={cfg_v['main']['best']['m']}$, learning rate "
            f"{cfg_v['main']['best']['lr']}). Authors' settings: those of their binary real-data experiment (K=50, $\\lambda=0.005$, "
            "$\\tau=0.001$, $m=20$, learning rate 0.001). The other rows change one setting of the main specification. All VIDS rows use the "
            "frozen hidden layers of the MLP of the same seed.")
vb = cfg_v["main"]["best"]
put("vidsK", str(vb["K"])); put("vidsTau", str(vb["tau"])); put("vidsM", str(vb["m"])); put("vidsLr", str(vb["lr"]))
put("vidsExtK", str(cfg_v["ext"]["best"]["K"])); put("vidsExtM", str(cfg_v["ext"]["best"]["m"])); put("vidsExtDistinct", str(cfg_v["ext"]["distinct"]))
lam52 = vdiag[vdiag.model == "vids_lam5e2"]
put("lamBigDiverged", str(int((lam52.val_final > 0.1).sum())))
put("valMlpMain", f"{100 * cfg_v['base_mlp_val']:.2f}")
put("valVidsTfifteen", f"{100 * cfg_v15['main']['best_val']:.2f}"); put("valMlpTfifteen", f"{100 * cfg_v15['base_mlp_val']:.2f}")

# toy validation
tg = toy.groupby("model").agg(["mean", "std"])
put("toyAccVids", f3(tg.loc["VIDS", ("acc", "mean")])); put("toyAccVidsSD", f3(tg.loc["VIDS", ("acc", "std")]))
put("toyAccMlp", f3(tg.loc["MLP", ("acc", "mean")]))
put("toyMaeVids", f3(tg.loc["VIDS", ("mae_true", "mean")])); put("toyMaeMlp", f3(tg.loc["MLP", ("mae_true", "mean")]))
put("toySdGap", f3(tg.loc["VIDS", ("sd_gap", "mean")])); put("toySdEdges", f3(tg.loc["VIDS", ("sd_edges", "mean")]))
put("toyReps", str(int(toy.rep.nunique())))

# ------------------------------------------------------------------ uncertainty table
LAB = {"ens5": "Deep ensemble", "mcd": "MC dropout", "sngp_res": "SNGP", "llla": "Last-layer Laplace", "vids": "VIDS", "vids_authors": "VIDS, authors' settings"}
rows = []
for _, r in unc.iterrows():
    late = uby[(uby.model == r.model) & (uby.scale == r.scale) & (uby.t >= 2021)].rel_val_resid.mean()
    rows.append(f"{LAB[r.model]} & {r.scale} & {r.spearman:.2f} & {r.r2_spline:.2f} & {b2(r.drop_u10)} & {b2(r.drop_p10)} & "
                f"{100 * r.matched10:+.3f} [{100 * r.matched10_lo:+.3f}, {100 * r.matched10_hi:+.3f}] & {pv(r.p_joint_holm)} & {late:.2f} \\\\")
    M_ = key2mac(r.model) + ("Prob" if r.scale == "prob" else "Logit")
    put(f"rho{M_}", f2(r.spearman)); put(f"rsq{M_}", f2(r.r2_spline)); put(f"dropU{M_}", b2(r.drop_u10)); put(f"dropP{M_}", b2(r.drop_p10))
    put(f"matched{M_}", f"{100 * r.matched10:+.3f}"); put(f"matched{M_}Lo", f"{100 * r.matched10_lo:+.3f}"); put(f"matched{M_}Hi", f"{100 * r.matched10_hi:+.3f}")
    put(f"pJoint{M_}", pv(r.p_joint_holm)); put(f"pJointRaw{M_}", pv(r.p_joint)); put(f"resLate{M_}", f2(late))
    put(f"matchedPct{M_}", f"{100 * r.matched10 / r.brier_all:+.1f}"); put(f"matchedAbsPct{M_}", f"{abs(100 * r.matched10 / r.brier_all):.1f}")
    put(f"cz{M_}", f"{r.c_z:+.2f}")
table("tab_unc.tex", "What epistemic uncertainty measures (fixed design, seed-averaged forecasts and uncertainties).", "tab:unc", "llrrrrrrr",
      "Model & Scale & $\\rho(u,\\hat p)$ & $R^2$ & \\shortstack{Drop top 10\\%\\\\by $u$} & \\shortstack{Drop top 10\\%\\\\by $\\hat p$} & "
      "\\shortstack{Drop top 10\\% by residual $u$\\\\within risk bins: $\\Delta$Brier} & \\shortstack{Joint test\\\\$p_{\\text{Holm}}$} & "
      "\\shortstack{Residual $u$\\\\2021--23}", rows, size="\\scriptsize", resize=True,
      notes="$u$: standard deviation of the predicted probability (prob) or of its logit (logit) across ensemble members, dropout masks or "
            "posterior draws. $\\rho$: Spearman correlation with the forecast. $R^2$: share of the variance of $\\log u$ explained by a cubic "
            "spline in $\\operatorname{logit}\\hat p$; the residual $r$ is what remains. Drop columns: Brier score $\\times10^{2}$ of the retained "
            "forecasts (all forecasts: about 1.61--1.67). $\\Delta$Brier $\\times10^{2}$: change in the Brier score when the 10\\% of forecasts "
            "with the highest $r$ are dropped within each of 20 bins of $\\hat p$, with a 95\\% firm-clustered bootstrap interval (random removal "
            "leaves it unchanged in expectation). Joint test: Wald test that $r$ and $r\\times\\operatorname{logit}\\hat p$ add nothing to a "
            "logistic recalibration of the outcome on $\\operatorname{logit}\\hat p$ (firm-clustered; Holm-adjusted over the twelve tests). "
            "Residual $u$ 2021--23: mean of $\\exp(r)$ in 2021--2023 (1 = no change relative to the whole test period).")
pr_ = unc[unc.scale == "prob"]
put("rhoProbMin", f2(pr_.spearman.min())); put("rhoProbMax", f2(pr_.spearman.max()))
put("rsqProbMin", f2(pr_.r2_spline.min()))
ubv = uby[(uby.scale == "prob")]
for k in ("vids", "ens5"):
    for t in (2021, 2023):
        put(f"uRel{key2mac(k)}{key2mac(str(t))}", f2(ubv[(ubv.model == k) & (ubv.t == t)].rel_val.iloc[0]))
late_res = uby[uby.t >= 2021].groupby(["model", "scale"]).rel_val_resid.mean()
put("resLateMin", f2(late_res.min())); put("resLateMax", f2(late_res.max()))

# ------------------------------------------------------------------ robustness
DES = [("Fixed design (main; ten seeds)", None), ("Rolling re-estimation", "rolling"), ("Market variables as within-year percentiles", "csrank"),
       ("Broader label: adds liquidations", "anyfail"), ("Financial firms included", "fin"), ("Training 2011--2015, tuned on 2016, test 2017--2023", "tfifteen")]
RM = ["logit_chs_platt", "logit_all", "logit_all_platt", "rf_platt", "lgbm_platt", "ens5_platt", "vids", "vids_platt"]
RML = ["CHS + Platt", "All 24 (raw)", "All 24 + Platt", "RF + Platt", "LGBM + Platt", "Ensemble + Platt", "VIDS (raw)", "VIDS + Platt"]
rows = ["\\multicolumn{9}{l}{\\emph{A. Brier score $\\times10^{2}$}} \\\\"]
ranks = {}
for lab, tag in DES:
    P_ = pooled.rename(columns=lambda c: c) if tag is None else ROB[tag]
    vals = [P_.loc[k, "brier_mean"] for k in RM]
    best = int(np.argmin(vals))
    cells = [("\\textbf{" + b2(v) + "}") if i == best else b2(v) for i, v in enumerate(vals)]
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    base = ["logit_chs", "logit_all", "rf", "lgbm", "mlp", "ens5", "mcd", "llla", "sngp_res" if tag is None else "sngp", "vids"]
    cand = [k for b in base for k in (b, b + "_platt") if k in P_.index]
    order = P_.loc[cand, "brier_mean"].sort_values()
    ranks[tag or "fixed"] = (list(order.index).index("vids_platt") + 1, len(order), order.index[0])
rows.append("\\midrule")
rows.append("\\multicolumn{9}{l}{\\emph{B. Expected over observed failures (E/O)}} \\\\")
for lab, tag in DES:
    P_ = pooled if tag is None else ROB[tag]
    rows.append(f"{lab} & " + " & ".join(f2(P_.loc[k, "eo_mean"]) for k in RM) + " \\\\")
rows.append("\\midrule")
rows.append("\\multicolumn{9}{l}{\\emph{C. Mean $|\\log\\mathrm{E/O}|$ across test years}} \\\\")
for lab, tag in DES:
    P_ = pooled if tag is None else ROB[tag]
    rows.append(f"{lab} & " + " & ".join(f2(P_.loc[k, "mal_eo_mean"]) for k in RM) + " \\\\")
table("tab_robust.tex", "Robustness: alternative designs. Bold: lowest Brier score in the row among the models shown.", "tab:robust",
      "p{4.2cm}rrrrrrrr", "Design & " + " & ".join(RML), rows, size="\\scriptsize", resize=True,
      notes="Alternative designs reuse the hyper-parameters selected on 2018 (except the 2011--2015 design, which is re-tuned on 2016) and use "
            "three seeds; Platt recalibration always uses the validation year of the design. The 2011--2015 design is tested on 2017--2023, so "
            "its levels are not comparable with the other rows.")
for tag, (rk, n, best) in ranks.items():
    put(f"rank{key2mac(tag)}", str(rk)); put(f"rankN{key2mac(tag)}", str(n)); put(f"best{key2mac(tag)}", NAMES.get(best, best))
put("vidsRankMin", str(min(v[0] for v in ranks.values()))); put("vidsRankMax", str(max(v[0] for v in ranks.values())))
put("robModelsN", str(max(v[1] for v in ranks.values())))
for tag in ("csrank", "anyfail", "fin", "tfifteen"):
    P_ = ROB[tag]
    for k in ("logit_all", "logit_all_platt", "ens5_platt", "vids", "vids_platt", "lgbm_platt", "rf_platt", "mlp_platt"):
        M_ = key2mac(k)
        put(f"rob{key2mac(tag)}Brier{M_}", b2(P_.loc[k, "brier_mean"])); put(f"rob{key2mac(tag)}Eo{M_}", f2(P_.loc[k, "eo_mean"]))
        put(f"rob{key2mac(tag)}Mal{M_}", f2(P_.loc[k, "mal_eo_mean"]))
cy = ROBYR["csrank"]
for k in ("vids_platt", "logit_all_platt", "lgbm_platt"):
    d = cy[cy.model == k].set_index("t")
    for t in yrs:
        put(f"robCsrankEoYr{key2mac(k)}{key2mac(str(t))}", f2(d.loc[t, "eo"]))

# ------------------------------------------------------------------ descriptives and codes (appendix)
VARS = C.FEATURE_SETS["all"]
LBL = {"nimta": "NIMTA", "tlmta": "TLMTA", "exret": "EXRET", "sigma": "SIGMA", "rsize": "RSIZE", "cashmta": "CASHMTA", "mb": "MB", "price": "PRICE",
       "size": "SIZE", "tlta": "TLTA", "wcta": "WCTA", "clca": "CLCA", "oeneg": "OENEG", "nita": "NITA", "futl": "FUTL", "intwo": "INTWO",
       "chin": "CHIN", "re_ta": "RE/TA", "ebit_ta": "EBIT/TA", "me_tl": "ME/TL", "sales_ta": "S/TA", "sgrowth": "SGROWTH", "ret12": "RET12",
       "bm_neg": "BMNEG"}
rows = []
for v in VARS:
    d = desc[desc["var"] == v].set_index(["period", "failed"])
    nd = 3
    cell = lambda per, f: f"{d.loc[(per, f), 'stat']:.{nd}f}"
    miss = d.loc[("train", 0), "missing"]
    rows.append(f"{LBL[v]}{'$^\\dagger$' if v in C.BINARY else ''} & {cell('train', 0)} & {cell('train', 1)} & {cell('test', 0)} & {cell('test', 1)} \\\\")
table("tab_desc.tex", "Predictors by period and outcome: medians (means for the binary indicators, marked $\\dagger$).", "tab:desc", "lrrrr",
      "& \\multicolumn{2}{c}{2011--2017} & \\multicolumn{2}{c}{2019--2023} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5} Predictor & no failure & failure & "
      "no failure & failure", rows, size="\\footnotesize", notes="Values before winsorisation and standardisation. Definitions in Table~\\ref{tab:vars}.")
CODEROWS = [("520-551", "520--551: moved to OTC; too few market makers or shareholders"), ("552", "552: price below acceptable level"),
            ("560-561", "560--561: insufficient capital, surplus or equity; insufficient float or assets"),
            ("570-573", "570--573: company request (incl.\\ 573, deregistration)"), ("574", "574: bankruptcy, declared insolvent"),
            ("580-583", "580--583: delinquent filing or fees; other rule violations"), ("584", "584: does not meet financial guidelines")]
rows = []
for g, lab in CODEROWS:
    r = codes.loc[g] if g in codes.index else pd.Series({"2011-2017": 0, "2018": 0, "2019-2023": 0})
    rows.append(f"{lab} & {int(r['2011-2017'])} & {int(r['2018'])} & {int(r['2019-2023'])} \\\\")
rows.append("\\midrule")
rows.append(f"All performance delistings & {int(codes['2011-2017'].sum())} & {int(codes['2018'].sum())} & {int(codes['2019-2023'].sum())} \\\\")
table("tab_codes.tex", "CRSP delisting codes of the failures.", "tab:codes", "p{8.3cm}rrr", "Code & 2011--2017 & 2018 & 2019--2023", rows,
      size="\\footnotesize", notes=f"Code 500 does not occur. Not counted as failures: codes above 584 ({MAC['codeAboveTrain']} delistings in "
      f"2011--2017 and {MAC['codeAboveTest']} in 2019--2023) and liquidations, codes 400--499 ({MAC['liqTrain']} and {MAC['liqTest']}; added in "
      "the broader label of Section~\\ref{sec:rob}).")

# ------------------------------------------------------------------ hyper-parameters (appendix)
dist = pd.read_csv(f"{R}/diagnostics_v2/tuning_distinct.csv").set_index("model")["distinct"]
def fmt(d): return ", ".join(f"{k}={v}" for k, v in d.items())
HP = [("LightGBM", fmt({k: v for k, v in cfg["lgbm"].items()}) + "; early stopping on 2018", int(dist["lgbm"])),
      ("Random forest", fmt(cfg["rf"]) + "; 400 trees", int(dist["rf"])),
      ("MLP (also ensemble members)", "hidden layers (128, 64), 8-dimensional last hidden layer, learning rate 0.003, weight decay 0.001; "
       "batch 512, up to 150 epochs, early stopping on 2018 (patience 15)", int(dist["mlp"])),
      ("MC dropout", f"MLP settings, dropout rate {cfg['mcd']['drop_p']}; 100 masks at prediction", int(dist["mcd"])),
      ("SNGP", f"residual network, width {cfg_s['best']['width']}, {cfg_s['best']['n_blocks']} blocks, spectral-norm bound "
       f"{cfg_s['best']['norm_c']}, 256 random features, scale {cfg_s['best']['scale']}, ridge {cfg_s['best']['ridge']}", 8),
      ("Last-layer Laplace", f"prior precision {cfg['llla']['prior_prec']}; 300 posterior draws", int(dist["llla"])),
      ("VIDS", f"K={vb['K']}, $\\lambda$={vb['lam']}, $\\tau$={vb['tau']}, m={vb['m']}, learning rate {vb['lr']}; fixed: J=10, n=1000, "
       "posterior network (256, 128, 64, 32); 300 posterior draws", int(cfg_v["main"]["distinct"]))]
rows = [f"{a} & {esc(b) if '$' not in b else b} & {c} \\\\" for a, b, c in HP]
table("tab_tuned.tex", "Hyper-parameters selected on the 2018 validation year (eight random draws per model, validation log loss).", "tab:tuned",
      "lp{8.7cm}r", "Model & Selected configuration & Distinct draws", rows, size="\\footnotesize",
      notes="Draws are made with replacement, so small grids yield fewer distinct configurations (MC dropout has three possible values). The "
            "logits have no hyper-parameters (scikit-learn with a negligible $L_2$ penalty, $C=10^{4}$, on standardised predictors).")


# ------------------------------------------------------------------ extra paired comparisons
ex = pd.read_csv(f"{EV}/extra_sig.csv")
for _, r in ex.iterrows():
    nm = f"ex{key2mac(r.A)}Vs{key2mac(r.B)}"
    put(nm, f"{1e3 * r['diff']:.2f}"); put(nm + "Lo", f"{1e3 * r.lo:.2f}"); put(nm + "Hi", f"{1e3 * r.hi:.2f}"); put(nm + "P", pv(r.p))

# ------------------------------------------------------------------ write macros
with open(os.path.join(OUT, "macros.tex"), "w") as f:
    for k, v in MAC.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(MAC, open(os.path.join(OUT, "macros.json"), "w"), indent=0)
print(len(MAC), "macros;", len(os.listdir(OUT)), "files in", OUT)
