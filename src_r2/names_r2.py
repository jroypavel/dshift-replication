"""Display names and groupings for the revised paper (revision 2)."""
NAMES = {
    "logit_altman": "Logit, Altman Z-score", "logit_altman5": "Logit, Altman ratios", "logit_ohlson": "Logit, Ohlson-type",
    "logit_chs": "Logit, CHS-type", "logit_all": "Logit, all 24 predictors",
    "rf": "Random forest", "lgbm": "LightGBM", "mlp": "MLP",
    "ens5": "Deep ensemble", "mcd": "MC dropout", "sngp_res": "SNGP", "sngp": "SNGP without residual layers",
    "llla": "Last-layer Laplace", "vids": "VIDS",
    "vids_authors": "VIDS, authors' settings", "vids_pre": "VIDS, started at the MLP head",
    "vids_lam1e3": "VIDS, $\\lambda=10^{-3}$", "vids_lam5e3": "VIDS, $\\lambda=5\\times10^{-3}$",
    "vids_lam1e2": "VIDS, $\\lambda=10^{-2}$", "vids_lam5e2": "VIDS, $\\lambda=5\\times10^{-2}$",
    "vids_fixedprior": "VIDS, prior without pseudo-test term", "vids_tau1e2": "VIDS, $\\tau=0.01$",
    "vids_malt": "VIDS, $m=200$", "vids_ext": "VIDS, 40-draw search",
    "logit_all_iwacct": "Logit, all, importance-weighted", "logit_chs_iwacct": "Logit, CHS-type, importance-weighted",
    "logit_all_iw": "Logit, all, importance-weighted (all-variable classifier)",
}
BASES_PLATT = ["mlp", "ens5", "mcd", "sngp_res", "sngp", "llla", "vids", "vids_authors", "vids_pre", "vids_lam1e3", "vids_lam5e3",
               "vids_lam1e2", "vids_lam5e2", "vids_fixedprior", "vids_tau1e2", "vids_malt", "vids_ext"]
BASES_TS = ["mcd", "sngp_res", "llla", "vids"]          # mlp_ts and ens5_ts come from the runs
for b in list(NAMES):
    NAMES.setdefault(b + "_platt", NAMES[b] + " + Platt")
for b in ["mlp", "ens5"] + BASES_TS:
    NAMES[b + "_ts"] = NAMES[b] + " + temperature"
NAMES["lgbm_iso"] = "LightGBM + isotonic"
POSTHOC = ["logit_all_platt", "logit_chs_platt", "lgbm_platt", "ens5_platt", "vids_platt"]
for b in POSTHOC:
    NAMES[b + "_lag"] = NAMES[b] + ", intercept from last year's rate"
    NAMES[b + "_lag3"] = NAMES[b] + ", intercept from 3-year rate"
    NAMES[b + "_em"] = NAMES[b] + ", EM prior shift"
MAIN = [("Finance benchmarks", ["logit_altman", "logit_altman5", "logit_ohlson", "logit_chs", "logit_all"]),
        ("Machine learning", ["rf", "lgbm", "mlp"]),
        ("Uncertainty-aware networks", ["ens5", "mcd", "sngp_res", "llla", "vids"])]
SIG_REFS = ["logit_chs_platt", "logit_all_platt", "rf_platt", "lgbm_platt", "mlp_platt", "ens5_platt", "mcd_platt",
            "sngp_res_platt", "llla_platt", "vids_platt"]
UNC = ["ens5", "mcd", "sngp_res", "llla", "vids", "vids_authors"]
