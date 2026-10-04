# README: how to run the code

Paper: *Does uncertainty-aware learning help forecast corporate distress under temporal distribution shift? Evidence from U.S. performance delistings, 2011–2023.*

This package contains everything needed to rebuild the analysis except the WRDS data, which are proprietary (see `README_data.md`). Aggregate results behind the paper are included in `results/`.

## 1. Layout

```
download_wrds_data.sh   downloads the WRDS extracts into wrds_data/
README_data.md          data sources, access, sample construction
README_run.md           this file
requirements.txt        Python packages (CPU only)
src/                    sample construction, models, VIDS, robustness runs
src_r2/                 evaluation, diagnostics, and the table and figure generators
results/                shipped aggregate results (no firm-level data)
wrds_data/              empty; the downloaded WRDS files go here
```

Every script uses relative paths (`../results`, `../wrds_data`), so **run the scripts from inside `src/` or `src_r2/`** (`cd src`). Python 3.13 is the version reported in the paper. Install packages with `pip install -r requirements.txt`. Everything runs on CPU.

## 2. Quick test (about a few minutes, once `panel.parquet` exists)

```bash
cd src
python3 run_experiments.py --quick --seeds 1 --out ../results/smoke
```

`--quick` shortens training and tuning. Use it only to check that the installation works; the paper's results use the full settings below.

## 3. Full pipeline

All commands are run from `src/` unless stated. Seeds are fixed (seed *k* = 0 to 9, ten seeds); hyper-parameter draws use fixed generators, so reruns reproduce the same configurations.

**Step 0. Data.** Follow `README_data.md`: download with `download_wrds_data.sh`, then

```bash
python3 build_panel.py                     # -> ../results/panel.parquet
```

**Step 1. Main design (fixed split: train 2011–2017, validate 2018, test 2019–2023).**

```bash
python3 run_experiments.py --design fixed --out ../results/main_v2 --seeds 10 --trials 8
python3 sngp_res.py --out ../results/sngp_res
python3 iw_logit.py --features all         # importance-weighted logits -> ../results/iw
python3 iw_logit.py --features acct        # accounting-variable domain classifier -> ../results/iwacct
```

- `run_experiments.py` tunes every model on the 2018 validation year (8 random draws, validation log loss), writes the chosen settings to `main_v2/best_configs.json`, then fits all models for ten seeds: logits (Altman Z, Altman ratios, Ohlson, CHS, all 24 predictors, each also Platt-recalibrated on 2018 so that the benchmarks get the same recalibration as the machine-learning models), random forest, LightGBM, MLP, deep ensemble, MC dropout, last-layer Laplace and the first-stage VIDS runs. Output: `main_v2/fixed/seed_<k>.pkl` and the test and validation indices.
- `sngp_res.py` fits the residual SNGP (tuned on 2018, writes `config.json`).
- `iw_logit.py` fits the importance-weighted logits (Shimodaira 2000) for each test year and writes the effective sample sizes.

**Step 2. VIDS (final runs).**

```bash
python3 vids_r2.py tune  --design fixed --mlp-cfg ../results/main_v2/best_configs.json \
                         --out ../results/vids_r2_cfg_fixed.json --ext 40
python3 vids_r2.py refit --src ../results/main_v2 --dst ../results/main_r2 \
                         --cfg ../results/vids_r2_cfg_fixed.json --mode full
```

`tune` searches VIDS settings on 2018 (8 draws; `--ext 40` adds the 40-draw search reported as an extension). `refit` retrains the base MLP of each seed (and checks that it reproduces the stored predictions), then fits VIDS with the tuned and the authors' settings and the ablations, and copies all other models' predictions into `main_r2/`. The check of the VIDS implementation on the authors' synthetic example is

```bash
mkdir -p ../results/vids_audit
python3 vids_toy.py keras                  # -> ../results/vids_audit/toy_logistic_keras.csv
python3 vids_trajectory.py                 # training trajectories
```

**Step 3. Rolling design and robustness designs.** Each is `run_experiments.py` followed by `vids_r2.py refit` (Steps 1 and 2) with a different output folder and flag (`sngp_res.py` and `iw_logit.py` are fixed-design only; the VIDS settings are reused from `vids_r2_cfg_fixed.json`, except for the 2011–2015 design, which has its own `vids_r2_cfg_t2015.json`); the shipped folders are `rolling_r2`, `rob_csrank_r2`, `rob_fin_r2`, `rob_anyfail_r2`, `rob_train2015_r2`.

| Design | Flag for `run_experiments.py` and `vids_r2.py` |
|---|---|
| Rolling (re-estimate every test year) | `--design rolling` (reuse the 2018 settings: `--tuned ../results/main_v2/best_configs.json --seeds 3 --core`; for the VIDS refit add `--design rolling --seeds 3`) |
| Market variables as within-year percentiles | `--cs-rank` |
| Include financial firms | `--include-financials` |
| Broader failure label (adds liquidations) | `--label y_any_fail` |
| Training years 2011–2015 | `--train-end 2015` (tuned separately; see `results/configs/best_configs_train2015.json` and `vids_r2_cfg_t2015.json`) |

The robustness runs use `--core` (skips the VIDS ablations).

**Step 4. Evaluation (run from `src_r2/`).**

```bash
cd ../src_r2
python3 evaluate_r2.py --res ../results/main_r2 --extra ../results/sngp_res ../results/iwacct ../results/iw
python3 evaluate_r2.py --res ../results/rolling_r2            # also rob_csrank_r2, rob_fin_r2 (add --with-financials),
                                                              # rob_anyfail_r2 (add --label y_any_fail), rob_train2015_r2
python3 extra_sig.py
python3 diag_r2.py                                            # shift diagnostics -> ../results/diag_r2
python3 tuning_distinct.py                                    # -> ../results/diagnostics_v2/tuning_distinct.csv
```

`evaluate_r2.py` adds Platt recalibration for every model, computes pooled and by-year accuracy (Brier score, log loss, expected/observed failures with simulation bands, AUC, average precision), firm-clustered bootstrap tests with Holm adjustment (1,000 draws), the explicit shift corrections, and the uncertainty tests. It writes to `<res>/eval_r2/`. `diag_r2.py` needs the WRDS delisting file for the delisting-code tables.

**Step 5. Tables and figures.**

```bash
python3 make_tables_r2.py      # -> ../paper_r2/generated/*.tex (tables and number macros used by main.tex)
python3 make_figures_r2.py     # -> ../paper_r2/figures/*.pdf
```

The paper's LaTeX includes these files with `\input` and `\includegraphics`. `make_figures_r2.py` uses a Times-compatible serif font if one is installed.

## 4. Shipped results versus rerun output

`results/` contains the aggregate outputs only (prediction files with firm identifiers and the panel are not shared). The folder names differ slightly from the paths the scripts write, because the evaluation folders were flattened:

| Shipped | Written by the scripts to |
|---|---|
| `results/main_r2/*.csv` (except `vids_r2_diag.csv`) | `results/main_r2/eval_r2/` |
| `results/rolling_r2/`, `rob_*_r2/` `*.csv` | `results/<same name>/eval_r2/` |
| `results/diagnostics/*.csv` | `results/diag_r2/` |
| `results/configs/best_configs_main.json` | `results/main_v2/best_configs.json` |
| `results/configs/sngp_residual_config.json` | `results/sngp_res/config.json` |
| `results/configs/vids_r2_cfg_*.json` | `results/vids_r2_cfg_*.json` |
| `results/diagnostics/iw_*_ess_by_year.csv` | `results/iw/ess_by_year.csv`, `results/iwacct/ess_by_year.csv` |
| `results/vids_port_check/` | `results/vids_audit/` |

To regenerate the tables without rerunning the models you would need the seed prediction files, which are not shared; rerun Steps 1–4 on your own WRDS extract instead.

## 5. Where each paper item comes from

| Paper item | Produced by |
|---|---|
| Sample, label and predictor tables (Tables 1–2), descriptives | `build_panel.py`, `diag_r2.py`, `make_tables_r2.py` |
| Test-period shift diagnostics (domain AUCs, year intercepts, failure codes) | `diag_r2.py` |
| Accuracy tables, significance tests, calibration by year, reliability | `evaluate_r2.py`, `make_tables_r2.py`, `make_figures_r2.py` |
| Shift corrections (importance weights, lagged-rate intercepts, EM prior shift, rolling) | `iw_logit.py`, `evaluate_r2.py`, rolling runs |
| VIDS runs, ablations, check on the authors' example | `vids_r2.py`, `vids_toy.py`, `vids_trajectory.py` |
| Uncertainty tests and risk–coverage | `evaluate_r2.py` |
| Robustness table | robustness runs of Step 3 |
| Hyper-parameter table (appendix) | `best_configs.json`, `vids_r2_cfg_fixed.json`, `tuning_distinct.py` |
