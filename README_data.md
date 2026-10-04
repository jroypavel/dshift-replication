# README: data

Paper: *Does uncertainty-aware learning help forecast corporate distress under temporal distribution shift? Evidence from U.S. performance delistings, 2011–2023.*

## 1. Summary

All firm-level data come from **Wharton Research Data Services (WRDS)**:

| Source | WRDS table | Provider |
|---|---|---|
| CRSP monthly stock file | `crsp.msf` | CRSP |
| CRSP names history (share and exchange codes, SIC) | `crsp.msenames` | CRSP |
| CRSP delistings | `crsp.msedelist` | CRSP |
| Compustat annual fundamentals | `comp.funda` | S&P Global |
| CRSP/Compustat Merged link table | `crsp.ccmxpf_lnkhist` | CRSP / S&P Global |
| Fama–French monthly factors (market return) | `ff.factors_monthly` | K. French (also public on his website) |

**These data are proprietary and are not included in this submission.** They are obtained through an institutional WRDS subscription whose terms do not allow redistribution. Instead, this package contains the script that downloads the data, the code that builds the analysis sample from it, and the aggregate result tables behind the paper. Anyone with WRDS access to the tables above can rebuild the sample from scratch.

## 2. How to obtain the data

1. You need a WRDS account whose institution subscribes to CRSP, Compustat, the CRSP/Compustat Merged database and Fama–French data.
2. Install the packages: `pip install wrds pandas pyarrow`.
3. From the top-level folder of this package, run

   ```bash
   WRDS_USERNAME=<your_wrds_username> bash download_wrds_data.sh 2010 2024 wrds_data parquet
   ```

   Arguments: start year, end year, output folder, file format. The script asks for your WRDS password through the `wrds` package. `SKIP_EXISTING=1` skips files that are already downloaded.

The code later looks for the files in `wrds_data/` (the folder beside `src/` and `results/`).

## 3. Files the script creates, and how they are used

| File in `wrds_data/` | Query (as in `download_wrds_data.sh`) | Used in the paper? |
|---|---|---|
| `crsp_msf.parquet` | `permno, date, ret, retx, prc, vol, shrout, cfacpr, cfacshr` from `crsp.msf`, dates 2010-01-01 to 2024-12-31 | Yes |
| `crsp_msenames.parquet` | `permno, namedt, nameendt, shrcd, exchcd, siccd, ticker, ncusip` from `crsp.msenames` (full history) | Yes (share code, exchange code, SIC) |
| `crsp_msedelist.parquet` | `permno, dlstdt, dlret, dlretx, dlstcd, dlpdt` from `crsp.msedelist`, delisting dates 2010-01-01 to 2024-12-31 | Yes (`dlstdt`, `dlstcd` only; delisting returns are never used) |
| `comp_funda.parquet` | `gvkey, datadate, fyear, fyr, sich AS sic, naicsh AS naics, seq, ceq, at, lt, revt, cogs, ib, sale, ni, act, lct, che, dlc, dltt, oibdp, xint, capx, ppent, txditc, pstk, re, ebit, oiadp, dp` from `comp.funda`; filters `indfmt='INDL'`, `datafmt='STD'`, `popsrc='D'`, `consol='C'`; fiscal years 2007–2024 | Yes (see section 4) |
| `crsp_ccm_link.parquet` | `gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt` from `crsp.ccmxpf_lnkhist`; `linktype IN ('LU','LC')`, `linkprim IN ('P','C')` | Yes |
| `ff_factors_monthly.parquet` | `date, mktrf, smb, hml, umd, rf` from `ff.factors_monthly`, 2010–2024 | Yes (`mktrf` + `rf` = market return) |
| `ff_fivefactors_monthly.parquet` | five-factor file | Downloaded as an option; not used |
| `crsp_dsf_monthly.parquet` | monthly volatility and Amihud illiquidity from `crsp.dsf` | Downloaded as an option; **not used** by any result in the paper (the 24 predictors do not include daily-data variables) |

Compustat items are used in lower case, as downloaded. Extra columns that the script downloads but the sample construction does not need (for example `revt`, `cogs`, `capx`, `ppent`) are harmless.

## 4. Building the analysis sample

Run from `src/`:

```bash
python3 build_panel.py            # reads ../wrds_data, writes ../results/panel.parquet
```

Options: `--data-dir`, `--out`, `--first-year` (default 2011), `--last-year` (default: last year with a complete 12-month outcome window, which is 2023 for data through December 2024).

What `build_panel.py` does (the paper's Section 3 describes the same steps):

1. **Universe.** CRSP common stocks (share codes 10 and 11) listed on the NYSE, NYSE American or Nasdaq (exchange codes 1, 2, 3), using the `msenames` record valid at each month-end.
2. **Forecast dates.** One observation per firm and year *t* = 2011, …, 2023 at 30 June of year *t*. A firm needs at least six monthly returns in the 12 months up to June *t* and positive market equity.
3. **Market variables.** Computed from the twelve monthly returns to June *t*: volatility (SIGMA), excess return over the CRSP value-weighted market (EXRET), cumulative return (RET12), relative size (RSIZE), log price capped at $15 (PRICE).
4. **Accounting variables.** From the latest fiscal year ending in the 12 months to 31 December of *t*−1 (so the data are 6 to 18 months old at the forecast date), linked through the CCM table (link must be valid at the fiscal year-end). The previous fiscal year is used only for changes (CHIN, INTWO, SGROWTH). Total assets and total liabilities must be positive.
5. **Predictors.** 24 predictors in four groups (CHS-type, Ohlson-type, Altman ratios, other), plus the Altman Z-score. Definitions are in Table 1 of the paper and in `build_panel.py`.
6. **Failure label.** `y_perf` = 1 if the stock is delisted for a performance reason (CRSP delisting codes 500 and 520–584) between 1 July of *t* and 30 June of *t*+1. Alternative labels are also stored: `y_liq` (liquidations, 400–499), `y_bk` (code 574, bankruptcy), `y_any_fail` (performance delisting or liquidation).
7. **Later filters** (in `src/common.py`, applied when the models run): financial firms (SIC 6000–6999) are dropped, firms missing any of the 17 never-imputed predictors are dropped, and five predictors (MB, CLCA, INTWO, CHIN, SGROWTH) are imputed with training-year medians. SIC is Compustat's historical `sich` if available, otherwise CRSP `siccd`.

### Columns of `results/panel.parquet`

`permno, gvkey, t` (forecast year), `date` (forecast date), `datadate` (fiscal year-end), `me` (market equity, $ millions); predictors `nimta, tlmta, exret, sigma, rsize, cashmta, mb, price, size, tlta, wcta, clca, oeneg, nita, futl, intwo, chin, re_ta, ebit_ta, me_tl, sales_ta, sgrowth, ret12, bm_neg, z_altman` (and `sigma_d`, `amihud` only if the optional daily file is present; they are not used); `sic, financial, utility`; `start, end` (outcome window); labels `y_perf, y_liq, y_bk, y_any_fail` and `y` (= `y_perf`).

### Numbers to check against after a rebuild

| Item | Paper |
|---|---|
| Firm-years after the share-code, exchange and positive-value filters | 46,134 |
| After excluding financial firms | 37,291 |
| Final sample (17 required predictors present) | 36,475 firm-years, 5,312 firms |
| Performance delistings (failures) | 664 in total: 337 in 2011–2017, 41 in 2018, 286 in 2019–2023 |
| Firm-years in training (2011–2017), validation (2018), test (2019–2023) | 19,341; 2,664; 14,470 |

CRSP and Compustat are revised over time, so an extract pulled later can differ slightly from the one used for the paper.

## 5. What can be shared

- Shared with the submission: `download_wrds_data.sh`, all code, tuned configurations and aggregate result tables (`results/`), which contain no firm-level data.
- Not shared: anything at the firm level (`wrds_data/`, `results/panel.parquet`, prediction files that carry firm identifiers).
