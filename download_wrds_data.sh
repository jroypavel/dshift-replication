#!/usr/bin/env bash
set -euo pipefail

# Usage: ./download_wrds_data.sh [START_YEAR] [END_YEAR] [OUTPUT_DIR] [FORMAT]
START_YEAR=${1:-"2010"}
END_YEAR=${2:-"2024"}
OUTPUT_DIR=${3:-"./wrds_data"}
FORMAT=${4:-"parquet"}
SKIP_EXISTING=${SKIP_EXISTING:-0}

mkdir -p "${OUTPUT_DIR}"

python3 -c "import wrds, pandas, pyarrow" 2>/dev/null || {
    echo "[!] Missing required Python packages."
    echo "    Run: pip install wrds pandas pyarrow"
    exit 1
}

if [[ -z "${WRDS_USERNAME:-}" ]]; then
    read -rp "Enter WRDS username: " WRDS_USERNAME
fi

export WRDS_USERNAME START_YEAR END_YEAR OUTPUT_DIR FORMAT SKIP_EXISTING

echo "============================================================"
echo " WRDS extraction: ${START_YEAR} to ${END_YEAR}"
echo " Output Directory: ${OUTPUT_DIR} (${FORMAT})"
echo " SKIP_EXISTING:    ${SKIP_EXISTING}"
echo "============================================================"

python3 - << 'EOF'
import os
import wrds
import pandas as pd

start_year = int(os.environ["START_YEAR"])
end_year = int(os.environ["END_YEAR"])
out_dir = os.environ["OUTPUT_DIR"]
fmt = os.environ["FORMAT"].lower()
username = os.environ["WRDS_USERNAME"]
skip_existing = os.environ["SKIP_EXISTING"] == "1"

def path_for(name):
    return os.path.join(out_dir, f"{name}.{fmt}")

def save_df(df, name):
    path = path_for(name)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"    --> Saved: {path} ({len(df):,} rows)")

def done(name):
    if skip_existing and os.path.exists(path_for(name)):
        print(f"[*] {path_for(name)} exists, skipping...")
        return True
    return False

print("[*] Connecting to WRDS...")
db = wrds.Connection(wrds_username=username)

try:
    # 1. CRSP Monthly Stock File (+ retx, vol)
    if not done("crsp_msf"):
        print("[*] Querying crsp.msf...")
        q = f"""
        SELECT permno, date, ret, retx, prc, vol, shrout, cfacpr, cfacshr
        FROM crsp.msf
        WHERE date >= '{start_year}-01-01' AND date <= '{end_year}-12-31'
        """
        msf = db.raw_sql(q, date_cols=['date'])
        save_df(msf, "crsp_msf")
        print(f"    CRSP msf date range: {msf.date.min().date()} to {msf.date.max().date()}")

    # 2. CRSP Names History (Exchange codes, Share codes)
    if not done("crsp_msenames"):
        print("[*] Querying crsp.msenames...")
        q = """
        SELECT permno, namedt, nameendt, shrcd, exchcd, siccd, ticker, ncusip
        FROM crsp.msenames
        """
        save_df(db.raw_sql(q, date_cols=['namedt', 'nameendt']), "crsp_msenames")

    # 3. CRSP Delisting Returns
    if not done("crsp_msedelist"):
        print("[*] Querying crsp.msedelist...")
        q = f"""
        SELECT permno, dlstdt, dlret, dlretx, dlstcd, dlpdt
        FROM crsp.msedelist
        WHERE dlstdt >= '{start_year}-01-01' AND dlstdt <= '{end_year}-12-31'
        """
        save_df(db.raw_sql(q, date_cols=['dlstdt', 'dlpdt']), "crsp_msedelist")

    # 4. Compustat Annual Fundamentals (FIXED: sich and naicsh)
    # Pulls 3 extra years back (from 2007) so lags and growth rates are available in 2010
    if not done("comp_funda"):
        print("[*] Querying comp.funda...")
        q = f"""
        SELECT gvkey, datadate, fyear, fyr, 
               sich AS sic, naicsh AS naics,
               seq, ceq, at, lt, revt, cogs, ib,
               sale, ni, act, lct, che, dlc, dltt, oibdp, xint,
               capx, ppent, txditc, pstk,
               re, ebit, oiadp, dp
        FROM comp.funda
        WHERE indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
          AND fyear >= {start_year - 3} AND fyear <= {end_year}
        """
        save_df(db.raw_sql(q, date_cols=['datadate']), "comp_funda")

    # 5. CRSP/Compustat CCM Link Table
    if not done("crsp_ccm_link"):
        print("[*] Querying crsp.ccmxpf_lnkhist...")
        q = """
        SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
        """
        save_df(db.raw_sql(q, date_cols=['linkdt', 'linkenddt']), "crsp_ccm_link")

    # 6. Fama-French Monthly Factors
    if not done("ff_factors_monthly"):
        print("[*] Querying ff.factors_monthly...")
        try:
            q = f"""
            SELECT date, mktrf, smb, hml, umd, rf
            FROM ff.factors_monthly
            WHERE date >= '{start_year}-01-01' AND date <= '{end_year}-12-31'
            """
            ff = db.raw_sql(q, date_cols=['date'])
        except Exception as e:
            print(f"    [!] 'umd' column not found ({e}); pulling 3 factors...")
            q = f"""
            SELECT date, mktrf, smb, hml, rf
            FROM ff.factors_monthly
            WHERE date >= '{start_year}-01-01' AND date <= '{end_year}-12-31'
            """
            ff = db.raw_sql(q, date_cols=['date'])
        save_df(ff, "ff_factors_monthly")

    # 7. Optional: Fama-French 5 factors
    if not done("ff_fivefactors_monthly"):
        print("[*] Querying ff.fivefactors_monthly...")
        try:
            q = f"""
            SELECT date, mktrf, smb, hml, rmw, cma, rf
            FROM ff.fivefactors_monthly
            WHERE date >= '{start_year}-01-01' AND date <= '{end_year}-12-31'
            """
            save_df(db.raw_sql(q, date_cols=['date']), "ff_fivefactors_monthly")
        except Exception as e:
            print(f"    [!] Skipped FF5 ({e})")

    # 8. Optional: Monthly return volatility + Amihud illiquidity from Daily CRSP
    if not done("crsp_dsf_monthly"):
        print("[*] Querying crsp.dsf (aggregated by month)...")
        try:
            q = f"""
            SELECT permno, date_trunc('month', date)::date AS mdate,
                   stddev(ret) AS sd_ret, count(ret) AS n_ret,
                   avg(abs(ret) / NULLIF(abs(prc) * vol, 0)) AS amihud
            FROM crsp.dsf
            WHERE date >= '{start_year}-01-01' AND date <= '{end_year}-12-31'
            GROUP BY permno, date_trunc('month', date)
            """
            save_df(db.raw_sql(q, date_cols=['mdate']), "crsp_dsf_monthly")
        except Exception as e:
            print(f"    [!] Skipped daily-volatility aggregate ({e})")

finally:
    db.close()
    print("[*] WRDS connection closed.")

EOF

echo "All datasets successfully downloaded."