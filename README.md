# dshift-replication

Replication code for **"Does uncertainty-aware learning help forecast corporate distress under temporal distribution shift? Evidence from U.S. performance delistings, 2011–2023"** (Juboraj R. Pavel, Old Dominion University).

The paper compares pooled logits, tree ensembles and uncertainty-aware neural networks (including VIDS, Slavutsky and Blei, 2025) for one-year failure forecasts of U.S. listed firms, estimated on 2011–2017 and tested on 2019–2023.

## Contents

| Path | What it is |
|---|---|
| `download_wrds_data.sh` | Downloads the CRSP, Compustat, CCM and Fama–French extracts from WRDS |
| `src/` | Sample construction, models, VIDS, robustness runs |
| `src_r2/` | Evaluation, diagnostics, table and figure generators |
| `results/` | Aggregate result tables and tuned configurations (no firm-level data) |
| `README_data.md` | Data sources, access, sample construction |
| `README_run.md` | How to run the code, step by step |
| `requirements.txt` | Python packages |

## Data

The firm-level data come from CRSP and Compustat through WRDS and are proprietary, so they are **not** in this repository. Anyone with WRDS access can rebuild the sample; see [`README_data.md`](README_data.md).

## Running the code

See [`README_run.md`](README_run.md).

## License

Code: MIT (see `LICENSE`). This does not cover the WRDS data, which remain under their providers' terms.
