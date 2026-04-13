# Security Policy

## Project Context

EOG_REM is a research project developing EOG-only digital biomarkers for REM Sleep Behavior Disorder (RBD) and Parkinson's Disease. 
The pipeline processes clinical polysomnography (PSG) data from the Danish Center for Sleep Medicine (DCSM). 
While this repository contains **code only** (no patient data), the sensitive nature of the underlying clinical data requires careful handling.

## Handling of Clinical Data

This project adheres to the following principles regarding clinical data:

- **No patient data is stored in this repository.** All PSG recordings and associated metadata remain on secured institutional infrastructure at DCSM/DTU.
- **Generated outputs** (feature CSVs, reports) use anonymised subject identifiers only.
- The `.gitignore` is configured to exclude data directories (`local_csv_eog/`, `*.csv`), environment files (`.env`), and conda environments (`BPML/`).
- Contributors must **never** commit raw PSG recordings, EDF files, patient identifiers, or any data that could be used to re-identify participants.

## For Contributors

- Do not commit data files, credentials, or environment secrets.
- Review `.gitignore` before staging changes to ensure sensitive files are excluded.
- If you need to reference file paths in code or documentation, use relative placeholder paths rather than absolute paths that reveal institutional directory structures.
- If you discover an accidental data exposure or any other security concern, contact the maintainers directly.

## Dependencies

The project relies on third-party packages managed via conda and pip (see `environment-win.yml` / `environment-mac.yml`). We recommend periodically auditing dependencies with tools such as `pip-audit` or `safety` to check for known vulnerabilities.
