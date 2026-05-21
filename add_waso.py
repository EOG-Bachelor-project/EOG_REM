# Filename: add_waso.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Computes WASO (Wake After Sleep Onset) from merged CSVs and
#              joins it onto an existing features.csv by subject_id.
#              Run this once to add WASO without rerunning the full pipeline.
#
# Usage:
#   python add_waso.py --merged-dir /path/to/merged_csvs --features-csv features_csv/features.csv

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================================
# Constants
# =====================================================================
DCSM_PATTERN = re.compile(r"(DCSM_\d+_[a-zA-Z])")
SLEEP_STAGES  = {"N1", "N2", "N3", "REM"}

# =====================================================================
# Functions
# =====================================================================
def compute_waso(merged_file: Path, fs: float = 250.0) -> dict:
    """
    Compute WASO from a single merged CSV.

    WASO = total wake time (in minutes) after the first sleep epoch
           (N1, N2, N3, or REM) until the end of the recording.

    Parameters
    ----------
    merged_file : Path
    fs : float
        Sampling frequency in Hz.\\
        **Default:** 250.0.

    Returns
    -------
    dict with keys: subject_id, waso_min
    """
    # ---- Subject ID ----
    raw_stem = merged_file.stem.replace(".csv", "")
    m        = DCSM_PATTERN.match(raw_stem)
    sid      = m.group(1) if m else raw_stem

    # ---- Load only the stage column ----
    df    = pd.read_csv(merged_file, usecols=["stage"], low_memory=False)
    stage = df["stage"].reset_index(drop=True)

    # ---- Find first sleep epoch ----
    first_sleep_idx = stage[stage.isin(SLEEP_STAGES)].index.min()

    if pd.isna(first_sleep_idx):
        print(f"  [WARN] {sid} — no sleep epochs found, WASO = NaN")
        return {"subject_id": sid, "waso_min": np.nan}

    # ---- Sum wake samples after first sleep epoch ----
    post_sleep      = stage.iloc[first_sleep_idx:]
    wake_samples    = (post_sleep == "W").sum()
    waso_min        = round(wake_samples / fs / 60.0, 4)

    print(f"  {sid:<25s}  first_sleep_idx={first_sleep_idx:,}  "
          f"wake_samples={wake_samples:,}  waso_min={waso_min:.2f}")

    return {"subject_id": sid, "waso_min": waso_min}


def add_waso_to_features(
        merged_dir:   str | Path,
        features_csv: str | Path,
        fs:           float = 250.0,
        pattern:      str   = "*_merged.csv.gz",
        overwrite:    bool  = False,
        ) -> pd.DataFrame:
    """
    Compute WASO for all merged CSVs and join onto features.csv.

    Parameters
    ----------
    merged_dir : str | Path
        Directory containing merged CSV files.
    features_csv : str | Path
        Path to existing features.csv to update.
    fs : float
        Sampling frequency.\\
        **Default:** 250.0.
    pattern : str
        Glob pattern for merged CSVs.\\
        **Default:** '*_merged.csv'.
    overwrite : bool
        If True, recompute WASO even if the column already exists.\\
        **Default:** False.

    Returns
    -------
    pd.DataFrame
        Updated features DataFrame with waso_min column added.
    """
    merged_dir   = Path(merged_dir)
    features_csv = Path(features_csv)

    # ---- Load existing features ----
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")
    feat_df = pd.read_csv(features_csv, low_memory=False)
    print(f"Loaded features: {feat_df.shape[0]} subjects, {feat_df.shape[1]} columns")

    if "waso_min" in feat_df.columns and not overwrite:
        print("  'waso_min' already exists in features.csv — use --overwrite to recompute")
        return feat_df

    # ---- Find merged CSVs ----
    files = sorted(merged_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {merged_dir}")
    print(f"Found {len(files)} merged CSVs\n")

    # ---- Compute WASO per subject ----
    rows = []
    for f in files:
        try:
            rows.append(compute_waso(f, fs=fs))
        except Exception as e:
            print(f"  [SKIP] {f.name} — {e}")

    waso_df = pd.DataFrame(rows)
    print(f"\nComputed WASO for {len(waso_df)} subjects")

    # ---- Join onto features ----
    if "waso_min" in feat_df.columns:
        feat_df = feat_df.drop(columns=["waso_min"])

    feat_df = feat_df.merge(waso_df, on="subject_id", how="left")

    n_nan = feat_df["waso_min"].isna().sum()
    print(f"  waso_min NaNs after join: {n_nan}")

    # ---- Save ----
    feat_df.to_csv(features_csv, index=False)
    print(f"\nSaved updated features.csv → {features_csv}  "
          f"({feat_df.shape[0]} subjects, {feat_df.shape[1]} columns)")

    return feat_df


# ================================================================================
# CLI
# ================================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute WASO from merged CSVs and add to features.csv."
    )
    parser.add_argument("--merged-dir", type=str, required=True,
                        help="Directory containing *_merged.csv.gz files")
    parser.add_argument("--features-csv", type=str, default="features_csv/features.csv",
                        help="Path to features.csv. Default: features_csv/features.csv")
    parser.add_argument("--fs", type=float, default=250.0,
                        help="Sampling frequency in Hz. Default: 250.0")
    parser.add_argument("--pattern", type=str, default="*_merged.csv.gz",
                        help="Glob pattern for merged CSVs. Default: '*_merged.csv.gz'")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute waso_min even if column already exists")

    args = parser.parse_args()

    add_waso_to_features(
        merged_dir   = args.merged_dir,
        features_csv = args.features_csv,
        fs           = args.fs,
        pattern      = args.pattern,
        overwrite    = args.overwrite,
    )