# Filename: test_gssc_features.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Test file for gssc_features.py — runs GSSC feature extraction on a single merged CSV.

# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
from features.gssc_feats import extract_gssc_features
import pandas as pd

# =====================================================================
# Paths
# =====================================================================
MERGED_FILE       = Path("C:/Users/AKLO0022/EOG_REM/merged_csv_eog/DCSM_2_a_contiguous_eog_merged.csv")
MERGED_FILE_UMAER = Path("C:/Users/AKLO0022/EOG_REM/merged_csv_eog/DCSM_2_a_contiguous_eog_merged_Umaer.csv")

# =====================================================================
# TEST
# =====================================================================
if __name__ == "__main__":

    # --- 1) Check input files exist ---
    print("=" * 60)
    print("Checking input files...")
    for label, path in [("Default", MERGED_FILE), ("Umaer", MERGED_FILE_UMAER)]:
        print(f"  {label}: {path} -> {'OK' if path.exists() else 'NOT FOUND'}")

    # --- 2) Test A — default merged CSV ---
    print("\n" + "=" * 60)
    print("TEST A: extract_gssc_features() — default")
    print("=" * 60)
    if not MERGED_FILE.exists():
        print("  SKIPPED — file not found")
    else:
        feats = extract_gssc_features(MERGED_FILE, fs=250.0)
        print("\nFeature results:")
        for k, v in feats.items():
            print(f"  {k:<45} {v}")

    # --- 3) Test B — Umaer merged CSV ---
    print("\n" + "=" * 60)
    print("TEST B: extract_gssc_features() — Umaer")
    print("=" * 60)
    if not MERGED_FILE_UMAER.exists():
        print("  SKIPPED — file not found")
    else:
        feats_umaer = extract_gssc_features(MERGED_FILE_UMAER, fs=250.0)
        print("\nFeature results:")
        for k, v in feats_umaer.items():
            print(f"  {k:<45} {v}")

    # --- 4) Sanity checks ---
    print("\n" + "=" * 60)
    print("Sanity checks:")
    if MERGED_FILE.exists():
        for k in ["rem_certainty", "rem_stability_index", "rem_fragmentation_index", "amount_of_rem"]:
            v = feats.get(k)
            status = "OK" if v is not None and str(v) != "nan" else "NaN — check prob_* columns"
            print(f"  {k:<45} {v}  [{status}]")

    # --- 5) Column diagnostics ---
    print("\n" + "=" * 60)
    print("Column diagnostics:")
    if MERGED_FILE.exists():
        df = pd.read_csv(MERGED_FILE, low_memory=False)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        print(f"\n  Probability columns: {prob_cols}")
        print(f"\n  Stage distribution:\n{df['stage'].value_counts().to_string()}")