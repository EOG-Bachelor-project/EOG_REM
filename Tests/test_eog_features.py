# Filename: test_eog_features.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Test file for eog_features.py — runs feature extraction on a single merged CSV.

# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
from features.eog_feats import extract_features, extract_features_batch
import pandas as pd

# =====================================================================
# Paths
# =====================================================================
MERGED_FILE       = Path("C:/Users/AKLO0022/EOG_REM/merged_csv_eog/DCSM_3_a_contiguous_eog_merged.csv")
MERGED_FILE_UMAER = Path("C:/Users/AKLO0022/EOG_REM/merged_csv_eog/DCSM_3_a_contiguous_eog_merged_Umaer.csv")

# =====================================================================
# TEST
# =====================================================================
if __name__ == "__main__":

    # --- 1) Check input files exist ---
    print("=" * 60)
    print("Checking input files...")
    for label, path in [("Default", MERGED_FILE), ("Umaer", MERGED_FILE_UMAER)]:
        print(f"  {label}: {path} -> {'OK' if path.exists() else 'NOT FOUND'}")

    # --- 2) Test A — default merged CSV (no EpochType) ---
    print("\n" + "=" * 60)
    print("TEST A: extract_features() — default (no Umaer)")
    print("=" * 60)
    if not MERGED_FILE.exists():
        print("  SKIPPED - file not found")
    else:
        feats = extract_features(MERGED_FILE, fs=250.0)
        print("\nFeature results:")
        for k, v in feats.items():
            print(f"  {k:<45} {v}")

    # --- 3) Test B — Umaer merged CSV (with EpochType) ---
    print("\n" + "=" * 60)
    print("TEST B: extract_features() - Umaer (with EpochType)")
    print("=" * 60)
    if not MERGED_FILE_UMAER.exists():
        print("  SKIPPED - file not found")
    else:
        feats_umaer = extract_features(MERGED_FILE_UMAER, fs=250.0)
        print("\nFeature results:")
        for k, v in feats_umaer.items():
            print(f"  {k:<45} {v}")

        # --- 4) Confirm phasic/tonic features are not NaN in Umaer version ---
        print("\n" + "=" * 60)
        print("Phasic/Tonic sanity check:")
        for k in ["phasic_epoch_count", "tonic_epoch_count", "phasic_fraction", "tonic_fraction"]:
            v = feats_umaer.get(k)
            status = "OK" if v is not None and str(v) != "nan" else "NaN — check EpochType column"
            print(f"  {k:<45} {v}  [{status}]")