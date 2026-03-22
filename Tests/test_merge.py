# Filename: test_merge.py
# Authors: Adam Klovborg & Rasmus Kleffel


# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
from preprocessing.merge import merge_all
import pandas as pd

# =====================================================================
# Paths
# =====================================================================
EOG_FILE    = Path("C:/Users/AKLO0022/EOG_REM/eog_csv/DCSM_1_a_contiguous_eog.csv")
GSSC_FILE   = Path("C:/Users/AKLO0022/EOG_REM/gssc_csv/DCSM_1_a_gssc.csv")
EVENTS_FILE = Path("C:/Users/AKLO0022/EOG_REM/extracted_rems/DCSM_1_a_extracted_rems.csv")
EM_FILE     = Path("C:/Users/AKLO0022/EOG_REM/detected_ems/DCSM_1_a_em.csv")
OUTPUT_DIR  = OUT_DIR = Path("merged_csv_eog")
OUTPUT_FILE = OUTPUT_DIR/f"{EOG_FILE.stem}_merged.csv"

# =====================================================================
# TEST
# =====================================================================
if __name__ == "__main__":

    # --- 1) Check input files exist ---
    print("=" * 60)
    print("Checking input files...")
    for label, path in [
        ("EOG", EOG_FILE), 
        ("GSSC", GSSC_FILE), 
        ("Events", EVENTS_FILE),
        ("EM", EM_FILE),
        ]:
        exists = path.exists()
        print(f"  {label}: {path} -> {'OK' if exists else 'NOT FOUND'}")
        if not exists:
            raise FileNotFoundError(f"{label} file not found: {path}")

    # --- 2) Preview input files ---
    print("\n" + "=" * 60)
    print("Previewing input files...")

    eog_df = pd.read_csv(EOG_FILE)
    print(f"\n[EOG] shape: {eog_df.shape}")
    print(f"  columns: {list(eog_df.columns)}")
    print(eog_df.head(3).to_string(index=False))

    gssc_df = pd.read_csv(GSSC_FILE)
    print(f"\n[GSSC] shape: {gssc_df.shape}")
    print(f"  columns: {list(gssc_df.columns)}")
    print(gssc_df.head(3).to_string(index=False))

    events_df = pd.read_csv(EVENTS_FILE)
    print(f"\n[Events] shape: {events_df.shape}")
    print(f"  columns: {list(events_df.columns)}")
    print(events_df.head(3).to_string(index=False))

    em_df = pd.read_csv(EM_FILE)
    print(f"\n[EM] shape: {em_df.shape}")
    print(f"  columns: {list(em_df.columns)}")
    print(em_df.head(3).to_string(index=False))

    # --- 3) Run merge_all ---
    print("\n" + "=" * 60)
    print("Running merge_all()...")
    merged_df = merge_all(
        eog_file     = EOG_FILE,
        gssc_file    = GSSC_FILE,
        events_file  = EVENTS_FILE,
        em_file      = EM_FILE,
        output_file  = OUTPUT_FILE,
    )

    # --- 4) Debug output ---
    print("\n" + "=" * 60)
    print("Inspecting merged output...")
    print(f"  Shape:   {merged_df.shape}")
    print(f"  Columns: {list(merged_df.columns)}")

    print(f"\n  GSSC stages:         {sorted(merged_df['stage'].dropna().unique())}")
    print(f"  REM event samples:  {merged_df['is_rem_event'].sum():,} samples")
    print(f"  EM event samples:    {merged_df['is_em_event'].sum():,}")
    print(f"  SEM samples:         {(merged_df['EM_Type'] == 'SEM').sum():,}")
    print(f"  REM EM samples:      {(merged_df['EM_Type'] == 'REM').sum():,}")
    print(f"  Phasic samples:      {(merged_df['EpochType'] == 'Phasic').sum():,}")
    print(f"  Tonic samples:       {(merged_df['EpochType'] == 'Tonic').sum():,}")
    print(f"  Time range:         {merged_df['time_sec'].min():.2f}s — {merged_df['time_sec'].max():.2f}s")

    print(f"\n  NaN counts per column:")
    nan_counts = merged_df.isna().sum()
    for col, n in nan_counts[nan_counts > 0].items():
        print(f"    {col}: {n}")
    
    print("\n" + "=" * 60)
    print(f"Done. Merged file saved to: {OUTPUT_FILE}")