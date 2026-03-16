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
EOG_FILE = Path("C:/Users/AKLO0022/EOG_REM/local_csv_eog/DCSM_1_a_contiguous_eog.csv")
GSSC_FILE = Path("C:/Users/AKLO0022/EOG_REM/gssc_csv/DCSM_1_a_gssc.csv")
EVENTS_FILE = Path("C:/Users/AKLO0022/EOG_REM/extracted_rems/DCSM_1_a_extracted_rems.csv")
OUTPUT_DIR =  OUT_DIR = Path("merged_csv_eog")
OUTPUT_FILE = OUTPUT_DIR/f"{EOG_FILE.stem}_merged.csv"

# =====================================================================
# TEST
# =====================================================================
if __name__ == "__main__":

    # --- 1) Check input files exist ---
    print("=" * 60)
    print("Checking input files...")
    for label, path in [("EOG", EOG_FILE), ("GSSC", GSSC_FILE), ("Events", EVENTS_FILE)]:
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

    # --- 3) Run merge_all ---
    print("\n" + "=" * 60)
    print("Running merge_all()...")
    merged_df = merge_all(
        eog_file=EOG_FILE,
        gssc_file=GSSC_FILE,
        events_file=EVENTS_FILE,
        output_file=OUTPUT_FILE,
    )

    # --- 4) Debug output ---
    print("\n" + "=" * 60)
    print("Inspecting merged output...")
    print(f"  Shape:   {merged_df.shape}")
    print(f"  Columns: {list(merged_df.columns)}")

    print(f"\n  REM events marked:  {merged_df['is_rem_event'].sum()} samples")
    print(f"  REM peaks marked:   {merged_df['is_rem_peak'].sum()} samples")
    print(f"  Unique event IDs:   {merged_df['event_id'].nunique()} events")
    print(f"  Unique stages:      {sorted(merged_df['stage'].dropna().unique())}")
    print(f"  Time range:         {merged_df['time_sec'].min():.2f}s — {merged_df['time_sec'].max():.2f}s")

    print(f"\n  NaN counts per column:")
    nan_counts = merged_df.isna().sum()
    for col, n in nan_counts[nan_counts > 0].items():
        print(f"    {col}: {n}")

    print(f"\nFirst 5 rows:")
    print(merged_df.head(5).to_string(index=False))

    print(f"\nFirst 5 REM event rows:")
    rem_rows = merged_df[merged_df["is_rem_event"] == True]
    print(rem_rows.head(5).to_string(index=False) if not rem_rows.empty else "  No REM events found.")

    print("\n" + "=" * 60)
    print(f"Done. Merged file saved to: {OUTPUT_FILE}")