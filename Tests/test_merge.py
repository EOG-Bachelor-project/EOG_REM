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
EOG_FILE        = Path("C:/Users/AKLO0022/EOG_REM/eog_csv/DCSM_2_a_contiguous_eog.csv")
GSSC_FILE       = Path("C:/Users/AKLO0022/EOG_REM/gssc_csv/DCSM_2_a_gssc.csv")
EVENTS_FILE     = Path("C:/Users/AKLO0022/EOG_REM/extracted_rems/DCSM_2_a_extracted_rems.csv")
EM_FILE         = Path("C:/Users/AKLO0022/EOG_REM/detected_ems/DCSM_2_a_em.csv")
SUBEPOCHS_FILE  = Path("C:/Users/AKLO0022/EOG_REM/detected_ems/DCSM_2_a_subepochs.csv")


OUTPUT_DIR  = Path("merged_csv_eog")
OUTPUT_FILE = OUTPUT_DIR/f"{EOG_FILE.stem}_merged.csv"
OUTPUT_FILE_UMAER = OUTPUT_DIR/f"{EOG_FILE.stem}_merged_Umaer.csv"

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
        ("SUB-EPOCHS", SUBEPOCHS_FILE)
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

     
    if SUBEPOCHS_FILE.exists():
        subepoch_df = pd.read_csv(SUBEPOCHS_FILE)
        print(f"\n[Subepochs] shape: {subepoch_df.shape} | columns: {list(subepoch_df.columns)}")
        print(subepoch_df.head(3).to_string(index=False))
 
# =====================================================================
# TEST A — default classifier (no subepochs_file)
# =====================================================================
print("\n" + "=" * 60)
print("TEST A: merge_all() — default classifier (use_Umaer=False)")
print("=" * 60)
try:
    merged_df = merge_all(
        eog_file = EOG_FILE,
        gssc_file = GSSC_FILE,
        events_file = EVENTS_FILE,
        em_file = EM_FILE,
        output_file = OUTPUT_FILE,
    )
 
    print("\nInspecting merged output (default)...")
    print(f" Shape: {merged_df.shape}")
    print(f" Columns: {list(merged_df.columns)}")
    print(f"\n GSSC stages: {sorted(merged_df['stage'].dropna().unique())}")
    print(f" REM event samples: {merged_df['is_rem_event'].sum():,}")
    print(f" EM event samples: {merged_df['is_em_event'].sum():,}")
    print(f" SEM samples: {(merged_df['EM_Type'] == 'SEM').sum():,}")
    print(f" REM EM samples: {(merged_df['EM_Type'] == 'REM').sum():,}")
    print(f" Phasic samples: {(merged_df['EpochType'] == 'Phasic').sum():,}")
    print(f" Tonic samples: {(merged_df['EpochType'] == 'Tonic').sum():,}")
    print(f" Time range: {merged_df['time_sec'].min():.2f}s — {merged_df['time_sec'].max():.2f}s")
    nan_counts = merged_df.isna().sum()
    print(f"\n NaN counts per column:")
    for col, n in nan_counts[nan_counts > 0].items():
        print(f" {col}: {n}")
    print(f"\nDone. Saved to: {OUTPUT_FILE}")
    
except Exception as e:
    print(f"TEST A FAILED: {e}")
 
# =====================================================================
# TEST B — Umaer classifier (with subepochs_file)
# =====================================================================
print("\n" + "=" * 60)
print("TEST B: merge_all() — Umaer classifier (use_Umaer=True)")
print("=" * 60)

try:
    if not SUBEPOCHS_FILE.exists():
        raise FileNotFoundError(
            f"Subepochs file not found: {SUBEPOCHS_FILE}\n"
            "Run em_to_csv(..., use_Umaer=True) first to generate it."
            )
 
    merged_df_umaer = merge_all(
        eog_file = EOG_FILE,
        gssc_file = GSSC_FILE,
        events_file = EVENTS_FILE,
        em_file = EM_FILE,
        output_file = OUTPUT_FILE_UMAER,
        subepochs_file = SUBEPOCHS_FILE,
    )
    
    print("\nInspecting merged output (Umaer)...")
    print(f" Shape: {merged_df_umaer.shape}")
    print(f" Columns: {list(merged_df_umaer.columns)}")
    print(f"\n GSSC stages: {sorted(merged_df_umaer['stage'].dropna().unique())}")
    print(f" REM event samples: {merged_df_umaer['is_rem_event'].sum():,}")
    print(f" EM event samples: {merged_df_umaer['is_em_event'].sum():,}")
    print(f" SEM samples: {(merged_df_umaer['EM_Type'] == 'SEM').sum():,}")
    print(f" REM EM samples: {(merged_df_umaer['EM_Type'] == 'REM').sum():,}")
    print(f" Phasic samples: {(merged_df_umaer['EpochType'] == 'Phasic').sum():,}")
    print(f" Tonic samples: {(merged_df_umaer['EpochType'] == 'Tonic').sum():,}")
    print(f" Unclassified samples: {(merged_df_umaer['EpochType'] == 'Unclassified').sum():,}")
    print(f" Time range: {merged_df_umaer['time_sec'].min():.2f}s — {merged_df_umaer['time_sec'].max():.2f}s")
    nan_counts = merged_df_umaer.isna().sum()
    print(f"\n NaN counts per column:")
    for col, n in nan_counts[nan_counts > 0].items():
        print(f" {col}: {n}")
    print(f"\nDone. Saved to: {OUTPUT_FILE_UMAER}")
    
except Exception as e:
    import traceback
    print(f"TEST B FAILED: {e}") 
    traceback.print_exc()