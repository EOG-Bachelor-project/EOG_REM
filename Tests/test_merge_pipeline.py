# test_merge_pipeline.py

# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
import pandas as pd

from preprocessing.edf_to_csv import edf_to_csv, OUT_DIR
from preprocessing.GSSC_to_csv import GSSC_to_csv, GSSC_DIR
from preprocessing.csv_merge import merge_csv_files

# Path of the EDF file to be tested
file_path = Path(
    "L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a/contiguous.edf"
)

# Output directory for merged CSV files
MERGED_DIR = Path("merged_csv")
MERGED_DIR.mkdir(parents=True, exist_ok=True)

# Extract session ID from file path
session_id = file_path.parent.name

print("\nTesting recording:")
print(file_path)

# 1) Create EOG CSV
edf_to_csv(file_path, OUT_DIR)
eog_csv = OUT_DIR / f"{session_id}_eog.csv"

# 2) Create GSSC CSV
GSSC_to_csv(file_path, GSSC_DIR)
gssc_csv = GSSC_DIR / f"{session_id}_gssc.csv"

# 3) Merge
merged_csv = MERGED_DIR / f"{session_id}_merged.csv"
merge_csv_files(eog_csv, gssc_csv, merged_csv)

# 4) Inspect
merged_df = pd.read_csv(merged_csv)

print("\nMerged columns:")
print(merged_df.columns.tolist())

print("\nMerged head:")
print(merged_df.head())

print("\nStage counts:")
print(merged_df["stage"].value_counts(dropna=False))