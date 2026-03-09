# test_merge_pipeline.py

# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
import pandas as pd

from preprocessing.edf_to_csv import edf_to_csv, OUT_DIR
from preprocessing.GSSC_to_csv import GSSC_to_csv
from preprocessing.csv_merge import merge_csv_files


# --------------------------------------------------
# Paths
# --------------------------------------------------
file_path = Path(
    "L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a/contiguous.edf"
)

GSSC_DIR = Path("gssc_csv")
MERGED_DIR = Path("merged_csv")

GSSC_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)

session_id = file_path.parent.name

print("\nTesting recording:")
print(file_path)


# --------------------------------------------------
# 1) Create EOG CSV
# --------------------------------------------------
edf_to_csv(file_path, OUT_DIR)
eog_csv = OUT_DIR / f"{session_id}_eog.csv"

print("\nEOG CSV created:")
print(eog_csv)


# --------------------------------------------------
# 2) Create GSSC CSV
# --------------------------------------------------
GSSC_to_csv(file_path)

gssc_csv = GSSC_DIR / f"{session_id}_gssc.csv"

print("\nGSSC CSV created:")
print(gssc_csv)


# --------------------------------------------------
# 3) Merge
# --------------------------------------------------
merged_csv = MERGED_DIR / f"{session_id}_merged.csv"

merge_csv_files(eog_csv, gssc_csv, merged_csv)

print("\nMerged CSV created:")
print(merged_csv)


# --------------------------------------------------
# 4) Inspect merged file
# --------------------------------------------------
merged_df = pd.read_csv(merged_csv)

print("\nMerged columns:")
print(merged_df.columns.tolist())

print("\nMerged head:")
print(merged_df.head())

print("\nStage counts:")
print(merged_df["stage"].value_counts(dropna=False))

