# Filename: merge_patient_info.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Merges patient demographic and diagnostic information from an Excel file with extracted features.

# =====================================================================
# Imports
# =====================================================================
import pandas as pd
from pathlib import Path

# =====================================================================
# Constants
# =====================================================================
KEEP_COLS = [
    "DCSM_ID", "Sex", "Age at recording", "CPAP", "BMI", "AHI", "DZ768D", "DR298A", "DG209", "DG47B2", "DG4772", "DG473",
    "DG4744", "ZM95300", "Validated", "Other neurological disorders", "Control", "PD(-RBD)", "PD(+RBD)", "iRBD","PLM",]

def merge_patient_info(
        feature_csv:  str | Path,
        patient_excel: str | Path,
        output_csv:   str | Path = "features_csv/features_with_info.csv",
        subject_col:  str = "DCSM_ID",
) -> pd.DataFrame:
    """
    Join patient demographic and diagnostic information from an Excel file
    onto the extracted feature CSV.

    Parameters
    ----------
    feature_csv : str | Path
        Path to the feature CSV (output of collect_features).
    patient_excel : str | Path
        Path to the Excel file containing patient information.
    output_csv : str | Path
        Path to save the merged CSV. Default is 'features_csv/features_with_info.csv'.
    subject_col : str
        Column name used to join the two files. Default is 'DCSM_ID'.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with patient info columns added.
    """
    # --- 1) Load feature CSV ---
    features_df = pd.read_csv(feature_csv)
    print(f"Features:  {features_df.shape[0]} subjects | {features_df.shape[1]} columns")
    print(f"DCSM_ID sample: {features_df[subject_col].head(3).tolist()}")

    # --- 2) Load Excel ---
    patient_df = pd.read_excel(patient_excel)
    print(f"\nExcel:     {patient_df.shape[0]} rows | {patient_df.shape[1]} columns")
    print(f"Excel columns: {patient_df.columns.tolist()}")

    # --- 3) Keep only relevant columns ---
    available = [c for c in KEEP_COLS if c in patient_df.columns]
    missing   = [c for c in KEEP_COLS if c not in patient_df.columns]
    if missing:
        print(f"\nWarning — columns not found in Excel and skipped: {missing}")

    patient_df = patient_df[available]
    print(f"\nKeeping {len(available)} columns from Excel")

    # --- 4) Join ---
    combined = pd.merge(features_df, patient_df, on=subject_col, how="left")
    print(f"\nMerged:    {combined.shape[0]} subjects | {combined.shape[1]} columns")

    n_unmatched = combined[available[1]].isna().sum()  # check first info col for NaNs
    if n_unmatched > 0:
        print(f"Warning — {n_unmatched} subjects had no match in Excel (check DCSM_ID format)")

    # --- 5) Save ---
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    return combined


if __name__ == "__main__":
    merge_patient_info(
        feature_csv   = "features_csv/features.csv",
        patient_excel = "path/to/patient_info.xlsx",  # ← update this
        output_csv    = "features_csv/features_with_info.csv",
    )
