# csv_merge.py

# =====================================================================
# Imports
# =====================================================================
import pandas as pd

# =====================================================================
# Functions
# =====================================================================
def merge_csv_files(eog_file: str, gssc_file: str, output_file: str) -> None:
    """
    Merges two CSV files based on a common column (e.g., "id") and saves the merged result to a new CSV file.

    Parameters
    ----------
    eog_file : str
        The path to the EOG CSV file.
    gssc_file : str
        The path to the GSSC CSV file.
    output_file : str
        The path to the output CSV file.
    
    Returns
    -------
    None
    """
    # 1) Read the two CSV files into pandas DataFrames
    eog_df = pd.read_csv(eog_file)
    gssc_df = pd.read_csv(gssc_file)

    # Check required columns
    if "id" not in eog_df.columns or "id" not in gssc_df.columns:
        raise ValueError("Both CSV files must contain an 'id' column for merging.")
    
    if "time_sec" not in eog_df.columns:
        raise ValueError("EOG CSV must contain a 'time_sec' column.")

    if "times" not in gssc_df.columns:
        raise ValueError("GSSC CSV must contain a 'times' column.")

    if "stages" not in gssc_df.columns:
        raise ValueError("GSSC CSV must contain a 'stages' column.")
    
    # 2) Check that both files belong to the same recording
    eog_ids = eog_df["id"].dropna().unique()
    gssc_ids = gssc_df["id"].dropna().unique()

    if len(eog_ids) != 1 or len(gssc_ids) != 1:
        raise ValueError("Each CSV file must contain exactly one unique 'id' value.")
    if eog_ids[0] != gssc_ids[0]:
        raise ValueError(f"ID mismatch: EOG ID '{eog_ids[0]}' does not match GSSC ID '{gssc_ids[0]}'.")
    
    # 3) Rename columns in gssc_df for clarity
    gssc_df = gssc_df.rename(columns={
        "times": "epoch_start",
        "stages": "stage"
    })

    # 4) Sort for merge_asof
    eog_df = eog_df.sort_values("time_sec").reset_index(drop=True)
    gssc_df = gssc_df.sort_values("epoch_start").reset_index(drop=True)

    # Merge by time
    merged_df = pd.merge_asof(
        eog_df,
        gssc_df,
        left_on="time_sec",
        right_on="epoch_start",
        direction="backward"
    )

    # Save merged result
    merged_df.to_csv(output_file, index=False)

    print(f"Merged file saved to: {output_file.name}")