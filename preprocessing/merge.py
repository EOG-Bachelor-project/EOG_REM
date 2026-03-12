# Filename: merge.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Utilities for merging EOG signals. GSSC sleep staging, and REM event annotations into a unifies CSV for downstream analysis.

# =====================================================================
# Imports
# =====================================================================
import pandas as pd
from pathlib import Path
from preprocessing.upsample import upsample_gssc_to_eog

# =====================================================================
# Functions
# =====================================================================

# —————————————————————————————————————————————————————————————————————
# Function to merge EOG and GSSC CSV files
# —————————————————————————————————————————————————————————————————————
def merge_csv_files(eog_file: str | Path, gssc_file: str | Path, output_file: str | Path) -> None:
    """
    Merges two CSV files based on a common column and saves the merged result to a new CSV file.\\
    The GSSC staging is upsampled to align with the EOG sample timeline before merging.

    Parameters
    ----------
    eog_file : str | Path
        The path to the EOG CSV file.
    gssc_file : str | Path
        The path to the GSSC CSV file.
    output_file : str | Path
        The path to the output CSV file.
    """

    # 1) Load EOG CSV to check for required columns
    eog_df = pd.read_csv(eog_file)

    if "time_sec" not in eog_df.columns:
        raise ValueError("EOG CSV must contain 'time_sec'.")

    # 2) Get upsampled GSSC staging aligned to EOG timeline
    gssc_df = upsample_gssc_to_eog(eog_file, gssc_file)

    # 3) Merge the original EOG data with the upsampled GSSC staging
    merged_df = pd.concat(
        [
            eog_df.reset_index(drop=True),
            gssc_df.drop(columns=["time_sec"]).reset_index(drop=True),
        ],
        axis=1,
    )
    
    # 4) Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged file saved to: {Path(output_file).name}")

# —————————————————————————————————————————————————————————————————————
# Function to merge CSV files with extract_rems results
# —————————————————————————————————————————————————————————————————————
def merge_edf_rem_events(eog_path: str, 
                         events_path: str,
                         time_col: str = "time_sec",
                         loc_col: str = "LOC",
                         roc_col: str = "ROC",
                         start_col: str = "Start",
                         end_col: str = "End",
                         peak_col: str = "Peak",
                         ) -> pd.DataFrame:
    """
    Merges EOG CSV data with REM event annotations based on a common time column.

    Parameters
    ----------
    eog_path : str
        The path to the EOG CSV file.
    events_path : str
        The path to the CSV file containing REM event annotations.
    time_col : str, optional
        The name of the time column in both CSV files. Default is "time_sec". 
    loc_col : str, optional
        The name of the LOC column in the EOG CSV file. Default is "LOC".
    roc_col : str, optional
        The name of the ROC column in the EOG CSV file. Default is "ROC".
    start_col : str, optional
        The name of the start column in the events CSV file. Default is "Start".
    end_col : str, optional
        The name of the end column in the events CSV file. Default is "End".
    peak_col : str, optional
        The name of the peak column in the events CSV file. Default is "Peak".

    Returns
    -------
    signal_df : pd.DataFrame
        A DataFrame containing the merged EOG data and REM event annotations.
    """

    # 1) Load CSV
    eog_df = pd.read_csv(eog_path)
    events_df = pd.read_csv(events_path)

    # 2) Check required columns
    # Check EOG CSV
    for col in [time_col, loc_col, roc_col]:
        if col not in eog_df.columns:
            raise ValueError(f"EOG CSV must contain '{col}' column.")
    # Check events CSV
    for col in [start_col, end_col]:
        if col not in events_df.columns:
            raise ValueError(f"Events CSV must contain '{col}' column.")

    # 3) Keep only needed EOG columns and rename to standard names
    signal_df = eog_df[[time_col, loc_col, roc_col]].copy()
    signal_df = signal_df.rename(columns={
        time_col: "time",
        loc_col: "LOC",
        roc_col: "ROC"
    })

    # 4) Add annotation columns
    signal_df["is_rem_event"] = False
    signal_df["event_id"] = pd.NA
    signal_df["is_rem_peak"] = False

    # 5) Mark samples belonging to each REM event
    for i, row in events_df.iterrows():
        start = row[start_col]
        end = row[end_col]

        mask = (signal_df["time"] >= start) & (signal_df["time"] <= end)
        signal_df.loc[mask, "is_rem_event"] = True
        signal_df.loc[mask, "event_id"] = i

        # 5.a) Mark peak if the column exists
        if peak_col in events_df.columns and pd.notna(row[peak_col]):
            peak = row[peak_col]
            peak_idx = (signal_df["time"] - peak).abs().idxmin()
            signal_df.loc[peak_idx, "is_rem_peak"] = True

    return signal_df