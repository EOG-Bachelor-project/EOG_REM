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

# —————————————————————————————————————————————————————————————————————
# Function to merge EOG, GSSC staging, and REM event annotations
# —————————————————————————————————————————————————————————————————————
def merge_all(
        eog_file:    str | Path,
        gssc_file:   str | Path,
        events_file: str | Path,
        output_file: str | Path,
        time_col:    str = "time_sec",
        loc_col:     str = "LOC",
        roc_col:     str = "ROC",
        start_col:   str = "Start",
        end_col:     str = "End",
        peak_col:    str = "Peak",
        ) -> pd.DataFrame:
    """
    Merges EOG signals, GSSC sleep staging, and REM event annotations into a single unified pandas DataFrame and saves it as a CSV file.

    The GSSC staging is upsampled to math the EOG sample timeline. REM events are joined by interval lookup, preserving all original event columns (Start, End, Peak, etc.) alongside per-sample boolean flags.

    Parameters
    ----------
    eog_file : str | Path
        Path to EOG CSV file with columns ['time_sec', 'LOC', 'ROC'].
    gssc_file : str | Path
        Path to GSSC CSV file with columns ['epoch_start', 'stage', 'prob_*'].
    events_file : str | Path
        Path to REM events CSV file with at minimum Start and End columns.
    output_file : str | Path
        Path where the merged CSV will be saved.
    time_col : str
        Name of time column in the EOG CSV. Default is 'time_sec'.
    loc_col : str 
        Name of LOC column in the EOG CSV. Default is 'LOC'.
    roc_col : str 
        Name of ROC column in the EOG CSV. Default is 'ROC'.
    start_col : str
        Name of the event start column int the evnets CSV. Default is 'Start'.
    end_col : str 
        Name of the event end column int the evnets CSV. Default is 'End'.
    peak_col : str 
        Name of the peak column int the evnets CSV. Default is 'Peak'.

    Returns
    -------
    pd.DataFrame
        Merged pandas DataFrame with one row per EOG sample containing EOG signals, GSSC staging/probabilities, and REM event annotations.
    """

    # --- 1) Load EOG ---
    print("Loading EOG file: ", {eog_file.parent.name})
    eog_df = pd.read_csv(eog_file)
    for col in [time_col, loc_col, roc_col]:
        if col not in eog_df.columns:
            raise ValueError(f"EOG CSV must contain `{col}` column.")

    # --- 2) Upsample GSSC to EOG timeline ---
    print("\nUpsample GSSC file to the EOG timeline")
    gssc_up = upsample_gssc_to_eog(eog_file, gssc_file)

    # --- 3) Merge EOG and GSSC ---
    merged_df = pd.concat(
        [
            eog_df.reset_index(drop=True),
            gssc_up.drop(columns=[time_col]).reset_index(drop=True),
        ],
        axis=1
    )

    # --- 4) Load REM events ---
    events_df = pd.read_csv(events_file)
    for col in [start_col, end_col]:
        if col not in events_df.columns:
            raise ValueError(f"Events CSV must contain `{col}` column.")
    

    # --- 5) Add per-sample REM annotation columns ---
    merged_df["is_rem_event"] = False
    merged_df["event_id"] = pd.NA

    # Copy all original event columns into per-sample columns (prefixed with "event_")
    # Include Start and End so plot function can access event_Start and event_End
    event_extra_cols = [c for c in events_df.columns]
    for col in event_extra_cols:
        merged_df[f"event_{col}"] = pd.NA

    # --- 6) Assign event metadata to each sample within an event window ---
    for i, row in events_df.iterrows():
        mask = (merged_df[time_col] >= row[start_col]) & (merged_df[time_col] <= row[end_col])
        merged_df.loc[mask, "is_rem_event"] = True
        merged_df.loc[mask, "event_id"] = 1
        for col in event_extra_cols:
            merged_df.loc[mask, f"event_{col}"] = row[col]

    # --- 7) Mark peak sample ---
    merged_df["is_rem_peak"] = False
    if peak_col in events_df.columns:
        for _, row in events_df.iterrows():
            if pd.notna(row[peak_col]):
                peak_idx = (merged_df[time_col]-row[peak_col]).abs().idxmin()
                merged_df.loc[peak_idx, "is_rem_peak"] = True

    # --- 8) Save to CSV ---
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved to: {output_file.name}")

    return merged_df