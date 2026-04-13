# Filename: merge.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Utilities for merging EOG signals. GSSC sleep staging, and REM event annotations into a unifies CSV for downstream analysis.

# =====================================================================
# Imports
# =====================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing.upsample import upsample_gssc_to_eog

# =====================================================================
# Functions
# =====================================================================

# —————————————————————————————————————————————————————————————————————————————————————————————————
# Helper - fast interval merge
# —————————————————————————————————————————————————————————————————————————————————————————————————
def _merge_events_fast(
        merged_df:  pd.DataFrame,
        events_df:  pd.DataFrame,
        time_col:   str,
        start_col:  str,
        end_col:    str,
        peak_col:   str,
        prefix:     str,
        flag_col:   str,
        ) -> pd.DataFrame:
    """
    Merges event-level annotations into the per-sample DataFrame using
    merge_asof instead of a slow for-loop.
 
    For each sample, finds the event whose start <= time_sec <= end and
    copies all event columns across with the given prefix. Samples outside
    any event window get NA.
 
    Parameters
    ----------
    merged_df : pd.DataFrame
        Per-sample DataFrame.
    events_df : pd.DataFrame
        Event-level DataFrame with start_col, end_col columns.
    time_col : str
        Name of the time column in merged_df.
    start_col : str
        Name of the event start column in events_df.
    end_col : str
        Name of the event end column in events_df.
    peak_col : str
        Name of the event peak column in events_df.
    prefix : str
        Prefix added to all event columns in the output.
        e.g. 'event_' or 'em_'
    flag_col : str
        Name of the boolean flag column marking samples inside an event.
        e.g. 'is_rem_event' or 'is_em_event'
 
    Returns
    -------
    pd.DataFrame
        merged_df with added columns:
        - {flag_col}          : bool, True if sample is inside an event
        - {prefix}event_id    : int or NA, which event this sample belongs to
        - {prefix}is_peak     : bool, True if this is the peak sample
        - {prefix}{col}       : for every column in events_df
    """
 
    events_df = events_df.copy().reset_index(drop=True)
    events_df["_event_id"] = events_df.index

    # Ensure merge columns are numeric (prevents dtype mismatch from mixed CSVs)
    merged_df[time_col]    = pd.to_numeric(merged_df[time_col], errors="coerce")
    events_df[start_col]   = pd.to_numeric(events_df[start_col], errors="coerce")
    events_df[end_col]     = pd.to_numeric(events_df[end_col], errors="coerce")
    
    # Sort for merge_asof
    merged_df     = merged_df.sort_values(time_col).reset_index(drop=True)
    events_sorted = events_df.sort_values(start_col).reset_index(drop=True)
 
    # Prefix all event columns except the start column (used as merge key)
    rename_map     = {c: f"{prefix}{c}" for c in events_df.columns if c != start_col}
    events_renamed = events_sorted.rename(columns=rename_map)
 
    # Backward merge — each sample gets the most recent event that started before it
    merged_df = pd.merge_asof(
        merged_df,
        events_renamed,
        left_on=time_col,
        right_on=start_col,
        direction="backward",
    )
 
    # Mask out samples that fall after the event end
    end_prefixed = f"{prefix}{end_col}"
    id_prefixed  = f"{prefix}_event_id"
    inside       = merged_df[time_col] <= merged_df[end_prefixed]
 
    # Boolean flag
    merged_df[flag_col] = inside
 
    # Prefixed event_id — avoids collision when merging both REM events and EM events
    merged_df[f"{prefix}event_id"] = merged_df[id_prefixed].where(inside, other=pd.NA)
 
    # Zero out all prefixed columns for samples outside events
    for col in [c for c in merged_df.columns if c.startswith(prefix)]:
        merged_df.loc[~inside, col] = pd.NA
 
    merged_df = merged_df.drop(columns=[id_prefixed], errors="ignore")
 
    # Mark peak samples — prefixed so REM peaks and EM peaks don't collide
    merged_df[f"{prefix}is_peak"] = False
    if peak_col in events_df.columns:
        for _, row in events_df.iterrows():
            if pd.notna(row[peak_col]):
                peak_idx = (merged_df[time_col] - row[peak_col]).abs().idxmin()
                merged_df.loc[peak_idx, f"{prefix}is_peak"] = True
 
    return merged_df

# —————————————————————————————————————————————————————————————————————————————————————————————————
# MAIN MERGE FUNCTION - combine EOG, GSSC, and REM events into a single DataFrame and save as CSV
# —————————————————————————————————————————————————————————————————————————————————————————————————
def merge_all(
        eog_file:           str | Path,
        gssc_file:          str | Path,
        events_file:        str | Path,
        em_file:            str | Path,
        output_file:        str | Path,
        subepochs_file:     str | Path | None = None,
        time_col:           str = "time_sec",
        loc_col:            str = "LOC",
        roc_col:            str = "ROC",
        start_col:          str = "Start",
        end_col:            str = "End",
        peak_col:           str = "Peak",
        ) -> pd.DataFrame:
    """
    Merges EOG signals, GSSC sleep staging, REM event annotations, 
    and eye movement classifications (SEM/REM, Phasic/Tonic) into a single 
    unified per-sample DataFrame and saves it as CSV.

    Supports two EM classification modes:

    Default (use_Umaer=False in em_to_csv): 
    - Pass only `em_file`. EpochType is read directly from the EM event rows
       and propagated to each sample that falls inside a detected EM.
    
    Umaer (use-umaer=True is em_to_csv):
    - Pass both `em_file` and `subepochs_file`. EpochType is read from the sub-epoch 
      DataFrame and each sample is labelled with the EpochType of the sub-epoch it falls in.


    Parameters
    ----------
    eog_file : str | Path
        Path to EOG CSV file with columns ['time_sec', 'LOC', 'ROC'].
    gssc_file : str | Path
        Path to GSSC CSV file with columns ['epoch_start', 'stage', 'prob_*'].
    events_file : str | Path
        Path to REM events CSV file with at minimum Start and End columns.
    em_file : str | Path
        Path to CSV file containing detected eye movement events with their classifications (SEM/REM, Phasic/Tonic).
    output_file : str | Path
        Path where the merged CSV will be saved.
    subepochs_file : str | Path | None
        Default is **None**
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
        Merged DataFrame with one row per EOG sample containing:

        From EOG: \\
            `time_sec`, `LOC`, `ROC`
 
        From GSSC: \\
            `stage`, `prob_w`, `prob_n1`, `prob_n2`, `prob_n3`, `prob_rem`
 
        From REM events (extract_rems_from_edf): \\
            `is_rem_event`        — bool, sample is inside a REM event \\
            `event_event_id`      — which REM event \\
            `event_is_peak`       — bool, sample is the REM event peak \\
            `event_{col}`         — all other columns from events_file
 
        From EM classifications (em_to_csv): \\
            `is_em_event`         — bool, sample is inside a detected EM \\
            `em_event_id`         — which EM event \\
            `em_is_peak`          — bool, sample is the EM peak \\
            `EM_Type`             — 'SEM' or 'REM' (top-level shortcut) \\
            `EpochType`           — 'Phasic', 'Tonic', or 'Non-REM' (top-level shortcut) \\
            `em_{col}`            — all other columns from em_file
    """
    # --- Convert all paths up front so .name, .is_file() etc always work ---
    eog_file    = Path(eog_file)
    gssc_file   = Path(gssc_file)
    events_file = Path(events_file)
    output_file = Path(output_file)
    if subepochs_file is not None:
        subepochs_file = Path(subepochs_file)

    # --- Validate input files ---
    for file in [eog_file, gssc_file, events_file, em_file]:
        if not Path(file).is_file():
            raise FileNotFoundError(f"File not found: {file}")
    if subepochs_file is not None and not subepochs_file.is_file():
        raise FileNotFoundError(f"Subepochs file not found: {subepochs_file}")  

    # --- 1) Load EOG ---
    print("=" * 60)
    print(f"Loading EOG file: {eog_file.name} ...")
    eog_df = pd.read_csv(eog_file)
    for col in [time_col, loc_col, roc_col]:
        if col not in eog_df.columns:
            raise ValueError(
                f"EOG CSV must contain `{col}` column."
                f"Found columns: {list(eog_df.columns)}"
            )
    print(f"    {len(eog_df):,} samples  |  columns: {list(eog_df.columns)}")

    # --- 2) Upsample GSSC to EOG timeline ---
    print("\nUpsample GSSC file to the EOG timeline...")
    gssc_up = upsample_gssc_to_eog(eog_file, gssc_file)
    print(f"    {len(gssc_up):,} rows after upsample")

    # --- 3) Merge EOG and GSSC ---
    print("\nMerging EOG + GSSC...")
    merged_df = pd.concat(
        [
            eog_df.reset_index(drop=True),
            gssc_up.drop(columns=[time_col]).reset_index(drop=True),
        ],
        axis=1
    )
    print(f"    Merged shape: {merged_df.shape}")

    # --- 4) Load and merge REM events ---
    print("\nMerging REM events...")
    events_df = pd.read_csv(events_file)
    for col in [start_col, end_col]:
        if col not in events_df.columns:
            raise ValueError(
                f"Events CSV must contain `{col}` column."
                f"Found columns: {list(events_df.columns)}"
            )
    print(f"    {len(events_df):,} REM events found")

    merged_df = _merge_events_fast(
        merged_df  = merged_df,
        events_df  = events_df,
        time_col   = time_col,
        start_col  = start_col,
        end_col    = end_col,
        peak_col   = peak_col,
        prefix     = "event_",
        flag_col   = "is_rem_event",
    )
    print(f"    {merged_df['is_rem_event'].sum():,} samples inside REM events")

    # --- 5) Merge EM classifications ---
    print("\nMerging EM classifications...")
    em_df = pd.read_csv(em_file)
    for col in [start_col, end_col, "EM_Type"]:
        if col not in em_df.columns:
            raise ValueError(f"EM CSV must contain '{col}'. Found: {list(em_df.columns)}")
    print(f"  {len(em_df)} eye movements | "
          f"SEM: {(em_df['EM_Type'] == 'SEM').sum()} | "
          f"REM: {(em_df['EM_Type'] == 'REM').sum()}")
 
    merged_df = _merge_events_fast(
        merged_df  = merged_df,
        events_df  = em_df,
        time_col   = time_col,
        start_col  = start_col,
        end_col    = end_col,
        peak_col   = peak_col,
        prefix     = "em_",
        flag_col   = "is_em_event",
    )
 
    # Pull EM_Type and EpochType to top level for easy access in plot
    merged_df["EM_Type"]   = merged_df["em_EM_Type"]
    print(f"    SEM samples:    {(merged_df['EM_Type'] == 'SEM').sum():,}")
    print(f"    REM EM samples: {(merged_df['EM_Type'] == 'REM').sum():,}")
 

    # --- 6) Add EpochType column ---
    if subepochs_file is not None:
        print(f"\nMerging Umaer sub_epoch classification")
        subepoch_df = pd.read_csv(subepochs_file)
        for col in ["SubEpochStart", "SubEpochEnd", "EpochType"]:
            if col not in subepoch_df.columns:
                raise ValueError(
                    f"Subepochs CSV must contain '{col}'. "
                    f"Found: {list(subepoch_df.columns)}")
            
        merged_df["EpochType"] = pd.NA
        times = merged_df[time_col].values

        for _, row in subepoch_df.iterrows():
            mask = (times >= row["SubEpochStart"]) & (times < row["SubEpochEnd"])
            merged_df.loc[mask, "EpochType"] = row["EpochType"]
        
        counts = subepoch_df["EpochType"].value_counts()
        print(f"    Sub-epochs - Phasic: {counts.get('Phasic', 0)} | "
              f"Tonic: {counts.get('Tonic', 0)} | "
              f"Unclassified: {counts.get('Unclassified', 0)}")
    else:
        merged_df["EpochType"] = merged_df["em_EpochType"] if "em_EpochType" in merged_df.columns else pd.NA

    # --- 7) Save ---
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"Saved: {output_file.name}")
    print(f"Shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
    print("=" * 60)

    return merged_df