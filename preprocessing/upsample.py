# Filename: upsample.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Upsample GSSC epoch-level staging to the EOG sample timeline. 

# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
import pandas as pd

# =====================================================================
# Function
# =====================================================================
def upsample_gssc_to_eog(eog_file: str | Path, gssc_file: str | Path) -> pd.DataFrame:
    """
    Upsamples GSSC epoch-level sleep staging to align with higher-frequency EOG sample timeline using backward merge.

    Parameters
    ----------
    eog_file : str | Path
        Path to EOG CSV with columns like ['time', 'LOC', 'ROC'].
    gssc_file : str | Path
        Path to GSSC CSV with columns like
        ['stages', 'times', 'prob_w', 'prob_n1', 'prob_n2', 'prob_n3', 'prob_rem'].
    
    Returns
    -------
    upsampled_df : pd.DataFrame
        A DataFrame with one row per EOG sample, containing the original EOG columns plus the upsampled GSSC staging and probabilities.
    """
    # 1) Read CSV files and save as dataframes
    eog_df = pd.read_csv(eog_file)
    gssc_df = pd.read_csv(gssc_file)

    if "time_sec" not in eog_df.columns:
        raise ValueError("EOG CSV must contain 'time_sec'.")
    
    # 2) Normalize GSSC column names
    gssc_df = gssc_df.rename(columns={
        "Times": "epoch_start",
        "times": "epoch_start",
        "Stages": "stage",
        "stages": "stage"
    })

    if "epoch_start" not in gssc_df.columns:
        raise ValueError("GSSC CSV must contain 'epoch_start'.")

    if "stage" not in gssc_df.columns:
        raise ValueError("GSSC CSV must contain 'stage'.")

    # 3) Sort dataframes and reset index
    eog_df = eog_df.sort_values("time_sec").reset_index(drop=True)
    gssc_df = gssc_df.sort_values("epoch_start").reset_index(drop=True)

    # 4) Save as new dataframe
    upsampled_df = pd.merge_asof(
        eog_df[["time_sec"]],
        gssc_df,
        left_on="time_sec",
        right_on="epoch_start",
        direction="backward",
    )

    return upsampled_df