# upsample.py

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
    Upsample GSSC epoch-level staging to the EOG sample timeline.

    Parameters
    ----------
    eog_file : str | Path
        Path to EOG CSV with columns like ['time', 'LOC', 'ROC'].
    gssc_file : str | Path
        Path to GSSC CSV with columns like
        ['stages', 'times', 'prob_w', 'prob_n1', 'prob_n2', 'prob_n3', 'prob_rem'].
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per EOG sample, containing the original EOG columns plus the upsampled GSSC staging and probabilities.
    """
    eog_df = pd.read_csv(eog_file)
    gssc_df = pd.read_csv(gssc_file)

    if "time_sec" not in eog_df.columns:
        raise ValueError("EOG CSV must contain 'time_sec'.")
    
    # normalize GSSC column names
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

    eog_df = eog_df.sort_values("time_sec").reset_index(drop=True)
    gssc_df = gssc_df.sort_values("epoch_start").reset_index(drop=True)

    upsampled_df = pd.merge_asof(
        eog_df[["time_sec"]],
        gssc_df,
        left_on="time_sec",
        right_on="epoch_start",
        direction="backward",
    )

    return upsampled_df