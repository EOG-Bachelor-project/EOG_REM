# Filename: rem_epoch_duration_feats.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Simple EOG feature extraction from a merged CSV file (output of merge_all).

# =========================================================================================================
# Imports
# =========================================================================================================

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def rem_epoch_duration_features (df: pd.DataFrame , fs: float, plot: bool = False) -> dict:
    """ 
    Duration statistics for each individual REM epoch. 

    Parameters
    -------
    df: pd.DataFrame 
        Full merged DataFrame containing a `stage` column with sleep stage labels.
    fs: float 
        sampling frequency of the signal [Hz].
    plot: bool (optional). 
        If True, display boxplot of REM epoch durations. Default is False
    
    Returns
    -------
    dict
        `rem_epoch_count`             : Number of distinct REM epochs.  
        `rem_epoch_mean_duration_min` : Mean REM epoch in minutes. 
        `rem_epoch_std_duration_min`  : Std of REM epoch in minutes. 
        `rem_epoch_min_duration_min`  : Shortest REM epoch duration in minutes. 
        `rem_epoch_max_duration_min`  : Longest REm epoch duration in minutes.
    """

    # --- Validate inputs ---
    required_cols = {'stage'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")
    
    # --- Identify REM blocks ---

    is_rem = (df["stage"] == "REM").astype(int)
    epoch_ids = (is_rem.diff().fillna(0) != 0).cumsum()
    rem_blocks = df[is_rem == 1].groupby(epoch_ids).size()
    rem_duration_min = rem_blocks / fs / 60.0 

    features: dict = {}

    # --- Retrun NaN if no REM Epochs are found 
    if rem_duration_min.empty: 
        print("No REM epochs found.")
        for k in ["rem_epoch_count", "rem_epoch_mean_duration_min", "rem_epoch_std_duration_min",
                  "rem_epoch_min_duration_min", "rem_epoch_max_duration_min"]:
            features[k] = np.nan
        return features
    
    # --- Compute duration statistics ---
    features["rem_epoch_count"]             = len(rem_duration_min)
    features["rem_epoch_mean_duration_min"] = round(float(rem_duration_min.mean()),3)
    features["rem_epoch_std_duration_min"]  = round(float(rem_duration_min.std()),3)
    features["rem_epoch_min_duration_min"]  = round(float(rem_duration_min.min()),3)
    features["rem_epoch_max_duration_min"]  = round(float(rem_duration_min.max()),3)

    print(f" REM epochs: {features['rem_epoch_count']} \\ mean: {features['rem_epoch_mean_duration_min']} min \\"
          f" std: {features['rem_epoch_std_duration_min']} \\ min: {features['rem_epoch_min_duration_min']} \\ "
          f" max : {features['rem_epoch_max_duration_min']}")
    
    # --- Optional Boxplot ---

    if plot: 

        fig, ax = plt.subplots(figsize =(4,6))
        sns.boxplot(y=rem_duration_min, ax = ax)
        ax.set_ylabel("Duration [min]")
        ax.set_xlabel("REM Epoch Durations")

        stats_text = (
            f"n = {features['rem_epoch_count']}\n"
            f"mean = {features['rem_epoch_mean_duration_min']} min\n"
            f"std = {features['rem_epoch_std_duration_min']} min\n"
            f"min = {features['rem_epoch_min_duration_min']} min\n"
            f"max = {features['rem_epoch_max_duration_min']} min"
        )

        ax.text(1.0, 0.5, stats_text, transform = ax.transAxes,
                verticalalignment = 'center', fontsize = 10,
                bbox=dict(boxstyle = 'round' , facecolor = 'lightyellow', alpha = 0.5))

        plt.tight_layout()
        plt.show()

    
    return features