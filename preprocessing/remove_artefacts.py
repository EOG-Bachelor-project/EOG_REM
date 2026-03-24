# Filename: remove_artefacts.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Removes artefatcs from EOG signal based on preset thresholds.

# =====================================================================
# Imports
# =====================================================================
import numpy as np
import pandas as pd




def remove_Artefacts(df: pd.DataFrame, loc: np.ndarray, roc: np.ndarray, Amplitude_thrsh: float = 50.0) -> pd.DataFrame:
    
    """
    Remove artefacts from EOG signal based on preset thresholds. 

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing REM events wich msut contain the follwing columns: 'LOCAbsValPeak', 'ROCAbsValPeak', 'LOCAbsRiseSlope', 'ROCAbsRiseSlope', 'LOCAbsFallSlope', 'ROCAbsFallSlope'.
    loc : np.ndarray
        Raw LOC signal .
    roc : np.ndarray
        Raw ROC signal .
    Amplitude_thresh : float (default = 50.0)
        The amplitude threshold for artefact removal. If LOCAbsVal or ROCAbsVal exxceeds it will be removed.

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with artefacts removed.
    """

    # --- Validate inputs ---
    required_cols = {'LOCAbsValPeak', 'ROCAbsValPeak', 'LOCAbsRiseSlope', 'ROCAbsRiseSlope', 'LOCAbsFallSlope', 'ROCAbsFallSlope'}
    missing = required_cols - set(df.columns)
    if missing: 
        raise ValueError(f"Input DataFrame is missing columns: {missing}")
    if loc.shape != roc.shape:
        raise ValueError (f"LOC and ROC must have the same shape. Got LOC: {loc.shape}, ROC: {roc.shape}")

    if df["LOCAbsVal", "ROCAbsVal"] > Amplitude_thrsh: 
        return None