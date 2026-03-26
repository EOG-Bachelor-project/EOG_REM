# Filename: remove_artefacts.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Removes artefatcs from EOG signal based on preset thresholds.

# =====================================================================
# Imports
# =====================================================================
import numpy as np
import pandas as pd




def remove_Artefacts(df: pd.DataFrame, loc: np.ndarray, roc: np.ndarray, Amplitude_thrsh: float = 300.0, fs: float = 250) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    
    """
    Remove artefacts from EOG signal based on preset thresholds. 

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing REM events. Must contain the columns: 'LOCAbsValPeak', 'ROCAbsValPeak' and column specified by `sample_col`
    loc : np.ndarray
        Raw LOC signal .
    roc : np.ndarray
        Raw ROC signal .
    Amplitude_thresh : float, optional
        The amplitude threshold in microvolts (µV). Rows where LOCAbsVal or ROCAbsVal exceeds this value are treated as artefacts and removed. Default is 300µV
    fs :float, optional
        Sampling frequency of the raw `loc` and `roc` signals in Hz.
        Used to convert timestamps to sample indices. Default is 128.0 Hz

    Returns
    -------
    df : pd.DataFrame
        The input DataFrame with artefact rows removed and index reset.
    loc_masked : np.ndarray
        Copy of `loc` with artefact samples set to NaN.
    roc_masked : np.ndarray
        Copy of `roc` with artefact samples set to NaN.
    """

    # --- Validate inputs ---
    required_cols = {'Start','Peak','End','LOCAbsValPeak', 'ROCAbsValPeak'}
    missing = required_cols - set(df.columns)
    if missing: 
        raise ValueError(f"Input DataFrame is missing columns: {missing}")
    if loc.shape != roc.shape:
        raise ValueError (f"LOC and ROC must have the same shape. Got LOC: {loc.shape}, ROC: {roc.shape}")

    # --- Create amplitude mask (True = clean, False = artefact) ---
    mask = (df["LOCAbsValPeak"] <= Amplitude_thrsh) & (df['ROCAbsValPeak']<= Amplitude_thrsh)

    n_removed = (~mask).sum()
    print(f"Artefacts removed: {n_removed} ({n_removed / len(df):.2%} of events)"
          f"— threshold: {Amplitude_thrsh} µV")
    
    # --- Mask raw signals over the full duration of each aretfact event --- 
    # Floor start and ceil end to ensure the full event is captured in the masking, at the cost of occasionally masking one extra sample at either boundary.

    loc_masked = loc.copy().astype(float)
    roc_masked = roc.copy().astype(float)

    artefact_events = df.loc[~mask, ['Start', 'End']]
    for _, event in artefact_events.iterrows():
        start_idx = int(np.floor(event['Start'] * fs))
        end_idx   = int(np.ceil(event['End']   * fs))
        loc_masked[start_idx:end_idx] = np.nan
        roc_masked[start_idx:end_idx] = np.nan


    return df[mask].reset_index(drop = True), loc_masked, roc_masked 