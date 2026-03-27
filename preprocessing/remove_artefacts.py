# Filename: remove_artefacts.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Removes artefatcs from EOG signal based on preset thresholds.

# =====================================================================
# Imports
# =====================================================================
import numpy as np
import pandas as pd

# =====================================================================
# Function
# =====================================================================
def remove_artefacts(
        df:                 pd.DataFrame, 
        loc:                np.ndarray, 
        roc:                np.ndarray, 
        amplitude_thresh:   float = 300.0, 
        fs:                 float = 128.0
        ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    
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
    amplitude_thresh : float, optional
        The amplitude threshold in microvolts (µV). Rows where LOCAbsVal or ROCAbsVal exceeds this value are treated as artefacts and removed. Default is **300 [µV]**
    fs :float, optional
        Sampling frequency of the raw `loc` and `roc` signals in Hz.
        Used to convert timestamps to sample indices. Default is **128.0 [Hz]**

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
    required_cols = {'Start', 'Peak', 'End', 'LOCAbsValPeak', 'ROCAbsValPeak'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")
    if not isinstance(loc, np.ndarray):
        raise ValueError(f"Input LOC signal must be a numpy array. Got: {type(loc)}")
    if not isinstance(roc, np.ndarray):
        raise ValueError(f"Input ROC signal must be a numpy array. Got: {type(roc)}")
    if loc.shape != roc.shape:
        raise ValueError(
            f"LOC and ROC must have the same shape. "
            f"Got LOC: {loc.shape}, ROC: {roc.shape}"
        )
    if amplitude_thresh <= 0:
        raise ValueError(f"Amplitude threshold must be positive. Got: {amplitude_thresh}")
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive. Got: {fs}")
    
    # --- Handle empty DataFrame gracefully ---
    if df.empty:
        print("Warning: DataFrame is empty — nothing to remove.")
        return df.copy().reset_index(drop=True), loc.copy().astype(float), roc.copy().astype(float)
    # --- Create amplitude mask (True = clean, False = artefact) ---
    mask = (df["LOCAbsValPeak"] <= amplitude_thresh) & (df['ROCAbsValPeak'] <= amplitude_thresh)

    n_removed = int((~mask).sum())
    
    print(
        f"Artefacts removed: {n_removed} / {len(df)} events "
        f"({n_removed / len(df):.2%}) — threshold: {amplitude_thresh} [µV]"
        )
    
    # --- Mask raw signals over the full duration of each aretfact event --- 
    # Floor start and ceil end to ensure the full event is captured in the masking, 
    # at the cost of occasionally masking one extra sample at either boundary.

    loc_masked = loc.copy().astype(float)
    roc_masked = roc.copy().astype(float)

    n_samples = len(loc_masked)
    artefact_events = df.loc[~mask, ['Start', 'End']]

    for _, event in artefact_events.iterrows():
        start_idx = int(np.floor(event['Start'] * fs))
        end_idx   = int(np.ceil(event['End'] * fs))

        # Clamp to valid array bounds
        start_idx = max(0, start_idx)
        end_idx   = min(n_samples, end_idx)

        loc_masked[start_idx:end_idx] = np.nan
        roc_masked[start_idx:end_idx] = np.nan


    return df[mask].reset_index(drop = True), loc_masked, roc_masked 

# ---------------------------------------------------------------------------
# Backward-compatible alias (preserves the original mixed-case name)
# ---------------------------------------------------------------------------
def remove_Artefacts(
    df:              pd.DataFrame,
    loc:             np.ndarray,
    roc:             np.ndarray,
    Amplitude_thrsh: float = 300.0,
    fs:              float = 128.0,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    
    """Backward-compatible alias for :func:`remove_artefacts`."""
    return remove_artefacts(df, loc, roc, amplitude_thresh=Amplitude_thrsh, fs=fs)