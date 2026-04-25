# Filename: eeg_to_csv.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Extracts EEG signals from pre-processed LOC and ROC signals
#              using the DTCWT-based method in eeg_signals_from_eog, and saves
#              the result as CSV. Signals are passed directly from extract_rems_from_edf
#              to avoid redundant preprocessing.

# =====================================================================
# Imports
# =====================================================================
import numpy as np
import pandas as pd
from pathlib import Path

from Tests.test_eeg_signals_from_eog import eeg_signals_from_eog

# =====================================================================
# Constants
# =====================================================================
EEG_DIR = Path("extracted_eeg")
EEG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Function
# =====================================================================
def eeg_to_csv(
        edf_path:      Path,
        loc:           np.ndarray,
        roc:           np.ndarray,
        loc_clean:     np.ndarray,
        roc_clean:     np.ndarray,
        out_dir:       Path = EEG_DIR,
        lights_path:   Path | None = None,
        method:        str = "subtract",
        fs:            int = 128,
) -> pd.DataFrame | None:
    """
    Extract EEG signals from pre-processed LOC and ROC signals and save as CSV.

    Signals are passed directly from extract_rems_from_edf — no reloading,
    resampling, or trimming is performed here as these steps are already
    done upstream.

    Parameters
    ----------
    edf_path : Path
        Path to the EDF file. Used only to derive session_id and output filename.
    loc : np.ndarray
        Raw LOC signal in µV, already resampled and trimmed (from extract_rems_from_edf).
    roc : np.ndarray
        Raw ROC signal in µV, already resampled and trimmed (from extract_rems_from_edf).
    loc_clean : np.ndarray
        EOG-filtered LOC signal in µV (result.data_filt[0] from detect_rem_jaec).
    roc_clean : np.ndarray
        EOG-filtered ROC signal in µV (result.data_filt[1] from detect_rem_jaec).
    out_dir : Path
        Directory where the output CSV will be saved. Default is **'extracted_eeg/'**.
    lights_path : Path | None
        Optional path to lights.txt. Used only to derive the time offset for
        the time_sec column. Default is **None**.
    method : str
        Method passed to eeg_signals_from_eog. Either 'subtract' or 'mask'.
        Default is **'subtract'**.
    fs : int
        Sampling rate of the signals in Hz. Default is **128 Hz**.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns ``time_sec``, ``EEG_LOC``, ``EEG_ROC``,
        saved as ``{session_id}_eeg.csv``.
        Returns None if signal is empty.
    """
    # --- Validate inputs ---
    if not isinstance(loc, np.ndarray) or not isinstance(roc, np.ndarray):
        raise ValueError("loc and roc must be numpy arrays.")
    if len(loc) == 0 or len(roc) == 0:
        print(f"Skipping {edf_path.parent.name} — empty signal.")
        return None
    if fs <= 0:
        raise ValueError(f"fs must be a positive integer, but got: {fs}")

    print(f"\nProcessing: {edf_path}")

    session_id = edf_path.parent.name

    print(f"    LOC range: {loc.min():.1f} to {loc.max():.1f} [µV]  |  length: {len(loc)} samples")
    print(f"    ROC range: {roc.min():.1f} to {roc.max():.1f} [µV]  |  length: {len(roc)} samples")

    # --- 1) Get lights_off offset for time vector ---
    lights_off = 0.0
    if lights_path is not None:
        from preprocessing.index_file import parse_lights_txt
        result = parse_lights_txt(lights_path)
        if result is not None:
            lights_off, _ = result
            print(f"    Lights off offset: {lights_off:.1f} [s]")
        else:
            print(f"    Lights times unavailable — using 0.0 [s] offset.")

    # --- 2) Extract EEG signals ---
    def _interp_nans(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy().astype(float)
        nans = np.isnan(arr)
        if nans.any():
            idx = np.arange(len(arr))
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
        return arr
    
    loc       = _interp_nans(loc)
    roc       = _interp_nans(roc)
    loc_clean = _interp_nans(loc_clean)
    roc_clean = _interp_nans(roc_clean)

    print(f"    loc NaNs after interp:       {np.isnan(loc).sum()}")
    print(f"    roc NaNs after interp:       {np.isnan(roc).sum()}")
    print(f"    loc_clean NaNs after interp: {np.isnan(loc_clean).sum()}")
    print(f"    roc_clean NaNs after interp: {np.isnan(roc_clean).sum()}")

    print(f"\nExtracting EEG signals using method='{method}'...")
    loc_eeg, roc_eeg = eeg_signals_from_eog(loc, roc, loc_clean, roc_clean, method=method)

    # --- 3) Build time vector and DataFrame ---
    time_sec = (np.arange(len(loc)) / fs) + lights_off
    eeg_df   = pd.DataFrame({
        "time_sec": time_sec,
        "EEG_LOC":  loc_eeg,
        "EEG_ROC":  roc_eeg,
    })

    # --- 4) Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}_eeg.csv"
    eeg_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"EEG LOC — min: {loc_eeg.min():.2f}, max: {loc_eeg.max():.2f}, mean: {loc_eeg.mean():.2f} [µV]")
    print(f"EEG ROC — min: {roc_eeg.min():.2f}, max: {roc_eeg.max():.2f}, mean: {roc_eeg.mean():.2f} [µV]")

    return eeg_df