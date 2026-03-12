# Filename: extract_rems_n.py 
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: extract REM from EDF recordings and save it in a CSV file.

# =====================================================================
# Imports
# =====================================================================
import mne 
import numpy as np
import sys
import os
import pandas as pd
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.channel_standardization import build_rename_map
from extract_rems import detect_rem_jaec
from gssc.infer import EEGInfer

# =====================================================================
# Constants
# =====================================================================
EXTRACT_REMS_DIR = Path("extracted_rems")
EXTRACT_REMS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Function
# =====================================================================
def extract_rems_from_edf(edf_path: Path, out_dir: Path = EXTRACT_REMS_DIR) -> pd.DataFrame | None:
    """
    Load one EDF file, rename EOG channels to canonical names, run GSSC
    sleep staging using EOG channels, detect REM events, and save the
    extracted REM events as a CSV file.

    Parameters
    ----------
    edf_path : Path
        The path to the input EDF file.
    out_dir : Path
        The directory where the output CSV file will be saved.

    Returns
    -------
    pd.DataFrame | None
        A dataframe containing extracted REM events if successful.
        Returns None if required EOG channels are missing.
    """
    print(f"\nProcessing: {edf_path}")

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    session_id = edf_path.parent.name

    # Load EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Rename channels for standardization
    rename_map = build_rename_map(raw.ch_names)
    print("Rename map:", rename_map)

    if rename_map:
        raw.rename_channels(rename_map)
    
    # Check required channels
    missing = [ch for ch in ["LOC", "ROC"] if ch not in raw.ch_names]
    if missing:
        print(f"Skipping {edf_path.parent.name} {edf_path.name} - missing channels: {missing}")
        return None

    # Set channel types 
    raw.set_channel_types({'LOC':'eog','ROC':'eog'})


    #GSCC staging EOG only 
    raw.filter(0.3,30, picks = ['LOC','ROC'])
    infer = EEGInfer(use_cuda = False)
    staging = infer.mne_infer(inst=raw, eeg=[], eog=['LOC', 'ROC'], eog_drop = False, filter = False)
    hypno_int = staging[0]
    
    # Resample EDF if edf isnt sampled at 128 Hz
    sf = raw.info["sfreq"]
    if sf != 128:
        print(f"Resampling from {sf} Hz to 128 Hz")
        raw = raw.copy().resample(128)
    loc = raw.get_data(picks=["LOC"])[0]
    roc = raw.get_data(picks=["ROC"])[0]
    
    # Upsampling hypnogram to match signal length
    samples_per_epoch = sf * 30
    hypno_up = np.repeat(hypno_int, samples_per_epoch)
    
    # Trim to multiple of 2^14 for dctwt( used in detect_rem_jaec)
    factor = 2**14
    trim = (min(len(loc), len(hypno_up)) // factor) * factor
    if trim == 0:
        print(f"Skipping {edf_path.name} - signal too short for dtcwt")
        return None

    loc = loc[:trim]
    roc = roc[:trim]
    hypno_up = hypno_up[:trim]

    # Rem detection
    result = detect_rem_jaec(loc, roc, hypno_up, method = 'ssc_threshold')

    # Create dataframe to retrun results
    df = result.summary()
    df['Stage'] = df['Stage'].map({
      0:'W', 
      1:'N1', 
      2:'N2', 
      3:'N3', 
      4:'REM'})
    
    out_path = out_dir / f"{session_id}_extracted_rems.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    return df
