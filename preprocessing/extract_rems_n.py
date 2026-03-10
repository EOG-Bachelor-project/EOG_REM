# extract_rems_from_edf.py

# =====================================================================
# Imports
# =====================================================================
import mne 
import numpy as np
import sys
import os
import pandas as pd
from pathlib import Path
from preprocessing.channel_standardization import build_rename_map
from extract_rems import detect_rem_jaec
from gssc.infer import EEGInfer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    # Set channel types 
    raw.set_channel_types({'LOC':'eog','ROC':'eog'})


    #GSCC staging EOG only - Skipping this
    #infer = EEGInfer()
    #annotations = infer.mne_infer(raw, eog=['LOC', 'ROC'])

    #sleepstage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
    #hypno_int = np.array([sleepstage_map[a['description']]for a in annotations])

    # Resample EDF if edf isnt sampled at 128 Hz
    sf = raw.info["sfreq"]
    if sf != 128:
        print(f"Resampling from {sf} Hz to 128 Hz")
        raw = raw.copy().resample(128)
        loc = raw.get_data(picks=["LOC"])[0]
        roc = raw.get_data(picks=["LOC"])[0]
    
    # Trim to multiple of 2^14 for dctwt( used in detect_rem_jaec)
    factor = 2**14
    trim = (len(loc) // factor) * factor
    if trim == 0:
        print(f"Skipping {edf_path.name} - signal too short for dtcwt")

    loc = loc[:trim]
    roc = roc[:trim]

    hypno_up = 0

    # Rem detection
    result = detect_rem_jaec(loc,roc,hypno_up,method = 'ssc_threshold')

    # Create dataframe to retrun results
    df = result.summary()
    ## df['Stage'] = df['Stage'].map({
    #  0:'W', 
    #  1:'N1', 
    #  2:'N2', 
    #  3:'N3', 
    #  4:'REM'})
    ##
    out_path = out_dir / f"{session_id}_extracted_rems.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    return df