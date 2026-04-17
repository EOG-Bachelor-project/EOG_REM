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
from gssc.infer import EEGInfer

from preprocessing.channel_standardization import build_rename_map
from preprocessing.index_file import parse_lights_txt
from extract_rems import detect_rem_jaec
from preprocessing.GSSC_to_csv import GSSC_to_csv
from preprocessing.remove_artefacts import remove_artefacts

# =====================================================================
# Constants
# =====================================================================
EXTRACT_REMS_DIR = Path("extracted_rems")
EXTRACT_REMS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Function
# =====================================================================
def extract_rems_from_edf(
        edf_path:    Path, 
        raw:         mne.io.Raw | None = None,
        pre_load:    bool = False,
        out_dir:     Path = EXTRACT_REMS_DIR,
        lights_path: Path | None = None,
        gssc_df:     pd.DataFrame | None = None,
        ) -> pd.DataFrame | None:
    """
    Load one EDF file, rename EOG channels to canonical names, run GSSC sleep staging using EOG channels, detect REM events, and save the extracted REM events as a CSV file.\\
    If a lights.txt path is provided, the signal is trimmed to the sleep period before detection

    Parameters
    ----------
    edf_path : Path
        The path to the input EDF file.
    raw : mne.io.Raw | None
        Pre-loaded MNE Raw obeject with channels already renamed. 
        If provided, the EDF is not re-read from disk and ``pre_load`` is ignored. \\
        A copy is made internally so the caller's object is not mutated. \\
        Default is **None** (load from from ``edf_path``).
    pre_load : bool
        If True mne.io.read_raw_edf(preload = True). \\
        If False mne.io.read_raw_edf(preload = False). \\
        Default is **False**.
    out_dir : Path
        The directory where the output CSV file will be saved.
    lights_path : Path | None
        Optional path to lights.txt file. If provided, the CSV is trimmed to the sleep period.
    gssc_df : Path | None
        Pre-computed GSSC staging dataframe (output of GSSC_to_csv).
        Pass this in to avoid running GSSC a second time.

    Returns
    -------
    pd.DataFrame | None
        A dataframe containing extracted REM events if successful.
        Returns None if required EOG channels are missing.
    """
    # --- Validate inputs ---
    if not isinstance(edf_path, Path):
        raise ValueError(f"edf_path must be a Path object. Got: {type(edf_path)}")
    if not isinstance(out_dir, Path):
        raise ValueError(f"out_dir must be a Path object. Got: {type(out_dir)}")
    if lights_path is not None and not isinstance(lights_path, Path):
        raise ValueError(f"lights_path must be a Path object or None. Got: {type(lights_path)}")
    if gssc_df is not None and not isinstance(gssc_df, pd.DataFrame):
        raise ValueError(f"gssc_df must be a pandas DataFrame or None. Got: {type(gssc_df)}")

    print(f"\nProcessing: {edf_path}")

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    session_id = edf_path.parent.name

    # --- Load EDF ---
    if raw is None:
        raw = mne.io.read_raw_edf(edf_path, preload=pre_load, verbose=False)
        print(" Loaded raw:", raw)
        print(" preload was set to:", pre_load)
        print(" sfreq:", raw.info["sfreq"],"[Hz]")

        rename_map = build_rename_map(raw.ch_names)
        print("\nRename map:", rename_map)

        if rename_map:
            raw.rename_channels(rename_map)
    else:
        raw = raw.copy()
    
    # --- Check required channels ---
    missing = [ch for ch in ["LOC", "ROC"] if ch not in raw.ch_names]
    if missing:
        print(f"Skipping {edf_path.parent.name} {edf_path.name} - missing channels: {missing}")
        return None

    # --- Set channel types ---
    raw.set_channel_types({'LOC':'eog','ROC':'eog'})

    # --- Trim to lights-off/lights-on window if provided  ---
    # We do this before staging and detection, so that staging/detection only operates on the sleep period.
    if lights_path is not None:
        lights_off, lights_on = parse_lights_txt(lights_path)
        raw = raw.crop(tmin=lights_off, tmax=min(lights_on, raw.times[-1]))
        print(f"    Trimmed to sleep period: {lights_off:.1f} [s] - {lights_on:.1f} [s].") 


    # --- GSCC staging EOG only ---
    if gssc_df is None:
        gssc_df = GSSC_to_csv(edf_path, lights_path=lights_path)
    stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    hypno_int = gssc_df["stage"].map(stage_map).fillna(0).astype(int).values
        
    
    # --- Resample EDF ---
    # If EDF is not already at 128 [Hz], resample it. 
    # This is required for the REM detection to work properly, 
    # and also ensures that the sample indices in the output 
    # CSV match the sample indices of the raw signals we use for detection.
    sf = raw.info["sfreq"]
    if sf != 128:
        print(f"\nResampling from {sf} [Hz] to 128 [Hz]")
        raw = raw.copy().resample(128)
        sf = 128
    loc = raw.get_data(picks=["LOC"])[0] * 1e6 # V to uV
    roc = raw.get_data(picks=["ROC"])[0] * 1e6

    print(f"    Signal length: {len(loc)} samples at {sf} [Hz] = {len(loc)/sf:.1f} [s]")
    print(f"    hypno_int epochs: {len(hypno_int)} epochs = {len(hypno_int) * 30:.1f} [s]")
    
    # --- Upsampling hypnogram --- 
    # We do this to match signal length 
    samples_per_epoch = sf * 30
    hypno_up = np.repeat(hypno_int, samples_per_epoch)
    
    # --- Trim to multiple of 2^14 for dctwt( used in detect_rem_jaec) ---
    factor = 2**14
    trim = (min(len(loc), len(hypno_up)) // factor) * factor
    if trim == 0:
        print(f"Skipping {edf_path.name} - signal too short for dtcwt")
        return None

    loc = loc[:trim]
    roc = roc[:trim]
    hypno_up = hypno_up[:trim]

    # Print info before we use detect_rem_jaec
    print(f"    LOC range: {loc.min():.2f} to {loc.max():.2f}   |   LOC length: {len(loc)}")
    print(f"    ROC range: {roc.min():.2f} to {roc.max():.2f}   |   ROC length: {len(roc)}")
    print(f"    hypno_up length: {len(hypno_up)}")
    print(f"    REM samples in hypno_up: {(hypno_up == 4).sum()}")
    print(f"    Total epochs: {len(hypno_up) / (128*30):.1f}")


    # --- Rem detection ---
    result = detect_rem_jaec(loc, roc, hypno_up, method = 'ssc_threshold')

    # --- Create dataframe to retrun results ---
    df = result.summary()

    df['Stage'] = df['Stage'].map({
      0:'W', 
      1:'N1', 
      2:'N2', 
      3:'N3', 
      4:'REM'})
    
    # --- Remove artefacts ---
    # Called before the lights_off offset block below, so df['Start']/df['End']
    # are still in signal-relative time (t=0) and match the sample indices of loc/roc.
    print("\nRemoving artefacts...")
    df, loc, roc = remove_artefacts(df, loc, roc, amplitude_thresh=300.0, fs=sf)
    print(f"Clean REM events remaining: {len(df)}")
 
    # --- Offset event times ---
    # This will allign with EOG `time_sec` (which starts at `lights_off`)
    # `detect_rem_jaec()` operates on a signal starting at 0, so we add `lights_off` 
    # to make Start, End, Peak match the absolute time reference in the EOG CSV
    if lights_path is not None:
        print(f"Offsetting event time by `lights_off` = {lights_off:.1f} [s]")
        for col in ["Start", "Peak", "End"]:
            if col in df.columns:
                df[col] = df[col] + lights_off

    out_path = out_dir / f"{session_id}_extracted_rems.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    return df, loc, roc, result  
