# Filename: em_to_csv.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description:  Loads an EDF file, detects eye movements using detect_em, 
#               classifies REM epochs as Phasic/Tonic, and saves the result as CSV.
#               Mirrors the structure of edf_to_csv, GSSC_to_csv, and extract_rems_n.

# =====================================================================
# Imports
# =====================================================================
import mne                  # For loading EDF files and handling raw EEG/EOG data
import numpy as np          # For numerical operations, especially with arrays
import pandas as pd         # For DataFrame manipulation and saving to CSV
from pathlib import Path    # For handling file paths
 
from preprocessing.channel_standardization import build_rename_map
from preprocessing.index_file import parse_lights_txt
from analysis.detect_em import detect_em, classify_rem_epochs_Umaer

# =====================================================================
# Constants
# =====================================================================
EM_DIR = Path("detected_ems")
EM_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Function
# =====================================================================
def em_to_csv(
        edf_path:         Path,
        hypno_int:        np.ndarray,
        raw:              mne.io.Raw | None = None,
        out_dir:          Path = EM_DIR,
        lights_path:      Path | None = None,
        Dur_Thresh_SEM:   float = 0.5,
        fs_target:        int = 128,
        pre_load:         bool = False,
        
        # classify_rem_epochs_Umaer params
        psg_epoch_sec:    float = 30.0,
        sub_epoch_len: float = 4.0,
        phasic_dur_thresh:  float = 1.0,

        ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Load one EDF file, detect eye movements, classify them as SEM/REM and
    Phasic/Tonic, and save the result as CSV.
 
    Reuses the GSSC staging from gssc_df so GSSC never runs twice.
 
    Uses classify_rem_epochs_Umaer() to produce a sub-epoch DataFrame (one row per 4-second sub-epoch inside REM).
    Saves two CSV's: ``{session_id}_em.csv`` and ``{session_id}_subepochs.csv``
 
    Parameters
    ----------
    edf_path : Path
        Path to the EDF file.
    hypno_int : np.ndarray
        Integer hypnogram from GSSC staging (one value per 30 s epoch). \\
        Used by classify_rem_epochs_Umaer to determine Phasic/Tonic.
    raw : mne.io.Raw | None
        Pre-loaded MNE Raw obeject with channels already renamed. 
        If provided, the EDF is not re-read from disk and ``pre_load`` is ignored. \\
        A copy is made internally so the caller's object is not mutated. \\
        Default is **None** (load from from ``edf_path``).
    out_dir : Path
        Directory where the output CSV will be saved. Default is **'detected_ems/'**.
    lights_path : Path | None
        Optional path to lights.txt. If provided, signal is cropped to the sleep period before detection.
    Dur_Thresh_SEM : float
        Duration threshold in seconds for SEM classification. Default is **0.5 [s]**. \\
        Eye movements longer than this are classified as SEM. Classification is duration-only
        (amplitude threshold has been dropped).
    fs_target : int
        Target sampling rate in Hz. Signal is resampled if needed. Default is **128 Hz**.
    pre_load : bool
        If True mne.io.read_raw_edf(preload = True). \\
        If False mne.io.read_raw_edf(preload = False). \\
        Default is **False**.
    psg_epoch_sec : float
        Duration of each PSG scoring epoch in seconds. Default is **30.0 [s]**. \\
        Must match the epoch length used to build hypno_int.
    sub_epoch_len : float
        Length of sub-epochs to classify in seconds. Default is **4.0 [s]**.
    phasic_dur_thresh : float
        Minimum total EM duration [s] within a sub-epoch required for Phasic classification.
        Default is **1.0 [s]**.
 
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] | None
        Tuple of (em_df, subepoch_df) where:
        - em_df: DataFrame with one row per detected eye movement containing:
          `Start`, `Peak`, `End`, `Duration`,
          `LOCAbsValPeak`, `ROCAbsValPeak`, `MeanAbsValPeak`,
          `LOCAbsRiseSlope`, `ROCAbsRiseSlope`,
          `LOCAbsFallSlope`, `ROCAbsFallSlope`,
          `Stage`, `EM_Type`.
        - subepoch_df: DataFrame with one row per 4-second sub-epoch inside REM containing:
          `SubEpochStart`, `SubEpochEnd`, `EpochIdx`, `EpochType`.
 
        Returns None if required channels are missing or signal is too short.
    """
    # --- Validation and setup ---
    if not isinstance(hypno_int, np.ndarray):
        raise ValueError(f"hypno_int must be a numpy array, but got type: {type(hypno_int)}")
    if fs_target <= 0:
        raise ValueError(f"fs_target must be a positive integer, but got: {fs_target}")
    if sub_epoch_len <= 0:
        raise ValueError(f"sub_epoch_len must be positive, but got: {sub_epoch_len}")
    if phasic_dur_thresh <= 0:
        raise ValueError(f"phasic_dur_thresh must be positive, but got: {phasic_dur_thresh}")
 
    print(f"\nProcessing: {edf_path}")
 
    session_id = edf_path.parent.name
   
    # --- 1) Load EDF ---
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
 
    # --- 2) Check required channels ---
    missing = [ch for ch in ["LOC", "ROC"] if ch not in raw.ch_names]
    if missing:
        print(f"Skipping {session_id} — missing channels: {missing}")
        return None
 
    raw.set_channel_types({"LOC": "eog", "ROC": "eog"})
 
    # --- 3) Crop to lights window ---
    lights_off = 0.0                                                        # Default to 0 if no lights.txt provided, so times remain absolute
    if lights_path is not None: 
        lights_off, lights_on = parse_lights_txt(lights_path)               # Returns lights_off and lights_on in seconds
        raw = raw.crop(tmin=lights_off, tmax=min(lights_on, raw.times[-1])) # Crop raw signal to the lights off/on times to focus on the sleep period
        print(f"    Cropped to lights window: {lights_off:.1f} - {lights_on:.1f} [s]")
 
    # ---4) Resample if needed ---
    sf = raw.info["sfreq"]  
    if sf != fs_target:
        print(f"\nResampling {sf} [Hz] to {fs_target} [Hz]")
        raw = raw.copy().resample(fs_target)
        sf  = fs_target
 
    # --- 5) Filter ---
    raw.filter(0.1, 30, picks=["LOC", "ROC"])
 
    # --- 6) Extract signals in µV ---
    # detect_rem_jaec expects µV — raw.get_data() returns volts so we convert
    loc_uv = raw.get_data(picks=["LOC"])[0] * 1e6
    roc_uv = raw.get_data(picks=["ROC"])[0] * 1e6
    print(f"    \nLOC range: {loc_uv.min():.1f} to {loc_uv.max():.1f} [µV]")
    print(f"    ROC range: {roc_uv.min():.1f} to {roc_uv.max():.1f} [µV]")
 
    # --- 7) Build upsampled hypnogram ---
    samples_per_epoch = int(sf * psg_epoch_sec)
    hypno_up          = np.repeat(hypno_int, samples_per_epoch)
    print(f"\nUpsampled hypnogram to match signal length: {len(hypno_up)} samples")
 
    # --- 8) Trim to match lengths and multiple of 2^14 (required by dtcwt) ---
    factor = 2 ** 14
    trim = (min(len(loc_uv), len(hypno_up)) // factor) * factor
    print(f"    len(loc_uv) = {len(loc_uv)} | len(hypno_up) = {len(hypno_up)} | factor={factor}")
    print(f"    trim = {trim}")
    if trim == 0:
        print(f" Skipping {session_id} — signal too short for dtcwt")
        return None
 
    loc_uv   = loc_uv[:trim]
    roc_uv   = roc_uv[:trim]
    hypno_up = hypno_up[:trim]
    print(f"Signal length after trim: {trim} samples = {trim/sf:.1f} [s]")
 
    # --- 9) Detect eye movements ---
    em_df = detect_em(
        loc            = loc_uv,
        roc            = roc_uv,
        hypno_up       = hypno_up,
        fs             = sf,
        Dur_Thresh_SEM = Dur_Thresh_SEM,
    )
 
    # --- 10) Classify Phasic / Tonic ---
    print(f"\nRunning classify_rem_epochs_Umaer...")
    subepoch_df = classify_rem_epochs_Umaer(
        df                 = em_df,
        loc                = loc_uv,
        roc                = roc_uv,
        hypno_int          = hypno_int,
        sf                 = sf,
        epoch_len          = int(psg_epoch_sec),
        sub_epoch_len      = sub_epoch_len,
        phasic_dur_thresh  = phasic_dur_thresh,
    )
 
    # --- 11) Offset times to absolute time reference ---
    if lights_path is not None:
        print(f"\nOffsetting EM times by lights_off = {lights_off:.1f} [s]")
        for col in ["Start", "Peak", "End"]:
            if col in em_df.columns:
                em_df[col] = em_df[col] + lights_off
        for col in ["SubEpochStart", "SubEpochEnd"]:
            if col in subepoch_df.columns:
                subepoch_df[col] = subepoch_df[col] + lights_off
 
    # --- 12) Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}_em.csv"
    em_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Total EMs: {len(em_df)} | "
          f"SEM: {(em_df['EM_Type'] == 'SEM').sum()} | "
          f"REM: {(em_df['EM_Type'] == 'REM').sum()}")
 
    subepoch_path = out_dir / f"{session_id}_subepochs.csv"
    subepoch_df.to_csv(subepoch_path, index=False)
    print(f"Saved: {subepoch_path}")
    return em_df, subepoch_df