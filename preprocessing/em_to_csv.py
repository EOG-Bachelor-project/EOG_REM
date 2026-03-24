# Filename: em_to_csv.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description:  Loads an EDF file, detects eye movements using detect_em, classifies REM epochs as Phasic/Tonic, and saves the result as CSV.
#               Mirrors the structure of edf_to_csv, GSSC_to_csv, and extract_rems_n.

# =====================================================================
# Imports
# =====================================================================
import mne
import numpy as np
import pandas as pd
from pathlib import Path
 
from preprocessing.channel_standardization import build_rename_map
from preprocessing.index_file import parse_lights_txt
from analysis.detect_em import detect_em, classify_rem_epochs, classify_rem_epochs_Umaer

# =====================================================================
# Constants
# =====================================================================
EM_DIR = Path("detected_ems")
EM_DIR.mkdir(parents=True, exist_ok=True)
 
# =====================================================================
# Functions
# =====================================================================

# =====================================================================
# Function
# =====================================================================
def em_to_csv(
        edf_path:         Path,
        gssc_df:          pd.DataFrame,
        hypno_int:        np.ndarray,
        out_dir:          Path = EM_DIR,
        lights_path:      Path | None = None,
        Dur_Thresh_SEM:   float = 0.5,
        Amp_Thresh_SEM:   float = 50.0,
        epoch_sec:        float = 4.0,
        psg_epoch_sec:    float = 30.0,
        amp_thresh_rem:   float = 150.0,
        dur_thresh_rem:   float = 0.5,
        amp_thresh_tonic: float = 25.0,
        fs_target:        int = 128,
        ) -> pd.DataFrame | None:
    """
    Load one EDF file, detect eye movements, classify them as SEM/REM and
    Phasic/Tonic, and save the result as a CSV file.
 
    Reuses the GSSC staging from gssc_df so GSSC never runs twice.
 
    Parameters
    ----------
    edf_path : Path
        Path to the EDF file.
    gssc_df : pd.DataFrame
        Pre-computed GSSC staging dataframe (output of GSSC_to_csv).\\
        Must contain 'stage' column with string stage labels.
    hypno_int : np.ndarray
        Integer hypnogram from GSSC staging (one value per 30 s epoch). \\
        Used by classify_rem_epochs to determine Phasic/Tonic.
    out_dir : Path
        Directory where the output CSV will be saved. Default 'detected_ems/'.
    lights_path : Path | None
        Optional path to lights.txt. If provided, signal is cropped to the
        sleep period before detection.
    Dur_Thresh_SEM : float
        Duration threshold in seconds for SEM classification. Default **0.5 [s]**. \\
        Eye movements longer than this are classified as SEM.
    Amp_Thresh_SEM : float
        Amplitude threshold in µV for SEM classification. Default **50 [µV]**. \\
        Eye movements below this amplitude are classified as SEM.
    epoch_sec : float
        Duration of each analysis epoch in seconds for Phasic/Tonic classification. Default **4.0 [s]**. This is independent of the 30-second.\\
        PSG scoring epoch - see classify_rem_epochs for details.
    psg_epoch_sec : float
        eDefault **30.0 [s]**.
    amp_thresh_rem : float
        Amplitude threshold in µV for classifying an eye movement as REM. Default **150 [µV]**. \\
        Eye movements with MeanAbsValPeak above this threshold are classified as REM.
    dur_thresh_rem : float
        Duration threshold in seconds for classifying an eye movement as REM. Default **0.5 [s]**. \\
        Eye movements with Duration below this threshold are classified as REM.
    amp_thresh_tonic : float
        Amplitude threshold in µV for classifying an eye movement as Tonic. Default **25 [µV]**. \\
        Eye movements with MeanAbsValPeak below this threshold are classified as Tonic.
    fs_target : int
        Target sampling rate in Hz. Signal is resampled if needed. Default **128 Hz**.
 
    Returns
    -------
    pd.DataFrame | None
        DataFrame with one row per detected eye movement containing:
        - `Start`, `Peak`, `End`, `Duration`,
        - `LOCAbsValPeak`, `ROCAbsValPeak`, `MeanAbsValPeak`,
        - `LOCAbsRiseSlope`, `ROCAbsRiseSlope`,
        - `LOCAbsFallSlope`, `ROCAbsFallSlope`,
        - `Stage`, `EM_Type`, `EpochIdx`, `EpochType`.

        Returns None if required channels are missing or signal is too short.
    """
    # --- Validation and setup ---
    if not isinstance(hypno_int, np.ndarray):
        raise ValueError(f"hypno_int must be a numpy array, but got type: {type(hypno_int)}")
    if epoch_sec <= 0:
        raise ValueError(f"epoch_sec must be a positive number, but got: {epoch_sec}")
    if fs_target <= 0:
        raise ValueError(f"fs_target must be a positive integer, but got: {fs_target}")
 
    print(f"\nProcessing: {edf_path}")
 
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
 
    session_id = edf_path.parent.name
 
    # --- 1) Load EDF ---
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
 
    # --- 2) Rename channels ---
    rename_map = build_rename_map(raw.ch_names)
    if rename_map:
        raw.rename_channels(rename_map)
    print(f"Rename map: {rename_map}")
 
    # --- 3) Check required channels ---
    missing = [ch for ch in ["LOC", "ROC"] if ch not in raw.ch_names]
    if missing:
        print(f"Skipping {session_id} — missing channels: {missing}")
        return None
 
    raw.set_channel_types({"LOC": "eog", "ROC": "eog"})
 
    # --- 4) Crop to lights window ---
    lights_off = 0.0                                          # Default to 0 if no lights.txt provided, so times remain absolute
    if lights_path is not None: 
        lights_off, lights_on = parse_lights_txt(lights_path) # Returns lights_off and lights_on in seconds
        raw = raw.crop(tmin=lights_off, tmax=lights_on)       # Crop raw signal to the lights off/on times to focus on the sleep period
        print(f"    Cropped to lights window: {lights_off:.1f} - {lights_on:.1f} [s]")
        print(f"    raw.times[0] after crop: {raw.times[0]:.1f} [s] (should be 0.0)")

        # Trim hypno_int to match cropped signal
        first_epoch = int(lights_off // psg_epoch_sec)
        last_epoch = int(lights_off // psg_epoch_sec)
        hypno_int = hypno_int[first_epoch:last_epoch]
        print(f"    hypno_int trimmed to epochs [{first_epoch}:{last_epoch}] = {len(hypno_int)} epochs")
 
    # --- 5) Resample if needed ---
    sf = raw.info["sfreq"]  
    if sf != fs_target:
        print(f"    Resampling {sf} Hz → {fs_target} Hz")
        raw = raw.copy().resample(fs_target)
        sf  = fs_target
 
    # --- 6) Filter ---
    raw.filter(0.1, 30, picks=["LOC", "ROC"])
 
    # --- 7) Extract signals in µV ---
    # detect_rem_jaec expects µV — raw.get_data() returns volts so we convert
    loc_uv = raw.get_data(picks=["LOC"])[0] * 1e6
    roc_uv = raw.get_data(picks=["ROC"])[0] * 1e6
    print(f"    LOC range: {loc_uv.min():.1f} to {loc_uv.max():.1f} [µV]")
    print(f"    ROC range: {roc_uv.min():.1f} to {roc_uv.max():.1f} [µV]")
 
    # --- 8) Build upsampled hypnogram ---
    samples_per_epoch = int(sf * psg_epoch_sec)
    hypno_up          = np.repeat(hypno_int, samples_per_epoch)
    print(f"Upsampled hypnogram to match signal length: {len(hypno_up)} samples")
 
    # --- 9) Trim to match lengths and multiple of 2^14 (required by dtcwt) ---
    factor = 2 ** 14
    trim   = (min(len(loc_uv), len(hypno_up)) // factor) * factor
    if trim == 0:
        print(f" Skipping {session_id} — signal too short for dtcwt")
        return None
 
    loc_uv   = loc_uv[:trim]
    roc_uv   = roc_uv[:trim]
    hypno_up = hypno_up[:trim]
    print(f"Signal length after trim: {trim} samples = {trim/sf:.1f} [s]")
 
    # --- 10) Detect eye movements ---
    em_df = detect_em(
        loc            = loc_uv,
        roc            = roc_uv,
        hypno_up       = hypno_up,
        Dur_Thresh_SEM = Dur_Thresh_SEM,
        Amp_Thresh_SEM = Amp_Thresh_SEM,
    )
 
    # --- 11) Classify Phasic / Tonic ---

    # Debug prints before we run classify_rem_epochs
    print(f"\nhypno_int length: {len(hypno_int)} | REM epochs {(hypno_int==4).sum()}")
    print(f"Frist few Peak times in em_df: {em_df['Peak'].head(5).values}")
    print(f"Expected epoch indices from peaks: {(em_df['Peak'].head(5).values // epoch_sec).astype(int)}")

    em_df = classify_rem_epochs(
        df               = em_df,
        hypno_int        = hypno_int,
        epoch_sec        = epoch_sec,
        amp_thresh_rem   = amp_thresh_rem,
        dur_thresh_rem   = dur_thresh_rem,
        amp_thresh_tonic = amp_thresh_tonic,
    )
 
    # --- 12) Offset times to absolute time reference ---
    # detect_rem_jaec operates on a signal starting at 0
    # so we add lights_off to align Start/Peak/End with time_sec in the EOG CSV
    if lights_path is not None:
        print(f"\nOffsetting EM times by lights_off = {lights_off:.1f} [s]")
        for col in ["Start", "Peak", "End"]:
            if col in em_df.columns:
                em_df[col] = em_df[col] + lights_off
 
    # --- 13) Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}_em.csv"
    em_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Total EMs: {len(em_df)} | "
          f"SEM: {(em_df['EM_Type'] == 'SEM').sum()} | "
          f"REM: {(em_df['EM_Type'] == 'REM').sum()}")
 
    return em_df