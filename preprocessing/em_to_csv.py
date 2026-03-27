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
        fs_target:        int = 128,
        pre_load:         bool = False,

        # classify_rem_epochs (default) params
        epoch_sec:        float = 4.0,
        psg_epoch_sec:    float = 30.0,
        min_rapid:        int = 1,
        
        
        # Umaer params (pased through to classify_rem_epochs_Umaer)
        use_Umaer : bool = False, 
        sub_epoch_len: float = 4.0,
        window_len: float = 2.0,
        min_separation: float = 8.0,
        amp_thresh_rem: float = 150,
        dur_thresh_rem: float = 0.5,
        amp_thresh_tonic: float = 25.0,

        ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Load one EDF file, detect eye movements, classify them as SEM/REM and
    Phasic/Tonic, and save the result as a CSV file.
 
    Reuses the GSSC staging from gssc_df so GSSC never runs twice.

    Two classifiers are available, selected via `use_Umaer`:
    - **False** (default): uses classify_rem_epochs(), and annotates each EM event with EpochIdx and EpochType. \\
                           Saves one CSV: ``{session_id}_em.csv``
    - **True**: uses classify_rem_epochs_Umaer(), and produces a sub-epoch DataFrame (one row per 4-second sub-epoch inside REM). \\
            Saves two CSV's: ``{session_id}_em.csv`` and ``{session_id}_subepochs.csv``

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
        Duration threshold in seconds for SEM classification. Default is **0.5 [s]**. \\
        Eye movements longer than this are classified as SEM.
    Amp_Thresh_SEM : float
        Amplitude threshold in µV for SEM classification. Default is **50 [µV]**. \\
        Eye movements below this amplitude are classified as SEM.
    fs_target : int
        Target sampling rate in Hz. Signal is resampled if needed. Default is**128 Hz**.
     pre_load : bool
        If True mne.io.read_raw_edf(preload = True). \\
        If False mne.io.read_raw_edf(preload = False). \\
        Default is **False**.

    **classify_rem_epochs params (used when use_Umaer=False)**
    
    epoch_sec : float
        Duration of each analysis epoch in seconds for Phasic/Tonic classification. Default is **4.0 [s]**. This is independent of the 30-second.\\
        PSG scoring epoch - see classify_rem_epochs for details.
    psg_epoch_sec : float
        Duration of each PSG scoring epoch in seconds Default is **30.0 [s]**.\\
        Must match the epoch length used to build hypno_int. Used to trim hypno_int to align with the cropped signal.
    min_rapid : int
        Minimum number of REMs per epoch to classify as Phasic. Default is **1**.
    
    **classify_rem_epochs_Umaer params (used when use_Umaer=True)**

    use_Umaer : bool
        If True use `classify_rem_epochs_Umaer()` instead of `classify_rem_epochs()`
        Default is **False**
    sub_epoch_len : float
        Length of sub-epochs to classify in seconds. Default is **4.0 [s]**
    window_len : float 
        Length of each adjacent window used for Phasic/Tonic detection in seconds. Default is **2.0 [s]**
    min_seperation : float 
        Default is **8.0 [s]**
    amp_thresh_rem : float
        Default is **150 [µV]** 
    dur_thresh_rem : float
        Default is **0.5 [s]**
    amp_thresh_tonic : float
        Maximum mean absolute amplitude [µV] in both 2-second windows for Tonic classification. Default is **25 [µV]**
 
    Returns
    -------
    pd.DataFrame | None
        When use_Umaer=False: DataFrame with one row per detected eye movement containing:
        - `Start`, `Peak`, `End`, `Duration`,
        - `LOCAbsValPeak`, `ROCAbsValPeak`, `MeanAbsValPeak`,
        - `LOCAbsRiseSlope`, `ROCAbsRiseSlope`,
        - `LOCAbsFallSlope`, `ROCAbsFallSlope`,
        - `Stage`, `EM_Type`, `EpochIdx`, `EpochType`.

    tuple[pd.DataFrame, pd.DataFrame] | None
        When use_Umaer=True: Tuple of (em_df, subepoch_df) where:
        - em_df: Same EM event DataFRame as above (without EpochIdx/EpochType columns).
        - subepoch_df: Datafram with one row per 4-second sub-epoch inside REM containing: `SubEpochStart`, `SubEpochEnd`, `EpochIdx`, `EpochType`  

        Returns None if required channels are missing or signal is too short.
    """
    # --- Validation and setup ---
    if not isinstance(hypno_int, np.ndarray):
        raise ValueError(f"hypno_int must be a numpy array, but got type: {type(hypno_int)}")
    if fs_target <= 0:
        raise ValueError(f"fs_target must be a positive integer, but got: {fs_target}")
        
    if not use_Umaer:
        if epoch_sec <= 0:
            raise ValueError(f"epoch_sec must be positive, but got: {epoch_sec}")
        if psg_epoch_sec <= 0:
            raise ValueError(f"psg_epoch_sec must be positive, but got: {psg_epoch_sec}")
        if min_rapid < 1:
            raise ValueError(f"min_rapid must be >= 1, but got: {min_rapid}")
    

    
    if use_Umaer:
        if sub_epoch_len <= 0:
            raise ValueError(f"sub_epoch_len must be positive, but got: {sub_epoch_len}")
        if window_len <= 0:
            raise ValueError(f"window_len must be positive, but got: {window_len}")
        if sub_epoch_len != 2 * window_len:
            raise ValueError(
                f"sub_epoch_len ({sub_epoch_len}) must be exactly 2 * window_len({window_len}). "
                f"Got 2 * window_len = {2 * window_len} [s]"
            )
        if psg_epoch_sec % sub_epoch_len != 0:
            raise ValueError(
                f"sub_epoch_len ({sub_epoch_len}) must divide evenly into psg_epoch_sec ({psg_epoch_sec}). "
                f"Got remainder: {psg_epoch_sec % sub_epoch_len} [s]."
            )
        if min_separation < sub_epoch_len:
            raise ValueError(
                f"min_separation ({min_separation}) must be >= sub_epoch_len ({sub_epoch_len}). "
                f"Otherwise adjacent sub-epochs can never both be kept."
            )
 
    print(f"\nProcessing: {edf_path}")
 
    #if not edf_path.exists():
    #    raise FileNotFoundError(f"EDF file not found: {edf_path}")
 
    session_id = edf_path.parent.name
 
    # --- 1) Load EDF ---
    raw = mne.io.read_raw_edf(edf_path, preload=pre_load, verbose=False)
    print(" Loaded raw:", raw)
    print(" preload was set to:", pre_load)
    print(" sfreq:", raw.info["sfreq"],"Hz")
 
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

        # hypno_int covers the full recording from t=0 (one entry per psg_epoch_sec).
        # After cropping, the signal starts at t=0 but corresponds to lights_off in
        # the original recording. Slice hypno_int so index 0 = lights_of

        # Trim hypno_int to match cropped signal
        #first_epoch = int(lights_off // psg_epoch_sec)
        #last_epoch = int(lights_on // psg_epoch_sec)
        #hypno_int = hypno_int[first_epoch:last_epoch]
        #print(f"    hypno_int trimmed to epochs [{first_epoch}:{last_epoch}] = {len(hypno_int)} epochs")
 
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
 
    # --- 10) Detect eye movements ---
    em_df = detect_em(
        loc            = loc_uv,
        roc            = roc_uv,
        hypno_up       = hypno_up,
        Dur_Thresh_SEM = Dur_Thresh_SEM,
        Amp_Thresh_SEM = Amp_Thresh_SEM,
    )
 
    # --- 11) Classify Phasic / Tonic ---
    if use_Umaer == True:
        print(f"\nRunning classify_rem_epochs_Umaer...")
        subepoch_df = classify_rem_epochs_Umaer(
            df                  = em_df,
            loc                 = loc_uv,
            roc                 = roc_uv,
            hypno_int           = hypno_int,
            sf                  = sf,
            epoch_len           = int(psg_epoch_sec),
            sub_epoch_len       = sub_epoch_len,
            window_len          = window_len,
            min_separation      = min_separation,
            amp_thresh_rem      = amp_thresh_rem,
            dur_thresh_rem      = dur_thresh_rem, 
            amp_thresh_tonic    = amp_thresh_tonic,
        )
    else:
        print(f"\nRunning classify_rem_epochs...")
        # Debug prints before we run classify_rem_epochs
        print(f"\nhypno_int length: {len(hypno_int)} | REM epochs {(hypno_int==4).sum()}")
        print(f"Frist few Peak times in em_df: {em_df['Peak'].head(5).values}")
        print(f"Expected epoch indices from peaks: {(em_df['Peak'].head(5).values // epoch_sec).astype(int)}")

        em_df = classify_rem_epochs(
            df            = em_df,
            hypno_int     = hypno_int,
            epoch_sec     = epoch_sec,
            psg_epoch_sec = psg_epoch_sec,
            min_rapid     = min_rapid,
        )
 
    # --- 12) Offset times to absolute time reference ---
    # detect_rem_jaec operates on a signal starting at 0
    # so we add lights_off to align Start/Peak/End with time_sec in the EOG CSV
    if lights_path is not None:
        print(f"\nOffsetting EM times by lights_off = {lights_off:.1f} [s]")
        for col in ["Start", "Peak", "End"]:
            if col in em_df.columns:
                em_df[col] = em_df[col] + lights_off
        if use_Umaer:
            for col in ["SubEpochStart", "SubEpochEnd"]:
                if col in subepoch_df.columns:
                    subepoch_df[col] = subepoch_df[col] + lights_off
 
    # --- 13) Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}_em.csv"
    em_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Total EMs: {len(em_df)} | "
          f"SEM: {(em_df['EM_Type'] == 'SEM').sum()} | "
          f"REM: {(em_df['EM_Type'] == 'REM').sum()}")
    
    if use_Umaer:
        subepoch_path = out_dir / f"{session_id}_subepochs.csv"
        subepoch_df.to_csv(subepoch_path, index = False)
        print(f"Saved: {subepoch_path}")
        # Add prints
        return em_df, subepoch_df
    
    return em_df