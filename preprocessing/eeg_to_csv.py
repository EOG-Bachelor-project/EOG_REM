# Filename: eeg_to_csv.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Loads an EDF file, extracts EEG signals from LOC and ROC using
#              the DTCWT-based method in eeg_signals_from_eog, and saves the result as CSV.
#              Mirrors the structure of em_to_csv.

# =====================================================================
# Imports
# =====================================================================
import mne
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing.channel_standardization import build_rename_map
from preprocessing.index_file import parse_lights_txt
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
        hypno_int:     np.ndarray,
        out_dir:       Path = EEG_DIR,
        lights_path:   Path | None = None,
        method:        str = "subtract",
        fs_target:     int = 128,
        pre_load:      bool = False,
        psg_epoch_sec: float = 30.0,
) -> pd.DataFrame | None:
    """
    Load one EDF file, extract EEG signals from LOC and ROC using the
    DTCWT-based method, and save the result as CSV.

    Parameters
    ----------
    edf_path : Path
        Path to the EDF file.
    hypno_int : np.ndarray
        Integer hypnogram from GSSC staging (one value per 30 s epoch).
    out_dir : Path
        Directory where the output CSV will be saved. Default is **'extracted_eeg/'**.
    lights_path : Path | None
        Optional path to lights.txt. If provided, signal is cropped to the
        sleep period before extraction.
    method : str
        Method passed to eeg_signals_from_eog. Either 'subtract' or 'mask'.
        Default is **'subtract'**.
    fs_target : int
        Target sampling rate in Hz. Signal is resampled if needed. Default is **128 Hz**.
    pre_load : bool
        If True, mne.io.read_raw_edf(preload=True). Default is **False**.
    psg_epoch_sec : float
        Duration of each PSG scoring epoch in seconds. Default is **30.0 [s]**.
        Must match the epoch length used to build hypno_int.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns ``time_sec``, ``EEG_LOC``, ``EEG_ROC``,
        saved as ``{session_id}_eeg.csv``.
        Returns None if required channels are missing or signal is too short.
    """
    if not isinstance(hypno_int, np.ndarray):
        raise ValueError(f"hypno_int must be a numpy array, but got type: {type(hypno_int)}")
    if fs_target <= 0:
        raise ValueError(f"fs_target must be a positive integer, but got: {fs_target}")

    print(f"\nProcessing: {edf_path}")

    session_id = edf_path.parent.name

    # --- 1) Load EDF ---
    raw = mne.io.read_raw_edf(edf_path, preload=pre_load, verbose=False)
    print(f" Loaded raw: {raw}")
    print(f" Preload was set to: {pre_load}")
    print(f" sfreq: {raw.info['sfreq']} [Hz] | n_channels: {len(raw.ch_names)} | duration: {raw.n_times / raw.info['sfreq']:.1f} [s]")

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
    lights_off = 0.0
    if lights_path is not None:
        lights_off, lights_on = parse_lights_txt(lights_path)
        raw = raw.crop(tmin=lights_off, tmax=min(lights_on, raw.times[-1]))
        print(f"    Cropped to lights window: {lights_off:.1f} - {lights_on:.1f} [s]")

    # --- 5) Resample if needed ---
    sf = raw.info["sfreq"]
    if sf != fs_target:
        print(f"\nResampling {sf} [Hz] to {fs_target} [Hz]")
        raw = raw.copy().resample(fs_target)
        sf  = fs_target

    # --- 6) Filter ---
    raw.filter(0.1, 30, picks=["LOC", "ROC"])

    # --- 7) Extract signals in µV ---
    loc_uv = raw.get_data(picks=["LOC"])[0] * 1e6
    roc_uv = raw.get_data(picks=["ROC"])[0] * 1e6
    print(f"    \nLOC range: {loc_uv.min():.1f} to {loc_uv.max():.1f} [µV]")
    print(f"    ROC range: {roc_uv.min():.1f} to {roc_uv.max():.1f} [µV]")

    # --- 8) Build upsampled hypnogram ---
    samples_per_epoch = int(sf * psg_epoch_sec)
    hypno_up          = np.repeat(hypno_int, samples_per_epoch)
    print(f"\nUpsampled hypnogram to match signal length: {len(hypno_up)} samples")

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

    # --- 10) Extract EEG signals ---
    print(f"\nExtracting EEG signals using method='{method}'...")
    loc_eeg, roc_eeg = eeg_signals_from_eog(loc_uv, roc_uv, hypno_up, method=method)

    # --- 11) Build time vector and DataFrame ---
    time_sec = (np.arange(trim) / sf) + lights_off
    eeg_df   = pd.DataFrame({
        "time_sec": time_sec,
        "EEG_LOC":  loc_eeg,
        "EEG_ROC":  roc_eeg,
    })

    # --- 12) Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}_eeg.csv"
    eeg_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"EEG LOC — min: {loc_eeg.min():.2f}, max: {loc_eeg.max():.2f}, mean: {loc_eeg.mean():.2f} [µV]")
    print(f"EEG ROC — min: {roc_eeg.min():.2f}, max: {roc_eeg.max():.2f}, mean: {roc_eeg.mean():.2f} [µV]")

    return eeg_df