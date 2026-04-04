# Filename: edf_to_csv.py 
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Convert EDF recordings to CSV files by extracting and standardizing LOC and ROC channels for all indexed sessions.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import mne

from preprocessing.index_file import index_sessions, parse_lights_txt
from preprocessing.channel_standardization import build_rename_map

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Constants
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
RAW_ROOT = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")
OUT_DIR = Path("eog_csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target sampling frequency for all saved EOG CSVs.
# MNE upsamples all channels to the highest sfreq found in the EDF,
# so files with mixed-rate channels (e.g. 1000 Hz) must be resampled
# down to a consistent rate before saving.
FS_TARGET = 250  # Hz

# =====================================================================
# Functions
# =====================================================================

# 1 —————————————————————————————————————————————————————————————————————
# 1 Function to convert a single EDF file to CSV
# 1 —————————————————————————————————————————————————————————————————————
def edf_to_csv(
    edf_path:    Path,
    pre_load:    bool = False,
    out_dir:     Path = OUT_DIR,
    lights_path: Path | None = None,
    fs_target:   int = FS_TARGET,
    ) -> None:
    """
    Load one EDF file, rename EOG channels to canonical names, and save the full signal matrix as a CSV file locally.

    Parameters
    ----------
    edf_path : Path
        The path to the input EDF file.
    pre_load : bool
        If True mne.io.read_raw_edf(preload = True). \\
        If False mne.io.read_raw_edf(preload = False). \\
        Default is **False**.
    out_dir : Path
        The directory where the output CSV file will be saved. \\
        By default, it is set to OUT_DIR, which is a directory named "local_csv_eog" in the current working directory.
    lights_path : Path | None
        Optional path to lights.txt file. If provided, the CSV is trimmed to the sleep period.
    fs_target : int
        The target sampling frequency for the saved CSV file. Default is **250 [Hz]**.\\
        Set to match the expected downstream pipeline frequency.

    Returns
    -------
    None
    """
    print(f"\nProcessing: {edf_path}")

    # --- 1) Load EDF ---
    raw = mne.io.read_raw_edf(edf_path, preload=pre_load, verbose=False)
    print(" Loaded raw:", raw)
    print(" preload was set to:", pre_load)
    print(" sfreq:", raw.info["sfreq"],"[Hz]")

    # --- 2) Rename EOG channels ---
    rename_map = build_rename_map(raw.ch_names)
    print("\nRename map:", rename_map)

    if rename_map:
        raw.rename_channels(rename_map)

    # --- 3) Check if both LOC and ROC channels are present after renaming ---
    missing = [ch for ch in ["LOC", "ROC"] if ch not in raw.ch_names]
    if missing:
        print(f"Skipping {edf_path.name} - missing channels: {missing}")
        return
    
    # --- 4) Resample to target frequency if needed ---
    sf = raw.info["sfreq"]
    if sf != fs_target:
        print(f"\nResampling from {sf} [Hz] to {fs_target} [Hz].")
        if not raw.preload:
            raw.load_data()
        raw = raw.resample(fs_target)

    # --- 5) Extract data for LOC and ROC channels ---
    loc = raw.get_data(picks=["LOC"])[0] * 1e6 # Convert V to µV
    roc = raw.get_data(picks=["ROC"])[0] * 1e6 # Convert V to µV

    # --- 5b) Unit sanity check ---
    loc_max = float(np.abs(loc).max())
    roc_max = float(np.abs(roc).max())
    print(f"    LOC range after conversion: {loc.min():.2f} to {loc.max():.2f} [µV]")
    print(f"    ROC range after conversion: {roc.min():.2f} to {roc.max():.2f} [µV]")

    if loc_max < 1.0 or roc_max < 1.0:
        raise ValueError(
            f"EOG signal suspiciously small after V→µV conversion "
            f"(LOC max={loc_max:.4f}, ROC max={roc_max:.4f} µV). "
            f"Check if EDF stores values in an unexpected unit."
        )
    if loc_max > 5000.0 or roc_max > 5000.0:
        raise ValueError(
            f"EOG signal suspiciously large after V→µV conversion "
            f"(LOC max={loc_max:.1f}, ROC max={roc_max:.1f} µV). "
            f"EDF may already store values in µV — check physical_dimension in header."
        )
    
    # --- 5c) Mask artefact samples on continuous signal ---
    # Any sample where |LOC| or |ROC| exceeds the threshold is set to NaN.
    # This ensures the EOG CSV is clean before merging and feature extraction.
    ARTEFACT_THRESH_UV = 300.0  # µV

    artefact_mask = (np.abs(loc) > ARTEFACT_THRESH_UV) | (np.abs(roc) > ARTEFACT_THRESH_UV)
    n_masked = int(artefact_mask.sum())
    n_total  = len(loc)

    loc = loc.astype(float)
    roc = roc.astype(float)
    loc[artefact_mask] = np.nan
    roc[artefact_mask] = np.nan

    print(f"    Artefact samples masked: {n_masked:,} / {n_total:,} "
          f"({n_masked / n_total:.2%}) — threshold: {ARTEFACT_THRESH_UV:.0f} µV")

    # --- 6) Create a DataFrame with time and EOG channels (signals in µV) ---
    df = pd.DataFrame({
        "time_sec": raw.times,
        "LOC": loc,   # µV
        "ROC": roc,   # µV
    })

    # --- 7) Trim to lights-off/lights-on window
    if lights_path is not None:
        lights_off, lights_on = parse_lights_txt(lights_path)
        df = df[(df["time_sec"] >= lights_off) & (df["time_sec"] <= lights_on)].reset_index(drop=True)
        print(f"    Trimmed to sleep period: {len(df)} samples remaining.")

    # --- 8) Save to CSV ---
    patient_id = edf_path.parent.name
    out_path = out_dir / f"{patient_id}_{edf_path.stem}_eog.csv"

    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")

# 2 —————————————————————————————————————————————————————————————————————
# 2 Function to convert all EDF files to CSV
# 2 —————————————————————————————————————————————————————————————————————
def convert_all_edfs(raw_root: Path = RAW_ROOT, out_dir: Path = OUT_DIR) -> None:
    """
    Find all EDF files using index_sessions and convert them to CSV.

    Parameters
    ----------
    raw_root : Path
        Root folder containing the EDF session folders.
    out_dir : Path
        Folder where CSV files should be saved.

    Returns
    -------
    None
    """
    records = index_sessions(
        root_dir=raw_root,
        edf=True,
        csv=False,
        txt=False,
        strict=False,
    )

    for rec in records:
        if rec.edf_path is None:
            continue

        try:
            edf_to_csv(rec.edf_path, out_dir, lights_path=rec.txt_path)
        except Exception as e:
            print(f"Failed for {rec.patient_id}: {e}")