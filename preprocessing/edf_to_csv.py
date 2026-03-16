# Filename: edf_to_csv.py 
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Convert EDF recordings to CSV files by extracting and standardizing LOC and ROC channels for all indexed sessions.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

from pathlib import Path
import pandas as pd
import mne

from preprocessing.index_file import index_sessions, parse_lights_txt
from preprocessing.channel_standardization import build_rename_map

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Constants
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
RAW_ROOT = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")
OUT_DIR = Path("eog_csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# Functions
# =====================================================================

# 1 —————————————————————————————————————————————————————————————————————
# 1 Function to convert a single EDF file to CSV
# 1 —————————————————————————————————————————————————————————————————————
def edf_to_csv(edf_path:    Path, 
               out_dir:     Path = OUT_DIR,
               lights_path: Path | None = None
               ) -> None:
    """
    Load one EDF file, rename EOG channels to canonical names, and save the full signal matrix as a CSV file locally.

    Parameters
    ----------
    edf_path : Path
        The path to the input EDF file.
    out_dir : Path
        The directory where the output CSV file will be saved. \\
        By default, it is set to OUT_DIR, which is a directory named "local_csv_eog" in the current working directory.
    lights_path : Path | None
        Optional path to lights.txt file. If provided, the CSV is trimmed to the sleep period.
    
    Returns
    -------
    None
    """
    print(f"\nProcessing: {edf_path}")

    # --- 1) Load EDF ---
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # --- 2) Rename EOG channels ---
    rename_map = build_rename_map(raw.ch_names)
    print("Rename map:", rename_map)

    if rename_map:
        raw.rename_channels(rename_map)

    # --- 3) Check if both LOC and ROC channels are present after renaming ---
    missing = [ch for ch in ["LOC", "ROC"] if ch not in raw.ch_names]
    if missing:
        print(f"Skipping {edf_path.name} - missing channels: {missing}")
        return

    # --- 4) Extract data for LOC and ROC channels ---
    loc = raw.get_data(picks=["LOC"])[0]
    roc = raw.get_data(picks=["ROC"])[0]

    # --- 5) Create a DataFrame with time and EOG channels ---
    df = pd.DataFrame({
        "time_sec": raw.times,
        "LOC": loc,
        "ROC": roc,
    })

    # --- 6) Trim to lights-off/lights-on window
    if lights_path is not None:
        lights_off, lights_on = parse_lights_txt(lights_path)
        df = df[(df["time_sec"] >= lights_off) & (df["time_sec"] <= lights_on)].reset_index(drop=True)
        print(f" Trimmed to sleep period: {len(df)} samples remaining.")

    # --- 7) Save to CSV ---
    patient_id = edf_path.parent.name
    out_path = out_dir / f"{patient_id}_{edf_path.stem}_eog.csv"

    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

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