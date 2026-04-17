# Filename: GSSC_to_csv.py 
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Runs GSSC sleep staging on EOG channels from EDF files and sev per-epoch stage predictions as CSV.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations
from pathlib import Path
import mne
import pandas as pd
import torch
import gssc.networks
torch.serialization.add_safe_globals([gssc.networks.ResSleep])
from gssc.infer import EEGInfer

from preprocessing.index_file import parse_lights_txt
from preprocessing.channel_standardization import build_rename_map

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Paths
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
GSSC_DIR = Path("gssc_csv")
GSSC_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Function
# =====================================================================

# GSSC test function
def GSSC_to_csv(
        edf_path:    str | Path, 
        raw:         mne.io.Raw | None = None,
        pre_load:    bool = False,
        out_dir:     Path = GSSC_DIR,
        lights_path: Path | None = None
        ) -> pd.DataFrame:
    """
    Load one EDF file, run GSSC inference, and save the result as CSV. \\
    If a lights.txt path is provided, the output is trimmed to the lights-off/lights-on window.

    Parameters
    ----------
    edf_path : str | Path
        The path to the EDF file to be loaded.
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
    
    Returns
    -------
    pd.DataFrame
        The trimmed staging datafram
    """
    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    session_id = edf_path.parent.name
    
    print(f"\nProcessing: {edf_path}")

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

    # --- 2) Pick channels ---
    picks = ["LOC", "ROC"]
    missing = [ch for ch in picks if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing expected channels: {missing}. Available: {raw.ch_names}")

    raw.pick(picks)

    # --- 3) Set channel types (Helps GSSC choose) ---
    raw.set_channel_types({"ROC": "eog", "LOC": "eog"})

    # --- 4) Load data (Required for GSSC) ---
    raw.load_data()

    # --- 5) Filter signal manually ---
    raw.filter(0.1,30, picks = picks) 

    # --- 6) Run inference ---
    infer = EEGInfer(use_cuda = False)
    stages, times, probs = infer.mne_infer(inst=raw, eeg=[], eog=["LOC", "ROC"], eog_drop=False, filter=False)
    
    df = pd.DataFrame(data={
        "epoch_start": times, 
        "stage": stages,
        })
    
    # Add probalities to dataframe and rename stages to string
    df[["prob_w", "prob_n1", "prob_n2", "prob_n3", "prob_rem"]] = probs

    df["stage"] = df["stage"].map({
        0: "W",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM",
    })

    # --- 7) Trim to lights-off/lights-on window
    if lights_path is not None:
        lights_off, lights_on = parse_lights_txt(lights_path)
        df = df[(df["epoch_start"] >= lights_off) & (df["epoch_start"] <= lights_on)].reset_index(drop=True)
        print(f"    Trimmed to sleep period: {len(df)} samples remaining.")

    # --- 8) Save as CSV ---
    out_path = out_dir / f"{session_id}_gssc.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}") 

    # Return the trimmed dataframe so callers (e.g. extract_rems_from_edf)
    # can reuse it directly, so there is no need to run GSSC a second time.
    return df