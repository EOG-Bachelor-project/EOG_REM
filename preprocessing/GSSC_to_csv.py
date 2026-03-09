# GSSC_to_csv.py

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
from preprocessing.channel_standardization import build_rename_map
from gssc.infer import EEGInfer

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Constants
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
edf = Path("l:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a/contiguous.edf")
channels = ['LOC', 'ROC']
print("Exists:", edf.exists())


# =====================================================================
# Function
# =====================================================================

# GSSC test function
def GSSC_to_csv(folder: str | Path):
    """
    Load EDF file from the specified folder path, run the GSSC inference, and return the results as a DataFrame.

    Parameters
    ----------
    folder_path : str | Path
        The path to the folder containing the EDF file to be loaded.
    """
    folder = Path(folder)
    edfs = list(folder.glob("*.edf"))
    if not edfs:
        raise FileNotFoundError(f"No EDF files found in: {folder}")
    edf_path = edfs[0]
    print("Using EDF:", edf_path)

    # 1) Read header only
    raw = mne.io.read_raw_edf(edf_path, preload=False)
    
    print("Loaded raw:", raw)
    print("Channels:", raw.ch_names[:20], "..." if len(raw.ch_names) > 20 else "")
    print("sfreq:", raw.info["sfreq"])

    # 2) Rename EOG channels 
    rename_map = build_rename_map(raw.ch_names)
    print("Rename map:", rename_map)

    if rename_map:
        raw.rename_channels(rename_map)

    # Check if both LOC and ROC channels are present after renaming
    missing = [ch for ch in ["LOC", "ROC"] if ch not in raw.ch_names]
    if missing:
        print(f"Skipping {edf_path.name} - missing channels: {missing}")
        return

    # 3) Pick channels
    picks = ["LOC", "ROC"]
    missing = [ch for ch in picks if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing expected channels: {missing}. Available: {raw.ch_names}")
    raw.pick(picks)

    # 4) Load data (Required for GSSC)
    raw.load_data()

    # 5) Set channel types (Helps GSSC choose)
    raw.set_channel_types({"ROC": "eog", "LOC": "eog"})

    # 6) Run inference
    infer = EEGInfer()
    stages, times, probs = infer.mne_infer(inst=raw)
    
    df = pd.DataFrame(data={
        "Stages": stages, 
         "Times": times
         })
    # Add probalities to dataframe and rename stages to string
    df[["P_W", "P_N1", "P_N2", "P_N3", "P_REM"]] = probs
    df['Stages'] = df['Stages'].map({0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'})
    

    print(df)
    # 7) Convert datframe to csv and save it 
    df.to_csv('gssc_results.csv', index=False)
    print("Saved: gssc_results.csv")

    return stages, times, probs

