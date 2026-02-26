# GSSC_test.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations

from pathlib import Path
import mne
import pandas as pd
import torch
import gssc.networks
torch.serialization.add_safe_globals([gssc.networks.ResSleep])

from gssc.infer import EEGInfer

# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Predefined variables
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
edf = Path("l:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a/contiguous.edf")
channels = ['EOGH-A1', 'EOGV-A2']
print("Exists:", edf.exists())

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# GSSC test function
def test_GSSC(folder: str | Path):
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

    # 2) Pick channels
    picks = ["C3", "EOGH", "EOGV"]
    missing = [ch for ch in picks if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing expected channels: {missing}. Available: {raw.ch_names}")
    raw.pick(picks)

    # 3) Load data (Required for GSSC)
    raw.load_data()

    # 4) Set channel types (Helps GSSC choose)
    raw.set_channel_types({"C3": "eeg", "EOGH": "eog", "EOGV": "eog"})

    # 5) Run inference
    infer = EEGInfer()
    stages, times = infer.mne_infer(inst=raw)

    df = pd.DataFrame({
        "time_sec": times,
        "stage_numeric": stages
    })

    print(df.head())
    return stages, times

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

test_GSSC(r"L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a")