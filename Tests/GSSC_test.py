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
edf = Path(r"C:\Users\rasmu\Desktop\6. Semester\Bachelor Projekt\Test edf filer\cfs-visit5-800331.edf") #--- Change Path back to correct edf
channels = ['EOG-H', 'EOG-V']
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
    picks = ["EOG-H", "EOG-V"]
    missing = [ch for ch in picks if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing expected channels: {missing}. Available: {raw.ch_names}")
    raw.pick(picks)

    # 3) Load data (Required for GSSC)
    raw.load_data()

    # 4) Set channel types (Helps GSSC choose)
    raw.set_channel_types({"EOG-H": "eog", "EOG-V": "eog"})
    
    # 5) Filter manually 
    raw.filter(0.3, 30., picks=['EOG-H', 'EOG-V'])

    # 6) Run inference
    infer = EEGInfer(use_cuda=False)
    stages, times, probs = infer.mne_infer(inst=raw, eeg = [], eog = ['EOG-H','EOG-V'], eog_drop=False, filter=False)
    
    df = pd.DataFrame(data={
        "Stages": stages, 
         "Times": times
         })
    
    df[["P_W", "P_N1", "P_N2", "P_N3", "P_REM"]] = probs
    
    print(df)

    return stages, times, probs

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

test_GSSC(r"C:\Users\rasmu\Desktop\6. Semester\Bachelor Projekt\Test edf filer") # --- Change Path to correct folder --- #