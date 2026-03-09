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
# Paths
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
GSSC_DIR = Path("gssc_csv")
GSSC_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Function
# =====================================================================

# GSSC test function
def GSSC_to_csv(edf_path: str | Path, out_dir: Path = GSSC_DIR) -> None:
    """
    Load one EDF file, run GSSC inference, and save the result as CSV.

    Parameters
    ----------
    edf_path : str | Path
        The path to the EDF file to be loaded.
    out_dir : Path
        The directory where the output CSV file will be saved.
    """
    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    session_id = edf_path.parent.name

    print("Using EDF:", edf_path)

    # 1) Read EDF
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

    print("Loaded raw:", raw)
    print("Channels:", raw.ch_names[:20], "..." if len(raw.ch_names) > 20 else "")
    print("sfreq:", raw.info["sfreq"])

    # 2) Rename EOG channels 
    rename_map = build_rename_map(raw.ch_names)
    print("Rename map:", rename_map)

    if rename_map:
        raw.rename_channels(rename_map)

    # 3) Require LOC and ROC
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

    

    out_path = out_dir / f"{session_id}_gssc.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

