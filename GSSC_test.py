# GSSC_test.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations

from pathlib import Path
import mne
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

    # Load EDF signal via MNE
    raw = mne.io.read_raw_edf(edf_path, preload=False)
    
    print("Loaded raw:", raw)
    print("Channels:", raw.ch_names[:20], "..." if len(raw.ch_names) > 20 else "")
    print("sfreq:", raw.info["sfreq"])
    raw.set_channel_types({"EOGH-A1": "eog", "EOGV-A2": "eog",})

    # Make inferencer
    infer = EEGInfer()

    # Run inferencer
    hypnogram = infer.mne_infer(inst=raw)
    print(hypnogram)

    return hypnogram

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

test_GSSC(r"L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a")