# GSSC_test.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations

from pathlib import Path
import mne
import torch
import gssc.networks
from gssc.infer import EEGInfer

torch.serialization.add_safe_globals([gssc.networks.ResSleep])


# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Predefined variables
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
edf = Path("l:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a/contiguous.edf")
print("EDF Path:", edf)
print("Exists:", edf.exists())

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# GSSC test function
def test_GSSC(edf_path: str | Path):
    """
    Load EDF file from the specified folder path, run the GSSC inference, and return the results as a DataFrame.

    Parameters
    ----------
    folder_path : str | Path
        The path to the folder containing the EDF file to be loaded.
    """

    # Load EDF signal via MNE
    raw = mne.io.read_raw_edf(edf_path, preload=False)

    # Make inferencer
    infer = EEGInfer()

    # Run inferencer
    hypnogram = infer.mne_infer(inst=raw)
    print(hypnogram)

    return hypnogram

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

test_GSSC(Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a/contiguous.edf"))