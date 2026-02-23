# GSSC_test.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import mne

from gssc.infer import EEGInfer

# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Predefined variables
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠

file_path = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a")
edf_file = next(file_path.glob("*.edf"))
channels = ['EOGH-A1', 'EOGV-A2']

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# GSSC test function
def test_GSSC(folder_path: str):
    """
    Load EDF file from the specified folder path, run the GSSC inference, and return the results as a DataFrame.

    Parameters
    ----------
    folder_path : str | Path
        The path to the folder containing the EDF file to be loaded.
    """

    # Load EDF signal via MNE
    raw = mne.io.read_raw_edf(folder_path, preload=False)

    # Make inferencer
    infer = EEGInfer()

    # Run inferencer
    hypnogram = infer.mne_infer(raw)
    print(hypnogram)

    return hypnogram

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

test_GSSC(Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_1_a\contiguous.edf"))