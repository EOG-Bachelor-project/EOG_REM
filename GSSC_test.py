# GSSC_test.py

# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Imports
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import mne

from gssc.infer import EEGInfer

# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Predefined variables
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠

file_path = "L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_2_a"

# ------------------------------------------------------------------------------------------

def channels_to_pick(path: str | Path) -> list[str]:
    """
    Get the list of channel names to pick from the EDF file based on the folder name.

    Parameters
    ----------
    path : str | Path
        The path to the folder containing the EDF file.

    Returns
    -------
    list[str]
        A list of channel names to pick from the EDF file.
    """
    folder = Path(path).resolve()

    edf_file = next(folder.glob("*.edf"), None)
    if edf_file is None:
        raise FileNotFoundError(f"No EDF file found in folder: {folder}")
    
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose="ERROR")
    return raw.ch_names

def test_GSSC(
        folder_path: str | Path,
        channelse: str | list[str] | None = None,
        )-> pd.DataFrame:
    """
    Load EDF file from the specified folder path, run the GSSC inference, and return the results as a DataFrame.

    Parameters
    ----------
    folder_path : str | Path
        The path to the folder containing the EDF file to be loaded.
    channels : str | list[str] | None, optional
        The channel(s) to use for inference. Can be a single channel name, a list of channel names, or None to use all channels. Default is None.
    """
    folder = Path(folder_path).resolve()
    print(f"\nTesting GSSC on: {folder}")

    edf_file = next(folder.glob("*.edf"), None)
    if edf_file is None:
        raise FileNotFoundError(f"No EDF file found in folder: {folder}")  
    
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose="ERROR")

    if channelse is not None:
        if isinstance(channelse, str):
            channelse = [channelse]
        raw.pick(channelse)
    infer = EEGInfer()
    results = infer.run_inference(raw)
    return results
# ------------------------------------------------------------------------------------------

print(f"Channels to pick: {channels_to_pick(file_path)}")
print(test_GSSC(file_path, channelse="EOGH-A1 EOGV-A2"))






