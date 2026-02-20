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
channels = ['EOGH-A1', 'EOGV-A2']

# ------------------------------------------------------------------------------------------
# FUNCTIONS 
# ------------------------------------------------------------------------------------------

# Channel selection function
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


# GSSC test function
def test_GSSC(folder_path: str | Path, channelse: str | list[str] | None = None,) -> pd.DataFrame:
    """
    Load EDF file from the specified folder path, run the GSSC inference, and return the results as a DataFrame.

    Parameters
    ----------
    folder_path : str | Path
        The path to the folder containing the EDF file to be loaded.
    channels : str | list[str] | None, optional
        The channel(s) to use for inference. Can be a single channel name, a list of channel names, or None to use all channels. Default is None.
        E.g., "EOGH-A1 EOGV-A2" or ["EOGH-A1", "EOGV-A2"].
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the GSSC inference, with columns for the predicted classes and their corresponding probabilities.
    """

    folder = Path(folder_path).resolve()
    print(f"\nTesting GSSC on: {folder}")

    edf_file = next(folder.glob("*.edf"), None)
    if edf_file is None:
        raise FileNotFoundError(f"No EDF file found in folder: {folder}")  
    
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose="ERROR")

    # If channels are specified, pick them from the raw object
    if channelse is not None:
        if isinstance(channelse, str):
            channelse = [channelse]
        raw.pick(channelse)

    # Cant find given channels in the EDF file
    if len(raw.ch_names) == 0:
        raise ValueError(f"No channels found in EDF file: {edf_file} after picking channels: {channelse}")
    
    infer = EEGInfer()
    results = infer.run_inference(raw)
    return results

# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Run the test
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
#print(f"Channels to pick: {channels_to_pick(file_path)}")
print(test_GSSC(file_path, channelse=channels))