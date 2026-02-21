# TEST_LOAD_EDF.py

from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
import numpy as np
import mne

file_path = "L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_2_a"

def load_files(
        folder_path: str | Path,
        *,
        picks: Optional[Iterable[str]] = None,
        rename_channels: Optional[dict[str, str]] = None,
        set_channel_types: Optional[dict[str, str]] = None,
        verbose: str | bool | None = "ERROR",
        plot_edf: bool = True,
):
    """
    Load EDF, CSV, and txt files from the specified folder path, and return the contents of set files.

    Parameters
    ----------
    folder_path : str | Path
        The path to the folder containing the files to be loaded.
    picks : Optional[Iterable[str]], optional
        An optional list of channel names to load from the EDF file. If None, all channels will be loaded. Default is None.
    rename_channels : Optional[dict[str, str]], optional
        An optional dictionary mapping old channel names to new channel names. If None, no renaming will be performed. Default is None.
    set_channel_types : Optional[dict[str, str]], optional
        An optional dictionary mapping channel names to their types (e.g., 'eeg', 'eog', 'emg'). If None, no channel type setting will be performed. Default is None.
    verbose : str | bool | None, optional
        The verbosity level for logging messages. Can be a string (e.g., "ERROR", "WARNING", "INFO", "DEBUG"), a boolean (True for INFO, False for ERROR), or None (no logging). Default is "ERROR".
    plot_edf: bool
        Whether to plot the EDF data after loading. Default is True.
    """
    folder = Path(folder_path).resolve()
    print(f"\nLoading from: {folder}")

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    # =====================================================
    # Find files in the folder
    # =====================================================
    edf_file = next(folder.glob("*.edf"), None)
    csv_file = next(folder.glob("*.csv"), None)
    txt_file = next(folder.glob("*.txt"), None)
    ### NOTE: 
    #   'next()' is used to get the first file found, or None if no files are found.

    print(f"edf file: {edf_file}")
    print(f"csv file: {csv_file}")
    print(f"txt file: {txt_file}")

    if edf_file is None:
        raise FileNotFoundError(f"No EDF files found in folder: {folder}")
    if csv_file is None:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")
    if txt_file is None:
        raise FileNotFoundError(f"No TXT files found in folder: {folder}")

    # =====================================================
    # Load EDF file (no preloading)
    # =====================================================
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=verbose)
    ### NOTE: 
    #   'mne.io.read_raw_edf()' is used to read the EDF file, and the resulting Raw object is stored in the variable 'raw'.
    #   'preload=False' means that the data will not be loaded into memory immediately.
    #   'verbose=verbose' control the logging level of messages during the loading process.

    if rename_channels:
        raw.rename_channels(rename_channels)
    if set_channel_types:
        raw.set_channel_types(set_channel_types)
    if picks:
        raw.pick(picks)
    ### NOTE:
    #   If 'rename_channels' is provided, the channel names in the Raw object will be renamed according to the provided mapping.
    #   If 'set_channel_types' is provided, the channel types will be set according to the provided mapping.
    #   If 'picks' is provided, only the specified channels will be retained in the Raw object.

    # --- Metadata ---
    chan_types = raw.get_channel_types() # Get the types of channels in the EDF file
    chan = raw.ch_names                  # Get the name of all channels in the EDF file
    fs_used = raw.info['sfreq']          # Get the sampling frequency used in the EDF file
    dur = raw.times[-1]                  # Get the duration of the recording in seconds

    # Extract annotations from the EDF file
    ant_edf = raw.annotations 
    ### NOTE:
    #   Annotations are typically used to mark events or segments of interest in the data, such as sleep stages, artifacts, or other relevant occurrences.


    EDF_results = {
        "channel_types": chan_types,
        "channels": chan,
        "sampling_frequency": fs_used,
        "duration_seconds": dur,
        "annotations": ant_edf
    }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot EDF
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if plot_edf:
        raw_eeg_eog = raw.copy().pick(['eeg', 'eog'])

        if len(raw_eeg_eog.ch_names) == 0:
            print("No EEG/EOG channels found. Plotting all channels.")
            raw.plot(
                duration=30,
                n_channels=min(12, len(raw.ch_names)),
                scalings="auto",
                block=True
            )
        else:
            raw_eeg_eog.plot(
                duration=30,
                n_channels=min(12, len(raw_eeg_eog.ch_names)),
                scalings="auto",
                color=dict(eeg="blue", eog="red"),
                title="EEG (blue) + EOG (red)",
                block=True
            )

    # =====================================================
    # Load CSV file
    # =====================================================
    df = pd.read_csv(csv_file) if csv_file else None

    CSV_results = {
        "dataframe": df.head(),
        "shape": df.shape,
        "info": df.info()
    }

    # =====================================================
    # Load TXT file
    # =====================================================
    txt_lines = None
    if txt_file:
        text_lines = txt_file.read_text().splitlines()
    ### NOTE:
    #   If a TXT file is found, its content will be read as text and split into lines, which are stored in the variable 'text_lines'.

    print("Done.\n")

    return EDF_results, CSV_results, text_lines

r, data, tex = load_files(file_path)
print("EDF file results:\n", r)
#print("CSV file results:\n", data)
#print("TXT file lines:\n", tex)