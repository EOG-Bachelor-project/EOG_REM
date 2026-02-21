# TEST_LOAD_EDF.py

from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
import numpy as np
import mne
from pprint import pprint

file_path = "L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_2_a"


def load_files(
        folder_path: str | Path,
        *,
        picks: Optional[Iterable[str]] = None,
        rename_channels: Optional[dict[str, str]] = None,
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
    # 1. Load EDF file (no preloading)
    # =====================================================
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=verbose)
    ### NOTE: 
    #   'mne.io.read_raw_edf()' is used to read the EDF file, and the resulting Raw object is stored in the variable 'raw'.
    #   'preload=False' means that the data will not be loaded into memory immediately.
    #   'verbose=verbose' control the logging level of messages during the loading process.

    if rename_channels:
        raw.rename_channels(rename_channels)  # Rename channels according to the provided mapping
    if picks:
        raw.pick(picks)                       # Retain only the specified channels in the Raw object
    raw.set_channel_types({
        "EOGH-A1": "eog",
        "EOGV-A2": "eog",
        "EKG": "ecg",
        "CHIN": "emg",
        "Nasal": "resp",
        "Thorax": "resp",
        "Abdomen": "resp",
        "SNORE": "misc",
        "SpO2": "misc",
        "Pulse": "misc",
        "IBI": "misc",
        "TIBH": "emg",
        "TIBV": "emg",
        })

    # --- Metadata ---
    chan_types = raw.get_channel_types() # Get the types of channels in the EDF file
    chan = raw.ch_names                  # Get the name of all channels in the EDF file
    fs_used = raw.info['sfreq']          # Get the sampling frequency used in the EDF file
    dur = raw.times[-1]                  # Get the duration of the recording in seconds
    ant_edf = raw.annotations            # EDF+ annotations, if present, are stored in the 'annotations' attribute of the Raw object.
    ### NOTE:
    #   Annotations are typically used to mark events or segments of interest in the data, such as sleep stages, artifacts, or other relevant occurrences.

    EDF_results = {
        "channels": chan,
        "channel_types": chan_types,
        "sampling_frequency_hz": fs_used,
        "duration_seconds": dur,
        "n_annotations": len(ant_edf),
        "annotations": ant_edf,  # keep the object if you want
        "edf_path": str(edf_file),
    }

    # =====================================================
    # 2. Load CSV file
    # =====================================================
    df = pd.read_csv(csv_file) if csv_file else None

    CSV_results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": df.head(),  # small preview
        "csv_path": str(csv_file),
    }

    # =====================================================
    # 3. Load TXT file
    # =====================================================
    txt_lines = None
    if txt_file:
        text_lines = txt_file.read_text().splitlines()
    ### NOTE:
    #   If a TXT file is found, its content will be read as text and split into lines, which are stored in the variable 'text_lines'.

    TXT_results = {
        "txt_path": str(txt_file),
        "lines": txt_lines,  # full list
    }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collect results to return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # --- EDF results ---
    print("\nEDF SUMMARY")
    print("=" * 50)

    print(f"{'Number of Channels':<28} : {len(raw.ch_names)}")
    print(f"{'Sampling Frequency (Hz)':<28} : {raw.info['sfreq']}")
    print(f"{'Duration (min)':<28} : {round(raw.times[-1] / 60, 2)}")
    print(f"{'Number of Annotations':<28} : {len(raw.annotations)}")

    print("\nCHANNEL LIST")
    print("-" * 50)

    for i, (ch, ch_type) in enumerate(
        zip(raw.ch_names, raw.get_channel_types()), start=1
    ):
        print(f"{i:>3}. {ch:<25} ({ch_type})")

    print("=" * 50)

    # --- CSV results ---
    print("\nCSV SUMMARY")
    print("=" * 50)

    print(f"{'Rows':<20} : {df.shape[0]}")
    print(f"{'Columns':<20} : {df.shape[1]}")

    print("\nCOLUMN NAMES")
    print("-" * 50)

    for i, col in enumerate(df.columns, start=1):
        print(f"{i:>3}. {col}")

    print("=" * 50)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot EDF
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if plot_edf:
        raw_plot = raw.copy().pick(["eeg", "eog", "ecg", "emg", "resp", "misc"])

        raw_plot.plot(
            duration=30,
            n_channels=min(20, len(raw_plot.ch_names)),
            scalings="auto",
            color=dict(
                eeg="blue",
                eog="red",
                ecg="purple",
                emg="green",
                resp="orange",
                misc="black",
            ),
            title="PSG (colored by channel type)",
            block=True,
            )


    return EDF_results, CSV_results, TXT_results

# --- Test ---
edf_res, csv_res, txt_res = load_files(file_path)

pprint(edf_res)
pprint(csv_res)
pprint(txt_res)