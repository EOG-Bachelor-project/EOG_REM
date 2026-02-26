# TEST_LOAD_EDF.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
import numpy as np
import mne
import threading
import time
import sys
from pprint import pprint

# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# Predefined variables
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
file_path = "L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_2_a"

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Helper function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# Simple ANSI color codes for terminal output
BOLD = "\033[1m"
RESET = "\033[0m"

# Function to print elapsed time
'''
def time_elapsed(start_time: float, end_time: float, label: str = "Total") -> None:
    """Print elapsed time."""
    elapsed = end_time - start_time
    if elapsed < 60:
        print(f"{label} time: {elapsed:.2f} seconds")
    elif elapsed < 3600:
        print(f"{label} time: {elapsed / 60:.2f} minutes")
    else:
        print(f"{label} time: {elapsed / 3600:.2f} hours")
'''

# Live timer class to show progress during loading
class LiveTimer:
    def __init__(self, label="Loading"):
        self.label = label
        self._running = False
        self._thread = None
        self.start_time = None

    def _run(self):
        while self._running:
            elapsed = time.perf_counter() - self.start_time
            sys.stdout.write(f"\r{self.label}... {elapsed:6.1f} sec")
            sys.stdout.flush()
            time.sleep(1)

    def start(self):
        self.start_time = time.perf_counter()
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()
        elapsed = time.perf_counter() - self.start_time
        print(f"\r{BOLD}{self.label}{RESET} finished in {elapsed:.2f} sec")



# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
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
   
    timer = LiveTimer("Loading files")
    timer.start()

    folder = Path(folder_path).expanduser()
    print(f"\nLoading from: {folder}")

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    # =====================================================
    # 1. Find files in the folder
    # =====================================================
    edf_file = next(folder.glob("*.edf"), None)
    csv_file = next(folder.glob("*.csv"), None)
    txt_file = next(folder.glob("*.txt"), None)
    ### NOTE: 
    #   'next()' is used to get the first file found, or None if no files are found.

    print(f"{BOLD}edf file:{RESET} {edf_file}")
    print(f"{BOLD}csv file:{RESET} {csv_file}")
    print(f"{BOLD}txt file:{RESET} {txt_file}")

    if edf_file is None:
        raise FileNotFoundError(f"No EDF files found in folder: {folder}")
    if csv_file is None:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")
    if txt_file is None:
        raise FileNotFoundError(f"No TXT files found in folder: {folder}")

    # =====================================================
    # 2. Load EDF file (no preloading)
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
    # 3. Load CSV file
    # =====================================================
    df = pd.read_csv(csv_file) if csv_file else None

    CSV_results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": df.head(),  # small preview
        "csv_path": str(csv_file),
    }

    # =====================================================
    # 4. Load TXT file
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

    timer.stop()
    return EDF_results, CSV_results, TXT_results

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
edf_res, csv_res, txt_res = load_files(file_path)

pprint(edf_res)
pprint(csv_res)
pprint(txt_res)