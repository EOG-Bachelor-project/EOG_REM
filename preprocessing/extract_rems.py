import mne 
import numpy as np
import sys
import os
import pandas as pd
from pathlib import Path
from preprocessing.channel_standardization import build_rename_map
from extract_rems import detect_rem_jaec
from gssc.infer import EEGInfer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


Extract_REMs_DIR = Path('extracted_rems') 
Extract_REMs_DIR.mkdir (parents=True, exist_ok=True)

def extract_rems(edf_path: str, out_dir: Path = Extract_REMs_DIR):

    """
    Load one EDF file, rename channels, run GSSC inference, detect REMs, and save the result as CSV.

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

    raw = mne.io.read_raw_edf(edf_path,preload = False )


# Rename channels for standardization
    rename_map = build_rename_map(raw.ch_names)
    print("Rename map:", rename_map)
    if rename_map:
        raw.rename_channels(rename_map)

# Set channel types 
    raw.set_channel_types({'LOC':'eog','ROC':'eog'})

#GSCC staging EOG only
    infer = EEGInfer()
    annotations = infer.mne_infer(raw, eog=['LOC', 'ROC'])

    sleepstage_map = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
    hypno_int = np.array([sleepstage_map[a['description']]for a in annotations])

# Upsample hypnogram to match signal length
    sf = raw.info["sfreq"]
    samples_per_epoch = int(sf*30)
    hypno_up = np.repeat(hypno_int, samples_per_epoch)
# Extract loc and roc 
    loc = raw.get_data(picks =["LOC"])[0]
    roc = raw.get_data(picks =["ROC"])[0]

# Trim or pad to match signal length
    n_samples = loc.shape[0]
    if len(hypno_up) > n_samples:
        hypno_up = hypno_up[:n_samples]
    elif len(hypno_up) < n_samples:
        hypno_up = np.pad(hypno_up, (0, n_samples-len(hypno_up)),mode='edge')

# Trim to multiple of 2^14 for dctwt( used in detect_rem_jaec)
    factor = 2**14
    trim = (len(loc) // factor) * factor
    loc = loc[:trim]
    roc = roc[:trim]
    hypno_up = hypno_up[:trim]

# Rem detection
    result = detect_rem_jaec(loc,roc,hypno_up,method = 'ssc_threshold')

# Create dataframe to retrun results
    df = result.summary()
    df['Stage'] = df['Stage'].map({
        0:'W', 
        1:'N1', 
        2:'N2', 
        3:'N3', 
        4:'REM'})
    out_path = out_dir / f"{session_id}_extracted_rems.csv"
    df.to_csv(out_path, index=False)
    return df

extract_rems("C:\\Users\\rasmu\\Desktop\\6. Semester\\Bachelor Projekt\\Test edf filer\\BOGN00022.edf")