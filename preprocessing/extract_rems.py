import mne 
import numpy as np
import sys
import os
import pandas as pd
from channel_standardization import build_rename_map
from extract_rems import detect_rem_jaec
from gssc.infer import EEGInfer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def extract_rems(raw):
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
    df['Stage'] = df['Stage'].map({0:'W', 1:'N1', 2:'N2', 3:'N3', 4:'REM'})

    return df