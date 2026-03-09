import mne 
import yasa
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Path to locate edf file 

edf_path = "C:\\Users\\rasmu\\Desktop\\6. Semester\\Bachelor Projekt\\Test edf filer\\BOGN00022.edf"

# Read edf file
raw = mne.io.read_raw_edf(edf_path,preload = True)

#Print channel names to ensure correct channels are being used
print(raw.ch_names)
print("ny ting ")
print(raw.annotations)

# Auto-stage using YASA
sls = yasa.SleepStaging(raw, eeg_name="C3M2")  
hypno = sls.predict()
hypno_int = yasa.hypno_str_to_int(hypno)

# Upsample hypnogram to match signal length
sf = raw.info["sfreq"]
hypno_up = yasa.hypno_upsample_to_data(hypno_int, sf_hypno=1/30, data=raw.get_data()[0], sf_data=sf)

# Extract loc and roc 
loc = raw.get_data(picks =["E1M2"])[0]
roc = raw.get_data(picks =["E2M2"])[0]

# Trim to multiple of 2^14
factor = 2**14
trim = (len(loc) // factor) * factor
loc = loc[:trim]
roc = roc[:trim]
hypno_up = hypno_up[:trim]

# Extract code and test it

from extract_rems import detect_rem_jaec
result = detect_rem_jaec(loc,roc,hypno_up,method = 'ssc_threshold')
print('summary')
print(result.summary())

df = result.summary()
df['Stage'] = yasa.hypno_int_to_str(df['Stage']).to_numpy()
df.to_csv('rem_results.csv', index=False)