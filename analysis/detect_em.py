# Filename: detect_em.py 
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: 

# =====================================================================
# Imports
# =====================================================================
import numpy as np
import pandas as pd 
from extract_rems import detect_rem_jaec


# =====================================================================
# Function
# =====================================================================
def detect_em(loc: np.ndarray, roc: np.ndarray, hypno_up: np.ndarray, fs: int=128) -> pd.DataFrame:
    

    #Run dtection to get cleaned signals and events 
    result = detect_rem_jaec(loc, roc, hypno_up, method='ssc_threshold')
    df = result.summary()

    # Recover cleaned signals from REMResults
    loc_clean = result.data_filt[0]
    roc_clean = result.data_filt[1]

    # Define Mean absolute peak amplitude from both channels
    df['MeanAbsValPeak'] = (df['LOCAbsValPeak'] + df['ROCAbsValPeak']) / 2

    # Define thresholds 

    Dur_Thresh_SEM = 0.5 # seconds change if needed 
    Amp_Thresh_SEM = 50.0 # 50 microvolt change if needed 

    # Classify Slow eye movement 

    is_slow_em = (df['Duration'] > Dur_Thresh_SEM) | (df['MeanAbsValPeak'] < Amp_Thresh_SEM)
    df['EM_Type'] = np.where(is_slow_em, 'SEM', 'REM')

    # Remapping of stage integers 
    stage_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    df['Stage'] = df['Stage'].map(stage_map)

    df = df[['Start', 'Peak', 'End', 'Duration',
        'LOCAbsValPeak', 'ROCAbsValPeak', 'MeanAbsValPeak',
        'LOCAbsRiseSlope', 'ROCAbsRiseSlope',
        'LOCAbsFallSlope', 'ROCAbsFallSlope',
        'Stage', 'EM_Type']]
    
    return df.reset_index(drop = True)



# pass the dataframe from detect_em as input to this function
def classify_rem_epochs(df: pd.DataFrame, hypno_int: np.ndarray, fs: int = 128, epoch_len: int = 30, min_rapid: int = 1) -> pd.DataFrame:

    df = df.copy()
    df['EpochIdx'] = (df['Peak'] // epoch_len).astype(int)

    # Count number of REMs in each REM epoch
    rem_epoch_indices = np.where(hypno_int == 4)[0]

    # For each epoch index count the number of REM's
    rem_counts = (
        df[df['EM_Type'] == 'REM']
        .groupby('EpochIdx')
        .size()
        .to_dict()
    )

    # Make a function to classify each 4 second epoch as Phasic or Tonic REM based on the number of REMs in the epoch and the min_rapid threshold
    def epoch_type(row):
        epoch_idx = row['EpochIdx']
        if epoch_idx not in rem_epoch_indices:
            return 'Non-REM'
        n_rapid = rem_counts.get(epoch_idx, 0)
        return 'Phasic' if n_rapid >= min_rapid else 'Tonic'

    df['EpochType'] = df.apply(epoch_type, axis=1)

    return df.reset_index(drop=True)