# Filename: REM_classification_test.py 
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Classify REM sleep stages based on eye movement data and classification from:
# Rosenblum Y, Bogdány T, Nádasy LB, Chen X, Kovács I, Gombos F, Ujma P, Bódizs R, Adelhöfer N, Simor P, Dresler M. Aperiodic neural activity distinguishes between phasic and tonic REM sleep. J Sleep Res. 2025 Aug;34(4):e14439. doi: 10.1111/jsr.14439. Epub 2024 Dec 26. PMID: 39724862; PMCID: PMC12215217. 

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations
import pandas as pd 
import numpy as np
#from preprocessing.extract_rems_n import extract_rems_from_edf

df = pd.read_csv(r'C:/Users/AKLO0022/EOG_REM/extracted_rems/DCSM_1_a_extracted_rems.csv')

def classify_REM(df: pd.DataFrame, epoch_duration: float = 4.0)-> pd.DataFrame:

    """
    Classifies 4-second REM epochs as Phasic or Tonic based on eye movement activity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame output from detect_rem_jaec() containing the following columns:
        - Start           : float, start time of the eye movement in seconds
        - Peak            : float, ...
        - End             : float, end time of the eye movement in seconds
        - Duration        : float, duration of the eye movement in seconds
        - LOCAbsValPeak   : float, absolute LOC amplitude at peak in volts
        - ROCAbsValPeak   : float, absolute ROC amplitude at peak in volts
        - LOCAbsRiseSlope : float, absolute LOC rise slope
        - ROCAbsRiseSlope : float, absolute ROC rise slope
        - LOCAbsFallSlope : float, absolute LOC fall slope
        - ROCAbsFallSlope : float, absolute ROC fall slope
        - Stage           : str, sleep stage — only rows with 'REM' are processed

    epoch_duration : float, optional
        Duration of each epoch in seconds. Default is 4.0.

    Classification criteria
    -----------------------
    Phasic:
        At least 1 phasic EM candidate in each of two adjacent 2-second windows.
        A phasic EM candidate must satisfy:
        - LOCAbsValPeak or ROCAbsValPeak > 150 µV (150e-6 V)
        - Duration < 0.5 s

    Tonic:
        No EMs present in the epoch, or all EMs have:
        - LOCAbsValPeak < 25 µV (25e-6 V)
        - ROCAbsValPeak < 25 µV (25e-6 V)

    Epochs must be at least 8 seconds apart to avoid contamination between states.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per kept epoch containing:
        - EpochStart : float, start time of the epoch in seconds
        - EpochEnd   : float, end time of the epoch in seconds
        - REM_Type   : str, 'Phasic' or 'Tonic'
        - EM_count   : int, number of eye movements detected in the epoch
    """

    if 'Stage' not in df.columns:
        raise ValueError ("DataFrame must contain a 'Stage' column")
    
    # Get REM stages only
    df = df[df['Stage'] == 'REM'].copy()

    # Define phasic REM
    df['is_phasic_REM'] = (
        (df['ROCAbsValPeak'] > 150e-6) | (df['LOCAbsValPeak'] > 150e-6)
    ) & (df['Duration'] > 0.5)

    # Build mini epochs of 4 seconds
    epoch_starts = np.arange(df['Start'].min(), df['Start'].max(), epoch_duration)

    # Create empty list to store results from for loop 
    results = []
    # Create for loop to classify each epoch
    for epoch_start in epoch_starts:
        epoch_end = epoch_start + epoch_duration
        #split the epoch into two 2 second windows
        mid = epoch_start + epoch_duration / 2  

        # Get EMs in this epoch
        epoch_ems = df[(df['Start'] >= epoch_start) & (df['Start'] < epoch_end)]

        # Split into two 2-second windows
        window1 = epoch_ems[epoch_ems['Start'] <  mid] # We use < mid and not <= mid so the EMs at the exact midpoints do not overlap between the two windows.
        window2 = epoch_ems[epoch_ems['Start'] >= mid]

        # Phasic: at least 1 phasic EM candidate in each adjacent 2-sec window
        phasic_in_w1 = window1['is_phasic_REM'].any()
        phasic_in_w2 = window2['is_phasic_REM'].any()
        is_phasic = phasic_in_w1 and phasic_in_w2

        # Tonic: no EMs at all, and all amplitudes below 25 µV
        no_ems = len(epoch_ems) == 0
        low_amp = (
            (epoch_ems['ROCAbsValPeak'] < 25 * 1e6) & 
            (epoch_ems['LOCAbsValPeak'] < 25 * 1e6)
        ).all() if len(epoch_ems) > 0 else True
        is_tonic = no_ems or low_amp

    results.append({
    'EpochStart': round(epoch_start, 4),
    'EpochEnd':   round(epoch_end, 4),
    'REM_Type':   'Phasic' if is_phasic else 'Tonic' if is_tonic else 'Unclassified',
    'EM_count':   len(epoch_ems),
    })
    # Convert the ruslts to a DataFrame 
    epochs_df = pd.DataFrame(results)

    # Remove epochs within 8 seconds of each other to avoid contamination between states
    # Keep only epochs where the previous kept epoch was >= 8 sec away
    kept = []
    last_kept_start = -np.inf

    for _, row in epochs_df.iterrows(): 
        if row['EpochStart'] - last_kept_start >= 8.0:
            kept.append(row)
            last_kept_start = row['EpochStart']

    return pd.DataFrame(kept).reset_index(drop=True)

# test it 
result = classify_REM(df)
print(result)

# 4 seocond epochs amplitude above 150 mV and shorther than 500 ms # build in define phasic REM 
# 4 second epoch was categorized as phais if at least 2 consecutive EMs were detected in adjecent 2 second time window. # 
# Segments were scored as tonic if no EMs occured amplitude below 25mV in adjecent 2 second time winbdow.
# To avoid contamination between states segments were only selected if they were at least 8 sec apart from eachother. 