# Filename: detect_em.py 
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: 

# =====================================================================
# Imports
# =====================================================================
import numpy as np
import pandas as pd 
from extract_rems import detect_rem_jaec
from pathlib import Path


# =====================================================================
# Functions
# =====================================================================

# 1 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 1 Detect eye movements and classify them as Rapid Eye Movements (REMs) or Slow Eye Movements (SEMs) based on duration and amplitude thresholds.
# 1 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def detect_em(
        loc:            np.ndarray, 
        roc:            np.ndarray, 
        hypno_up:       np.ndarray,
        Dur_Thresh_SEM: float = 0.5,
        Amp_Thresh_SEM: float = 50.0
        ) -> pd.DataFrame:
    """
    The function takes in the LOC and ROC EOG signals, as well as the hypnogram, and returns
    a DataFrame with detected eye movement events, their characteristics, and classifications.

    Detects eye movements in EOG signals and classifies them as Rapid Eye Movements (REMs) or Slow Eye Movements (SEMs) based on duration and amplitude thresholds. 

    The function uses the `detect_rem_jaec` algorithm to identify eye movement events and then applies criteria to classify them.

    The thresholds for classifying an eye movement as a SEM can be adjusted by changing the `Dur_Thresh_SEM` and `Amp_Thresh_SEM` parameters.

    The function classifies an eye movement as REM if it does not meet the criteria for being a SEM, 
    meaning it has a duration less than or equal to the duration threshold and an amplitude greater than or equal to the amplitude threshold.
    

    Parameters
    ----------
    loc : np.ndarray
        The EOG signal from the left eye (LOC).
    roc : np.ndarray
        The EOG signal from the right eye (ROC).
    hypno_up : np.ndarray
        The hypnogram indicating sleep stages, upsampled to match the EOG signal length.
    Dur_Thresh_SEM : float, optional
        Duration threshold for classifying an eye movement as a Slow Eye Movement (SEM) in seconds, by default **0.5 [s]**.
    Amp_Thresh_SEM : float, optional
        Amplitude threshold for classifying an eye movement as a Slow Eye Movement (SEM) in microvolts, by default **50.0 [μV]**.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing detected eye movement events with the following columns:
        - 'Start': Start time of the eye movement event (in seconds).
        - 'Peak': Time of the peak amplitude of the eye movement event (in seconds).
        - 'End': End time of the eye movement event (in seconds).
        - 'Duration': Duration of the eye movement event (in seconds).
        - 'LOCAbsValPeak': Absolute value of the peak amplitude in the LOC channel (in microvolts).
        - 'ROCAbsValPeak': Absolute value of the peak amplitude in the ROC channel (in microvolts).
        - 'MeanAbsValPeak': Mean absolute value of the peak amplitude across both channels (in microvolts).
        - 'LOCAbsRiseSlope': Absolute value of the rise slope in the LOC channel (in microvolts/second).
        - 'ROCAbsRiseSlope': Absolute value of the rise slope in the ROC channel (in microvolts/second).
        - 'LOCAbsFallSlope': Absolute value of the fall slope in the LOC channel (in microvolts/second).
        - 'ROCAbsFallSlope': Absolute value of the fall slope in the ROC channel (in microvolts/second).
        - 'Stage': Sleep stage during which the eye movement event occurred (W, N1, N2, N3, REM).
        - 'EM_Type': Classification of the eye movement event as 'SEM' (Slow Eye Movement) or 'REM' (Rapid Eye Movement) based on duration and amplitude thresholds.
    """
    # --- Validate inputs ---
    if loc.shape != roc.shape:
        raise ValueError(
            f"LOC and ROC signals must have the same shape." 
            f"Got LOC shape: {loc.shape}, ROC shape: {roc.shape}"
            )
    if loc.shape[0] != hypno_up.shape[0]:
        raise ValueError(
            f"LOC/ROC signals and hypnogram must have the same length." 
            f"Got LOC/ROC length: {loc.shape[0]}, hypnogram length: {hypno_up.shape[0]}"
            )
    if Dur_Thresh_SEM <= 0:
        raise ValueError(f"Duration threshold for SEM must be positive. Got: {Dur_Thresh_SEM}")
    if Amp_Thresh_SEM <= 0:
        raise ValueError(f"Amplitude threshold for SEM must be positive. Got: {Amp_Thresh_SEM}")
    
    # Insure that the input signals are arrays
    if not isinstance(loc, np.ndarray):
        loc = np.array(loc)
    if not isinstance(roc, np.ndarray):
        roc = np.array(roc)
    if not isinstance(hypno_up, np.ndarray):
        hypno_up = np.array(hypno_up)

    # ---- 1) Run dtection to get cleaned signals and events ----
    print("\nRunning REM detection algorithm...")
    result = detect_rem_jaec(loc, roc, hypno_up, method='ssc_threshold')
    df = result.summary()
    print(f"    Detected {len(df)} eye movement events.")

    # 2) Define Mean absolute peak amplitude from both channels
    df['MeanAbsValPeak'] = (df['LOCAbsValPeak'] + df['ROCAbsValPeak']) / 2

    # ---- 3) Define thresholds for SEM ----
    Dur_Thresh_SEM = Dur_Thresh_SEM  # [s]  - change if needed 
    Amp_Thresh_SEM = Amp_Thresh_SEM  # [µV] - change if needed 
    print(f"    Using duration threshold for SEM classification: {Dur_Thresh_SEM} [s]")
    print(f"    Using amplitude threshold for SEM classification: {Amp_Thresh_SEM} [μV]")

    # ---- 4) Classify Slow eye movement ----
    is_slow_em = (df['Duration'] > Dur_Thresh_SEM) | (df['MeanAbsValPeak'] < Amp_Thresh_SEM) 
    df['EM_Type'] = np.where(is_slow_em, 'SEM', 'REM')
    # NOTE: 
    #   `is_slow_em`    - Slow EM if duration is greater than threshold or amplitude is less than threshold.
    #   `df['EM_Type']` - If the EM meets the criteria for being a SEM, it is classified as 'SEM', otherwise it is classified as 'REM'. 
    #                     np.where returns 'SEM' for rows where `is_slow_em` is True and 'REM' for rows where `is_slow_em` is False.
    n_sem = is_slow_em.sum() 
    n_rem = (~is_slow_em).sum()
    print(f"    Classified: {n_rem} REM events | {n_sem} SEM events")

    # ---- 5) Remapping of stage integers ----
    stage_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'} 
    df['Stage'] = df['Stage'].map(stage_map) 

    df = df[[
        'Start', 'Peak', 'End', 'Duration',
        'LOCAbsValPeak', 'ROCAbsValPeak', 'MeanAbsValPeak',
        'LOCAbsRiseSlope', 'ROCAbsRiseSlope',
        'LOCAbsFallSlope', 'ROCAbsFallSlope',
        'Stage', 'EM_Type'
        ]]
    
    df = df.reset_index(drop=True) # Reset index to ensure it starts from 0 and is sequential after filtering and processing.
    return df

# 2 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 2 Classify REM epochs as Phasic or Tonic based on the number of REMs detected in each epoch.
# 2 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# pass the dataframe from detect_em as input to this function
def classify_rem_epochs(
        df:               pd.DataFrame, 
        hypno_int:        np.ndarray, 
        epoch_sec:        float = 4,
        psg_epoch_sec:    float = 30.0,
        min_rapid:        int = 1,
        ) -> pd.DataFrame:
    """
    Classifies REM epochs as Phasic or Tonic by analysing eye movement activity within each ``epoch_sec``-length window.
 
    Each epoch is split into two equal half-windows. Classification criteria:
 
    - **Phasic**: at least 1 qualifying REM candidate in *each* half-window.\\
      A qualifying candidate satisfies:\\
      ``MeanAbsValPeak ≥ amp_thresh_rem`` AND ``Duration < dur_thresh_rem``.\\
    - **Tonic**: zero EMs detected in the epoch AND mean absolute amplitudebelow ``amp_thresh_tonic`` in both half-windows.\\
    - **Unclassified**: EMs present but Phasic criteria not met.\\
    - **Non-REM**: epoch does not fall inside a REM PSG epoch.
 
    - **Phasic REM** is characterized by bursts of rapid eye movements
    - **Tonic REM** is characterized by a relative absence of rapid eye movements.

    Phasic REM epochs are defined as those containing at least `min_rapid` REMs, while Tonic REM epochs contain fewer than `min_rapid` REMs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing detected eye movement events with their characteristics and classifications.
    hypno_int : np.ndarray
        Hypnogram as an array of integers representing sleep stages (0: W, 1: N1, 2: N2, 3: N3, 4: REM).
    epoch_sec : float, optional
        Duration of each analysis epoch in seconds. Default is **4.0 [s]**.
    psg_epoch_sec : float, optional
        Duration of each PSG scoring epoch in seconds, used to determine which analysis epochs fall inside REM sleep. Default is **30.0 [s]**
    min_rapid : int, optional
        Minimum number of REMs required in each half-window for an epoch to be classified as Phasic. Default is **1**.


    Returns
    -------
    pd.DataFrame
        A DataFrame with an additional column 'EpochType' classifying each REM epoch as 'Phasic', 'Tonic', or 'Non-REM' based on the number of REMs detected in each epoch and the sleep stage indicated by the hypnogram.
    """
    # --- Validate inputs ---
    required_cols = {'EM_Type', 'Peak', 'MeanAbsValPeak', 'Duration'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame must contain {missing}. "
            "Run `detect_em` first."
        )
    if not isinstance(hypno_int, np.ndarray):
        hypno_int = np.array(hypno_int)
        print(f"Converted hypnogram to numpy array with shape: {hypno_int.shape}")
    if not isinstance(min_rapid, int) or min_rapid < 1:
        raise ValueError(f"min_rapid must be a positive integer. Got: {min_rapid}")
    if epoch_sec <= 0:
        raise ValueError(f"epoch_sec must be positive, got: {epoch_sec}")
 
    print(f"\nClassifying REM epochs | epoch_sec={epoch_sec} [s] | psg_epoch_sec={psg_epoch_sec} [s] | min_rapid={min_rapid}")

    df = df.copy()      
    half = epoch_sec / 2.0 # Half-window duration for Phasic classification criteria.

    # --- 1) Assign each event to an epoch index ---
    df['EpochIdx'] = (df['Peak'] // epoch_sec).astype(int)  
    # NOTE:
    #   The floor division operator (//) is used to determine which epoch each event belongs to, 
    #   and the result is converted to an integer type for indexing purposes.

    # --- 2) Determine which epoch indices fall inside REM sleep ---
    # hypno_int is one value per 30-second PSG epoch.
    # We convert those to epoch_sec-length indices by scaling.
    psg_epoch_sec = psg_epoch_sec
    rem_epoch_indices = set()
    for psg_idx in np.where(hypno_int == 4)[0]:
        t_start              = psg_idx * psg_epoch_sec
        t_end                = (psg_idx + 1) * psg_epoch_sec
        first_analysis_epoch = int(t_start // epoch_sec)
        last_analysis_epoch  = int((t_end - 1e-9) // epoch_sec)  # - epsilon to avoid boundary overlap
        for idx in range(first_analysis_epoch, last_analysis_epoch + 1):
            rem_epoch_indices.add(idx)

    print(f"    Total PSG epochs in hypnogram: {len(hypno_int)}")
    print(f"    REM PSG epochs: {(hypno_int == 4).sum()}")
    print(f"    Analysis epochs ({epoch_sec} [s]) inside REM: {len(rem_epoch_indices)}")

    # --- 3) Flag qualifying Phasic candidates ---
    # An EM qualifies as a Phasic candidate if it has high amplitude AND short duration
    rem_counts = (
        df[df['EM_Type'] == 'REM']  # Filter to REM events only
        .groupby('EpochIdx')         # Group by epoch index
        .size()                      # Count events per epoch
        .to_dict()                   # Convert to dict for fast lookup
    )

    # --- 4) Classify each epoch ---
    def epoch_type(row):
        epoch_idx = row['EpochIdx']
        if epoch_idx not in rem_epoch_indices:
            return 'Non-REM'
        n_rapid = rem_counts.get(epoch_idx, 0)
        return 'Phasic' if n_rapid >= min_rapid else 'Tonic'
 
    df['EpochType'] = df.apply(epoch_type, axis=1)
 
    # Drop internal helper column
    df = df.drop(columns=['_is_rapid'])

    # --- 5) Summary ---
    counts = df.drop_duplicates('EpochIdx')['EpochType'].value_counts()
    print(f"\nEpoch classification summary (one row per unique {epoch_sec} [s] epoch):\n{counts.to_string()}")

    df = df.reset_index(drop=True)
    return df


# 3 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 3 Classify REM epochs based on paper shared by co-supervisor Umaer. 
# 3 ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def classify_rem_epochs_Umaer(
        df:                 pd.DataFrame,
        loc:                np.ndarray,
        roc:                np.ndarray,
        hypno_int:          np.ndarray,
        sf:                 float,
        epoch_len:          int = 30, 
        sub_epoch_len:      float = 4.0,
        window_len:         float = 2.0,
        min_separation:     float = 8.0,
        amp_thresh_rem :    float = 150.0,
        dur_thresh_rem :    float = 0.5,
        amp_thresh_tonic:   float = 25.0,
        ) -> pd.DataFrame: 
    """
    Classifies 4-second subepochs within REM epochs as Phasic or TOnic based on eye movement activity.

    Phasic sub-epoch are defined as those containing at least one qualifying EM in each of the two adjecent 2 second windows. 
    Tonic  sub-epoch are defined as those containing no EMs at all and a mean absolute below `amp_thresh_tonic` in both 2 second windows.
    To avoid contamination between the two stages segments are only selected if they are at least `min_separation` seconds apart.
    Since this is based on a paper it is not recommended to change parameters without a good reason and understanding of the underlying method. 

    Parameters
    ----------
    df : pd.DataFrame
        Output of detect_em() — must contain `Start`, `Peak`, `End`, `Duration`, `MeanAbsValPeak`, `EM_Type` columns.
    loc : np.ndarray
        Raw LOC signal (same length as used in detect_em).
    roc : np.ndarray
        Raw ROC signal (same length as used in detect_em).
    hypno_int : np.ndarray
        Hypnogram as an array of integers representing sleep stages (0: W, 1: N1, 2: N2, 3: N3, 4: REM).
    sf : float
        Sampling frequency in Hz.
    epoch_len : int, optional
        Length of a scored epoch in seconds, by default **30 [s]**.
    sub_epoch_len : float, optional
        Length of sub-epochs to classify in seconds, by default **4.0 [s]**.
    window_len : float, optional
        Length of each adjacent window used for Phasic/Tonic detection in seconds, by default **2.0 [s]**.
    min_separation : float, optional
        Minimum gap in seconds between a Phasic and a Tonic segment, by default **8.0 [s]**.
    amp_thresh_rem : float, optional
        Minimum mean peak amplitude [µV] for an EM to count toward Phasic detection, by default **150.0 [μV]**.
    dur_thresh_rem : float, optional
        Maximum duration [s] for an EM to count toward Phasic detection, by default **0.5 [s]**.
    amp_thresh_tonic : float, optional
        Maximum mean absolute amplitude [µV] in both 2-second windows for Tonic classification, by default **25.0 [μV]**.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per 4-second sub-epoch inside a REM scored epoch, with the following columns:
        - 'SubEpochStart': Start time of the sub-epoch (in seconds).
        - 'SubEpochEnd'  : End time of the sub-epoch (in seconds).
        - 'EpochIdx'     : Parent 30-second epoch index.
        - 'EpochType'    : Classification of the sub-epoch as 'Phasic', 'Tonic', or 'Unclassified'.
    """
    # --- Validate inputs ---
    required_cols = {"Start", "Peak", "End", "Duration", "MeanAbsValPeak", "EM_Type"}
    missing = required_cols - set(df.columns)
    if missing: 
        raise ValueError(f"Input DataFrame is missing columns: {missing}")
    if loc.shape != roc.shape:
        raise ValueError (f"LOC and ROC must have the same shape. Got LOC: {loc.shape}, ROC: {roc.shape}")
    if sub_epoch_len != 2 * window_len:
        raise ValueError (f"sub_epoch_len({sub_epoch_len} [s]) must be exactly 2 x window_len ({window_len} [s])")
    
    df = df.copy()
    total_samples = len(loc)
    total_duration = total_samples/sf 
    print(f"Total signal duration: {total_duration:.2f} [s] | Total samples: {total_samples} | Sampling frequency: {sf} [Hz]")

     # --- 1) Filter EMs that qualify as rapid for Phasic detection ---
    rapid_mask = (
        (df["MeanAbsValPeak"]>= amp_thresh_rem) &
        (df["Duration"]< dur_thresh_rem)    
    )

    rapid_df = df[rapid_mask].reset_index(drop=True) # Store the rows that meet the criteria for later use
    print(f"EMs qualifying for Phasic detection (amp≥{amp_thresh_rem} [µV], dur<{dur_thresh_rem} [s]): {len(rapid_df)}")

    # --- 2) Build list of all 4 seconds sub-epochs inside REM epochs --- 
    rem_epoch_indices = set(np.where(hypno_int == 4)[0])

    sub_epochs = []
    t=0.0
    while t + sub_epoch_len <= total_duration: # Stops when the last sub-epoch would exceed signal duration
        parent_epoch = int(t // epoch_len) # Determine which 30 second epoch this sub epoch belongs to
        if parent_epoch in rem_epoch_indices: # Only keep sub epochs if they are in REM 
            sub_epochs.append({
                'SubEpochStart': t,
                'SubEpochEnd':   t + sub_epoch_len,
                'EpochIdx':      parent_epoch,
                'EpochType':     'Unclassified', # Will be filled in later based on EM activaity #### min seperation might cause problems here
            })
        t = round(t + sub_epoch_len, 6)

    result_df = pd.DataFrame(sub_epochs)
    if result_df.empty:
        print("No REM sub-epochs found.")
        return result_df
    
    print(f"Total 4 second sub-epochs inside REM: {len(result_df)}")

    # --- 3) classify each sub-epoch ---
    def classify_sub_epoch(row):
        t0, t1 = row["SubEpochStart"], row["SubEpochEnd"]
        w_mid = t0 + window_len # Splits the window in 2 for the phasic detection rule

        # Phasic: at least 1 qualifying epoch in each adjecent 2 second window
        in_w1 = rapid_df[(rapid_df['Peak'] >= t0)    & (rapid_df['Peak'] < w_mid)] # Check for qualifying EM's in fist 2 second window
        in_w2 = rapid_df[(rapid_df['Peak'] >= w_mid) & (rapid_df['Peak'] < t1)] # Check for qualifting EM's in second 2 second window
        if len(in_w1) >= 1 and len(in_w2) >= 1: # Atleast 1 qualifying EM in each window
            return 'Phasic'
        
        # Tonic: 
        # If there are EMs but they do not meet the criteria for being Phasic We classify as unclassified. If no EMs are dtected code moves to next step.
        any_em_in_epoch = df[(df['Peak'] >= t0) & (df['Peak'] < t1)]
        if len(any_em_in_epoch) > 0:
            return 'Unclassified'
        
        def mean_abs_amp(start_s , end_s):
            s0, s1 = int(start_s * sf), int(end_s * sf)
            s0, s1 = max(0, s0), min(total_samples, s1)
            return (np.mean(np.abs(loc[s0:s1])) + np.mean(np.abs(roc[s0:s1]))) / 2
        
        # if zero EMs are detected and the mean absolute amplitude in both windows is below the threshold, classify as Tonic
        if mean_abs_amp(t0, w_mid) < amp_thresh_tonic and mean_abs_amp(w_mid, t1) < amp_thresh_tonic:
            return 'Tonic'

        return 'Unclassified'

    result_df['EpochType'] = result_df.apply(classify_sub_epoch, axis=1)

    # ---- 4) Separation rule: discard segments within 8s of each other ----
    classified = result_df[result_df['EpochType'].isin(['Phasic', 'Tonic'])].copy()
    classified = classified.sort_values('SubEpochStart').reset_index()

    last_kept_end = -np.inf # start at negative infinity to ensure first segment is kept
    for _, row in classified.iterrows():
        if row['SubEpochStart'] >= last_kept_end + min_separation:
            last_kept_end = row['SubEpochEnd']
        else:
            result_df.loc[row['index'], 'EpochType'] = 'Unclassified' # Mark as unclassified if it is too close to the previous one

    # ---- 5) Summary ----
    counts = result_df['EpochType'].value_counts()
    print(f"Sub-epoch classification summary:\n{counts.to_string()}")

    result_df = result_df.reset_index(drop=True)
    print(f" result_df shape: {result_df.shape} |"
          f" Total sub-epochs: {len(result_df)} |"
          f" Phasic: {counts.get('Phasic', 0)} |" 
          f" Tonic: {counts.get('Tonic', 0)} |"
          f" Unclassified: {counts.get('Unclassified', 0)}"
        )
    return result_df
