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
        Duration threshold for classifying an eye movement as a Slow Eye Movement (SEM) in seconds, by default 0.5 seconds.
    Amp_Thresh_SEM : float, optional
        Amplitude threshold for classifying an eye movement as a Slow Eye Movement (SEM) in microvolts, by default 50.0 microvolts.
    
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
        raise ValueError(f"LOC and ROC signals must have the same shape. Got LOC shape: {loc.shape}, ROC shape: {roc.shape}")
    if loc.shape[0] != hypno_up.shape[0]:
        raise ValueError(f"LOC/ROC signals and hypnogram must have the same length. Got LOC/ROC length: {loc.shape[0]}, hypnogram length: {hypno_up.shape[0]}")
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
    print("Running REM detection algorithm...")
    result = detect_rem_jaec(loc, roc, hypno_up, method='ssc_threshold')
    df = result.summary()

    # ---- 2) Recover cleaned signals from REMResults ----
    loc_clean = result.data_filt[0]
    roc_clean = result.data_filt[1]
    print(f"Original LOC signal shape: {loc.shape}   |   Original ROC signal shape: {roc.shape}")
    print(f"Cleaned LOC signal shape: {loc_clean.shape}   |   Cleaned ROC signal shape: {roc_clean.shape}")

    # 3) Define Mean absolute peak amplitude from both channels
    df['MeanAbsValPeak'] = (df['LOCAbsValPeak'] + df['ROCAbsValPeak']) / 2

    # ---- 4) Define thresholds for SEM ----
    Dur_Thresh_SEM = Dur_Thresh_SEM  # [seconds] - change if needed 
    Amp_Thresh_SEM = Amp_Thresh_SEM  # [microvolts] - change if needed 
    print(f"Using duration threshold for SEM classification: {Dur_Thresh_SEM} s")
    print(f"Using amplitude threshold for SEM classification: {Amp_Thresh_SEM} μV")

    # ---- 5) Classify Slow eye movement ----
    is_slow_em = (df['Duration'] > Dur_Thresh_SEM) | (df['MeanAbsValPeak'] < Amp_Thresh_SEM) 
    df['EM_Type'] = np.where(is_slow_em, 'SEM', 'REM')
    # NOTE: 
    #   `is_slow_em`    - Slow EM if duration is greater than threshold or amplitude is less than threshold.
    #   `df['EM_Type']` - If the EM meets the criteria for being a SEM, it is classified as 'SEM', otherwise it is classified as 'REM'. 
    #                     np.where returns 'SEM' for rows where `is_slow_em` is True and 'REM' for rows where `is_slow_em` is False.

    # ---- 6) Remapping of stage integers ----
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

# pass the dataframe from detect_em as input to this function
def classify_rem_epochs(df:         pd.DataFrame, 
                        hypno_int:  np.ndarray, 
                        epoch_len:  int = 30,
                        min_rapid:  int = 1,
                        ) -> pd.DataFrame:
    """
    Classifies REM epochs as Phasic or Tonic based on the number of Rapid Eye Movements (REMs) detected in each epoch.

    - **Phasic REM** is characterized by bursts of rapid eye movements
    - **Tonic REM** is characterized by a relative absence of rapid eye movements.

    Phasic REM epochs are defined as those containing at least `min_rapid` REMs, while Tonic REM epochs contain fewer than `min_rapid` REMs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing detected eye movement events with their characteristics and classifications.
    hypno_int : np.ndarray
        Hypnogram as an array of integers representing sleep stages (0: W, 1: N1, 2: N2, 3: N3, 4: REM).
    min_rapid : int, optional
        Minimum number of REMs required in an epoch to classify it as Phasic REM, by default 1.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with an additional column 'EpochType' classifying each REM epoch as 'Phasic', 'Tonic', or 'Non-REM' based on the number of REMs detected in each epoch and the sleep stage indicated by the hypnogram.
    """

    # --- Validate inputs ---
    # DataFrame must contain 'EM_Type' column with classifications of eye movements
    if 'EM_Type' not in df.columns:
        raise ValueError("Input DataFrame must contain 'EM_Type' column with classifications of eye movements.")
    # Hypnogram must be a 1D array of integers representing sleep stages
    if not isinstance(hypno_int, np.ndarray):
        raise ValueError(f"Hypnogram must be a numpy array, but got type: {type(hypno_int)}")
    # Ensure that epoch_len and min_rapid are integers
    for i in [min_rapid, epoch_len]:
        if not isinstance(i, int):
            raise ValueError(f"{i} must be an integer, but got type: {type(i)}")
    # Hypnogram values must be positive integers
    if epoch_len <= 0:
        raise ValueError(f"epoch_len must be a positive integer, but got: {epoch_len}")
    # Minimum number of rapid eye movements must be a non-negative integer
    if min_rapid < 0:
        raise ValueError(f"min_rapid must be a non-negative integer, but got: {min_rapid}")
    
    print(f"Classifying REM epochs with epoch length: {epoch_len} [s] and minimum REM count for Phasic classification: {min_rapid}")
    
    df = df.copy()                                          # Ensure that the input DataFrame is a copy to avoid modifying the original DataFrame outside of this function
    df['EpochIdx'] = (df['Peak'] // epoch_len).astype(int)  # Calculate the epoch index for each eye movement event based on its peak time and the specified epoch length. 
    # NOTE:
    #   The floor division operator (//) is used to determine which epoch each event belongs to, 
    #   and the result is converted to an integer type for indexing purposes.

    # ---- 1) Count number of REMs in each REM epoch ----
    # Get indices of epochs that are classified as REM in the hypnogram
    rem_epoch_indices = np.where(hypno_int == 4)[0]
    print(f"Total number of 4-second epochs in hypnogram: {len(hypno_int)}")
    
    # For each epoch index count the number of REM's
    rem_counts = (
        df[df['EM_Type'] == 'REM'] # Filter the DataFrame to include only rows where 'EM_Type' is 'REM'
        .groupby('EpochIdx')       # Group the filtered DataFrame by the 'EpochIdx' column, which contains the index of the epoch to which each eye movement event belongs.
        .size()                    # Count the number of REM events in each epoch group, returning a Series with 'EpochIdx' as the index and the count of REM events in that epoch as the value.
        .to_dict()                 # Convert the resulting Series to a dictionary where the keys are epoch indices and the values are the counts of REM events in those epochs.
    )

    # ---- 2) Helper Function to Classify each 4-second epoch as Phasic or Tonic REM ----
    def epoch_type(row):
        """
        Classifies each epoch as 'Phasic', 'Tonic', or 'Non-REM' 
        based on the number of REMs detected in that epoch and 
        whether the epoch is classified as REM in the hypnogram.
        """
        epoch_idx = row['EpochIdx'] 
        if epoch_idx not in rem_epoch_indices:
            return 'Non-REM'
        n_rapid = rem_counts.get(epoch_idx, 0)
        return 'Phasic' if n_rapid >= min_rapid else 'Tonic'
    
    # Apply `epoch_type` function to each row of the DataFrame to classify each epoch as 'Phasic', 'Tonic', or 'Non-REM'.
    df['EpochType'] = df.apply(epoch_type, axis=1)
    print(f"Classification of REM epochs completed. Sample of classified epochs:\n{df[['EpochIdx', 'EpochType']].drop_duplicates().head()}")

    # Reset index to ensure it starts from 0 and is sequential after adding new columns and classifications.
    df = df.reset_index(drop=True)

    return df