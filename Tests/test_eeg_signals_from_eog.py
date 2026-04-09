# Filename: test_eeg_signals_from_eog.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: File to test extracting eeg signals from EM_detect by subtracting mask applied in extract_rems.py provided by our main supervisor Andreas Brink-Kjaer.

# =====================================================================
# Imports
# =====================================================================
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from extract_rems import detect_rem_jaec 
import numpy as np
import dtcwt



def eeg_signals_from_eog(loc: np.ndarray, roc:np.ndarray, hypno_up: np.ndarray, method: str = 'subtract')-> tuple[np.ndarray, np.ndarray]:

    """
    Extract EEG signals by removing EOG activity from LOC and ROC signals.

    Parameters
    ----------
    loc : np.ndarray
        Raw LOC signal in µV.
    roc : np.ndarray
        Raw ROC signal in µV.
    hypno_up : np.ndarray
        Hypnogram upsampled to the signal sampling rate.
    method : Literal['subtract', 'mask'], optional
        Method used to extract EEG signal.
        - 'subtract' : Subtracts the EOG-filtered signal from the raw signal.
        - 'mask'     : Reconstructs signal using the inverse DTCWT gain mask.
        Default is 'subtract'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        loc_eeg : EEG-like signal extracted from LOC in µV.
        roc_eeg : EEG-like signal extracted from ROC in µV.
    """


    # --- Validate inputs --- 

    if not isinstance (loc, np.ndarray) or not isinstance (roc, np.ndarray): 
        raise TypeError ("loc and roc must me numpy arrays.")
    if loc.shape != roc.shape:
        raise ValueError (f"loc and roc must be the same shape. loc shape:{loc.shape}\\roc shape: {roc.shape}")
    if method not in ('subtract', 'mask'):
        raise ValueError (f"method must be either 'mask' or 'subtract', got {method}")

    # --- Extract EEG singals ---

    dtcwt_transform = dtcwt.Transform1d(biort='near_sym_b', qshift='qshift_b')


    if method == 'subtract':

        result = detect_rem_jaec(loc, roc, hypno_up, method = 'ssc_threshold')

        loc_clean = result.data_filt[0]
        roc_clean = result.data_filt[1]

        loc_eeg = loc - loc_clean
        roc_eeg = roc - roc_clean

    elif method == 'mask': 
        
        inverted_mask = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] # Inverse of mask in `extract_rems.py`
        
        loc_dtcwt = dtcwt_transform.forward(loc, nlevels=14)
        roc_dtcwt = dtcwt_transform.forward(roc, nlevels=14)

        loc_eeg = dtcwt_transform.inverse(loc_dtcwt, inverted_mask)
        roc_eeg = dtcwt_transform.inverse(roc_dtcwt, inverted_mask)

    # --- Amplitude characteristic of the EEG signals ---
    print(f"LOC EEG — min: {loc_eeg.min():.2f}, max: {loc_eeg.max():.2f}, mean: {loc_eeg.mean():.2f}")
    print(f"ROC EEG — min: {roc_eeg.min():.2f}, max: {roc_eeg.max():.2f}, mean: {roc_eeg.mean():.2f}")

    return loc_eeg, roc_eeg 