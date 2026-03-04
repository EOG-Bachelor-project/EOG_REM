import os
import sys
import numpy as np
import pandas as pd
import yasa
import dtcwt
from scipy.ndimage import minimum_filter1d, maximum_filter1d

def detect_rem_jaec(loc, roc, hypno_up, method='original'):
    
    # Fixed threshold
    fs = 128

    # Thresholds
    T_angle = 0.9 * np.pi
    T_amp = 600
    P_th = 92
    dur_hole = 2.0 * fs
    dur_em = 2.5 * fs
    
    # Initialize transform
    dtcwt_transform = dtcwt.Transform1d(biort='near_sym_b', qshift='qshift_b')

    # DTCWT
    loc_dtcwt = dtcwt_transform.forward(loc, nlevels=14)
    roc_dtcwt = dtcwt_transform.forward(roc, nlevels=14)

    # Difference angle (arctan)
    diff_angle = [np.angle(x) - np.angle(y) for x, y in zip(roc_dtcwt.highpasses, loc_dtcwt.highpasses)]
    diff_angle = [np.mod(a + np.pi, 2 * np.pi) - np.pi for a in diff_angle]
    diff_angle = [np.min(np.concatenate([np.abs(a), 2*np.pi - np.abs(a)], 1), 1) for a in diff_angle]
    angle_mask = [np.expand_dims(1.0*(a > T_angle), 1) for a in diff_angle]

    # Set sub-threshold values to zero
    loc_dtcwt_angle_corrected = []
    roc_dtcwt_angle_corrected = []
    for i in range(len(angle_mask)):
        loc_dtcwt_angle_corrected.append(loc_dtcwt.highpasses[i] * angle_mask[i])
        roc_dtcwt_angle_corrected.append(roc_dtcwt.highpasses[i] * angle_mask[i])
    loc_dtcwt.highpasses = tuple(loc_dtcwt_angle_corrected)
    roc_dtcwt.highpasses = tuple(roc_dtcwt_angle_corrected)

    # Initialain mask
    gain_mask = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

    # Clean signal
    loc_clean = dtcwt_transform.inverse(loc_dtcwt, gain_mask)
    roc_clean = dtcwt_transform.inverse(roc_dtcwt, gain_mask)

    # Difference signal
    A_diff = np.abs(roc_clean - loc_clean)

    # Threshold (original extremely sensitive to sleep stage distribution and fixes positives at 8 %)
    if method == 'original':
        T_pth = np.percentile(A_diff[A_diff < T_amp], P_th)
    elif method == 'ssc_threshold':
        if not np.isscalar(hypno_up):
            T_pth_loc = np.mean([np.percentile(np.abs(loc[hypno_up == ssc]), P_th) for ssc in np.unique(hypno_up)])
            T_pth_roc = np.mean([np.percentile(np.abs(roc[hypno_up == ssc]), P_th) for ssc in np.unique(hypno_up)])
            T_pth = np.mean([T_pth_loc, T_pth_roc])
        else:
            T_pth = np.mean([np.percentile(np.abs(loc), P_th), np.percentile(np.abs(roc), P_th)])
    else:
        raise Exception("method ['original', 'ssc_threshold'].")

    # EM Candidates
    if method == 'original':
        EM_cand = np.logical_and(A_diff < T_amp, A_diff > T_pth)
        EM_cand_combined = maximum_filter1d(EM_cand, dur_hole)
        EM = minimum_filter1d(EM_cand_combined, dur_em)
        return EM

    elif method == 'ssc_threshold':

        # Threshold A diff
        EM_cand = np.logical_and(A_diff < T_amp, A_diff > T_pth)

        # Identify the start and end of each True interval in X
        dEM_cand = np.diff(EM_cand.astype(int))
        starts = np.where(dEM_cand == 1)[0] + 1
        ends = np.where(dEM_cand == -1)[0] + 1
        
        # If X starts with a True, we need to account for the initial interval
        if EM_cand[0]:
            starts = np.insert(starts, 0, 0)
        # If X ends with a True, we need to account for the final interval
        if EM_cand[-1]:
            ends = np.append(ends, len(EM_cand))
        
        pks = []
        for start, end in zip(starts, ends):
            peak_index = start + np.argmax(A_diff[start:end])
            pks.append(peak_index)
        pks = np.array(pks)

        # Initialize params
        pks_params = {}
        pks_params['left_bases'] = []
        pks_params['right_bases'] = []

        # Find local minimas in A_diff that are less than 5 to select start and end
        minima = (np.diff(np.sign(np.diff(A_diff))) > 0).nonzero()[0] + 1
        minima = np.insert(minima, 0, 0)
        minima = np.insert(minima, -1, len(A_diff) - 1)
        minima = minima[A_diff[minima] < (T_pth/2)]

        for peak in pks:
            # Find the closest local minimum before the peak
            before_minima = minima[minima < peak]
            if len(before_minima) > 0:
                closest_before_minimum = before_minima[-1]
            else:
                closest_before_minimum = np.max([peak - 1, 0])
            pks_params['left_bases'].append(closest_before_minimum)
            
            # Find the closest local minimum after the peak
            after_minima = minima[minima > peak]
            if len(after_minima) > 0:
                closest_after_minimum = after_minima[0] - 1
            else:
                closest_after_minimum = np.min([peak + 1, len(A_diff) - 1])
            pks_params['right_bases'].append(closest_after_minimum)

        pks_params['left_bases']  = np.array(pks_params['left_bases'])
        pks_params['right_bases']  = np.array(pks_params['right_bases'])

        # Stage
        if not np.isscalar(hypno_up):
            # The sleep stage at the beginning of the REM is considered.
            rem_sta = hypno_up[pks_params["left_bases"]]
        else:
            rem_sta = np.zeros(pks.shape)

        # Follows YASA (https://raphaelvallat.com/yasa/build/html/_modules/yasa/detection.html#rem_detect)
        # Timing
        pks_params["Start"] = pks_params["left_bases"] / fs
        pks_params["Peak"] = pks / fs
        pks_params["End"] = pks_params["right_bases"] / fs
        pks_params["Duration"] = pks_params["End"] - pks_params["Start"]

        # Absolute LOC / ROC value at peak (filtered)
        pks_params["LOCAbsValPeak"] = abs(loc_clean[pks])
        pks_params["ROCAbsValPeak"] = abs(roc_clean[pks])

        # Absolute rising and falling slope
        dist_pk_left = (pks - pks_params["left_bases"]) / fs
        dist_pk_right = (pks_params["right_bases"] - pks) / fs
        locrs = (loc_clean[pks] - loc_clean[pks_params["left_bases"]]) / dist_pk_left
        rocrs = (roc_clean[pks] - roc_clean[pks_params["left_bases"]]) / dist_pk_left
        locfs = (loc_clean[pks_params["right_bases"]] - loc_clean[pks]) / dist_pk_right
        rocfs = (roc_clean[pks_params["right_bases"]] - roc_clean[pks]) / dist_pk_right
        pks_params["LOCAbsRiseSlope"] = abs(locrs)
        pks_params["ROCAbsRiseSlope"] = abs(rocrs)
        pks_params["LOCAbsFallSlope"] = abs(locfs)
        pks_params["ROCAbsFallSlope"] = abs(rocfs)
        pks_params["Stage"] = rem_sta  # Sleep stage

        # Convert to Pandas DataFrame
        df = pd.DataFrame(pks_params)

        # Keep only useful channels
        df = df[
            [
                "Start",
                "Peak",
                "End",
                "Duration",
                "LOCAbsValPeak",
                "ROCAbsValPeak",
                "LOCAbsRiseSlope",
                "ROCAbsRiseSlope",
                "LOCAbsFallSlope",
                "ROCAbsFallSlope",
                "Stage",
            ]
        ]

        if np.isscalar(hypno_up):
            df = df.drop(columns=["Stage"])
        else:
            df["Stage"] = df["Stage"].astype(int)

        df = df.reset_index(drop=True)
        return yasa.REMResults(events=df, data=np.vstack((loc, roc)), sf=fs, ch_names=['LOC', 'ROC'], hypno=hypno_up, data_filt=np.vstack((loc_clean, roc_clean)))

    else:
        raise Exception("method ['original', 'ssc_threshold'].")
