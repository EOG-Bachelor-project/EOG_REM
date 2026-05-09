# Filename: extra_feats.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Extra feature extraction from merged CSVs.
#              Follows the same pattern as eog_feats.py, bout_feats.py, etc.
#              Computes spectral, phasic/tonic structure, EM morphology,
#              and sleep architecture features.
#
# Usage:
#       python features/extra_feats.py merged_csv_eog/DCSM_1_a_merged.csv.gz   # Single subject (for testing)
#       python features/extra_feats.py merged_csv_eog/                         # Full batch

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch

# =============================================================================
# Constants
# =============================================================================
_DCSM_PATTERN = re.compile(r"^(DCSM_\d+_\w+)")
FEATURES_DIR  = Path("C:/Users/AKLO0022/EOG_REM/features_csv/features.csv") 

# Frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "gamma": (30.0, 45.0), 
}
# NOTE: Gamma is often defined 30 Hz and above, but we use 30-45 Hz to avoid line noise at 50/60 Hz 

# Only load what we need — keeps memory low on long recordings
_USECOLS_REQUIRED = ["time_sec", "stage", "EEG_LOC", "EEG_ROC"]
_USECOLS_OPTIONAL = [
    "EpochType",
    # EM event columns
    "Start_x",               # EM start time
    "em_Peak",               # EM peak time
    "em_End",                # EM end time
    "em_Duration",           # EM duration
    "em_MeanAbsValPeak",     # EM amplitude
    "em_EM_Type",            # EM type (SEM / REM)
    "is_em_event",           # bool flag
    "em_LOCAbsRiseSlope",    # LOC rise slope
    "em_ROCAbsRiseSlope",    # ROC rise slope
    "em_LOCAbsFallSlope",    # LOC fall slope
    "em_ROCAbsFallSlope",    # ROC fall slope
    "em_SubEpochStart",      # for phasic/tonic bout deduplication
    ]

SUB_EPOCH_LEN_S = 4.0

# =============================================================================
# Helpers
# =============================================================================

# ———— Load only needed columns ————
def _load(merged_file: Path) -> pd.DataFrame:
    """Load only the columns we need. Handles .csv and .csv.gz."""
    peek    = pd.read_csv(merged_file, nrows=0).columns.tolist()                # Get columns in file
    usecols = [c for c in _USECOLS_REQUIRED + _USECOLS_OPTIONAL if c in peek]   # Only load columns that exist in the file
    missing = [c for c in _USECOLS_REQUIRED if c not in peek]                   # Check for missing required columns
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return pd.read_csv(merged_file, usecols=usecols, low_memory=False)          # Load with only needed columns

# ———— REM mask ————
def _rem_mask(df: pd.DataFrame) -> pd.Series:
    return df["stage"].str.upper().str.strip() == "REM"

# ———— EpochType mask ————
def _epoch_type_mask(df: pd.DataFrame, epoch_type: str) -> pd.Series: 
    if "EpochType" not in df.columns:           # If missing, return all False (no phasic/tonic epochs)
        return pd.Series(False, index=df.index) 
    return df["EpochType"] == epoch_type        # Mask for the specified epoch type (e.g., "Phasic" or "Tonic")

# ———— Band power via Welch PSD ————
def _band_power(signal: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """
    Absolute band power via Welch PSD.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array of EEG samples.
    fs : float
        Sampling frequency in Hz.
    fmin : float
        Lower frequency bound of the band (Hz).
    fmax : float
        Upper frequency bound of the band (Hz).
    """
    if len(signal) < fs * 2:                            # Need at least 2 seconds of data for a reliable PSD estimate
        return np.nan
    nperseg = min(int(fs * 4), len(signal))             # Use 4-second windows or the full signal if shorter        
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)  # Compute PSD using Welch's method
    idx = (freqs >= fmin) & (freqs <= fmax)             # Indices of frequencies within the desired band
    if idx.sum() == 0:
        return np.nan
    return float(np.trapezoid(psd[idx], freqs[idx]))    # Integrate PSD over the band to get absolute power

# ———— Get EEG signal (average LOC + ROC) ————
def _get_eeg(df: pd.DataFrame) -> np.ndarray | None:
    """Average LOC + ROC EEG channels."""
    has_loc = "EEG_LOC" in df.columns and df["EEG_LOC"].notna().any()
    has_roc = "EEG_ROC" in df.columns and df["EEG_ROC"].notna().any()
    if has_loc and has_roc:
        return ((pd.to_numeric(df["EEG_LOC"], errors="coerce").fillna(0) +
                 pd.to_numeric(df["EEG_ROC"], errors="coerce").fillna(0)) / 2).values
    elif has_loc:
        return pd.to_numeric(df["EEG_LOC"], errors="coerce").values
    elif has_roc:
        return pd.to_numeric(df["EEG_ROC"], errors="coerce").values
    return None

# ———— Get sub-epoch series for REM —————
def _get_subepoch_series(df: pd.DataFrame) -> pd.Series:
    """Return deduplicated EpochType series for REM sub-epochs."""
    rem_df = df[_rem_mask(df)]
    if "EpochType" not in rem_df.columns or rem_df["EpochType"].isna().all():
        return pd.Series(dtype=str)
    if "em_SubEpochStart" in rem_df.columns:
        sub = rem_df.drop_duplicates(subset="em_SubEpochStart")
    else:
        sub = rem_df[rem_df["EpochType"].notna()].copy()
        sub["_bin"] = (pd.to_numeric(sub["time_sec"], errors="coerce") // SUB_EPOCH_LEN_S).astype(int)
        sub = sub.drop_duplicates(subset="_bin")
    return sub["EpochType"].reset_index(drop=True)

# ———— Identify bouts of a given label in the sub-epoch series ————
def _identify_bouts(types: pd.Series, label: str) -> list[int]:
    """
    Return list of bout lengths (in sub-epochs) for the given label.

    Parameters
    ----------
    types : pd.Series
        Series of sub-epoch types (e.g., "Phasic", "Tonic").
    label : str
        Label to identify bouts for (e.g., "Phasic" or "Tonic").
    """
    bouts, count = [], 0
    for t in types:
        if t == label:
            count += 1
        else:
            if count > 0:
                bouts.append(count)
            count = 0
    if count > 0:
        bouts.append(count)
    return bouts


def _get_em_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a deduplicated per-EM-event DataFrame.
    Uses Start_x as the deduplication key (actual column name in merged CSV).
    """
    if "is_em_event" in df.columns:
        em_df = df[df["is_em_event"] == True].copy()
    else:
        em_df = df[df["Start_x"].notna()].copy() if "Start_x" in df.columns else pd.DataFrame()

    if em_df.empty:
        return pd.DataFrame()

    if "Start_x" in em_df.columns:
        em_df = em_df.drop_duplicates(subset="Start_x")

    return em_df.reset_index(drop=True)


# =============================================================================
# Feature groups
# =============================================================================

def _spectral_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Band power (delta, theta, gamma) during REM overall, phasic, tonic.
    Plus theta/delta ratio per context.

    Features:
        eeg_{band}_{context}_power      (context: rem / phasic / tonic)
        eeg_theta_delta_ratio_{context}

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame for a single subject.
    fs : float
        Sampling frequency in Hz.
    
    Returns
    -------
    dict
        Flat dict of feature name  -> value.
    """
    feats: dict = {}          # Initialize empty dict
    eeg_full = _get_eeg(df)   # Get the EEG signal

    # Define the expected feature names for NaN defaults if EEG data is missing
    nan_feats = (
        [f"eeg_{b}_{ctx}_power" for b in BANDS for ctx in ("rem", "phasic", "tonic")]
        + [f"eeg_theta_delta_ratio_{ctx}" for ctx in ("rem", "phasic", "tonic")]
    )

    # If no EEG data is available, set all spectral features to NaN and return early
    if eeg_full is None:
        print("    [SKIP] No EEG columns found")
        return {k: np.nan for k in nan_feats}
    
    # Create masks for REM, phasic REM, and tonic REM contexts
    contexts = {
        "rem":    _rem_mask(df),
        "phasic": _rem_mask(df) & _epoch_type_mask(df, "Phasic"),
        "tonic":  _rem_mask(df) & _epoch_type_mask(df, "Tonic"),
    }

    # Calculate features for each context
    for ctx, mask in contexts.items():
        signal = eeg_full[mask.values]                  # Extract EEG samples for the current context using the boolean mask
        powers = {}                                     # Store band powers for ratio calculation
        for band, (fmin, fmax) in BANDS.items():
            bp = _band_power(signal, fs, fmin, fmax)    # Calculate absolute band power for the current band and context
            feats[f"eeg_{band}_{ctx}_power"] = bp       # FEAT: Absolute band power
            powers[band] = bp

        d = powers.get("delta", np.nan)     # Delta power
        t = powers.get("theta", np.nan)     # Theta power
        if d and not np.isnan(d) and d > 0:
            feats[f"eeg_theta_delta_ratio_{ctx}"] = round(t / d, 6) # FEAT: Theta/delta ratio if delta is valid and > 0
        else:
            feats[f"eeg_theta_delta_ratio_{ctx}"] = np.nan          # FEAT: Theta/delta ratio is NaN if delta is missing/invalid/zero

        # Print summary
        n = int(mask.sum())
        print(f"    {ctx:<8s}: {n:,} samples  |  "
              + "  ".join(f"{b}={feats.get(f'eeg_{b}_{ctx}_power', np.nan):.3e}"
                          for b in BANDS))

    return feats


def _phasic_tonic_structure_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Extended phasic/tonic structure.

    Feature:
        pt_transitions_per_min
        phasic_bout_p25_s / p75_s / p90_s
        tonic_bout_p25_s  / p75_s / p90_s
        phasic_longest_run_s
        tonic_longest_run_s
        phasic_first_latency_s
        phasic_long_bout_fraction
        tonic_long_bout_fraction

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame for a single subject.
    fs : float
        Sampling frequency in Hz.
    
    Returns
    -------
    dict
        Flat dict of feature name  -> value.
    """

    # Set all features to NaN by default; will fill in if we have the necessary data
    nan_keys = [
        "pt_transitions_per_min",
        "phasic_bout_p25_s", "phasic_bout_p75_s", "phasic_bout_p90_s",
        "tonic_bout_p25_s",  "tonic_bout_p75_s",  "tonic_bout_p90_s",
        "phasic_longest_run_s", "tonic_longest_run_s",
        "phasic_first_latency_s",
        "phasic_long_bout_fraction", "tonic_long_bout_fraction",
    ]
    feats = {k: np.nan for k in nan_keys}

    # Check for REM samples and EpochType column first
    rem_df = df[_rem_mask(df)]
    if rem_df.empty:
        print("    No REM samples — returning NaN defaults")
        return feats

    rem_min = len(rem_df) / fs / 60.0   # Total REM duration in minutes
    types   = _get_subepoch_series(df)  # Get the deduplicated EpochType series for REM sub-epochs (one entry per sub-epoch)

    # If there are no valid sub-epoch types, return NaN
    if types.empty:
        print("    No sub-epoch EpochType series — returning NaN defaults")
        return feats

    # Calculate transitions per minute in the sub-epoch series (changes in EpochType)
    transitions = int((types != types.shift()).sum()) - 1                                                   # Number of transitions
    feats["pt_transitions_per_min"] = round(max(transitions, 0) / rem_min, 4) if rem_min > 0 else np.nan    # FEAT: Transitions per minute, ensuring non-negative and handling zero REM duration
    print(f"    Transitions: {transitions}  ({feats['pt_transitions_per_min']:.3f}/min)")

    total_rem_s = rem_min * 60  # Total REM duration in seconds

    for label in ("Phasic", "Tonic"):
        prefix = label.lower()
        bouts  = _identify_bouts(types, label) # Get list of bout lengths for the current label 

        if bouts:
            dur_s = np.array(bouts) * SUB_EPOCH_LEN_S                                       # Convert bout lengths from sub-epochs to seconds
            feats[f"{prefix}_bout_p25_s"]      = round(float(np.percentile(dur_s, 25)), 4)  # FEAT: 25th percentile of bout durations in seconds
            feats[f"{prefix}_bout_p75_s"]      = round(float(np.percentile(dur_s, 75)), 4)  # FEAT: 75th percentile of bout durations in seconds
            feats[f"{prefix}_bout_p90_s"]      = round(float(np.percentile(dur_s, 90)), 4)  # FEAT: 90th percentile of bout durations in seconds
            feats[f"{prefix}_longest_run_s"]   = round(float(dur_s.max()), 4)               # FEAT: Longest bout duration in seconds
            long_total = dur_s[dur_s > 30].sum()                                            # Total time spent in long bouts (>30s)
            feats[f"{prefix}_long_bout_fraction"] = (                                       # FEAT: Fraction of REM spent in long bouts (>30s)
                round(long_total / total_rem_s, 4) if total_rem_s > 0 else np.nan
            )
            print(f"    {label}: {len(bouts)} bouts  "
                  f"p75={feats[f'{prefix}_bout_p75_s']:.1f}[s]  "
                  f"longest={feats[f'{prefix}_longest_run_s']:.1f}[s]")
        else:
            print(f"    {label}: 0 bouts")

    # First phasic bout latency from REM onset
    rem_onset = pd.to_numeric(rem_df["time_sec"], errors="coerce").iloc[0] # Time of the first REM sample, which we consider as REM onset
    if "EpochType" in rem_df.columns:
        phasic_rows = rem_df[rem_df["EpochType"] == "Phasic"] # Rows corresponding to phasic sub-epochs within REM
        if not phasic_rows.empty:
            first_phasic = pd.to_numeric(phasic_rows["time_sec"], errors="coerce").iloc[0]
            feats["phasic_first_latency_s"] = round(float(first_phasic - rem_onset), 4) # FEAT: Latency of first phasic sub-epoch from REM onset in seconds
            print(f"    First phasic latency: {feats['phasic_first_latency_s']:.1f}[s]")

    return feats


def _em_morphology_features(df: pd.DataFrame) -> dict:
    """
    EM morphology using rise/fall slopes from LOC and ROC channels, amplitude variance, and SEM fraction.

    Feature names:
        em_mean_rise_slope       : mean of (LOC + ROC) rise slopes
        em_mean_fall_slope       : mean of (LOC + ROC) fall slopes
        em_amplitude_variance    : variance of em_MeanAbsValPeak
        em_sem_fraction          : fraction of EMs classified as SEM

    Parameters    
    ----------
    df : pd.DataFrame
        Merged DataFrame for a single subject.

    Returns
    -------
    dict
        Flat dict of feature name  -> value.
    """
    feats = {
        "em_mean_rise_slope":    np.nan,
        "em_mean_fall_slope":    np.nan,
        "em_amplitude_variance": np.nan,
        "em_sem_fraction":       np.nan,
    }

    em_df = _get_em_table(df)
    if em_df.empty:
        print("    No valid EM events for morphology")
        return feats

    # Rise slope — average LOC and ROC
    if "em_LOCAbsRiseSlope" in em_df.columns and "em_ROCAbsRiseSlope" in em_df.columns:
        loc_rise = pd.to_numeric(em_df["em_LOCAbsRiseSlope"], errors="coerce")
        roc_rise = pd.to_numeric(em_df["em_ROCAbsRiseSlope"], errors="coerce")
        rise_mean = ((loc_rise + roc_rise) / 2).mean()
        feats["em_mean_rise_slope"] = round(float(rise_mean), 6) # FEAT: Mean of the average of LOC and ROC rise slopes across all EM events, rounded to 6 decimal places

    # Fall slope — average LOC and ROC
    if "em_LOCAbsFallSlope" in em_df.columns and "em_ROCAbsFallSlope" in em_df.columns:
        loc_fall = pd.to_numeric(em_df["em_LOCAbsFallSlope"], errors="coerce")
        roc_fall = pd.to_numeric(em_df["em_ROCAbsFallSlope"], errors="coerce")
        fall_mean = ((loc_fall + roc_fall) / 2).mean()
        feats["em_mean_fall_slope"] = round(float(fall_mean), 6) # FEAT: Mean of the average of LOC and ROC fall slopes across all EM events, rounded to 6 decimal places

    # Amplitude variance
    if "em_MeanAbsValPeak" in em_df.columns:
        amp = pd.to_numeric(em_df["em_MeanAbsValPeak"], errors="coerce").dropna()
        feats["em_amplitude_variance"] = round(float(amp.var()), 6) if len(amp) > 1 else np.nan # FEAT: Variance of EM amplitudes, rounded to 6 decimal places; NaN if fewer than 2 valid amplitude values

    # SEM fraction — use em_EM_Type (actual column name)
    type_col = "em_EM_Type" if "em_EM_Type" in em_df.columns else \
               "EM_Type"    if "EM_Type"    in em_df.columns else None
    if type_col:
        n_total = len(em_df)
        n_sem   = (em_df[type_col] == "SEM").sum()
        feats["em_sem_fraction"] = round(n_sem / n_total, 4) if n_total > 0 else np.nan # FEAT: Fraction of EM events classified as SEM, rounded to 4 decimal places

    print(f"    EM events: {len(em_df)}  "
          f"rise_slope={feats['em_mean_rise_slope']}  "
          f"fall_slope={feats['em_mean_fall_slope']}  "
          f"sem_frac={feats['em_sem_fraction']}")
    return feats


def _sleep_architecture_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Sleep architecture from stage column.

    Features:
        rem_latency_min    : minutes from sleep onset to first REM
        n_rem_cycles       : number of distinct REM periods
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame for a single subject.
    fs : float
        Sampling frequency in Hz.
    
    Returns
    -------
    dict
        Flat dict of feature name  -> value.
    """
    # Initialize features to NaN; will fill in if we have the necessary columns and data
    feats = {"rem_latency_min": np.nan, "n_rem_cycles": np.nan}

    # Check for required columns first
    if "stage" not in df.columns or "time_sec" not in df.columns:
        return feats

    stage = df["stage"].str.upper().str.strip()             # Clean stage labels
    time  = pd.to_numeric(df["time_sec"], errors="coerce")  # Convert time to numeric, coercing errors to NaN

    # Sleep onset is defined as the first sample of any sleep stage (N1, N2, N3, REM)
    sleep_mask = stage.isin(["N1", "N2", "N3", "REM"]) # Mask for any sleep stage
    if not sleep_mask.any():
        return feats
    sleep_onset = time[sleep_mask].iloc[0]  # Time of the first sleep stage sample, which we consider as sleep onset

    rem_mask = stage == "REM"
    if not rem_mask.any():
        return feats
    first_rem = time[rem_mask].iloc[0]

    feats["rem_latency_min"] = round(float((first_rem - sleep_onset) / 60.0), 4) # FEAT: REM latency in minutes, rounded to 4 decimal places

    rem_blocks = (rem_mask != rem_mask.shift()).cumsum()
    feats["n_rem_cycles"] = int(rem_blocks[rem_mask].nunique()) # FEAT: Number of distinct REM periods (contiguous blocks of REM samples)

    print(f"    REM latency: {feats['rem_latency_min']:.1f} [min]  |  "
          f"REM cycles: {feats['n_rem_cycles']}")
    return feats


# =============================================================================
# Single-subject entry point
# =============================================================================

def extract_extra_features( 
        merged_file: str | Path,
        subject_id:  str | None = None,
        fs:          float = 250.0,
        ) -> dict:
    """
    Extract extra features from a single merged CSV (.csv or .csv.gz).

    Parameters
    ----------
    merged_file : str | Path
        Path to the merged CSV (output of merge_all).
    subject_id : str | None
        Subject identifier. If None, parsed from filename.
    fs : float
        Sampling frequency [Hz]. Default is 250.0.

    Returns
    -------
    dict
        Flat dict of feature name -> value.
    """
    merged_file = Path(merged_file)
    raw_stem    = merged_file.name.replace(".csv.gz", "").replace(".csv", "")               # Remove extensions to get raw stem
    m           = _DCSM_PATTERN.match(raw_stem)                                             # Try to parse subject ID using regex pattern           
    sid         = subject_id if subject_id is not None else (m.group(1) if m else raw_stem) # Use provided subject_id or parsed ID or raw stem as fallback

    print(f"\n{'=' * 60}")
    print(f"Extracting extra features: {merged_file.name}")
    print(f"  subject_id : {sid}  |  fs : {fs} [Hz]")

    df = _load(merged_file)

    feats: dict = {"subject_id": sid}

    print(f"\n--- Spectral band power ---")
    feats.update(_spectral_features(df, fs))

    print(f"\n--- Phasic / tonic structure ---")
    feats.update(_phasic_tonic_structure_features(df, fs))

    print(f"\n--- EM morphology ---")
    feats.update(_em_morphology_features(df))

    print(f"\n--- Sleep architecture ---")
    feats.update(_sleep_architecture_features(df, fs))

    n_nan = sum(1 for v in feats.values() if isinstance(v, float) and np.isnan(v))
    print(f"\n  Features computed : {len(feats) - 1}")
    print(f"  NaN features      : {n_nan}")

    return feats


# =============================================================================
# Batch entry point
# =============================================================================

def extract_extra_features_batch(
        merged_dir:  str | Path,
        output_file: str | Path | None = None,
        fs:          float = 250.0,
        pattern:     str = "*_merged.csv*",
        ) -> pd.DataFrame:
    """
    Run extract_extra_features on every merged CSV in a directory.
    Skips already-processed subjects. Appends to output_file if it exists.

    Parameters
    ----------
    merged_dir : str | Path
        Directory containing merged CSVs.
    output_file : str | Path | None
        Output CSV path. Defaults to features_csv/extra_features.csv.
    fs : float
        Sampling frequency [Hz]. Default is 250.0.
    pattern : str
        Glob pattern. Default is '*_merged.csv'.

    Returns
    -------
    pd.DataFrame
        One row per subject with all extra features.
    """
    merged_dir = Path(merged_dir)                   # Ensure merged_dir is a Path object
    files      = sorted(merged_dir.glob(pattern))   # Find all files matching the pattern in the directory

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {merged_dir}")

    if output_file is None:
        output_file = FEATURES_DIR / "extra_features.csv" 
    output_file = Path(output_file)

    existing_df  = pd.DataFrame() # Initialize empty DataFrame for existing features; will load if output_file exists
    already_done = set()          # Set of subject_ids already processed, to skip in the batch run
    if output_file.exists():
        existing_df  = pd.read_csv(output_file, low_memory=False)
        if "subject_id" in existing_df.columns:
            already_done = set(existing_df["subject_id"].astype(str)) # Extract subject IDs from existing DataFrame to skip already-processed subjects
            print(f"  Found {len(already_done)} already-processed subjects in {output_file.name}")

    new_rows = []   # Initialize list to collect new feature rows for subjects processed in this batch run
    skipped  = 0    # Initialize counter for skipped subjects (already processed)
    for f in files:
        raw_stem = f.name.replace(".csv.gz", "").replace(".csv", "")    # Remove extensions to get raw stem for subject ID parsing
        m        = _DCSM_PATTERN.match(raw_stem)                        # Try to parse subject ID using regex pattern
        sid      = m.group(1) if m else raw_stem                        # Use parsed subject ID or raw stem as fallback if parsing fails
        if sid in already_done:
            skipped += 1
            continue
        try:
            row = extract_extra_features(f, fs=fs)  # Extract features for the current file and subject ID
            new_rows.append(row)                    # Append the resulting feature dict as a new row to the list of new rows
        except Exception as e:
            print(f"  [SKIP] {f.name} — {e}")

    if skipped:
        print(f"  Skipped {skipped} already-processed subjects")

    # Combine existing features with new features
    new_df = pd.DataFrame(new_rows)
    if not existing_df.empty and not new_df.empty:                          # If we have both existing and new data
        feature_df = pd.concat([existing_df, new_df], ignore_index=True)    # Combine existing and new DataFrames, ignoring the index to create a continuous index in the resulting DataFrame
    elif not new_df.empty:                                                  # If we only have new data (no existing data)        
        feature_df = new_df                                                 # Use the new DataFrame as the feature DataFrame if there is no existing data to combine with
    else:                                                                   # If we have no new data (i.e. all subjects were already processed) and possibly existing data
        feature_df = existing_df                                            # Use the existing DataFrame as the feature DataFrame if there are no new rows to add

    output_file.parent.mkdir(parents=True, exist_ok=True)   # Ensure the output directory exists; creates it if it doesn't
    feature_df.to_csv(output_file, index=False)             # Save the combined feature DataFrame to the specified output CSV file without the index column
    print(f"\nSaved -> {output_file}  "
          f"({feature_df.shape[0]} subjects, {feature_df.shape[1] - 1} features)")

    return feature_df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extra_feats.py <merged_csv_or_dir> [fs]")
        sys.exit(1)

    path = Path(sys.argv[1])
    fs   = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0

    if path.is_dir():
        df_out = extract_extra_features_batch(path, fs=fs)
        print(df_out.to_string(index=False))
    elif path.is_file():
        feats = extract_extra_features(path, fs=fs)
        for k, v in feats.items():
            print(f"  {k:<50}  {v}")
    else:
        print(f"Path not found: {path}")
        sys.exit(1)