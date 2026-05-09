# Filename: extra_feats.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Extra feature extraction from merged CSVs.
#              Follows the same pattern as eog_feats.py, bout_feats.py, etc.
#              Computes spectral, phasic/tonic structure, EM morphology,
#              and sleep architecture features.
#
# Usage:
#       python features/extra_feats.py merged_csv_eog/DCSM_1_a_merged.csv   # single subject (for testing)
#       python features/extra_feats.py merged_csv_eog/                      # Full batch


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

# Columns needed - load only these to keep memory low
_USECOLS_REQUIRED = ["time_sec", "stage", "EEG_LOC", "EEG_ROC"]
_USECOLS_OPTIONAL = [
    "EpochType",
    "em_Start", "em_Peak", "em_End", "em_Duration",
    "em_MeanAbsValPeak", "EM_Type", "is_em_event",
    "em_SubEpochStart",
]

SUB_EPOCH_LEN_S = 4.0

# =============================================================================
# Helpers
# =============================================================================

# ———— Load only needed columns ————
def _load(merged_file: Path) -> pd.DataFrame:
    peek   = pd.read_csv(merged_file, nrows=0).columns.tolist()                 # Get columns in file
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
def _band_power(
        signal: np.ndarray,
        fs: float, 
        fmin: float, 
        fmax: float,
        ) -> float:
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
    """
    Average LOC + ROC EEG channels. 
    Returns None if both missing.
    """
    has_loc = "EEG_LOC" in df.columns and df["EEG_LOC"].notna().any()           # Check if LOC column exists and has any non-NaN values
    has_roc = "EEG_ROC" in df.columns and df["EEG_ROC"].notna().any()           # Check if ROC column exists and has any non-NaN values       
    if has_loc and has_roc:
        return ((df["EEG_LOC"].fillna(0) + df["EEG_ROC"].fillna(0)) / 2).values # Average LOC and ROC, treating NaNs as 0 (only for averaging; actual NaN handling is done in _band_power)
    elif has_loc:
        return df["EEG_LOC"].values                                             # Use LOC if ROC missing
    elif has_roc:
        return df["EEG_ROC"].values                                             # Use ROC if LOC missing
    return None

# ———— Get sub-epoch series for REM —————
def _get_subepoch_series(df: pd.DataFrame) -> pd.Series:
    """
    Return deduplicated EpochType series for REM sub-epochs.
    If em_SubEpochStart exists, use it to deduplicate. Otherwise, deduplicate by 4-second bins.
    """
    rem_df = df[_rem_mask(df)]  
    if "EpochType" not in rem_df.columns or rem_df["EpochType"].isna().all(): # If no EpochType info, return empty series
        return pd.Series(dtype=str)
    if "em_SubEpochStart" in rem_df.columns:                                  # If em_SubEpochStart exists, use it to deduplicate (one row per sub-epoch)
        sub = rem_df.drop_duplicates(subset="em_SubEpochStart")
    else:                                                                     # Otherwise:
        sub = rem_df[rem_df["EpochType"].notna()].copy()                      # Only consider rows with valid EpochType
        sub["_bin"] = (sub["time_sec"] // SUB_EPOCH_LEN_S).astype(int)        # Bin by 4-second intervals
        sub = sub.drop_duplicates(subset="_bin")                              # Drop duplicates within each bin, keeping the first occurrence         
    return sub["EpochType"].reset_index(drop=True)                            # Return the deduplicated EpochType series for REM sub-epochs

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


# =============================================================================
# Feature groups
# =============================================================================

def _spectral_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Band power (delta, theta, gamma) during REM overall, phasic, and tonic.
    Also computes theta/delta ratio for each context.

    Features (per context x band + ratios):
        eeg_{band}_rem_power       : overall REM
        eeg_{band}_phasic_power    : phasic REM only
        eeg_{band}_tonic_power     : tonic REM only
        eeg_theta_delta_ratio_rem/phasic/tonic

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
    feats: dict = {}            # Initialize empty dict to store features
    eeg_full = _get_eeg(df)     # Get the combined EEG signal (average of LOC and ROC, or whichever is available)
    
    # 1) If no EEG data available, return NaN for all spectral features
    if eeg_full is None:
        print("    [SKIP] No EEG columns found")
        for band in BANDS:                                  
            for ctx in ("rem", "phasic", "tonic"):
                feats[f"eeg_{band}_{ctx}_power"] = np.nan 
        for ctx in ("rem", "phasic", "tonic"):
            feats[f"eeg_theta_delta_ratio_{ctx}"] = np.nan
        return feats

    # 2) Define contexts and their corresponding masks
    contexts = {
        "rem":    _rem_mask(df),
        "phasic": _rem_mask(df) & _epoch_type_mask(df, "Phasic"),
        "tonic":  _rem_mask(df) & _epoch_type_mask(df, "Tonic"),
    }

    # 3) Compute band power for each context and band, and store in feats dict
    for ctx, mask in contexts.items():
        signal = eeg_full[mask.values]  # Extract the EEG signal for the current context using the boolean mask
        powers = {}                     # Temporary dict to store band powers for ratio calculation
        for band, (fmin, fmax) in BANDS.items():
            bp = _band_power(signal, fs, fmin, fmax)    # Compute band power for the current band and context
            feats[f"eeg_{band}_{ctx}_power"] = bp       # feat: Store the computed band power in the features dict with a descriptive key
            powers[band] = bp                       

        # feat: eeg theta/delta ratio for the current context, only if delta power is available and > 0 to avoid division issues
        if powers.get("delta") and powers["delta"] > 0:
            feats[f"eeg_theta_delta_ratio_{ctx}"] = round(powers["theta"] / powers["delta"], 6)
        else:
            feats[f"eeg_theta_delta_ratio_{ctx}"] = np.nan

        n = int(mask.sum())
        print(f"    {ctx:<8s}: {n:,} samples  |  "
              + "  ".join(f"{b}={feats[f'eeg_{b}_{ctx}_power']:.3e}"
                          for b in BANDS if not np.isnan(feats.get(f"eeg_{b}_{ctx}_power", np.nan))))

    return feats


def _phasic_tonic_structure_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Extended phasic/tonic structure beyond what bout_feats.py computes.

    Features:
        pt_transitions_per_min         : phasic↔tonic switches per minute of REM
        phasic_bout_p25/p75/p90_s      : bout length percentiles [seconds]
        tonic_bout_p25/p75/p90_s
        phasic_longest_run_s           : longest consecutive phasic run [s]
        tonic_longest_run_s
        phasic_first_latency_s         : time from REM onset to first phasic bout [s]
        phasic_long_bout_fraction      : fraction of REM in phasic bouts > 30 s
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
    feats: dict = {} # Initialize empty dict to store features

    # Set all features to NaN by default; will fill in if we have the necessary data
    nan_keys = [
        "pt_transitions_per_min",
        "phasic_bout_p25_s", "phasic_bout_p75_s", "phasic_bout_p90_s",
        "tonic_bout_p25_s",  "tonic_bout_p75_s",  "tonic_bout_p90_s",
        "phasic_longest_run_s", "tonic_longest_run_s",
        "phasic_first_latency_s",
        "phasic_long_bout_fraction", "tonic_long_bout_fraction",
    ]
    for k in nan_keys:
        feats[k] = np.nan

    # 1) Check for REM samples and sub-epoch series; if missing, return NaN defaults
    rem_df = df[_rem_mask(df)]
    if rem_df.empty:
        print("    No REM samples — returning NaN defaults")
        return feats

    rem_min = len(rem_df) / fs / 60.0   # Total REM duration in minutes
    types   = _get_subepoch_series(df)  # Get the deduplicated EpochType series for REM sub-epochs (one entry per sub-epoch)

    if types.empty:
        print("    No sub-epoch series — returning NaN defaults")
        return feats

    # 2) Transitions per minute
    transitions = int((types != types.shift()).sum()) - 1                                                # Count number of changes 'EpochType' 
    feats["pt_transitions_per_min"] = round(max(transitions, 0) / rem_min, 4) if rem_min > 0 else np.nan # feat: Transitions per minute of REM, ensuring we don't divide by zero and that transitions can't be negative
    print(f"    Transitions: {transitions}  ({feats['pt_transitions_per_min']:.3f}/min)")

    for label in ("Phasic", "Tonic"):
        prefix = label.lower()
        bouts  = _identify_bouts(types, label) # Get list of bout lengths for the current label 

        if bouts:
            dur_s = np.array(bouts) * SUB_EPOCH_LEN_S                                   # Convert bout lengths from number of sub-epochs to seconds         
            feats[f"{prefix}_bout_p25_s"] = round(float(np.percentile(dur_s, 25)), 4)   # feat: 25th percentile of bout lengths in seconds
            feats[f"{prefix}_bout_p75_s"] = round(float(np.percentile(dur_s, 75)), 4)   # feat: 75th percentile of bout lengths in seconds
            feats[f"{prefix}_bout_p90_s"] = round(float(np.percentile(dur_s, 90)), 4)   # feat: 90th percentile of bout lengths in seconds
            feats[f"{prefix}_longest_run_s"] = round(float(dur_s.max()), 4)             # feat: Longest consecutive run of the label in seconds
            long_total = dur_s[dur_s > 30].sum()                                        # Total time spent in bouts longer than 30 seconds
            total_rem_s = rem_min * 60                                                  # Total REM duration in seconds
            feats[f"{prefix}_long_bout_fraction"] = (                                   # feat: Fraction of REM spent in long bouts of the label, ensuring we don't divide by zero
                round(long_total / total_rem_s, 4) if total_rem_s > 0 else np.nan 
            )
            print(f"    {label} bouts: {len(bouts)}  p75={feats[f'{prefix}_bout_p75_s']:.1f}[s]  "
                  f"longest={feats[f'{prefix}_longest_run_s']:.1f}[s]")
        else:
            print(f"    {label}: 0 bouts")

    # 3) First phasic latency from REM onset
    rem_onset = rem_df["time_sec"].iloc[0] # Time of the first REM sample, which we consider as REM onset
    if "EpochType" in rem_df.columns:
        phasic_rows = rem_df[rem_df["EpochType"] == "Phasic"] # Rows corresponding to phasic sub-epochs within REM
        if not phasic_rows.empty:
            feats["phasic_first_latency_s"] = round(float(phasic_rows["time_sec"].iloc[0] - rem_onset), 4)  # feat: Time from REM onset to the first phasic sub-epoch in seconds
            print(f"    First phasic latency: {feats['phasic_first_latency_s']:.1f} [s]")

    return feats


def _em_morphology_features(df: pd.DataFrame) -> dict:
    """
    Eye movement morphology from em_Start / em_Peak / em_End columns.

    Features:
        em_mean_rise_time_s    : mean time from Start to Peak [s]
        em_mean_fall_time_s    : mean time from Peak to End [s]
        em_amplitude_variance  : variance of em_MeanAbsValPeak across all EMs
        em_sem_fraction        : fraction of EMs classified as SEM
    
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
        "em_mean_rise_time_s":   np.nan, 
        "em_mean_fall_time_s":   np.nan,
        "em_amplitude_variance": np.nan,
        "em_sem_fraction":       np.nan,
    }

    # 1) Build per-EM table from peak rows
    needed = {"em_Start", "em_Peak", "em_End"}
    if not needed.issubset(df.columns):
        print(f"    [SKIP] Missing EM morphology columns: {needed - set(df.columns)}")
        return feats

    # 2) One row per EM — drop duplicates on em_Start
    em_df = df[df["is_em_event"] == True].drop_duplicates(subset="em_Start").copy() \
        if "is_em_event" in df.columns \
        else df[df["em_Start"].notna()].drop_duplicates(subset="em_Start").copy()
    
    # Convert EM timing columns to numeric, coercing errors to NaN, and drop rows with invalid timings
    for col in ["em_Start", "em_Peak", "em_End"]:
        em_df[col] = pd.to_numeric(em_df[col], errors="coerce")

    # Drop rows where any of the EM timing columns are NaN (invalid)
    em_df = em_df.dropna(subset=["em_Start", "em_Peak", "em_End"])
    if em_df.empty:
        print("    No valid EM events for morphology")
        return feats
    
    # 3) Compute morphology features
    rise = (em_df["em_Peak"] - em_df["em_Start"]).clip(lower=0) 
    fall = (em_df["em_End"]  - em_df["em_Peak"]).clip(lower=0)

    feats["em_mean_rise_time_s"] = round(float(rise.mean()), 6) # feat: Mean time from EM start to peak in seconds, rounded to 6 decimal places
    feats["em_mean_fall_time_s"] = round(float(fall.mean()), 6) # feat: Mean time from EM peak to end in seconds, rounded to 6 decimal places

    # feat: EM amplitude variance
    if "em_MeanAbsValPeak" in em_df.columns:
        amp = pd.to_numeric(em_df["em_MeanAbsValPeak"], errors="coerce").dropna()
        feats["em_amplitude_variance"] = round(float(amp.var()), 6) if len(amp) > 1 else np.nan # feat: Variance of EM amplitudes, rounded to 6 decimal places; requires at least 2 valid amplitude values

    # feat: SEM fraction
    if "EM_Type" in em_df.columns:
        n_total = len(em_df)
        n_sem   = (em_df["EM_Type"] == "SEM").sum()
        feats["em_sem_fraction"] = round(n_sem / n_total, 4) if n_total > 0 else np.nan # feat: Fraction of EMs classified as SEM

    print(f"    EM events: {len(em_df)}  rise={feats['em_mean_rise_time_s']:.4f}[s]  "
          f"fall={feats['em_mean_fall_time_s']:.4f}[s]  "
          f"amp_var={feats['em_amplitude_variance']:.3e}  "
          f"SEM%={feats['em_sem_fraction']:.2%}"
          ) 
    return feats


def _sleep_architecture_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Sleep architecture features from the stage column.

    Features:
        rem_latency_min        : time from first sleep epoch to first REM [min]
        n_rem_cycles           : number of distinct REM periods
    
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
    feats = {
        "rem_latency_min": np.nan,
        "n_rem_cycles":    np.nan,
    }

    if "stage" not in df.columns or "time_sec" not in df.columns:
        return feats

    stage = df["stage"].str.upper().str.strip() # Normalize stage labels to uppercase and strip whitespace for consistent processing
    time  = df["time_sec"]                      # Time in seconds from the start of the recording, used to calculate latencies and durations

    # 1) First non-wake sample = sleep onset
    sleep_mask = stage.isin(["N1", "N2", "N3", "REM"])
    if not sleep_mask.any():
        return feats
    sleep_onset = time[sleep_mask].iloc[0]

    # 2) First REM sample
    rem_mask = stage == "REM" 
    if not rem_mask.any():
        return feats
    first_rem = time[rem_mask].iloc[0] # Time of the first REM sample

    feats["rem_latency_min"] = round(float((first_rem - sleep_onset) / 60.0), 4) # feat: REM latency in minutes, rounded to 4 decimal places

    # 3)Count REM cycles = number of distinct REM runs
    rem_blocks = (rem_mask != rem_mask.shift()).cumsum()    # Identify consecutive blocks of REM and non-REM
    n_cycles   = int(rem_blocks[rem_mask].nunique())        # Count unique REM blocks to get number of REM cycles
    feats["n_rem_cycles"] = n_cycles                        # feat: Number of distinct REM periods (runs of consecutive REM samples)

    print(f"    REM latency: {feats['rem_latency_min']:.1f} min  |  REM cycles: {n_cycles}")
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
    Extract extra features from a single merged CSV.

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
    raw_stem    = merged_file.stem.replace(".csv", "")
    m           = _DCSM_PATTERN.match(raw_stem)
    sid         = subject_id if subject_id is not None else (m.group(1) if m else raw_stem)

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
        pattern:     str = "*_merged.csv",
        ) -> pd.DataFrame:
    """
    Run extract_extra_features on every merged CSV in a directory.

    Already-processed subjects are skipped (reads existing output_file).
    Results are appended and saved after every subject.

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
    merged_dir = Path(merged_dir)
    files      = sorted(merged_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {merged_dir}")

    if output_file is None:
        output_file = FEATURES_DIR / "extra_features.csv"
    output_file = Path(output_file)

    # Skip already-processed subjects
    existing_df  = pd.DataFrame()
    already_done = set()
    if output_file.exists():
        existing_df  = pd.read_csv(output_file, low_memory=False)
        if "subject_id" in existing_df.columns:
            already_done = set(existing_df["subject_id"].astype(str))
            print(f"  Found {len(already_done)} already-processed subjects in {output_file.name}")

    new_rows = []
    skipped  = 0
    for f in files:
        m   = _DCSM_PATTERN.match(f.stem.replace(".csv", ""))
        sid = m.group(1) if m else f.stem.replace(".csv", "")
        if sid in already_done:
            skipped += 1
            continue
        try:
            row = extract_extra_features(f, fs=fs)
            new_rows.append(row)
        except Exception as e:
            print(f"  [SKIP] {f.name} — {e}")

    if skipped:
        print(f"  Skipped {skipped} already-processed subjects")

    new_df = pd.DataFrame(new_rows)
    if not existing_df.empty and not new_df.empty:
        feature_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif not new_df.empty:
        feature_df = new_df
    else:
        feature_df = existing_df

    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_file, index=False)
    print(f"\nSaved → {output_file}  "
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