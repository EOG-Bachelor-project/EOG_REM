# Filename: eog_features.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Simple EOG feature extraction from a merged CSV file (output of merge_all).

# =========================================================================================================
# Imports
# =========================================================================================================
from __future__ import annotations # for type hinting of class methods that return instances of the class itself
 
import numpy as np                 # for numerical operations
import pandas as pd                # for data manipulation and analysis
from pathlib import Path           # for handling file paths
 
# =========================================================================================================
# Constants
# =========================================================================================================
FEATURES_DIR = Path("features_csv")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
 
# =========================================================================================================
# Helper
# =========================================================================================================

def _load_and_validate(merged_file: Path) -> pd.DataFrame:
    """Load merged CSV and check required columns are present."""
    required = {"time_sec", "LOC", "ROC", "stage", "is_rem_event", "is_em_event", "EM_Type"}
    df = pd.read_csv(merged_file, low_memory=False)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Merged CSV is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
    return df
 
 
def _rem_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Return only samples scored as REM sleep by GSSC."""
    return df[df["stage"] == "REM"].copy()

# =========================================================================================================
# Feature groups
# =========================================================================================================

def _stage_distribution_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Distribution of sleep stage derived from GSSC staging.

    Features
    --------
    `total_recording_min`       : Total recording duration in minutes.
    `rem_duration_min`          : Total time scored as REM sleep in minutes.
    `rem_fraction`              : REM duration / total recording duration.
    `n_rem_epochs`              : Number of distinct consecutive REM epochs (GSSC-level).
    `stage_frac_W/N1/N2/N3`     : Fraction of recording in each non-REM stage.
    """
    # ---- 1) Calculate total recording duration in minutes ----
    n_total = len(df)
    total_min = n_total / fs / 60.0

    print(f"    Total recording: {total_min:.2f} [min]  ({n_total:,} samples at {fs} [Hz])")
    
    # ---- 2) Calculate duration and fraction of each stage ----
    feats: dict = {"total_recording_min": round(total_min, 3)}

    stages = ["W", "N1", "N2", "N3", "REM"]
    for s in stages:
        n_s     = (df["stage"] == s).sum()
        dur_min = n_s / fs / 60.0
        frac    = n_s / n_total if n_total > 0 else np.nan
        label   = "rem" if s == "REM" else s.lower()
        feats[f"{label}_duration_min"] = round(dur_min, 3)
        feats[f"{label}_fraction"]     = round(frac, 4)
 
    print(f"    Stages - " + "  |  ".join([f"{s}: {round((df['stage']==s).sum()/n_total*100,1)}%" for s in stages]))
    

    # ---- 3) Count distinct consecutive REM blocks ----
    # NOTE:
    #       Proxy for number of REM episodes, but depends on how GSSC scores REM (e.g. minimum duration for a REM episode).
    #       If GSSC has a minimum duration for REM episodes, this will underestimate the true number of REM episodes,
    #       but is still informative about the structure of REM sleep in the recording.
    stage_blocks = (df["stage"] != df["stage"].shift()).cumsum()
    n_rem_epochs = int(
        df[df["stage"] == "REM"].groupby(stage_blocks)["stage"].count().shape[0]
    )
    feats["n_rem_epochs"] = n_rem_epochs
    print(f"    Distinct REM epochs: {n_rem_epochs}")
 
    return feats

#——————————————————————————————————————————————————————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————————————————————————————————————————————————————

def _rem_epoch_duration_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Duration statistics for each individual REM epoch (consecutive REM blocks).

    Features
    --------
    `rem_epoch_count`             : Number of distinct REM epochs.
    `rem_epoch_mean_duration_min` : Mean REM epoch duration [minutes].
    `rem_epoch_std_duration_min`  : Std of REM epoch durations [minutes].
    `rem_epoch_min_duration_min`  : Shortest REM epoch duration [minutes].
    `rem_epoch_max_duration_min`  : Longest REM epoch duration [minutes].
    """

    # ---- 1) Identify consecutive REM blocks ----
    is_rem = (df["stage"] == "REM").astype(int)
    epoch_ids = (is_rem.diff().fillna(0) != 0).cumsum()
    rem_blocks = df[is_rem == 1].groupby(epoch_ids).size()
    rem_duration_min = rem_blocks / fs / 60.0

    feats: dict = {}

    # ---- 2) Return NaN defaults if no REM found ----
    if rem_duration_min.empty:
        print("    No REM epochs found — returning NaN defaults")
        for k in ["rem_epoch_count", "rem_epoch_mean_duration_min",
                  "rem_epoch_std_duration_min", "rem_epoch_min_duration_min",
                  "rem_epoch_max_duration_min"]:
            feats[k] = np.nan
        return feats

    # ---- 3) Compute duration statistics ----
    feats["rem_epoch_count"]             = int(len(rem_duration_min))
    feats["rem_epoch_mean_duration_min"] = round(float(rem_duration_min.mean()), 3)
    feats["rem_epoch_std_duration_min"]  = round(float(rem_duration_min.std()), 3)
    feats["rem_epoch_min_duration_min"]  = round(float(rem_duration_min.min()), 3)
    feats["rem_epoch_max_duration_min"]  = round(float(rem_duration_min.max()), 3)

    print(f"    REM epochs: {feats['rem_epoch_count']}  |  "
          f"mean: {feats['rem_epoch_mean_duration_min']} [min]  |  "
          f"std: {feats['rem_epoch_std_duration_min']}  |  "
          f"min: {feats['rem_epoch_min_duration_min']}  |  "
          f"max: {feats['rem_epoch_max_duration_min']}")

    return feats

#——————————————————————————————————————————————————————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————————————————————————————————————————————————————

def _rem_event_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Features derived from detected REM eye movement events (extract_rems_n.py output), computed only over samples scored as REM sleep.
 
    Features
    --------
    `rem_event_count`                   : Total number of distinct REM events in REM sleep.
    `rem_event_rate_per_min`            : REM events per minute of REM sleep.
    `rem_event_mean_duration_s`         : Mean event duration in seconds (if event_Duration present).
    `rem_event_median_duration_s`       : Median event duration in seconds.
    `rem_event_mean_loc_amp_uv`         : Mean LOC absolute peak amplitude across all events [µV].
    `rem_event_mean_roc_amp_uv`         : Mean ROC absolute peak amplitude across all events [µV].
    `rem_event_mean_loc_rise_slope`     : Mean LOC absolute rise slope [µV/s].
    `rem_event_mean_roc_rise_slope`     : Mean ROC absolute rise slope [µV/s].
    """

    # ---- 1) Filter to REM sleep samples and compute duration ----
    rem_df = _rem_samples(df)
    rem_min = len(rem_df) / fs / 60.0
    print(f"    REM duration: {rem_min:.2f} [min]  ({len(rem_df):,} samples)")
    
    feats: dict = {}
    
    # ---- 2) Return NaN defaults if no REM data or event column is missing ----
    if rem_df.empty or "is_rem_event" not in rem_df.columns:
        feats["rem_event_count"]           = 0
        feats["rem_event_rate_per_min"]    = np.nan
        feats["rem_event_mean_duration_s"] = np.nan
        feats["rem_event_median_duration_s"] = np.nan
        feats["rem_event_mean_loc_amp_uv"] = np.nan
        feats["rem_event_mean_roc_amp_uv"] = np.nan
        feats["rem_event_mean_loc_rise_slope"] = np.nan
        feats["rem_event_mean_roc_rise_slope"] = np.nan
        return feats
    
    # ---- 3) Deduplicate to one row per event ----
    if "event_Peak" in rem_df.columns:
        n_events   = int(rem_df["event_Peak"].dropna().nunique())
        event_meta = (
            rem_df[rem_df["event_Peak"].notna()]
            .drop_duplicates(subset="event_Peak")
        )
    else:
        is_event   = (rem_df["is_rem_event"] == True).astype(int)
        n_events   = int(((is_event.diff() == 1) | (is_event.iloc[:1] == 1)).sum())
        event_meta = pd.DataFrame()

    print(f"    Distinct REM events: {n_events}  |  rate: {round(n_events / rem_min, 4) if rem_min > 0 else 'N/A'} [1/min]")
    
    # ---- 4) Compute event count and rate ----
    feats["rem_event_count"]        = n_events
    feats["rem_event_rate_per_min"] = round(n_events / rem_min, 4) if rem_min > 0 else np.nan
 
    # ---- 5) Compute duration statistics ----
    if "event_Duration" in rem_df.columns and not event_meta.empty:
        dur = event_meta["event_Duration"].dropna()
        feats["rem_event_mean_duration_s"]   = round(float(dur.mean()), 4) if len(dur) else np.nan
        feats["rem_event_median_duration_s"] = round(float(dur.median()), 4) if len(dur) else np.nan
    else:
        feats["rem_event_mean_duration_s"]   = np.nan
        feats["rem_event_median_duration_s"] = np.nan
    
    print(f"    Duration — mean: {feats['rem_event_mean_duration_s']} [s]  |  median: {feats['rem_event_median_duration_s']} [s]")
 
    # ---- 6) Compute mean peak amplitude per channel ----
    for ch, col in [("loc", "event_LOCAbsValPeak"), ("roc", "event_ROCAbsValPeak")]:
        if col in rem_df.columns and not event_meta.empty:
            vals = event_meta[col].dropna()
            feats[f"rem_event_mean_{ch}_amp_uv"] = round(float(vals.mean()), 4) if len(vals) else np.nan
        else:
            feats[f"rem_event_mean_{ch}_amp_uv"] = np.nan
    
    print(f"    Amplitude — LOC: {feats['rem_event_mean_loc_amp_uv']} [µV]  |  ROC: {feats['rem_event_mean_roc_amp_uv']} [µV]")
 
    # ---- 7) Compute mean rise slope per channel ----
    for ch, col in [("loc", "event_LOCAbsRiseSlope"), ("roc", "event_ROCAbsRiseSlope")]:
        if col in rem_df.columns and not event_meta.empty:
            vals = event_meta[col].dropna()
            feats[f"rem_event_mean_{ch}_rise_slope"] = round(float(vals.mean()), 4) if len(vals) else np.nan
        else:
            feats[f"rem_event_mean_{ch}_rise_slope"] = np.nan

    print(f"    Rise slope  — LOC: {feats['rem_event_mean_loc_rise_slope']} [µV/s]  |  ROC: {feats['rem_event_mean_roc_rise_slope']} [µV/s]")
 
    return feats

#——————————————————————————————————————————————————————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————————————————————————————————————————————————————

def _em_classification_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Features derived from EM type classifications (SEM / REM) in REM sleep.
 
    Features
    --------
    `sem_count_rem_sleep`         : Number of SEMs during REM sleep.
    `rem_em_count_rem_sleep`      : Number of rapid EMs during REM sleep.
    `sem_rate_per_min`            : SEMs per minute of REM sleep.
    `rem_em_rate_per_min`         : Rapid EMs per minute of REM sleep.
    `sem_fraction`                : SEMs / (SEMs + rapid EMs). Indicator of slow-eye-movement dominance.
    `rem_em_fraction`             : Rapid EMs / total EMs.
    `sem_mean_duration_s`         : Mean SEM duration in seconds.
    `rem_em_mean_duration_s`      : Mean rapid EM duration in seconds.
    `sem_mean_amp_uv`             : Mean SEM peak amplitude (mean of LOC+ROC) in [µV].
    `rem_em_mean_amp_uv`          : Mean rapid EM peak amplitude in [µV].
    """

    # ---- 1) Filter to REM sleep samples and compute duration ----
    rem_df = _rem_samples(df)
    rem_min = len(rem_df) / fs / 60.0

    print(f"    REM duration: {rem_min:.2f} [min]  ({len(rem_df):,} samples)")

    feats: dict = {}
    
    # ---- 2) Return NaN defaults if no REM data or EM_Type column is missing ----
    if rem_df.empty or "EM_Type" not in rem_df.columns:
        for k in [
            "sem_count_rem_sleep", "rem_em_count_rem_sleep",
            "sem_rate_per_min", "rem_em_rate_per_min",
            "sem_fraction", "rem_em_fraction",
            "sem_mean_duration_s", "rem_em_mean_duration_s",
            "sem_mean_amp_uv", "rem_em_mean_amp_uv",
        ]:
            feats[k] = np.nan
        return feats
    
    # ---- 3) Deduplicate to one row per EM event ----
    if "em_event_id" in rem_df.columns:
        em_events = (
            rem_df[rem_df["em_event_id"].notna()]
            .drop_duplicates(subset="em_event_id")
        )
    else:
        em_events = rem_df[rem_df["is_em_event"] == True]
 
    sem_df     = em_events[em_events["EM_Type"] == "SEM"]
    rem_em_df  = em_events[em_events["EM_Type"] == "REM"]
    n_sem      = len(sem_df)
    n_rem_em   = len(rem_em_df)
    n_total_em = n_sem + n_rem_em

    print(f"    EM events — SEM: {n_sem}  |  REM: {n_rem_em}  |  total: {n_total_em}")
    
    # ---- 4) Compute counts, rates and fractions ----
    feats["sem_count_rem_sleep"]    = n_sem
    feats["rem_em_count_rem_sleep"] = n_rem_em
    feats["sem_rate_per_min"]       = round(n_sem / rem_min, 4)    if rem_min > 0 else np.nan
    feats["rem_em_rate_per_min"]    = round(n_rem_em / rem_min, 4) if rem_min > 0 else np.nan
    feats["sem_fraction"]           = round(n_sem / n_total_em, 4) if n_total_em > 0 else np.nan
    feats["rem_em_fraction"]        = round(n_rem_em / n_total_em, 4) if n_total_em > 0 else np.nan

    print(f"    Rates — SEM: {feats['sem_rate_per_min']} [1/min]  |  REM EM: {feats['rem_em_rate_per_min']} [1/min]  |  SEM fraction: {feats['sem_fraction']}")

    # ---- 5) Compute mean duration per EM type ----
    for label, sub_df in [("sem", sem_df), ("rem_em", rem_em_df)]:
        dur_col = "em_Duration"
        if dur_col in sub_df.columns and not sub_df.empty:
            vals = sub_df[dur_col].dropna()
            feats[f"{label}_mean_duration_s"] = round(float(vals.mean()), 4) if len(vals) else np.nan
        else:
            feats[f"{label}_mean_duration_s"] = np.nan

    print(f"    Duration — SEM: {feats['sem_mean_duration_s']} [s]  |  REM EM: {feats['rem_em_mean_duration_s']} [s]")
 
    # ---- 6) Compute mean amplitude per EM type ----
    # NOTE:
    #       Uses em_MeanAbsValPeak if available, otherwise averages LOC and ROC peak amplitudes.
    for label, sub_df in [("sem", sem_df), ("rem_em", rem_em_df)]:
        if "em_MeanAbsValPeak" in sub_df.columns and not sub_df.empty:
            vals = sub_df["em_MeanAbsValPeak"].dropna()
            feats[f"{label}_mean_amp_uv"] = round(float(vals.mean()), 4) if len(vals) else np.nan
        elif "em_LOCAbsValPeak" in sub_df.columns and "em_ROCAbsValPeak" in sub_df.columns:
            mean_amp = ((sub_df["em_LOCAbsValPeak"] + sub_df["em_ROCAbsValPeak"]) / 2).dropna()
            feats[f"{label}_mean_amp_uv"] = round(float(mean_amp.mean()), 4) if len(mean_amp) else np.nan
        else:
            feats[f"{label}_mean_amp_uv"] = np.nan

    print(f"    Amplitude — SEM: {feats['sem_mean_amp_uv']} [µV]  |  REM EM: {feats['rem_em_mean_amp_uv']} [µV]")
 
    return feats

#——————————————————————————————————————————————————————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————————————————————————————————————————————————————

def _em_stage_count_features(df: pd.DataFrame) -> dict:
    """
    Total EM counts broken down by sleep stage across the full recording.

    Features
    --------
    `em_count_n1`     : Total EM count during N1 sleep.
    `em_count_n2`     : Total EM count during N2 sleep.
    `em_count_n3`     : Total EM count during N3 sleep.
    `em_count_rem`    : Total EM count during REM sleep.
    `em_count_wake`   : Total EM count during Wake.
    """

    feats: dict = {}

    # ---- 1) Return NaN defaults if required columns are missing ----
    if "stage" not in df.columns or "EM_Type" not in df.columns:
        print("    'stage' or 'EM_Type' column missing — returning NaN defaults")
        for k in ["em_count_n1", "em_count_n2", "em_count_n3", "em_count_rem", "em_count_wake"]:
            feats[k] = np.nan
        return feats

    # ---- 2) Deduplicate to one row per EM event ----
    if "em_event_id" in df.columns:
        em_events = df[df["em_event_id"].notna()].drop_duplicates(subset="em_event_id")
    elif "em_is_peak" in df.columns:
        em_events = df[df["em_is_peak"] == True]
    else:
        em_events = df[df["EM_Type"].notna()]

    em_count_total = len(em_events)

    # ---- 3) Count EM events per sleep stage ----
    stage_col = em_events["stage"]
    stage_map = {
        "em_count_n1":   stage_col == "N1",
        "em_count_n2":   stage_col == "N2",
        "em_count_n3":   stage_col == "N3",
        "em_count_rem":  stage_col == "REM",
        "em_count_wake": stage_col == "W",
    }
    for feat_key, mask in stage_map.items():
        feats[feat_key] = int(mask.sum())

    print(f"    Stage counts — "
          f"N1: {feats['em_count_n1']} ({round(feats['em_count_n1'] / em_count_total * 100, 1) if em_count_total > 0 else 'N/A'}%)  |  "
          f"N2: {feats['em_count_n2']} ({round(feats['em_count_n2'] / em_count_total * 100, 1) if em_count_total > 0 else 'N/A'}%)  |  "
          f"N3: {feats['em_count_n3']} ({round(feats['em_count_n3'] / em_count_total * 100, 1) if em_count_total > 0 else 'N/A'}%)  |  "
          f"REM: {feats['em_count_rem']} ({round(feats['em_count_rem'] / em_count_total * 100, 1) if em_count_total > 0 else 'N/A'}%)  |  "
          f"Wake: {feats['em_count_wake']} ({round(feats['em_count_wake'] / em_count_total * 100, 1) if em_count_total > 0 else 'N/A'}%)")

    # ---- 4) Sanity check - staged sum should equal total EM events ----
    em_count_staged = sum(feats[k] for k in stage_map)
    print(f"    Sanity check - total EM events: {em_count_total}  |  staged sum: {em_count_staged}")

    return feats

#——————————————————————————————————————————————————————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————————————————————————————————————————————————————

def _phasic_tonic_features(df: pd.DataFrame) -> dict:
    """
    Features derived from Phasic / Tonic sub-epoch classification (EpochType column).
    Only meaningful if the Umaer sub-epoch classifier was run (subepochs_file passed to merge_all).
 
    Features
    --------
    `phasic_epoch_count`      : Number of 4-second sub-epochs classified as Phasic.
    `tonic_epoch_count`       : Number of 4-second sub-epochs classified as Tonic.
    `phasic_fraction`         : Phasic / (Phasic + Tonic). Key RBD biomarker candidate.
    `tonic_fraction`          : Tonic / (Phasic + Tonic).
    """
    
    # ---- 1) Return NaN defaults if EpochType column is missing ----
    feats: dict = {}

    if "EpochType" not in df.columns:
        print("    'EpochType' column missing — returning NaN defaults")
        for k in ["phasic_epoch_count", "tonic_epoch_count", "phasic_fraction", "tonic_fraction"]:
            feats[k] = np.nan
        return feats
    
    # ---- 2) Filter to REM sleep samples ----
    rem_df = _rem_samples(df)
    print(f"    REM samples: {len(rem_df):,}")

    # ---- 3) Deduplicate to one row per sub-epoch ----
    # NOTE:
    #       Uses em_SubEpochStart to identify distinct sub-epochs if available,
    #       otherwise falls back to counting consecutive EpochType blocks.
    if "em_SubEpochStart" in rem_df.columns:
        subepoch_df = rem_df.drop_duplicates(subset="em_SubEpochStart")
    else:
        subepoch_df = rem_df[rem_df["EpochType"].notna()].copy()
        subepoch_df = subepoch_df[
            subepoch_df["EpochType"] != subepoch_df["EpochType"].shift()
        ]

    print(f"    Sub-epochs found: {len(subepoch_df):,}  |  types: {subepoch_df['EpochType'].value_counts().to_dict()}")

    # ---- 4) Compute counts and fractions ----
    n_phasic = int((subepoch_df["EpochType"] == "Phasic").sum())
    n_tonic  = int((subepoch_df["EpochType"] == "Tonic").sum())
    n_total  = n_phasic + n_tonic

    feats["phasic_epoch_count"] = n_phasic
    feats["tonic_epoch_count"]  = n_tonic
    feats["phasic_fraction"]    = round(n_phasic / n_total, 4) if n_total > 0 else np.nan
    feats["tonic_fraction"]     = round(n_tonic  / n_total, 4) if n_total > 0 else np.nan

    print(f"    Phasic: {n_phasic}  |  Tonic: {n_tonic}  |  Phasic fraction: {feats['phasic_fraction']}")

    return feats

#——————————————————————————————————————————————————————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————————————————————————————————————————————————————

def _eog_amplitude_features(df: pd.DataFrame) -> dict:
    """
    Simple amplitude statistics of the raw LOC and ROC signals during REM sleep.
    These are signal-level summary features — no windowing or frequency analysis.
 
    Features
    --------
    `rem_loc_mean_abs_uv`     : Mean |LOC| amplitude during REM sleep [µV].
    `rem_roc_mean_abs_uv`     : Mean |ROC| amplitude during REM sleep [µV].
    `rem_loc_std_uv`          : Std of LOC signal during REM sleep [µV].
    `rem_roc_std_uv`          : Std of ROC signal during REM sleep [µV].
    `rem_loc_p95_uv`          : 95th percentile of |LOC| during REM sleep [µV].
    `rem_roc_p95_uv`          : 95th percentile of |ROC| during REM sleep [µV].
    """

    # ---- 1) Filter to REM sleep samples ----
    rem_df = _rem_samples(df)

    print(f"    REM samples: {len(rem_df):,}")

    feats: dict = {}

    # ---- 2) Compute amplitude statistics per channel ----
    for ch in ["LOC", "ROC"]:
        label = ch.lower()
        if ch in rem_df.columns and not rem_df.empty:
            sig = rem_df[ch].dropna()
            feats[f"rem_{label}_mean_abs_uv"] = round(float(sig.abs().mean()), 4)
            feats[f"rem_{label}_std_uv"]      = round(float(sig.std()), 4)
            feats[f"rem_{label}_p95_uv"]      = round(float(sig.abs().quantile(0.95)), 4)
        else:
            feats[f"rem_{label}_mean_abs_uv"] = np.nan
            feats[f"rem_{label}_std_uv"]      = np.nan
            feats[f"rem_{label}_p95_uv"]      = np.nan

        print(f"    {ch} — mean: {feats[f'rem_{label}_mean_abs_uv']} [µV]  |  std: {feats[f'rem_{label}_std_uv']} [µV]  |  p95: {feats[f'rem_{label}_p95_uv']} [µV]")

    return feats

 
# =========================================================================================================
# =========================================================================================================
# Main extraction function
# =========================================================================================================
# =========================================================================================================

def extract_features(
        merged_file: str | Path,
        subject_id:  str | None = None,
        fs:          float = 250.0,
        ) -> dict:
    """
    Extract all simple EOG features from a single merged CSV file.
 
    The merged CSV must be the output of ``merge_all()`` and contain at minimum:
    ``time_sec``, ``LOC``, ``ROC``, ``stage``, ``is_rem_event``, ``is_em_event``, ``EM_Type``.
 
    Parameters
    ----------
    merged_file : str | Path
        Path to the merged CSV file for one subject/session.
    subject_id : str | None
        Optional subject identifier included in the returned dict. 
        If None, the file stem is used.
    fs : float
        Sampling frequency of the EOG signal in [Hz]. Default is **250.0 Hz**.
 
    Returns
    -------
    feats : dict
        Flat dictionary of feature name → value for this subject.
    """
    merged_file = Path(merged_file)
    sid = subject_id if subject_id is not None else merged_file.stem
 
    print(f"\n{'=' * 60}")
    print(f"Extracting features: {merged_file.name}")
    print(f"  subject_id : {sid}  |  fs : {fs} [Hz]")
 
    df = _load_and_validate(merged_file)
 
    feats: dict = {"subject_id": sid}
 
    print(f"\n--- Stage distribution ---")
    feats.update(_stage_distribution_features(df, fs))

    print(f"\n--- REM epoch duration ---")         
    feats.update(_rem_epoch_duration_features(df, fs)) 

    print(f"\n--- REM event features ---")
    feats.update(_rem_event_features(df, fs))
 
    print(f"\n--- EM classification features ---")
    feats.update(_em_classification_features(df, fs))

    print(f"\n--- EM stage count features ---")      
    feats.update(_em_stage_count_features(df))        
 
    print(f"\n--- Phasic / Tonic features ---")
    feats.update(_phasic_tonic_features(df))
 
    print(f"\n--- EOG amplitude features ---")
    feats.update(_eog_amplitude_features(df))
 
    n_nan = sum(1 for v in feats.values() if isinstance(v, float) and np.isnan(v))
    print(f"\n  Features computed : {len(feats) - 1}")
    print(f"  NaN features      : {n_nan}")
 
    return feats
 
# =========================================================================================================
# Batch extraction
# =========================================================================================================

def extract_features_batch(
        merged_dir:  str | Path,
        output_file: str | Path | None = None,
        fs:          float = 250.0,
        pattern:     str = "*_merged.csv",
        ) -> pd.DataFrame:
    """
    Run ``extract_features`` on every merged CSV in a directory and
    collect results into a single DataFrame.
 
    Parameters
    ----------
    merged_dir : str | Path
        Directory containing merged CSV files (output of merge_all).
    output_file : str | Path | None
        If provided, save the feature DataFrame to this CSV path.
        Default saves to ``features_csv/features.csv``.
    fs : float
        Sampling frequency. Default is **250.0 Hz**.
    pattern : str
        Glob pattern to match merged CSVs. Default is ``'*_merged.csv'``.
 
    Returns
    -------
    pd.DataFrame
        One row per subject with all extracted features.
    """
    
    merged_dir = Path(merged_dir)
    files = sorted(merged_dir.glob(pattern))
 
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {merged_dir}"
        )
 
    print(f"\nFound {len(files)} merged CSV(s) in {merged_dir}")
 
    rows = []
    for f in files:
        try:
            row = extract_features(f, fs=fs)
            rows.append(row)
        except Exception as e:
            print(f"  [SKIP] {f.name} — {e}")
 
    feature_df = pd.DataFrame(rows)
 
    if output_file is None:
        output_file = FEATURES_DIR / "features.csv"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_file, index=False)
    print(f"\nFeature table saved → {output_file}  ({feature_df.shape[0]} subjects, {feature_df.shape[1]-1} features)")
 
    return feature_df
 
 
# =========================================================================================================
# Entry point
# =========================================================================================================
 
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print("Usage: python eog_features.py <merged_csv_or_dir> [fs]")
        sys.exit(1)
 
    path = Path(sys.argv[1])
    sampling_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
 
    if path.is_dir():
        df_out = extract_features_batch(path, fs=sampling_freq)
        print(df_out.to_string(index=False))
    elif path.is_file():
        feats = extract_features(path, fs=sampling_freq)
        for k, v in feats.items():
            print(f"  {k:<45} {v}")
    else:
        print(f"Path not found: {path}")
        sys.exit(1)