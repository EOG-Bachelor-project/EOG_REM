# Filename: bout_feats.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Phasic/Tonic bout-level features from merged CSV (output of merge_all).
#              A "bout" is a run of consecutive 4-second sub-epochs of the same type
#              (Phasic or Tonic) with no gaps or intervening epochs of a different type.

# =========================================================================================================
# Imports
# =========================================================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================================================================
# Constants
# =========================================================================================================
SUB_EPOCH_LEN_S = 4.0  # duration of each sub-epoch in seconds
FEATURES_DIR = Path("features_csv")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================================================================
# Helpers
# =========================================================================================================

def _rem_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Return only samples scored as REM sleep by GSSC."""
    return df[df["stage"] == "REM"].copy()


def _get_subepoch_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate the merged DataFrame to one row per 4-second sub-epoch
    inside REM sleep, returning a DataFrame with at least an 'EpochType' column.

    Uses em_SubEpochStart if available, otherwise falls back to time_sec binning.
    Mirrors the deduplication logic in eog_feats._phasic_tonic_features().
    """
    # 1) Filter to REM samples
    rem_df = _rem_samples(df)
    if rem_df.empty or "EpochType" not in rem_df.columns:
        return pd.DataFrame(columns=["EpochType"])

    # 2) Deduplicate to one row per sub-epoch
    if "em_SubEpochStart" in rem_df.columns:
        subepoch_df = rem_df.drop_duplicates(subset="em_SubEpochStart").copy()

    # Fallback: if em_SubEpochStart is not available, use time_sec binning to deduplicate
    else:
        subepoch_df = rem_df[rem_df["EpochType"].notna()].copy()
        subepoch_df["_epoch_bin"] = (subepoch_df["time_sec"] // SUB_EPOCH_LEN_S).astype(int)
        subepoch_df = subepoch_df.drop_duplicates(subset="_epoch_bin")

    return subepoch_df.reset_index(drop=True)


def _identify_bouts(
        types: pd.Series, 
        label: str
        ) -> list[int]:
    """
    Given a Series of EpochType values (in temporal order), identify bouts
    of consecutive sub-epochs matching ``label`` and return a list of bout
    lengths (in number of sub-epochs).

    Parameters
    ----------
    types : pd.Series
        Ordered EpochType values, e.g. ['Phasic', 'Phasic', 'Tonic', 'Phasic', ...].
    label : str
        The type to look for ('Phasic' or 'Tonic').

    Returns
    -------
    list[int]
        Length of each bout in number of sub-epochs.
    """
    bouts: list[int] = []
    current_len = 0

    # Iterate through the sequence of types and count consecutive runs of the target label
    for val in types:
        if val == label:                  # If the current sub-epoch matches the target label (e.g. 'Phasic') 
            current_len += 1              # Increment current bout length
        else:
            if current_len > 0:           # If in the middle of a bout and hit a different label, end the bout
                bouts.append(current_len) # Save the completed bout length
            current_len = 0

    # flush last bout
    if current_len > 0:
        bouts.append(current_len)

    return bouts


def _bout_stats(
        bouts: list[int], 
        label: str, 
        rem_min: float
        ) -> dict:
    """
    Compute summary statistics for a list of bout lengths (in sub-epochs).

    Parameters
    ----------
    bouts : list[int]
        List of bout lengths in number of sub-epochs.
    label : str
        Bout type label ('Phasic' or 'Tonic') for naming features.
    rem_min : float
        Total REM duration in minutes, used for rate calculation.
    
    Returns
    -------
    dict
        Dictionary of feature name → value for this bout type.
    """
    prefix = label.lower()
    feats: dict = {}

    if not bouts:
        for suffix in [
            "bout_count", "bout_mean_duration_s", "bout_max_duration_s",
            "bout_min_duration_s", "bout_std_duration_s", "bout_median_duration_s",
            "bout_rate_per_min", "bout_total_duration_s",
        ]:
            feats[f"{prefix}_{suffix}"] = np.nan
        return feats

    durations_s = np.array(bouts) * SUB_EPOCH_LEN_S

    feats[f"{prefix}_bout_count"]             = len(bouts)
    feats[f"{prefix}_bout_mean_duration_s"]   = round(float(durations_s.mean()), 4)
    feats[f"{prefix}_bout_max_duration_s"]    = round(float(durations_s.max()), 4)
    feats[f"{prefix}_bout_min_duration_s"]    = round(float(durations_s.min()), 4)
    feats[f"{prefix}_bout_std_duration_s"]    = round(float(durations_s.std(ddof=1)), 4) if len(bouts) > 1 else 0.0
    feats[f"{prefix}_bout_median_duration_s"] = round(float(np.median(durations_s)), 4)
    feats[f"{prefix}_bout_rate_per_min"]      = round(len(bouts) / rem_min, 4) if rem_min > 0 else np.nan
    feats[f"{prefix}_bout_total_duration_s"]  = round(float(durations_s.sum()), 4)

    return feats


# =========================================================================================================
# Main extraction function
# =========================================================================================================

def phasic_tonic_bout_features(df: pd.DataFrame, fs: float = 250.0) -> dict:
    """
    Bout-level features for Phasic and Tonic REM sub-epochs.

    A **bout** is a run of consecutive 4-second sub-epochs classified as the
    same type (Phasic or Tonic), with no gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged DataFrame (output of merge_all) containing at minimum
        ``time_sec``, ``stage``, and ``EpochType`` columns.
    fs : float
        Sampling frequency of the EOG signal in [Hz]. Default is **250.0 Hz**.

    Returns
    -------
    dict
        Flat dictionary of feature name → value.

    Features (16 total — 8 per type)
    --------
    For each of {phasic, tonic}:
        ``{type}_bout_count``             : Number of bouts.
        ``{type}_bout_mean_duration_s``   : Mean bout duration [seconds].
        ``{type}_bout_max_duration_s``    : Longest bout [seconds].
        ``{type}_bout_min_duration_s``    : Shortest bout [seconds].
        ``{type}_bout_std_duration_s``    : Std of bout durations [seconds].
        ``{type}_bout_median_duration_s`` : Median bout duration [seconds].
        ``{type}_bout_rate_per_min``      : Bouts per minute of total REM sleep.
        ``{type}_bout_total_duration_s``  : Total time in bouts [seconds].
    """
    feats: dict = {}

    # ---- 1) Compute REM duration ----
    rem_df = _rem_samples(df)
    rem_min = len(rem_df) / fs / 60.0
    print(f"    REM duration: {rem_min:.2f} [min]  ({len(rem_df):,} samples)")

    # ---- 2) Get deduplicated sub-epoch sequence ----
    subepoch_df = _get_subepoch_series(df)

    if subepoch_df.empty:
        print("    No sub-epochs found — returning NaN defaults")
        for label in ["phasic", "tonic"]:
            feats.update(_bout_stats([], label.capitalize(), rem_min))
        return feats

    types = subepoch_df["EpochType"]
    print(f"    Sub-epochs: {len(types):,}  |  types: {types.value_counts().to_dict()}")

    # ---- 3) Identify bouts and compute stats ----
    for label in ["Phasic", "Tonic"]:
        bouts = _identify_bouts(types, label)
        feats.update(_bout_stats(bouts, label, rem_min))

        if bouts:
            dur_arr = np.array(bouts) * SUB_EPOCH_LEN_S
            print(
                f"    {label} bouts: {len(bouts)}  |  "
                f"mean: {dur_arr.mean():.1f} s  |  "
                f"max: {dur_arr.max():.1f} s  |  "
                f"min: {dur_arr.min():.1f} s"
            )
        else:
            print(f"    {label} bouts: 0")

    # ---- 4) Phasic <-> Tonic transitions ----
    transitions = (types != types.shift()).sum() - 1  # subtract 1 to get number of transitions not boundaries
    feats["phasic_tonic_transitions"] = int(transitions) if transitions >= 0 else 0
    print(f"    Phasic <-> Tonic transitions: {feats['phasic_tonic_transitions']}")

    return feats


# =========================================================================================================
# Batch extraction
# =========================================================================================================

def extract_bout_features(
        merged_file: str | Path,
        subject_id:  str | None = None,
        fs:          float = 250.0,
        ) -> dict:
    """
    Extract phasic/tonic bout features from a single merged CSV.

    Parameters
    ----------
    merged_file : str | Path
        Path to the merged CSV file for one subject/session.
    subject_id : str | None
        Optional subject identifier. If None, the file stem is used.
    fs : float
        Sampling frequency in [Hz]. Default is **250.0 Hz**.

    Returns
    -------
    dict
        Flat dictionary of feature name → value for this subject.
    """
    merged_file = Path(merged_file)
    sid = subject_id if subject_id is not None else merged_file.stem

    print(f"\n{'=' * 60}")
    print(f"Extracting bout features: {merged_file.name}")
    print(f"  subject_id : {sid}  |  fs : {fs} [Hz]")

    df = pd.read_csv(merged_file, low_memory=False)

    feats: dict = {"subject_id": sid}

    print(f"\n--- Phasic / Tonic bout features ---")
    feats.update(phasic_tonic_bout_features(df, fs))

    n_nan = sum(1 for v in feats.values() if isinstance(v, float) and np.isnan(v))
    print(f"\n  Features computed : {len(feats) - 1}")
    print(f"  NaN features      : {n_nan}")

    return feats


def extract_bout_features_batch(
        merged_dir:  str | Path,
        output_file: str | Path | None = None,
        fs:          float = 250.0,
        pattern:     str = "*_merged.csv",
        ) -> pd.DataFrame:
    """
    Run ``extract_bout_features`` on every merged CSV in a directory.

    Parameters
    ----------
    merged_dir : str | Path
        Directory containing merged CSV files (output of merge_all).
    output_file : str | Path | None
        If provided, save the feature DataFrame to this CSV path.
    fs : float
        Sampling frequency. Default is **250.0 Hz**.
    pattern : str
        Glob pattern to match merged CSVs. Default is ``'*_merged.csv'``.

    Returns
    -------
    pd.DataFrame
        One row per subject with all extracted bout features.
    """
    merged_dir = Path(merged_dir)
    files = sorted(merged_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {merged_dir}"
        )

    # ---- Load existing CSV to skip already-processed subjects ----
    if output_file is None:
        output_file = FEATURES_DIR / "bout_features.csv"
    output_file = Path(output_file)

    existing_df = pd.DataFrame()
    already_done = set()
    if output_file.exists():
        existing_df = pd.read_csv(output_file, low_memory=False)
        if "subject_id" in existing_df.columns:
            already_done = set(existing_df["subject_id"].astype(str).values)
            print(f"  Found {len(already_done)} already-processed subjects in {output_file.name}")

    # ---- Process only new subjects ----
    new_rows = []
    skipped = 0
    for f in files:
        sid = f.stem
        if sid in already_done:
            skipped += 1
            continue
        try:
            row = extract_bout_features(f, fs=fs)
            new_rows.append(row)
        except Exception as e:
            print(f"  [SKIP] {f.name} — {e}")

    if skipped:
        print(f"  Skipped {skipped} already-processed subjects")

    # ---- Concat old + new ----
    new_df = pd.DataFrame(new_rows)
    if not existing_df.empty and not new_df.empty:
        feature_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif not new_df.empty:
        feature_df = new_df
    else:
        feature_df = existing_df

    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_file, index=False)
    print(f"\nFeature table saved -> {output_file}  "
          f"({feature_df.shape[0]} subjects, {feature_df.shape[1]-1} features)")

    return feature_df