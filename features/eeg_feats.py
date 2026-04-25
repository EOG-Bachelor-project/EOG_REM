# Filename: eeg_feats.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: EEG feature extraction from a merged CSV file (output of merge_all).
#              Extracts band power features per sleep stage using Welch's method.

# =========================================================================================================
# Imports
# =========================================================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch

# =========================================================================================================
# Constants
# =========================================================================================================
EEG_BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}
EEG_COLS = ["EEG_LOC", "EEG_ROC"]
STAGES   = ["W", "N1", "N2", "N3", "REM"]

# =========================================================================================================
# Helper
# =========================================================================================================
def _load_and_validate(merged_file: Path) -> pd.DataFrame:
    """Load merged CSV and check required columns are present."""
    required = {"time_sec", "stage", "EEG_LOC", "EEG_ROC"}
    df = pd.read_csv(merged_file, low_memory=False)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Merged CSV is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
    return df

# =========================================================================================================
# Feature groups
# =========================================================================================================

def _eeg_band_power_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Band power features per EEG channel per sleep stage using Welch's method.

    Features (per channel × stage × band)
    --------
    `eeg_{ch}__{stage}__delta`       : Delta band power [µV²/Hz].
    `eeg_{ch}__{stage}__theta`       : Theta band power [µV²/Hz].
    `eeg_{ch}__{stage}__alpha`       : Alpha band power [µV²/Hz].
    `eeg_{ch}__{stage}__beta`        : Beta band power [µV²/Hz].
    `eeg_{ch}__{stage}__total`       : Total band power (sum of all bands) [µV²/Hz].
    `eeg_{ch}__{stage}__theta_ratio` : Theta / total power ratio.
    """
    feats: dict = {}
    nperseg     = int(4.0 * fs)
    min_samples = int(10.0 * fs)

    for col in EEG_COLS:
        if col not in df.columns:
            print(f"    {col} not found — skipping")
            continue

        ch = col.lower()  # eeg_loc or eeg_roc

        for stage in STAGES:
            mask = df["stage"] == stage
            n_samples = mask.sum()

            if n_samples < min_samples:
                print(f"    {col} | {stage}: only {n_samples} samples (< {min_samples}) — filling NaN")
                for band in EEG_BANDS:
                    feats[f"{ch}__{stage.lower()}__{band}"] = np.nan
                feats[f"{ch}__{stage.lower()}__total"]       = np.nan
                feats[f"{ch}__{stage.lower()}__theta_ratio"] = np.nan
                continue

            sig = df.loc[mask, col].dropna().values
            f, psd = welch(sig, fs=fs, nperseg=min(nperseg, len(sig)))

            total = 0.0
            for band, (lo, hi) in EEG_BANDS.items():
                band_mask = (f >= lo) & (f <= hi)
                power = float(np.trapz(psd[band_mask], f[band_mask]))
                feats[f"{ch}__{stage.lower()}__{band}"] = round(power, 6)
                total += power

            feats[f"{ch}__{stage.lower()}__total"] = round(total, 6)
            feats[f"{ch}__{stage.lower()}__theta_ratio"] = (
                round(feats[f"{ch}__{stage.lower()}__theta"] / total, 6)
                if total > 0 else np.nan
            )

            print(
                f"    {col} | {stage} ({n_samples:,} samples) — "
                f"delta: {feats[f'{ch}__{stage.lower()}__delta']:.4f}  |  "
                f"theta: {feats[f'{ch}__{stage.lower()}__theta']:.4f}  |  "
                f"alpha: {feats[f'{ch}__{stage.lower()}__alpha']:.4f}  |  "
                f"beta:  {feats[f'{ch}__{stage.lower()}__beta']:.4f}  |  "
                f"theta_ratio: {feats[f'{ch}__{stage.lower()}__theta_ratio']:.4f}"
            )

    return feats

# =========================================================================================================
# Batch extraction
# =========================================================================================================
FEATURES_DIR = Path("features_csv")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def extract_eeg_features_batch(
        merged_dir:  str | Path,
        output_file: str | Path | None = None,
        fs:          float = 128.0,
        pattern:     str = "*_merged.csv",
) -> pd.DataFrame:
    """
    Run ``extract_eeg_features`` on every merged CSV in a directory and
    collect results into a single DataFrame.

    Parameters
    ----------
    merged_dir : str | Path
        Directory containing merged CSV files (output of merge_all).
    output_file : str | Path | None
        If provided, save the feature DataFrame to this CSV path.
        Default saves to ``features_csv/eeg_features.csv``.
    fs : float
        Sampling frequency. Default is **128.0 Hz**.
    pattern : str
        Glob pattern to match merged CSVs. Default is ``'*_merged.csv'``.

    Returns
    -------
    pd.DataFrame
        One row per subject with all extracted EEG features.
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
            row = extract_eeg_features(f, fs=fs)
            rows.append(row)
        except Exception as e:
            print(f"  [SKIP] {f.name} — {e}")

    feature_df = pd.DataFrame(rows)

    if output_file is None:
        output_file = FEATURES_DIR / "eeg_features.csv"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_file, index=False)
    print(f"\nEEG feature table saved to: {output_file}  ({feature_df.shape[0]} subjects, {feature_df.shape[1]-1} features)")

    return feature_df


# =========================================================================================================
# Main extraction function
# =========================================================================================================

def extract_eeg_features(
        merged_file: str | Path,
        subject_id:  str | None = None,
        fs:          float = 128.0,
) -> dict:
    """
    Extract all EEG features from a single merged CSV file.

    The merged CSV must be the output of ``merge_all()`` and contain at minimum:
    ``time_sec``, ``stage``, ``EEG_LOC``, ``EEG_ROC``.

    Parameters
    ----------
    merged_file : str | Path
        Path to the merged CSV file for one subject/session.
    subject_id : str | None
        Optional subject identifier. If None, the file stem is used.
    fs : float
        Sampling frequency of the EEG signal in [Hz]. Default is **128.0 Hz**.

    Returns
    -------
    dict
        Flat dictionary of feature name → value for this subject.
    """
    merged_file = Path(merged_file)
    sid = subject_id if subject_id is not None else merged_file.stem

    print(f"\n{'=' * 60}")
    print(f"Extracting EEG features: {merged_file.name}")
    print(f"  subject_id : {sid}  |  fs : {fs} [Hz]")

    df = _load_and_validate(merged_file)

    feats: dict = {"subject_id": sid}

    print(f"\n--- EEG band power features ---")
    feats.update(_eeg_band_power_features(df, fs))

    n_nan = sum(1 for v in feats.values() if isinstance(v, float) and np.isnan(v))
    print(f"\n  Features computed : {len(feats) - 1}")
    print(f"  NaN features      : {n_nan}")

    return feats