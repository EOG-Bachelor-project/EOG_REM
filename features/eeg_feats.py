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
import re

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
_DCSM_PATTERN = re.compile(r"(DCSM_\d+_[a-zA-Z])") 

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

    for stage in STAGES:
        mask      = df["stage"] == stage
        n_samples = mask.sum()

        # ---- Average LOC and ROC into a single signal ----
        loc = df.loc[mask, "EEG_LOC"].dropna().values
        roc = df.loc[mask, "EEG_ROC"].dropna().values
        min_len = min(len(loc), len(roc))

        if min_len < min_samples:
            print(f"    -> NaN: {stage}: only {min_len} usable samples, need {min_samples} "
                  f"(= 10s * {fs} Hz) for reliable Welch PSD — all {stage} bands set to NaN")
            for band in EEG_BANDS:
                feats[f"eeg__{stage.lower()}__{band}"] = np.nan
            feats[f"eeg__{stage.lower()}__total"]       = np.nan
            feats[f"eeg__{stage.lower()}__theta_ratio"] = np.nan
            continue

        sig = (loc[:min_len] + roc[:min_len]) / 2.0

        f, psd = welch(sig, fs=fs, nperseg=min(nperseg, len(sig)))

        total = 0.0
        for band, (lo, hi) in EEG_BANDS.items():
            band_mask = (f >= lo) & (f <= hi)
            power = float(np.trapz(psd[band_mask], f[band_mask]))
            feats[f"eeg__{stage.lower()}__{band}"] = round(power, 6)
            total += power

        feats[f"eeg__{stage.lower()}__total"] = round(total, 6)
        feats[f"eeg__{stage.lower()}__theta_ratio"] = (
            round(feats[f"eeg__{stage.lower()}__theta"] / total, 6)
            if total > 0 else np.nan
        )
        if total == 0:
            print(f"    -> NaN: {stage}: theta_ratio = NaN because total band power is 0 (no signal energy)")

        print(
            f"    {stage} ({n_samples:,} samples) — "
            f"delta: {feats[f'eeg__{stage.lower()}__delta']:.4f}  |  "
            f"theta: {feats[f'eeg__{stage.lower()}__theta']:.4f}  |  "
            f"alpha: {feats[f'eeg__{stage.lower()}__alpha']:.4f}  |  "
            f"beta:  {feats[f'eeg__{stage.lower()}__beta']:.4f}  |  "
            f"theta_ratio: {feats[f'eeg__{stage.lower()}__theta_ratio']:.4f}"
        )
        # ---- Overall theta/beta ratio (across all stages combined) ----
    loc_all = df["EEG_LOC"].dropna().values
    roc_all = df["EEG_ROC"].dropna().values
    min_len = min(len(loc_all), len(roc_all))

    if min_len >= min_samples:
        sig_all = (loc_all[:min_len] + roc_all[:min_len]) / 2.0
        f_all, psd_all = welch(sig_all, fs=fs, nperseg=min(nperseg, len(sig_all)))

        theta_power = float(np.trapz(
            psd_all[(f_all >= 4.0) & (f_all <= 8.0)],
            f_all[(f_all >= 4.0) & (f_all <= 8.0)]
        ))
        beta_power = float(np.trapz(
            psd_all[(f_all >= 13.0) & (f_all <= 30.0)],
            f_all[(f_all >= 13.0) & (f_all <= 30.0)]
        ))

        feats["eeg__overall__theta_beta_ratio"] = (
            round(theta_power / beta_power, 6) if beta_power > 0 else np.nan
        )
        print(f"    Overall theta/beta ratio: {feats['eeg__overall__theta_beta_ratio']:.4f}")
    else:
        feats["eeg__overall__theta_beta_ratio"] = np.nan
        print(f"    -> NaN: overall theta/beta ratio — only {min_len} samples, need {min_samples} for Welch PSD")

    nan_feats = [k for k, v in feats.items() if isinstance(v, float) and np.isnan(v)]
    if nan_feats:
        print(f"    -> NaN features ({len(nan_feats)}): {', '.join(nan_feats)}")

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

    # ---- Load existing CSV to skip already-processed subjects ----
    if output_file is None:
        output_file = FEATURES_DIR / "eeg_features.csv"
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
        m = _DCSM_PATTERN.match(f.stem.replace(".csv", ""))
        sid = m.group(1) if m else f.stem.replace(".csv", "")
        if sid in already_done:
            skipped += 1
            continue
        try:
            row = extract_eeg_features(f, fs=fs)
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


# =========================================================================================================
# Main extraction function
# =========================================================================================================

def extract_eeg_features(
        merged_file: str | Path,
        subject_id:  str | None = None,
        fs:          float = 128.0,
) -> dict:
    """
    Band power features per sleep stage using Welch's method.
    LOC and ROC signals are averaged into a single signal before computing PSD.

    Parameters
    ----------
    df : pd.DataFrame
        Merged CSV DataFrame containing ``stage``, ``EEG_LOC``, ``EEG_ROC`` columns.
    fs : float
        Sampling frequency of the EEG signal in [Hz].

    Returns
    -------
    dict
        Flat dictionary of feature name → value containing:

        Per stage (W, N1, N2, N3, REM): \\
            `eeg__{stage}__delta`       — Delta band power [µV²/Hz] \\
            `eeg__{stage}__theta`       — Theta band power [µV²/Hz] \\
            `eeg__{stage}__alpha`       — Alpha band power [µV²/Hz] \\
            `eeg__{stage}__beta`        — Beta band power [µV²/Hz] \\
            `eeg__{stage}__total`       — Total band power [µV²/Hz] \\
            `eeg__{stage}__theta_ratio` — Theta / total power ratio

        Overall (all stages combined): \\
            `eeg__overall__theta_beta_ratio` — Theta / beta power ratio
    """
    merged_file = Path(merged_file)

    raw_stem = merged_file.stem.replace(".csv", "")
    m = _DCSM_PATTERN.match(raw_stem)
    sid = subject_id if subject_id is not None else (m.group(1) if m else raw_stem)

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