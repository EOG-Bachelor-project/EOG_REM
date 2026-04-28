# Filename: gssc_features.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: GSSC probability feature extraction from a merged CSV file (output of merge_all).
#              Computes per-subject features from GSSC stage probability outputs, capturing
#              REM sleep staging confidence and micro-sleep structure. These features correspond
#              directly to the micro-sleep structure category in Cesari et al. (2021), which was
#              the highest-ranked feature category for RBD identification.

# =========================================================================================================
# Imports
# =========================================================================================================
from __future__ import annotations # for type hinting of class methods that return instances of the class itself
 
import numpy as np                 # for numerical operations
import pandas as pd                # for data manipulation and analysis
from pathlib import Path           # for handling file paths
import re

# =========================================================================================================
# Constants
# =========================================================================================================
FEATURES_DIR = Path("features_csv")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

PROB_COLS = ["prob_w", "prob_n1", "prob_n2", "prob_n3", "prob_rem"]

_DCSM_PATTERN = re.compile(r"(DCSM_\d+_[a-zA-Z])") 

# =========================================================================================================
# Helpers
# =========================================================================================================

def _load_and_validate(merged_file: Path) -> pd.DataFrame:
    """Load merged CSV and check required columns are present."""
    required = {"time_sec", "stage"} | set(PROB_COLS)
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

def _gssc_probability_features(df: pd.DataFrame) -> dict:
    """
    Features derived from GSSC stage probability outputs during REM sleep.
    These capture the confidence and stability of the staging model within REM,
    reflecting micro-sleep structure — the highest-ranked feature category in Cesari et al.

    Features
    --------
    `rem_mean_prob_rem`         : Mean probability of REM during REM-scored samples. High = confident REM staging.
    `rem_mean_prob_w`           : Mean probability of Wake during REM sleep. High = REM instability.
    `rem_mean_prob_n1`          : Mean probability of N1 during REM sleep.
    `rem_mean_prob_n2`          : Mean probability of N2 during REM sleep.
    `rem_mean_prob_n3`          : Mean probability of N3 during REM sleep.
    `rem_certainty`             : Fraction of REM samples where prob_rem > 0.5. Key RBD marker from Cesari et al.
    `rem_mean_prob_nrem`        : Mean probability of any NREM stage (N1+N2+N3) during REM sleep. Proxy for REM instability.
    `rem_high_wake_prob_frac`   : Fraction of REM samples where prob_w > 0.2. Captures wake intrusions into REM.
    """

    # ---- 1) Check probability columns are present ----
    feats: dict = {}

    if not all(c in df.columns for c in PROB_COLS):
        missing = [c for c in PROB_COLS if c not in df.columns]
        print(f"    Probability columns missing: {missing} — returning NaN defaults")
        for k in [
            "rem_mean_prob_rem", "rem_mean_prob_w", "rem_mean_prob_n1",
            "rem_mean_prob_n2", "rem_mean_prob_n3", "rem_certainty",
            "rem_mean_prob_nrem", "rem_high_wake_prob_frac",
        ]:
            feats[k] = np.nan
        return feats

    # ---- 2) Filter to REM sleep samples ----
    rem_df = _rem_samples(df)
    print(f"    REM samples: {len(rem_df):,}")

    if rem_df.empty:
        print("    No REM samples found — returning NaN defaults")
        for k in [
            "rem_mean_prob_rem", "rem_mean_prob_w", "rem_mean_prob_n1",
            "rem_mean_prob_n2", "rem_mean_prob_n3", "rem_certainty",
            "rem_mean_prob_nrem", "rem_high_wake_prob_frac",
        ]:
            feats[k] = np.nan
        return feats

    # ---- 3) Compute mean probability per stage during REM sleep ----
    for col, label in [
        ("prob_rem", "rem_mean_prob_rem"),
        ("prob_w",   "rem_mean_prob_w"),
        ("prob_n1",  "rem_mean_prob_n1"),
        ("prob_n2",  "rem_mean_prob_n2"),
        ("prob_n3",  "rem_mean_prob_n3"),
    ]:
        feats[label] = round(float(rem_df[col].mean()), 4)

    print(f"    Mean probs — REM: {feats['rem_mean_prob_rem']}  |  W: {feats['rem_mean_prob_w']}  |  N1: {feats['rem_mean_prob_n1']}  |  N2: {feats['rem_mean_prob_n2']}  |  N3: {feats['rem_mean_prob_n3']}")

    # ---- 4) Compute REM certainty ----
    # NOTE:
    #       Directly corresponds to 'REM certainty (th=0.5)' in Cesari et al. Table 5 —
    #       one of the top discriminating features between PDnonRBD and PD+RBD.
    #       Lower certainty = more staging ambiguity, associated with RBD.
    feats["rem_certainty"] = round(float((rem_df["prob_rem"] > 0.5).mean()), 4)
    print(f"    REM certainty (prob_rem > 0.5): {feats['rem_certainty']}")

    # ---- 5) Compute combined NREM probability and wake intrusion fraction ----
    feats["rem_mean_prob_nrem"]      = round(float((rem_df["prob_n1"] + rem_df["prob_n2"] + rem_df["prob_n3"]).mean()), 4)
    feats["rem_high_wake_prob_frac"] = round(float((rem_df["prob_w"] > 0.2).mean()), 4)
    print(f"    Mean NREM prob: {feats['rem_mean_prob_nrem']}  |  High wake prob fraction (>0.2): {feats['rem_high_wake_prob_frac']}")

    return feats

#——————————————————————————————————————————————————————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————————————————————————————————————————————————————

def _gssc_rem_stability_features(df: pd.DataFrame, fs: float) -> dict:
    """
    Features describing REM sleep stability and fragmentation derived from GSSC staging.
    These are macro-level structural features of REM sleep across the night.

    Features
    --------
    `rem_stability_index`       : Mean prob_rem within REM epochs, weighted by epoch length.
                                  Higher = more stable REM. Corresponds to REM stability index in Cesari et al.
    `rem_fragmentation_index`   : Number of REM-to-nonREM transitions per hour of REM sleep.
                                  Higher = more fragmented REM. Associated with RBD.
    `rem_w_transition_frac`     : Fraction of REM-to-nonREM transitions that go directly to Wake.
                                  Elevated in RBD due to arousal from dream enactment.
    `amount_of_rem`             : Fraction of REM samples where prob_rem > 0.5 out of total recording.
                                  Corresponds directly to 'Amount of REM (th=0.5)' in Cesari et al.
    """

    # ---- 1) Calculate total recording duration and REM duration ----
    n_total = len(df)
    total_min = n_total / fs / 60.0
    feats: dict = {}

    rem_df = _rem_samples(df)
    rem_min = len(rem_df) / fs / 60.0
    print(f"    Total: {total_min:.2f} min  |  REM: {rem_min:.2f} min  ({len(rem_df):,} samples)")

    if rem_df.empty:
        print("    No REM samples — returning NaN defaults")
        for k in ["rem_stability_index", "rem_fragmentation_index", "rem_w_transition_frac", "amount_of_rem"]:
            feats[k] = np.nan
        return feats

    # ---- 2) Compute REM stability index ----
    # NOTE:
    #       Mean prob_rem during REM-scored samples. Higher = the model is more
    #       consistently confident in REM, i.e. more stable REM sleep.
    if "prob_rem" in df.columns:
        feats["rem_stability_index"] = round(float(rem_df["prob_rem"].mean()), 4)
    else:
        feats["rem_stability_index"] = np.nan
    print(f"    REM stability index: {feats['rem_stability_index']}")

    # ---- 3) Compute REM fragmentation index (REM-to-nonREM transitions per hour) ----
    # NOTE:
    #       Count stage transitions out of REM into any other stage.
    #       Divide by REM duration in hours to normalise across subjects.
    stage_series = df["stage"].reset_index(drop=True)
    transitions_out_of_rem = (
        (stage_series.shift(1) == "REM") & (stage_series != "REM")
    ).sum()
    rem_hours = rem_min / 60.0
    feats["rem_fragmentation_index"] = round(transitions_out_of_rem / rem_hours, 4) if rem_hours > 0 else np.nan
    print(f"    REM fragmentation index: {feats['rem_fragmentation_index']} transitions/hour")

    # ---- 4) Compute fraction of REM exits that go directly to Wake ----
    rem_to_w = (
        (stage_series.shift(1) == "REM") & (stage_series == "W")
    ).sum()
    feats["rem_w_transition_frac"] = round(rem_to_w / transitions_out_of_rem, 4) if transitions_out_of_rem > 0 else np.nan
    print(f"    REM —> W transition fraction: {feats['rem_w_transition_frac']}")

    # ---- 5) Compute amount of REM (Cesari et al. definition) ----
    # NOTE:
    #       Fraction of ALL recording samples where prob_rem > 0.5.
    #       Directly mirrors 'Amount of REM (th=0.5)' from Cesari et al. Table 5.
    #       RBD patients show significantly lower values than controls.
    if "prob_rem" in df.columns:
        feats["amount_of_rem"] = round(float((df["prob_rem"] > 0.5).mean()), 4)
    else:
        feats["amount_of_rem"] = np.nan
    print(f"    Amount of REM (prob_rem > 0.5): {feats['amount_of_rem']}")

    return feats

# =========================================================================================================
# =========================================================================================================
# Main extraction function
# =========================================================================================================
# =========================================================================================================

def extract_gssc_features(
        merged_file: str | Path,
        subject_id:  str | None = None,
        fs:          float = 250.0,
        ) -> dict:
    """
    Extract all GSSC probability features from a single merged CSV file.

    The merged CSV must be the output of ``merge_all()`` and contain at minimum:
    ``time_sec``, ``stage``, ``prob_w``, ``prob_n1``, ``prob_n2``, ``prob_n3``, ``prob_rem``.

    Parameters
    ----------
    merged_file : str | Path
        Path to the merged CSV file for one subject/session.
    subject_id : str | None
        Optional subject identifier. If None, the file stem is used.
    fs : float
        Sampling frequency of the EOG signal in [Hz]. Default is **250.0 Hz**.

    Returns
    -------
    feats : dict
        Flat dictionary of feature name —> value for this subject.
    """
    merged_file = Path(merged_file)

    raw_stem = merged_file.stem.replace(".csv", "")
    m = _DCSM_PATTERN.match(raw_stem)
    sid = subject_id if subject_id is not None else (m.group(1) if m else raw_stem)

    print(f"\n{'=' * 60}")
    print(f"Extracting GSSC features: {merged_file.name}")
    print(f"  subject_id : {sid}  |  fs : {fs} [Hz]")

    df = _load_and_validate(merged_file)

    feats: dict = {"subject_id": sid}

    print(f"\n--- GSSC probability features ---")
    feats.update(_gssc_probability_features(df))

    print(f"\n--- GSSC REM stability features ---")
    feats.update(_gssc_rem_stability_features(df, fs))

    n_nan = sum(1 for v in feats.values() if isinstance(v, float) and np.isnan(v))
    print(f"\n  Features computed : {len(feats) - 1}")
    print(f"  NaN features      : {n_nan}")

    return feats


# =========================================================================================================
# Batch extraction
# =========================================================================================================

def extract_gssc_features_batch(
        merged_dir:  str | Path,
        output_file: str | Path | None = None,
        fs:          float = 250.0,
        pattern:     str = "*_merged.csv",
        ) -> pd.DataFrame:
    """
    Run ``extract_gssc_features`` on every merged CSV in a directory and
    collect results into a single DataFrame.

    Parameters
    ----------
    merged_dir : str | Path
        Directory containing merged CSV files (output of merge_all).
    output_file : str | Path | None
        If provided, save the feature DataFrame to this CSV path.
        Default saves to ``features_csv/gssc_features.csv``.
    fs : float
        Sampling frequency. Default is **250.0 Hz**.
    pattern : str
        Glob pattern to match merged CSVs. Default is ``'*_merged.csv'``.

    Returns
    -------
    pd.DataFrame
        One row per subject with all extracted GSSC features.
    """
    merged_dir = Path(merged_dir)
    files = sorted(merged_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {merged_dir}"
        )

    # ---- Load existing CSV to skip already-processed subjects ----
    if output_file is None:
        output_file = FEATURES_DIR / "gssc_features.csv"
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
            row = extract_gssc_features(f, fs=fs)
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
# Entry point
# =========================================================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gssc_features.py <merged_csv_or_dir> [fs]")
        sys.exit(1)

    path = Path(sys.argv[1])
    sampling_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0

    if path.is_dir():
        df_out = extract_gssc_features_batch(path, fs=sampling_freq)
        print(df_out.to_string(index=False))
    elif path.is_file():
        feats = extract_gssc_features(path, fs=sampling_freq)
        for k, v in feats.items():
            print(f"  {k:<45} {v}")
    else:
        print(f"Path not found: {path}")
        sys.exit(1)