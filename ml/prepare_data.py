# Filename: prepare_data.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Loads the feature CSV, assigns group labels for binary and multiclass
#              classification, drops subjects with NaN values, and splits into
#              train/test sets. All parameters are configurable.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# =====================================================================
# Constants
# =====================================================================
GROUP_COLS      = ["Control", "PD(-RBD)", "PD(+RBD)", "iRBD", "PLM"]
ID_COLS         = ["subject_id", "DCSM_ID"]
DEFAULT_SEED    = 42

# ANSI helpers
BOLD    = "\033[1m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
RESET   = "\033[0m"


# =====================================================================
# Helper — assign group label
# =====================================================================
def _assign_group(df: pd.DataFrame) -> pd.Series:
    """
    Assign a single diagnostic group label to each subject based on the
    1/0 indicator columns from the patient Excel.

    Priority: iRBD > PD(+RBD) > PD(-RBD) > Control.
    PLM subjects are mapped to Control.
    Subjects with no group info (all zeros or all NaN) get 'Unknown'.

    Returns
    -------
    pd.Series
        Group label for each row.
    """

    groups = pd.Series("Unknown", index=df.index)

    available = [c for c in GROUP_COLS if c in df.columns]
    if not available:
        return groups

    # Subjects that have any group info at all
    has_info = df[available].notna().any(axis=1) & (df[available].sum(axis=1) > 0)
    groups[has_info] = "Control"

    if "PD(-RBD)" in df.columns:
        groups[df["PD(-RBD)"] == 1] = "PD(-RBD)"
    if "PD(+RBD)" in df.columns:
        groups[df["PD(+RBD)"] == 1] = "PD(+RBD)"
    if "iRBD" in df.columns:
        groups[df["iRBD"] == 1] = "iRBD"

    # PLM → Control
    if "PLM" in df.columns:
        groups[df["PLM"] == 1] = "Control"

    return groups


# =====================================================================
# Main data loading function
# =====================================================================
def load_features(
        feature_csv: str | Path,
        drop_nan:    bool = True,
        ) -> pd.DataFrame:
    """
    Load the feature CSV and assign group labels.

    Parameters
    ----------
    feature_csv : str | Path
        Path to features.csv (output of the extract step).
    drop_nan : bool
        If True, drop subjects that have any NaN in feature columns
        and print a warning listing them. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with all features plus a 'group' column.
    """

    df = pd.read_csv(feature_csv)
    print(f"Loaded: {feature_csv}  ({df.shape[0]} subjects, {df.shape[1]} columns)")

    # Assign group labels
    df["group"] = _assign_group(df)
    group_counts = df["group"].value_counts()
    print(f"\nGroup distribution:")

    for g, n in group_counts.items():
        print(f"  {g:12s}  n = {n}")

    # Warn about Unknown
    n_unknown = (df["group"] == "Unknown").sum()
    if n_unknown > 0:
        unknown_ids = df.loc[df["group"] == "Unknown", "subject_id"].tolist()
        print(f"\n  {YELLOW}WARNING{RESET}  {n_unknown} subjects have no group label and will be excluded:")

        for uid in unknown_ids[:10]:
            print(f"      - {uid}")

        if n_unknown > 10:
            print(f"      ... and {n_unknown - 10} more")

        df = df[df["group"] != "Unknown"].reset_index(drop=True)
        print(f"  Remaining: {len(df)} subjects")

    # Check for NaN values in feature columns
    feature_cols = _get_feature_cols(df)
    rows_with_nan = df[df[feature_cols].isna().any(axis=1)]

    if not rows_with_nan.empty:
        print(f"\n  {YELLOW}WARNING{RESET}  {len(rows_with_nan)} subject(s) have NaN values:")

        for _, row in rows_with_nan.iterrows():
            sid = row["subject_id"]
            missing = [c for c in feature_cols if pd.isna(row[c])]
            print(f"      {sid}: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")

        if drop_nan:
            df = df.dropna(subset=feature_cols).reset_index(drop=True)
            print(f"  Dropped — {len(df)} subjects remaining.")

        else:
            print(f"  Kept (drop_nan=False) — NaN values may cause errors in training.")

    return df


# =====================================================================
# Feature column helper
# =====================================================================
def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of numeric feature columns (excludes IDs, group labels, etc.)."""

    exclude = set(ID_COLS) | set(GROUP_COLS) | {"group"}

    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# =====================================================================
# Binary label assignment
# =====================================================================
def make_binary_labels(
        df:   pd.DataFrame,
        mode: str = "control_vs_all",
        ) -> pd.Series:
    """
    Create binary labels (0/1) from the group column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'group' column.
    mode : str
        Which binary split to use:
        - 'control_vs_all'  : Control=0, all disease groups=1
        - 'control_vs_irbd' : Control=0, iRBD=1 (drops PD subjects)

    Returns
    -------
    pd.Series
        Binary labels (0 or 1). Subjects that don't fit the split get NaN.
    """

    if mode == "control_vs_all":
        labels = pd.Series(np.nan, index=df.index)
        labels[df["group"] == "Control"] = 0
        labels[df["group"].isin(["iRBD", "PD(-RBD)", "PD(+RBD)"])] = 1
        return labels

    elif mode == "control_vs_irbd":
        labels = pd.Series(np.nan, index=df.index)
        labels[df["group"] == "Control"] = 0
        labels[df["group"] == "iRBD"] = 1
        return labels

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'control_vs_all' or 'control_vs_irbd'.")


# =====================================================================
# Multiclass label assignment
# =====================================================================
def make_multiclass_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create multiclass labels from the group column.

    Mapping:
        Control  → 0
        iRBD     → 1
        PD(-RBD) → 2
        PD(+RBD) → 3

    Returns
    -------
    pd.Series
        Integer labels. Subjects that don't fit get NaN.
    """

    label_map = {
        "Control":  0,
        "iRBD":     1,
        "PD(-RBD)": 2,
        "PD(+RBD)": 3,
    }

    return df["group"].map(label_map)


# =====================================================================
# Train/test split
# =====================================================================
def split_data(
        df:         pd.DataFrame,
        labels:     pd.Series,
        test_size:  float = 0.2,
        seed:       int = DEFAULT_SEED,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and labels into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns.
    labels : pd.Series
        Labels (binary or multiclass) aligned with df.
    test_size : float
        Fraction of data for testing. Default is 0.2 (80/20 split).
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Feature DataFrames and label Series for train and test.
    """
    # ---- 1) Identify valid rows (those with non-NaN labels) ----
    valid = labels.notna()  
    df_valid = df[valid].reset_index(drop=True)                     # Only keep rows with valid labels
    labels_valid = labels[valid].reset_index(drop=True).astype(int) # Convert labels to int (0/1 or 0/1/2/3)

    # ---- 2) Get feature matrix and label vector ----
    feature_cols = _get_feature_cols(df_valid) # Get feature columns from the valid subset
    X = df_valid[feature_cols]                 # Feature matrix
    y = labels_valid                           # Label vector  

    # ---- 3) Split into train/test sets ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    print(f"\nTrain/test split (test_size={test_size}, seed={seed}):")
    print(f"  Train: {len(X_train)} subjects")
    print(f"  Test:  {len(X_test)} subjects")
    print(f"  Features: {len(feature_cols)}")

    # Print class distribution
    print(f"\n  Train class distribution:")
    for label, count in y_train.value_counts().sort_index().items():
        print(f"    {label}: {count}")

    print(f"  Test class distribution:")
    for label, count in y_test.value_counts().sort_index().items():
        print(f"    {label}: {count}")

    return X_train, X_test, y_train, y_test


# =====================================================================
# Convenience — full pipeline in one call
# =====================================================================
def prepare(
        feature_csv:    str | Path,
        mode:           str = "binary",
        binary_mode:    str = "control_vs_all",
        test_size:      float = 0.2,
        seed:           int = DEFAULT_SEED,
        drop_nan:       bool = True,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load features, assign labels, and split — all in one call.

    Parameters
    ----------
    feature_csv : str | Path
        Path to features.csv.
    mode : str
        Options:
        - `'binary'`
        - `'multiclass'`

        Binary will create a 0/1 label, multiclass will create a 0/1/2/3 label. \\
        **Default is `'binary'`**.
    binary_mode : str
        IMPORTANT: Only used if `mode='binary'`\\
        Options:
        - `'control_vs_all'`
        - `'control_vs_irbd'`
        
        Determines how the binary labels are assigned. \\
        **Default is `'control_vs_all'`**.
    test_size : float
        Fraction for test set. So 0.2 means 80% train, 20% test. \\
        **Default value: 0.2**.
    seed : int
        Random seed. \\
        Default value: **42**.
    drop_nan : bool
        Drop subjects with NaN features. If False, NaN features will be replaced with the mean. \\
        **Default is `True`**.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    # ---- 1) Load features ----
    df = load_features(feature_csv, drop_nan=drop_nan)

    # ---- 2) Assign labels ----
    # For binary classification, assign 0/1 labels based on the specified binary_mode.
    # For multiclass classification, assign 0/1/2/3 labels based on the group column.
    # Subjects that don't fit the classification (e.g. PLM subjects in control_vs_irbd) will get NaN and be dropped in the next step.

    if mode == "binary":
        labels = make_binary_labels(df, mode=binary_mode)
        print(f"\nBinary classification: {binary_mode}")

    elif mode == "multiclass":
        labels = make_multiclass_labels(df)
        print(f"\nMulticlass classification: Control / iRBD / PD(-RBD) / PD(+RBD)")

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'binary' or 'multiclass'.")

    # ---- 3) Split into train/test sets ----
    n_valid = labels.notna().sum() # Subjects that have valid labels for this classification task
    n_dropped = len(df) - n_valid  # Subjects that don't fit the classification 

    if n_dropped > 0:
        print(f"  {n_dropped} subjects excluded (don't fit this classification task)")

    return split_data(df, labels, test_size=test_size, seed=seed)


# =====================================================================
# CLI — for quick testing
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for ML classification.")
    parser.add_argument("feature_csv", type=str, help="Path to features.csv")
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "multiclass"],
                        help="Classification mode (default: binary)")
    parser.add_argument("--binary-mode", type=str, default="control_vs_all",
                        choices=["control_vs_all", "control_vs_irbd"],
                        help="Binary split mode (default: control_vs_all)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare(
        feature_csv=args.feature_csv,
        mode=args.mode,
        binary_mode=args.binary_mode,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"\n{GREEN}Ready for training.{RESET}")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")