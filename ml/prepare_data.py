# Filename: prepare_data.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Loads the feature CSV and assigns group labels for binary and multiclass classification. 
#              Imputation is intentionally NOT done here — it is performed inside each CV fold in train.py 
#              to avoid data leakage.

# Pipeline overview:
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Load feature CSV and derive a 'group' label column from indicator columns
# 2. Assign integer labels based on the chosen classification mode:
#       binary     : Control=0, disease group(s)=1 (subjects outside the chosen binary split are excluded)
#       multiclass : Control=0, iRBD=1, PD(-RBD)=2, PD(+RBD)=3
# 3. Drop subjects with no valid label for the chosen task
# 4. Return feature matrix X, label vector y, and feature column names
#    (no splitting or imputation — both happen inside the CV fold in train.py)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# ================================================================================
# Imports
# ================================================================================

from __future__ import annotations  # for Python 3.10+ type hinting (e.g. list[str])

import numpy as np                  # for any numeric operations if needed in the future
import pandas as pd                 # for data manipulation and loading CSVs
from pathlib import Path            # for handling file paths in a platform-independent way


# =====================================================================
# Constants
# =====================================================================
GROUP_COLS   = ["Control", "PD(-RBD)", "PD(+RBD)", "iRBD", "PLM"]
ID_COLS      = ["subject_id", "DCSM_ID"]
DEFAULT_SEED = 42

BOLD   = "\033[1m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


# =====================================================================
# Helpers
# =====================================================================
def _assign_group(df: pd.DataFrame) -> pd.Series:
    """
    Derive a single diagnostic group label from the 1/0 indicator columns.

    Priority order: iRBD > PD(+RBD) > PD(-RBD) > Control.\\
    PLM subjects are folded into Control.\\
    Subjects with no group info get 'Unknown'.
    """
    groups = pd.Series("Unknown", index=df.index)           # Default to 'Unknown' for all subjects

    # Check which group indicator columns are actually present in the DataFrame
    available = [c for c in GROUP_COLS if c in df.columns]  
    if not available:
        return groups
    
    has_info = df[available].notna().any(axis=1) & (df[available].sum(axis=1) > 0)
    groups[has_info] = "Control"

    if "PD(-RBD)" in df.columns:
        groups[df["PD(-RBD)"] == 1] = "PD(-RBD)"
    if "PD(+RBD)" in df.columns:
        groups[df["PD(+RBD)"] == 1] = "PD(+RBD)"
    if "iRBD" in df.columns:
        groups[df["iRBD"] == 1] = "iRBD"
    if "PLM" in df.columns:
        groups[df["PLM"] == 1] = "Control"

    return groups


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding ID and group indicator columns."""
    exclude = set(ID_COLS) | set(GROUP_COLS) | {"group"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# =====================================================================
# Load
# =====================================================================
def load_features(feature_csv: str | Path) -> pd.DataFrame:
    """
    Load the feature CSV and attach a 'group' column.

    Parameters
    ----------
    feature_csv : str | Path
        Path to features.csv produced by the extraction pipeline.

    Returns
    -------
    pd.DataFrame
        Full feature DataFrame with a 'group' column added.
    """
    df = pd.read_csv(feature_csv)
    if "group" not in df.columns:
        df["group"] = _assign_group(df)
    print(f"Loaded {len(df)} subjects, {len(_get_feature_cols(df))} features")
    print(f"  Group counts:\n{df['group'].value_counts().to_string()}")
    return df


# =====================================================================
# Label assignment
# =====================================================================
def make_binary_labels(
        df: pd.DataFrame, 
        mode: str = "control_vs_all"
        ) -> pd.Series:
    """
    Assign binary labels (0 / 1).

    Modes
    -----
    control_vs_all   : Control=0, iRBD/PD(+RBD)=1  (PD(-RBD) excluded)
    control_vs_irbd  : Control=0, iRBD=1  (PD subjects excluded)
    control_vs_pd    : Control=0, PD(+RBD)=1  (iRBD and PD(-RBD) excluded)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'group' column with diagnostic group labels.
    mode : str
        Which binary classification task to prepare labels for.\\
        Options: 'control_vs_all', 'control_vs_irbd', 'control_vs_pd'.\\
        *Default 'control_vs_all'*.
    
    Returns
    -------
    pd.Series
        Binary label vector (0/1) aligned with the input DataFrame. 
        Subjects with no valid label for the chosen task will have NaN.
    """
    g = df["group"] # Get the 'group' column to map from    
    
    if mode == "control_vs_all":
        mapping = {"Control": 0, "iRBD": 1, "PD(+RBD)": 1}
    elif mode == "control_vs_irbd":
        mapping = {"Control": 0, "iRBD": 1}
    elif mode == "control_vs_pd":
        mapping = {"Control": 0, "PD(+RBD)": 1}
    else:
        raise ValueError(
            f"Unknown binary_mode '{mode}'. "
            "Use 'control_vs_all', 'control_vs_irbd', or 'control_vs_pd'."
        )
    return g.map(mapping)


def make_multiclass_labels(df: pd.DataFrame) -> pd.Series:
    """
    Assign multiclass labels.

    Mapping: Control=0, iRBD=1, PD(-RBD)=2, PD(+RBD)=3.
    """
    return df["group"].map({"Control": 0, "iRBD": 1, "PD(-RBD)": 2, "PD(+RBD)": 3})


# =====================================================================
# Convenience — full load + label in one call
# =====================================================================
def prepare(
        feature_csv:  str | Path,
        mode:         str = "binary",
        binary_mode:  str = "control_vs_all",
        ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load features and return (X, y, feature_cols).

    No imputation is done here. Imputation happens inside each CV fold in ``train.py``.

    Parameters
    ----------
    feature_csv : str | Path
        Path to features.csv produced by the extraction pipeline.
    mode : str
        'binary' or 'multiclass'. Determines the classification task and label assignment.\\
        *Default 'binary'*.
    binary_mode : str
        Only used when mode='binary'. \\
        Options: 'control_vs_all', 'control_vs_irbd', 'control_vs_pd'.\\
        *Default 'control_vs_all'*.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all subjects with valid labels).
    y : pd.Series
        Integer label vector aligned with X.
    feature_cols : list[str]
        Column names used as features.
    """
    # Load the full DataFrame with group labels
    df = load_features(feature_csv)

    # Assign labels according to the specified mode
    if mode == "binary":
        labels = make_binary_labels(df, mode=binary_mode)
        print(f"\nBinary mode: {binary_mode}")
    elif mode == "multiclass":
        labels = make_multiclass_labels(df)
        print(f"\nMulticlass mode: Control / iRBD / PD(-RBD) / PD(+RBD)")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'binary' or 'multiclass'.")

    # Get feature columns (numeric, excluding IDs and group indicators)
    feature_cols = _get_feature_cols(df)

    valid = labels.notna()      # Only keep subjects with valid labels for the chosen task
    n_excluded = (~valid).sum() # Count how many subjects are excluded due to missing/invalid labels
    if n_excluded > 0:
        print(f"  {n_excluded} subjects excluded (no valid label for this task)")

    X = df.loc[valid, feature_cols].reset_index(drop=True) # Feature matrix for valid subjects
    y = labels[valid].astype(int).reset_index(drop=True)   # Label vector for valid subjects

    print(f"  {len(X)} subjects retained")
    print(f"  Class distribution:\n{y.value_counts().sort_index().to_string()}")

    return X, y, feature_cols