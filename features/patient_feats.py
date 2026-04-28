# Filename: patient_feats.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Looks up patient diagnostic group labels (Control, iRBD, PD(-RBD), PD(+RBD))
#              from the patient Excel file and returns them as 1/0 features.
#              Follows the same pattern as eog_feats.py, gssc_feats.py, eeg_feats.py.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import re
import pandas as pd
from pathlib import Path


# =====================================================================
# Constants
# =====================================================================

# Columns to look up from the Excel file
GROUP_COLS = ["Control", "PD(-RBD)", "PD(+RBD)", "iRBD", "PLM"]

# Regex to extract DCSM_<number>_<letter> from subject_id
_DCSM_PATTERN = re.compile(r"(DCSM_\d+_[a-zA-Z])")

# Cache for the Excel DataFrame so we only read it once
_excel_cache: dict[str, pd.DataFrame] = {}


# =====================================================================
# Helper
# =====================================================================
def _extract_dcsm_id(subject_id: str) -> str:
    """
    Extract the DCSM_X_Y part from a subject_id string.

    Examples
    --------
    >>> _extract_dcsm_id("DCSM_2_a_contiguous_eog_merged")
    'DCSM_2_a'
    >>> _extract_dcsm_id("DCSM_15_b_contiguous_eog_merged_Umaer")
    'DCSM_15_b'
    """
    match = _DCSM_PATTERN.search(str(subject_id))
    if match:
        return match.group(1)
    return str(subject_id)


def _load_excel(patient_excel: str | Path) -> pd.DataFrame:
    """Load the patient Excel file, with caching so it's only read once."""
    key = str(patient_excel)
    if key not in _excel_cache:
        df = pd.read_excel(patient_excel)
        _excel_cache[key] = df
    return _excel_cache[key]


# =====================================================================
# Main extraction function
# =====================================================================
def extract_patient_features(
        merged_file:   str | Path,
        patient_excel: str | Path,
        subject_id:    str | None = None,
) -> dict:
    """
    Look up patient group labels from the Excel file for a single subject.

    Takes a subject_id (e.g. ``DCSM_2_a_contiguous_eog_merged``), extracts
    the DCSM ID (``DCSM_2_a``), and looks up the corresponding row in the
    patient Excel to get the group labels as 1/0 values.

    Parameters
    ----------
    merged_file : str | Path
        Path to the merged CSV file. Only used to derive subject_id if
        not provided explicitly.
    patient_excel : str | Path
        Path to the patient info Excel file with DCSM_ID and group columns.
    subject_id : str | None
        Optional subject identifier. If None, the file stem is used.

    Returns
    -------
    dict
        Dictionary with ``subject_id`` and group label columns (1 or 0).
    """
    merged_file = Path(merged_file)
    raw_stem = merged_file.stem.replace(".csv", "")
    m = _DCSM_PATTERN.match(raw_stem)
    sid = subject_id if subject_id is not None else (m.group(1) if m else raw_stem)
    dcsm_id = _extract_dcsm_id(sid)

    print(f"\n{'=' * 60}")
    print(f"Looking up patient info: {sid}")
    print(f"  DCSM_ID : {dcsm_id}")

    feats: dict = {"subject_id": sid, "DCSM_ID": dcsm_id}

    # --- Load Excel ---
    excel_df = _load_excel(patient_excel)

    if "DCSM_ID" not in excel_df.columns:
        print(f"  ERROR: Excel file has no 'DCSM_ID' column")
        for col in GROUP_COLS:
            feats[col] = 0
        return feats

    # --- Find matching row ---
    match = excel_df[excel_df["DCSM_ID"] == dcsm_id]

    if match.empty:
        print(f"  ⚠  No match found in Excel for {dcsm_id}")
        for col in GROUP_COLS:
            feats[col] = 0
        return feats

    # --- Extract group labels ---
    row = match.iloc[0]
    for col in GROUP_COLS:
        if col in row.index:
            val = row[col]
            feats[col] = int(val) if pd.notna(val) else 0
        else:
            feats[col] = 0

    # --- Print result ---
    group_str = ", ".join(f"{c}={feats[c]}" for c in GROUP_COLS)
    print(f"  Found:  {group_str}")

    return feats