# Filename: channel_standardization.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Standardizes EOG channel names py mapping Danish convention variants (EOGV, EOGH) to canonical LOC/ROC lables.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Helper function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# This function normalizes channel names by converting them to uppercase and stripping whitespace.
def normalize_channel_name(name: str) -> str:
    return name.upper().strip()

# =====================================================================
# Functions
# =====================================================================
def build_rename_map(ch_names: list[str]) -> dict[str, str]:
    """
    Builds a mapping from original channel names to standardized names based on common patterns. \\
    If a channel name contains "EOGV", it is renamed to "LOC", since the V in EOGV stands for "venstre" (left in Danish). \\
    If it contains "EOGH", it is renamed to "ROC", since the H in EOGH stands for "højre" (right in Danish). 

    Parameters
    ----------
    ch_names : list[str]
        A list of original channel names.

    Returns
    -------
    rename_map : dict[str, str]
        A dictionary mapping original channel names to standardized names.
    """
    
    rename_map = {}

    for ch in ch_names:
        # Normalize the channel name by converting it to uppercase and stripping whitespace
        name = ch.upper()

        # If the channel name contains "EOGV", rename it to "LOC". I.e. "EOGV-A1" becomes "LOC"
        if "EOGV" in name:
            rename_map[ch] = "LOC"

        # If the channel name contains "EOGH", rename it to "ROC". I.e. "EOGH-A2" becomes "ROC"
        elif "EOGH" in name:
            rename_map[ch] = "ROC"

    return rename_map