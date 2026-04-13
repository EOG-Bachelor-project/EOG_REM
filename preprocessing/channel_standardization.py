# Filename: channel_standardization.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Standardizes EOG channel names py mapping Danish convention variants (EOGV, EOGH) to canonical LOC/ROC lables.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations


# =====================================================================
# Functions
# =====================================================================
def build_rename_map(ch_names: list[str]) -> dict[str, str]:
    """
    Builds a mapping from original channel names to standardized LOC/ROC names.
 
    Strategy:
    1. Skip channels already named exactly "LOC" or "ROC".
    2. Find all channels containing "EOG" (case-insensitive).
    3. For each EOG channel, determine left/right from context clues:
       - Left indicators:  L, V (Danish: venstre)
       - Right indicators: R, H (Danish: højre)
    4. If exactly 2 EOG channels are found but neither has clear
       left/right indicators, assign by order (first = LOC, second = ROC).
 
    Parameters
    ----------
    ch_names : list[str]
        A list of original channel names from the EDF file.
 
    Returns
    -------
    rename_map : dict[str, str]
        A dictionary mapping original channel names to 'LOC' or 'ROC'.
        Only contains entries for channels that need renaming.
    """
    rename_map = {}
 
    # Check if LOC and ROC already exist exactly
    upper_names = [ch.upper().strip() for ch in ch_names]
    has_loc = "LOC" in upper_names
    has_roc = "ROC" in upper_names
 
    if has_loc and has_roc:
        return rename_map
 
    # Find all channels containing "EOG"
    eog_channels = [(ch, ch.upper().strip()) for ch in ch_names if "EOG" in ch.upper()]
 
    # Determine left/right for each EOG channel
    # Remove the "EOG" part and look at what's left for clues
    LEFT_CLUES  = {"L", "V"}   # L=left, V=venstre (Danish),
    RIGHT_CLUES = {"R", "H"}   # R=right, H=højre (Danish),
 
    loc_candidate = None
    roc_candidate = None
 
    for original, upper in eog_channels:
        # Strip "EOG" to look at remaining characters for clues
        remainder = upper.replace("EOG", "")
 
        # Check each character in the remainder for left/right clues
        is_left  = any(c in LEFT_CLUES for c in remainder)
        is_right = any(c in RIGHT_CLUES for c in remainder)
 
        if is_left and not is_right and loc_candidate is None:
            loc_candidate = original
        elif is_right and not is_left and roc_candidate is None:
            roc_candidate = original
 
    # Fallback: if exactly 2 EOG channels and we couldn't determine sides,
    # assign by order (first = LOC, second = ROC)
    if (loc_candidate is None or roc_candidate is None) and len(eog_channels) == 2:
        if loc_candidate is None and roc_candidate is None:
            loc_candidate = eog_channels[0][0]
            roc_candidate = eog_channels[1][0]
        elif loc_candidate is None:
            # We found ROC but not LOC — the other one must be LOC
            other = [ch for ch, _ in eog_channels if ch != roc_candidate]
            if other:
                loc_candidate = other[0]
        elif roc_candidate is None:
            # We found LOC but not ROC — the other one must be ROC
            other = [ch for ch, _ in eog_channels if ch != loc_candidate]
            if other:
                roc_candidate = other[0]
 
    # Build the rename map
    if loc_candidate is not None and loc_candidate.upper().strip() != "LOC" and not has_loc:
        rename_map[loc_candidate] = "LOC"
    if roc_candidate is not None and roc_candidate.upper().strip() != "ROC" and not has_roc:
        rename_map[roc_candidate] = "ROC"
 
    return rename_map