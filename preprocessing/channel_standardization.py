# channel_standardization.py

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Constants
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
CHANNEL_ALIASES = {
    # left eye
    "EOGH": "LOC",
    "LOC-A2": "LOC",
    "LEFT EOG": "LOC",
    "E1": "LOC",

    # right eye
    "EOGV": "ROC",
    "ROC-A1": "ROC",
    "RIGHT EOG": "ROC",
    "E2": "ROC",
}

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Helper functions
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# This function normalizes channel names by converting them to uppercase and stripping whitespace.
def normalize_channel_name(name: str) -> str:
    return name.upper().strip()

# =====================================================================
# Functions
# =====================================================================
def build_rename_map(ch_names: list[str]) -> dict[str, str]:
    rename_map = {}

    for ch in ch_names:
        norm = normalize_channel_name(ch)

        if norm in CHANNEL_ALIASES:
            rename_map[ch] = CHANNEL_ALIASES[norm]

    return rename_map
