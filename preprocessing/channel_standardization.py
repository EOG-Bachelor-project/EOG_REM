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
    "EOGV": "LOC",   
    "EOGV-Ref": "LOC",
    "EOGV-A2": "LOC",
    "eogl-a2": "LOC",

    # right eye
    "EOGH": "ROC",   
    "EOGH-Ref": "ROC",
    "EOGH-A1": "ROC",
    "eogr-a1": "ROC",

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
