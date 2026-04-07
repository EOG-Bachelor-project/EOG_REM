# Temporary diagnostic script — delete after use
import mne
from pathlib import Path

raw = mne.io.read_raw_edf(
    Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok/DCSM_1_a/contiguous.edf"),
    preload=False,
    verbose=False
)

for ch in raw.ch_names:
    if any(x in ch.upper() for x in ["EOG", "LOC", "ROC", "EOGV", "EOGH"]):
        idx = raw.ch_names.index(ch)
        print(f"{ch}: unit = {raw._raw_extras[0]['units'][idx]}")