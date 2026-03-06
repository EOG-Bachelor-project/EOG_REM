import mne
from pathlib import Path

from index_file import index_sessions
from channel_standardization import build_rename_map


ROOT = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")

records = index_sessions(ROOT)

rec = records[0]

raw = mne.io.read_raw_edf(rec.edf_path, preload=False, verbose=False)

print("Before:", raw.ch_names)

rename_map = build_rename_map(raw.ch_names)
print("Rename map:", rename_map)

raw.rename_channels(rename_map)

print("After:", raw.ch_names)