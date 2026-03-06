# inspect_channels.py

from pathlib import Path
import mne

from index_file import index_sessions

ROOT = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")

records = index_sessions(ROOT)

for rec in records[:5]:
    print("\n================================")
    print("Session: ", rec.patient_id)
    print("EDF: ", rec.edf_path)

    raw = mne.io.read_raw_edf(rec.edf_path, preload=False, verbose=False)

    print("Channels:")
    print(raw.ch_names)
