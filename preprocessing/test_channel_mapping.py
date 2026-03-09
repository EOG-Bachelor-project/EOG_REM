import mne
from pathlib import Path

from index_file import index_sessions
from edf_to_csv import edf_to_csv


ROOT = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")

records = index_sessions(ROOT)

# Take the first EDF
rec = records[0]

print("\nTesting conversion for:")
print(rec.edf_path)

edf_to_csv(rec.edf_path)