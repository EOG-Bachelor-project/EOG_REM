from pathlib import Path
import mne

from index_file import index_sessions


ROOT = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")

records = index_sessions(ROOT)

keywords = ["EOG", "LOC", "ROC", "E1", "E2"]
all_eog_names = set()

for rec in records:
    try:
        raw = mne.io.read_raw_edf(rec.edf_path, preload=False, verbose=False)

        for ch in raw.ch_names:
            ch_upper = ch.upper()
            if any(keyword in ch_upper for keyword in keywords):
                all_eog_names.add(ch)

    except Exception as e:
        print(f"FAILED for {rec.patient_id}: {e}")

print("\nUnique EOG-like channel names found:")
for ch in sorted(all_eog_names):
    print(ch)