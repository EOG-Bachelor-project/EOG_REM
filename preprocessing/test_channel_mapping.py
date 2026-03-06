import mne
import pandas as pd
from pathlib import Path

from preprocessing.index_file import index_sessions
from preprocessing.channel_standardization import build_rename_map


ROOT = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")
OUT_DIR = Path("local_csv_eog")
OUT_DIR.mkdir(exist_ok=True)


records = index_sessions(ROOT)

# ----------------------------------------------------
# Test only the first EDF file
# ----------------------------------------------------
rec = records[0]

print("\nTesting file:", rec.edf_path)

# Load EDF
raw = mne.io.read_raw_edf(rec.edf_path, preload=True, verbose=False)

print("\nBefore renaming:")
print(raw.ch_names)

# Build rename map
rename_map = build_rename_map(raw.ch_names)
print("\nRename map:", rename_map)

# Rename channels
raw.rename_channels(rename_map)

print("\nAfter renaming:")
print(raw.ch_names)

# ----------------------------------------------------
# Test EOG extraction
# ----------------------------------------------------
missing = [ch for ch in ["LOC", "ROC"] if ch not in raw.ch_names]

if missing:
    print("\nMissing channels:", missing)
else:
    print("\nLOC and ROC successfully detected")

    loc = raw.get_data(picks=["LOC"])[0]
    roc = raw.get_data(picks=["ROC"])[0]

    print("LOC shape:", loc.shape)
    print("ROC shape:", roc.shape)

    df = pd.DataFrame({
        "time_sec": raw.times,
        "LOC": loc,
        "ROC": roc
    })

    print("\nPreview of DataFrame:")
    print(df.head())

    # Save test CSV
    out_path = OUT_DIR / f"{rec.patient_id}_test_eog.csv"
    df.to_csv(out_path, index=False)

    print("\nSaved test CSV:", out_path)