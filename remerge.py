from pathlib import Path
from preprocessing.merge import merge_all

MERGED_DIR   = Path("merged_csv_eog")
EOG_DIR      = Path("eog_csv")
GSSC_DIR     = Path("gssc_csv")
REMS_DIR     = Path("extracted_rems")
EM_DIR       = Path("detected_ems")
EEG_DIR      = Path("eeg_csv")

import re
for eeg_file in sorted(EEG_DIR.glob("*_eeg.csv")):
    session_id = eeg_file.stem.replace("_eeg", "")
    
    # find matching files
    eog_files  = list(EOG_DIR.glob(f"{session_id}_*_eog.csv"))
    if not eog_files:
        # try .csv.gz
        eog_files = list(EOG_DIR.glob(f"{session_id}_*_eog.csv.gz"))
    if not eog_files:
        print(f"Skipping {session_id} — no EOG file found")
        continue

    eog_file       = eog_files[0]
    gssc_file      = GSSC_DIR / f"{session_id}_gssc.csv"
    events_file    = REMS_DIR / f"{session_id}_extracted_rems.csv"
    em_file        = EM_DIR   / f"{session_id}_em.csv"
    subepochs_file = EM_DIR   / f"{session_id}_subepochs.csv"

    # derive output filename from eog filename
    edf_stem    = eog_file.stem.replace(f"{session_id}_", "").replace("_eog", "")
    output_file = MERGED_DIR / f"{session_id}_{edf_stem}_eog_merged.csv"

    # check all inputs exist
    missing = [f for f in [eog_file, gssc_file, events_file, em_file] if not f.exists()]
    if missing:
        print(f"Skipping {session_id} — missing: {[f.name for f in missing]}")
        continue

    print(f"\nRe-merging: {session_id}")
    try:
        merge_all(
            eog_file       = eog_file,
            gssc_file      = gssc_file,
            events_file    = events_file,
            em_file        = em_file,
            output_file    = output_file,
            subepochs_file = subepochs_file,
            eeg_file       = eeg_file,
        )
    except Exception as e:
        print(f"  FAILED: {e}")