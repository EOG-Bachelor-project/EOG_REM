# index_file.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Helpers
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

@dataclass(frozen=True)
class PatientRecord:
    patient_id: str                 # Unique identifier for the patient: DCSM_x_a
    folder: Path                    # Path to the patient's folder containing the EDF, CSV, and TXT files
    edf_path: Optional[Path] = None # Path to the EDF file containing the PSG data
    csv_path: Optional[Path] = None # Path to the CSV file (if available)
    txt_path: Optional[Path] = None # Path to the TXT file (if available)


# Pattern
SESSION_RE = re.compile(r"^DCSM_(\d+)_a$")


# Expected file names within each patient folder
EDF_NAME = "contiguous.edf"
CSV_NAME = "hypnogram.csv"
TXT_NAME = "lights.txt"

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Function
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
def index_sessions(root_dir: str | Path, 
                   edf: bool = True, 
                   csv: bool = True,
                   txt: bool = True,
                   recursive: bool = False,
                   strict: bool = False,
                ) -> list[PatientRecord]:
    """
    Go through folders named DCSM_x_a and find contiguous.edf, hypnogram.csv, lights.txt for each

    Parameters
    ----------
        root_dir: str | Path
            The root directory containing the patient folders.
        edf : bool 
            Whether to include the EDF file path in the PatientRecord. Default is True.
        csv : bool
            Whether to include the CSV file path in the PatientRecord. Default is True.
        txt : bool 
            Whether to include the TXT file path in the PatientRecord. Default is True.
        recursive : bool
            - True  -> search inside subfolders (rglob)
            - False -> only look directly inside patient folder
        strict : bool
            - True -> raise error if any *requested* file is missing in a matching folder.
            - False -> skip folders missing any *requested* file.
    """
    root_dir = Path(root_dir)  # convert str -> Path if needed

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    # Helper function to find a file in a folder (optionally recursively)
    def find_file(folder: Path, filename: str) -> Optional[Path]:
        if recursive:
            return next(folder.rglob(filename), None)
        path = folder / filename
        return path if path.exists() else None
    
    records: list[PatientRecord] = []
    skipped = 0

    for folder in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        print("Found folder: ", folder.name)

        if not SESSION_RE.match(folder.name):
            print(" -> skipped (name does not match the pattern)")
            continue

        edf_path = folder / EDF_NAME if edf else None
        csv_path = folder / CSV_NAME if csv else None
        txt_path = folder / TXT_NAME if txt else None

        print(" EDF exists:", edf_path.exists())
        print(" CSV exists:", csv_path.exists())
        print(" TXT exists:", txt_path.exists())

        # Validate that the required files exist
        missing = []
        if edf and not edf_path.exists(): missing.append(EDF_NAME)
        if csv and not csv_path.exists(): missing.append(CSV_NAME)
        if txt and not txt_path.exists(): missing.append(TXT_NAME)
        
        if missing:
            skipped += 1
            msg = f"{folder.name} missing {missing} (recursive={recursive})"
            if strict:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                continue
            
        records.append(
            PatientRecord(
                patient_id=folder.name,
                folder=folder,
                edf_path=edf_path,
                csv_path=csv_path,
                txt_path=txt_path
            )   
        )

    if not records:
        raise RuntimeError(
            "No valid patient records found. "
            "Check root path or enable recursive=True."
        )
        
    print(f"Indexed {len(records)} session. Skipped {skipped}.")
    return records

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# Path
p = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")

# Only index EDF files
records = index_sessions(root_dir=p, edf=True, csv=False, txt=False, recursive=False)

