# index_file.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data container 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@dataclass(frozen=True)
class PatientRecord:
    """
    Structured container for one patient/session.

    Attributes
    ----------
    patient_id : str
        Folder name (e.g., DCSM_123_a)
    folder : Path
        Path to the patient directory
    edf_path : Optional[Path]
        Path to contiguous.edf (if requested and found)
    csv_path : Optional[Path]
        Path to hypnogram.csv (if requested and found)
    txt_path : Optional[Path]
        Path to lights.txt (if requested and found)
    """
    patient_id: str                 
    folder: Path                    
    edf_path: Optional[Path] = None 
    csv_path: Optional[Path] = None 
    txt_path: Optional[Path] = None 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Match folder names
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
    edf, csv, txt : bool
        Specify which file types should be searched for and included.
    recursive : bool
        - True  -> search inside subfolders (rglob)
        - False -> only look directly inside patient folder
    strict : bool
        - True -> raise error if any *requested* file is missing in a matching folder.
        - False -> skip folders missing any *requested* file.

    Returns
    -------
    list[PatientRecord]
        One record per valid patient folder.
    """
    # Ensure Path object
    root_dir = Path(root_dir)

    # Safety check
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    # ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
    # Internal helper to locate a file
    # ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
    def find_file(folder: Path, filename: str) -> Optional[Path]:
        """
        Search for a file either directly in the folder or recursively in subfolders.
        """
        if recursive:
            return next(folder.rglob(filename), None)
        path = folder / filename
        return path if path.exists() else None
    
    records: list[PatientRecord] = []
    skipped = 0

    # =====================================================================
    # Iterate through patient folders
    # =====================================================================
    for folder in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        print("Found folder: ", folder.name)

        if not SESSION_RE.match(folder.name):
            print(" -> skipped (name does not match the pattern)")
            continue

        edf_path = find_file(folder, EDF_NAME) if edf else None
        csv_path = find_file(folder, CSV_NAME) if csv else None
        txt_path = find_file(folder, TXT_NAME) if txt else None

        if edf:
            print(" EDF exists:", edf_path is not None)
        if csv:
            print(" CSV exists:", csv_path is not None)
        if txt:
            print(" TXT exists:", txt_path is not None)

        # =====================================================================
        # Check for missing requested files
        # =====================================================================
        missing = []
        if edf and edf_path is None: missing.append(EDF_NAME)
        if csv and csv_path is None: missing.append(CSV_NAME)
        if txt and txt_path is None: missing.append(TXT_NAME)
        
        if missing:
            skipped += 1
            msg = f"{folder.name} missing {missing} (recursive={recursive})"
            if strict:
                raise FileNotFoundError(msg)
            else:
                print("Skipping:", msg)
                continue
        
        # =====================================================================
        # Store the record
        # =====================================================================
        records.append(
            PatientRecord(
                patient_id=folder.name,
                folder=folder,
                edf_path=edf_path,
                csv_path=csv_path,
                txt_path=txt_path
            )   
        )

    # Final safety check
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