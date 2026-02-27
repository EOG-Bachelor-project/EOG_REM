# index_file.py

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Imports
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
from __future__ import annotations
import csv
import re
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data container 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@dataclass(frozen=True)
class SessionRecord:
    """
    Structured container for one session.

    Attributes
    ----------
    patient_id : str
        Full folder name (e.g., DCSM_123_a).
    patient_number : int
        Numeric patient identifier extracted from folder name (e.g., 123).
    session_type : str
        Session lable extracted from folder name (e.g., "a", "b", "c").
    folder : Path
        Path to the patient directory.
    edf_path : Optional[Path]
        Path to contiguous.edf (if requested and found).
    csv_path : Optional[Path]
        Path to hypnogram.csv (if requested and found).
    txt_path : Optional[Path]
        Path to lights.txt (if requested and found).
    """
    patient_id: str
    patient_number: int
    session_type: str                 
    folder: Path                    
    edf_path: Optional[Path] = None 
    csv_path: Optional[Path] = None 
    txt_path: Optional[Path] = None 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Simple ANSI color codes for terminal output
BOLD = "\033[1m"
RESET = "\033[0m"

# Match folder names
SESSION_RE = re.compile(r"^DCSM_(\d+)_([abc])$")


# Expected file names within each patient folder
EDF_NAME = "contiguous.edf"
CSV_NAME = "hypnogram.csv"
TXT_NAME = "lights.txt"

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Functions
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# —————————————————————————————————————————————————————————————————————
# Main function to index patient sessions
# —————————————————————————————————————————————————————————————————————
def index_sessions(root_dir: str | Path, 
                   edf: bool = True, 
                   csv: bool = True,
                   txt: bool = True,
                   recursive: bool = False,
                   strict: bool = False,
                ) -> list[SessionRecord]:
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
    
    records: list[SessionRecord] = []
    skipped = 0

    # --- 1) Iterate through patient folders ---
    for folder in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        print("\nFound folder: ", folder.name)

        match = SESSION_RE.match(folder.name)
        if not match:
            print(" -> skipped (name does not match the pattern)")
            continue

        patient_nr = int(match.group(1))
        session_tp = match.group(2) # "a", "b", "c", ...

        edf_path = find_file(folder, EDF_NAME) if edf else None
        csv_path = find_file(folder, CSV_NAME) if csv else None
        txt_path = find_file(folder, TXT_NAME) if txt else None

        if edf:
            print("  - EDF exists:", edf_path is not None)
        if csv:
            print("  - CSV exists:", csv_path is not None)
        if txt:
            print("  - TXT exists:", txt_path is not None)

        # --- 2) Check for missing requested files ---
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
                print(f" {BOLD}Skipping:{RESET}", msg)
                continue
        

        # --- 3) Store the record ---
        record_n = SessionRecord(
            patient_id=folder.name,
            patient_number=patient_nr,
            session_type=session_tp,
            folder=folder,
            edf_path=edf_path,
            csv_path=csv_path,
            txt_path=txt_path
        )  
        records.append(record_n)
        
        print(f" {BOLD}Added:{RESET} {record_n.patient_id}  (patient={record_n.patient_number}, session={record_n.session_type})")

    # --- 4) Final safety check ---
    if not records:
        raise RuntimeError(
            "No valid patient records found. "
            "Check root path or enable recursive=True."
        )
    
    # --- 5) Summary ---
    print("\n")
    print("="*50)
    print(f"Indexed {len(records)} session. Skipped {skipped}.")
    print("="*50)
    return records

# —————————————————————————————————————————————————————————————————————
# Function to convert the list of SessionRecords to a DataFrame
# —————————————————————————————————————————————————————————————————————

def records_to_df(records: Iterable[SessionRecord], out_root: str | Path | None = None, sort_alg: str = "stable") -> pd.DataFrame:
    """
    Converts SessionRecord objects into a pandas DataFrame, and optionally saves it as a CSV file.

    Parameters
    ----------
    records : Iterable[SessionRecord]
        An iterable of SessionRecord objects to be converted into a DataFrame.
    out_root : str | Path | None
        Optional path to save the DataFrame as a CSV file. If None, the DataFrame will not be saved. Default is None.
    sort_alg : str
        Sorting algorithm to use when sorting the DataFrame. Options include "stable", "quicksort", "mergesort", "heapsort". Default is "stable". \\
        See pandas documentation for more details: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html#pandas.DataFrame.sort_values

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the session records with columns corresponding to the attributes of SessionRecord.
    """
    # Ensure Path object for output root
    out_root = Path(out_root) if out_root is not None else None

    # Validate sorting algorithm
    if sort_alg not in ["stable", "quicksort", "mergesort", "heapsort"]:
        raise ValueError(f"Invalid sort_alg: {sort_alg}. Must be one of 'stable', 'quicksort', 'mergesort', 'heapsort'.")

    # Creat empy list to hold rows for DataFrame
    rows = []

    # --- 1) Convert each dataclass record into a dictionary row ---
    for r in records:
        # Define output directory per patient if out_root is provided
        out_dir = (out_root / r.patient_id) if out_root is not None else None

        row = {
            "patient_id": r.patient_id,
            "patient_number": r.patient_number,
            "session_type": r.session_type,
            "folder": str(r.folder),
            "edf_path": str(r.edf_path) if r.edf_path else None,
            "csv_path": str(r.csv_path) if r.csv_path else None,
            "txt_path": str(r.txt_path) if r.txt_path else None,
            "out_dir": str(out_dir) if out_dir else None,
        }
        rows.append(row)
    
    # --- 2) Create DataFrame ---
    df = pd.DataFrame(rows)

    # --- 3) Sort sessions ---
    df = df.sort_values(by=["patient_number", "session_type"], kind=sort_alg).reset_index(drop=True)

    return df

# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –
# Test
# – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - – - –

# Path
p = Path("L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok")

# Only index EDF files
records = index_sessions(root_dir=p, edf=True, csv=False, txt=False, recursive=False)
df = records_to_df(records)

print("\nPreview of DataFrame:")
print(df.head())