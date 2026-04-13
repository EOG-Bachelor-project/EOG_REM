# Filename: main.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Resume-friendly pipeline runner. Processes patients in configurable
#              batch sizes, skipping those already processed (merged CSV exists).
#              Run repeatedly until all patients are done.
#
# Usage:
#   python main.py <raw_root>                          # default batch_size=10
#   python main.py <raw_root> --batch-size 5
#
# Example:
#   python main.py "L:/Auditdata/RBD PD/PD-RBD Glostrup Database_ok" --batch-size 5

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations  # for Python 3.10+ type hinting features

import argparse                     # for command-line argument parsing
import time                         # for measuring runtime  
import traceback                    # for detailed error traces
import numpy as np                  # for numerical operations
import pandas as pd                 # for data manipulation
from pathlib import Path            # for filesystem paths

from preprocessing.index_file import index_sessions
from preprocessing.edf_to_csv import edf_to_csv
from preprocessing.GSSC_to_csv import GSSC_to_csv
from preprocessing.extract_rems_n import extract_rems_from_edf
from preprocessing.em_to_csv import em_to_csv
from preprocessing.merge import merge_all
from analysis.feat_report import collect_features, generate_report

# =====================================================================
# Constants — output directories
# =====================================================================
EOG_DIR      = Path("eog_csv")
GSSC_DIR     = Path("gssc_csv")
REMS_DIR     = Path("extracted_rems")
EM_DIR       = Path("detected_ems")
MERGED_DIR   = Path("merged_csv_eog")
FEATURES_DIR = Path("features_csv")
REPORTS_DIR  = Path("reports")
 
for d in [EOG_DIR, GSSC_DIR, REMS_DIR, EM_DIR, MERGED_DIR, FEATURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
 
# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
 
AMPLITUDE_THRESH_UV = 300.0  # artefact masking threshold [µV]
 
 
# =====================================================================
# Helper — check if a patient has already been processed
# =====================================================================
def _is_processed(session_id: str) -> bool:
    """
    Check whether the final merged CSV exists for this session.
    If it does, we assume all earlier stages succeeded and skip.
    """
    merged_path = MERGED_DIR / f"{session_id}_contiguous_eog_merged.csv"
    return merged_path.exists()
 
 
# =====================================================================
# Core — process one patient through the full pipeline
# =====================================================================
def process_patient(rec) -> bool:
    """
    Run stages 1-6 for a single patient session.
 
    Returns True if successful, False if an error occurred.
    """
    session_id = rec.patient_id
    edf_path   = rec.edf_path
    lights_path = rec.txt_path
 
    print(f"\n{'=' * 70}")
    print(f"  Processing: {BOLD}{session_id}{RESET}")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()
 
    try:
        # ── Stage 1: EDF → EOG CSV ──────────────────────────────────
        print(f"\n{BOLD}[1/6] EDF → EOG CSV{RESET}")
        edf_to_csv(edf_path, out_dir=EOG_DIR, lights_path=lights_path)
 
        # ── Stage 2: GSSC sleep staging ─────────────────────────────
        print(f"\n{BOLD}[2/6] GSSC sleep staging{RESET}")
        gssc_df = GSSC_to_csv(edf_path, out_dir=GSSC_DIR, lights_path=lights_path)
 
        # Build hypno_int from GSSC output (needed by stages 3 & 5)
        stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
        hypno_int = gssc_df["stage"].map(stage_map).fillna(0).astype(int).values
 
        # ── Stage 3: Extract REM events ─────────────────────────────
        print(f"\n{BOLD}[3/6] Extract REM events{RESET}")
        extract_rems_from_edf(
            edf_path=edf_path,
            out_dir=REMS_DIR,
            lights_path=lights_path,
            gssc_df=gssc_df,
        )
 
        # ── Stage 4: Mask artefacts in EOG CSV ──────────────────────
        print(f"\n{BOLD}[4/6] Mask artefacts in EOG CSV{RESET}")
        eog_csv_path = EOG_DIR / f"{session_id}_{edf_path.stem}_eog.csv"
        eog_df = pd.read_csv(eog_csv_path)
 
        artefact_mask = (
            (np.abs(eog_df["LOC"].values) > AMPLITUDE_THRESH_UV) |
            (np.abs(eog_df["ROC"].values) > AMPLITUDE_THRESH_UV)
        )
        n_masked = int(artefact_mask.sum())
        eog_df.loc[artefact_mask, "LOC"] = np.nan
        eog_df.loc[artefact_mask, "ROC"] = np.nan
        eog_df.to_csv(eog_csv_path, index=False)
        print(f"    Artefact samples masked: {n_masked:,} / {len(eog_df):,}")
 
        # ── Stage 5: Detect & classify eye movements ────────────────
        print(f"\n{BOLD}[5/6] Detect & classify eye movements{RESET}")
        em_result = em_to_csv(
            edf_path=edf_path,
            gssc_df=gssc_df,
            hypno_int=hypno_int,
            out_dir=EM_DIR,
            lights_path=lights_path,
            use_Umaer=True,
        )
 
        # ── Stage 6: Merge into unified CSV ─────────────────────────
        print(f"\n{BOLD}[6/6] Merge into unified CSV{RESET}")
 
        eog_file       = EOG_DIR  / f"{session_id}_{edf_path.stem}_eog.csv"
        gssc_file      = GSSC_DIR / f"{session_id}_gssc.csv"
        events_file    = REMS_DIR / f"{session_id}_extracted_rems.csv"
        em_file        = EM_DIR   / f"{session_id}_em.csv"
        subepochs_file = EM_DIR   / f"{session_id}_subepochs.csv"
        output_file    = MERGED_DIR / f"{session_id}_{edf_path.stem}_eog_merged.csv"
 
        merge_all(
            eog_file=eog_file,
            gssc_file=gssc_file,
            events_file=events_file,
            em_file=em_file,
            output_file=output_file,
            subepochs_file=subepochs_file,
        )
 
        elapsed = time.perf_counter() - t0
        print(f"\n{GREEN}✓ {session_id} completed in {elapsed:.1f}s ({elapsed/60:.1f} min){RESET}")
        return True
 
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n{RED}✗ {session_id} FAILED after {elapsed:.1f}s: {e}{RESET}")
        traceback.print_exc()
        return False
 
 
# =====================================================================
# Main — batch runner with resume logic
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="EOG_REM pipeline — resume-friendly batch runner",
    )
    parser.add_argument(
        "raw_root",
        type=str,
        help="Root directory containing patient session folders (e.g. DCSM_1_a, DCSM_2_a, ...)",
    )
    parser.add_argument(
        "--batch-size", "-n",
        type=int,
        default=10,
        help="Number of NEW patients to process before stopping (default: 10)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        default=False,
        help="Skip the final feature extraction step",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=250.0,
        help="Sampling frequency in Hz for feature extraction (default: 250.0)",
    )
 
    args = parser.parse_args()
    raw_root   = Path(args.raw_root)
    batch_size = args.batch_size
    fs         = args.fs
 
    print(f"\n{'#' * 70}")
    print(f"#  EOG_REM Pipeline Runner")
    print(f"#  Root:       {raw_root}")
    print(f"#  Batch size: {batch_size}")
    print(f"#  Fs:         {fs} Hz")
    print(f"{'#' * 70}")
 
    total_start = time.perf_counter()
 
    # ── 1) Index all sessions ────────────────────────────────────────
    print(f"\n{BOLD}Indexing sessions...{RESET}")
    records = index_sessions(
        root_dir=raw_root,
        edf=True,
        csv=False,
        txt=True,
        strict=False,
    )
    print(f"Found {len(records)} sessions total")
 
    # ── 2) Filter to sessions with EDF files ─────────────────────────
    records = [r for r in records if r.edf_path is not None]
    print(f"Sessions with EDF files: {len(records)}")
 
    # ── 3) Check which are already processed ─────────────────────────
    already_done = []
    to_process   = []
 
    for rec in records:
        if _is_processed(rec.patient_id):
            already_done.append(rec.patient_id)
        else:
            to_process.append(rec)
 
    print(f"\n{BOLD}Status:{RESET}")
    print(f"  Already processed: {len(already_done)}")
    print(f"  Remaining:         {len(to_process)}")
    print(f"  Will process:      {min(batch_size, len(to_process))}")
 
    if not to_process:
        print(f"\n{GREEN}All sessions are already processed!{RESET}")
 
        # Still run feature extraction if requested
        if not args.skip_features:
            print(f"\n{BOLD}Running feature extraction on all merged CSVs...{RESET}")
            combined = collect_features(
                merged_dir=MERGED_DIR,
                fs=fs,
                pattern="*_merged.csv",
            )
            csv_path = FEATURES_DIR / "all_features.csv"
            combined.to_csv(csv_path, index=False)
            print(f"Feature CSV saved → {csv_path}")
 
            report_path = REPORTS_DIR / "all_features.html"
            generate_report(combined, report_path, title="all_features")
            print(f"HTML report saved → {report_path}")
 
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        return
 
    # ── 4) Process batch ─────────────────────────────────────────────
    completed = 0
    failed    = 0
 
    for rec in to_process:
        if completed >= batch_size:
            print(f"\n{BOLD}Batch limit ({batch_size}) reached — stopping.{RESET}")
            break
 
        success = process_patient(rec)
        if success:
            completed += 1
        else:
            failed += 1
 
    # ── 5) Feature extraction ────────────────────────────────────────
    if not args.skip_features and completed > 0:
        print(f"\n{'=' * 70}")
        print(f"{BOLD}Running feature extraction on all merged CSVs...{RESET}")
        print(f"{'=' * 70}")
 
        pattern = "*_merged.csv"
 
        try:
            combined = collect_features(
                merged_dir=MERGED_DIR,
                fs=fs,
                pattern=pattern,
            )
            csv_path = FEATURES_DIR / "all_features.csv"
            combined.to_csv(csv_path, index=False)
            print(f"Feature CSV saved → {csv_path}")
 
            report_path = REPORTS_DIR / "all_features.html"
            generate_report(combined, report_path, title="all_features")
            print(f"HTML report saved → {report_path}")
 
            print(f"\n{GREEN}Features: {combined.shape[0]} subjects x {combined.shape[1]-1} features{RESET}")
        except Exception as e:
            print(f"\n{RED}Feature extraction failed: {e}{RESET}")
            traceback.print_exc()
 
    # ── 6) Summary ───────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    remaining = len(to_process) - completed - failed
 
    print(f"\n{'#' * 70}")
    print(f"#  {BOLD}Pipeline Summary{RESET}")
    print(f"#  Processed:  {completed}")
    print(f"#  Failed:     {failed}")
    print(f"#  Remaining:  {remaining}")
    print(f"#  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'#' * 70}")
 
    if remaining > 0:
        print(f"\n  Run again to process the next batch of patients.")
 
 
# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    main()
 