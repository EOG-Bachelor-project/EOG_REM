# Filename: main.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Resume-friendly pipeline runner. Processes patients in configurable
#              batch sizes, skipping those already processed (merged CSV exists).
#              Also supports running feature extraction and HTML report steps
#              on their own so you don't have to redo the slow preprocessing.
#
# Usage:
#   python main.py <raw_root>                     # same as before: process patients, batch=10
#   python main.py <raw_root> --batch-size 5      # smaller batch
#   python main.py extract                        # extract features from merged_csv_eog/
#   python main.py report                         # regenerate HTML from cached features (fast)
#   python main.py all <raw_root>                 # process + extract + report in one go
#
# Re-running specific stages:
#   The pipeline skips any stage whose output file already exists.
#   To re-run a specific stage, delete its output files and the merged CSV,
#   then run again. Examples:
#
# Re-run only EEG extraction + merge (stages 6-7):
#   rm eeg_csv/*_eeg.csv merged_csv_eog/*_merged.csv
#   python main.py <raw_root> --batch-size 9999
#
# Re-run only merge (stage 7):
#   rm merged_csv_eog/*_merged.csv
#   python main.py <raw_root> --batch-size 9999
#
# Re-run everything from scratch:
#   rm eog_csv/* gssc_csv/* extracted_rems/* detected_ems/* eeg_csv/* merged_csv_eog/*
#   python main.py <raw_root> --batch-size 9999
#
# Note: the merged CSV must always be deleted, because _is_processed()
#       checks for it to decide whether a patient needs processing at all.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations  # for Python 3.10+ type hinting features

import argparse                     # for command-line argument parsing
import sys                          # for sys.exit
import time                         # for measuring runtime  
import traceback                    # for detailed error traces
import numpy as np                  # for numerical operations
import pandas as pd                 # for data manipulation
import mne                          # for loading EDF files and handling raw EEG/EOG data
from pathlib import Path            # for filesystem paths

from preprocessing.index_file import index_sessions
from preprocessing.edf_to_csv import edf_to_csv
from preprocessing.GSSC_to_csv import GSSC_to_csv
from preprocessing.extract_rems_n import extract_rems_from_edf
from preprocessing.em_to_csv import em_to_csv
from preprocessing.merge import merge_all
from preprocessing.channel_standardization import build_rename_map
from preprocessing.eeg_to_csv import eeg_to_csv
from analysis.feat_report import collect_features, generate_report

# =====================================================================
# Constants — output directories
# =====================================================================
EOG_DIR      = Path("eog_csv")
GSSC_DIR     = Path("gssc_csv")
REMS_DIR     = Path("extracted_rems")
EM_DIR       = Path("detected_ems")
MERGED_DIR   = Path("merged_csv_eog")
MERGED_DIR_b = Path("merged_csv_eog_backup")
FEATURES_DIR = Path("features_csv")
REPORTS_DIR  = Path("reports")
EEG_DIR      = Path("eeg_csv")
 
for d in [EOG_DIR, GSSC_DIR, REMS_DIR, EM_DIR, MERGED_DIR, FEATURES_DIR, REPORTS_DIR, EEG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_FEATURE_CSV = FEATURES_DIR / "features.csv"
DEFAULT_REPORT_HTML = REPORTS_DIR  / "features_report.html"
 
# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
 
AMPLITUDE_THRESH_UV = 300.0  # artefact masking threshold [µV]
 
 
# =====================================================================
# Helper — check which stages have already been completed
# =====================================================================
def _is_processed(session_id: str) -> bool:
    """
    Check whether the final merged CSV exists for this session.
    If it does, we assume all earlier stages succeeded and skip.
    """
    merged_path = MERGED_DIR / f"{session_id}_contiguous_eog_merged.csv"
    return merged_path.exists()


def _check_existing_outputs(session_id: str, edf_stem: str) -> dict[str, bool]:
    """
    Check which intermediate output files already exist for a session.

    Returns a dict mapping stage names to True (output exists) or False.
    """
    return {
        "eog":       (EOG_DIR  / f"{session_id}_{edf_stem}_eog.csv").exists(),
        "gssc":      (GSSC_DIR / f"{session_id}_gssc.csv").exists(),
        "rems":      (REMS_DIR / f"{session_id}_extracted_rems.csv").exists(),
        "em":        (EM_DIR   / f"{session_id}_em.csv").exists(),
        "subepochs": (EM_DIR   / f"{session_id}_subepochs.csv").exists(),
        "eeg":       (EEG_DIR  / f"{session_id}_eeg.csv").exists(),
    }

 
def _wait_for_file(path: Path, timeout: float = 10.0, interval: float = 0.5) -> Path:
    """
    Wait for a file to appear on disk. Useful when a previous stage
    just wrote it but the OS hasn't flushed/synced yet.

    Parameters
    ----------
    path : Path
        Expected file path.
    timeout : float
        Max seconds to wait before raising FileNotFoundError.
        **Default is 10**.
    interval : float
        Seconds between checks.
        **Default is 0.5**.

    Returns
    -------
    Path
        The same path, once it exists.
    """
    elapsed = 0.0
    while not path.exists():
        if elapsed >= timeout:
            raise FileNotFoundError(
                f"File not found after {timeout}s: {path}\n"
                f"  (directory contents: {[p.name for p in path.parent.iterdir()] if path.parent.exists() else 'parent dir missing'})"
            )
        time.sleep(interval)
        elapsed += interval

    return path


def _compress_intermediates(session_id: str, edf_stem: str) -> float:
    """
    Gzip intermediate CSV files for a session after a successful merge.
    Compressed files are saved as .csv.gz and the uncompressed .csv is removed.

    If a stage needs to be re-run later, pandas reads .csv.gz transparently —
    but _check_existing_outputs looks for .csv, so the stage will re-run
    and produce a fresh .csv automatically.

    Returns the total disk space freed in MB.
    """
    import gzip
    import shutil

    intermediate_files = [
        EOG_DIR  / f"{session_id}_{edf_stem}_eog.csv",
        GSSC_DIR / f"{session_id}_gssc.csv",
        REMS_DIR / f"{session_id}_extracted_rems.csv",
        EM_DIR   / f"{session_id}_em.csv",
        EM_DIR   / f"{session_id}_subepochs.csv",
        EEG_DIR  / f"{session_id}_eeg.csv",
    ]

    freed_bytes = 0
    for f in intermediate_files:
        if f.exists():
            size_before = f.stat().st_size
            gz_path = f.with_suffix(".csv.gz")

            with open(f, "rb") as f_in:
                with gzip.open(gz_path, "wb", compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)

            size_after = gz_path.stat().st_size
            saved = size_before - size_after
            freed_bytes += saved
            f.unlink()  # remove original .csv, keep .csv.gz
            print(f"    📦 {f.name}: {size_before / (1024**2):.1f} MB → {size_after / (1024**2):.1f} MB ({saved / (1024**2):.1f} MB saved)")

    freed_mb = freed_bytes / (1024 ** 2)
    print(f"    ✓ Freed {freed_mb:.1f} MB for {session_id}")
    return freed_mb

 
# =====================================================================
# Core — process one patient through the full pipeline
# =====================================================================
def process_patient(rec, cleanup: bool = True) -> bool:
    """
    Run stages 1-7 for a single patient session, skipping any stage
    whose output file already exists on disk.

    Parameters
    ----------
    rec : SessionRecord
        A record containing patient_id, edf_path, and txt_path.
    cleanup : bool
        Whether to compress intermediate CSVs after merging to save disk space.
        **Default is True** (intermediates are deleted after merging).
    
    Returns
    -------
    bool
        True if processing succeeded, False if an error occurred.
    """
    session_id  = rec.patient_id
    edf_path    = rec.edf_path
    lights_path = rec.txt_path

    print(f"\n{'=' * 70}")
    print(f"  Processing: {BOLD}{session_id}{RESET}")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()

    try:
        # ── Check which outputs already exist ───────────────────────
        existing = _check_existing_outputs(session_id, edf_path.stem)
 
        skip_summary = [f"{name}: {'EXISTS' if exists else 'missing'}"
                        for name, exists in existing.items()]
        print(f"  Intermediate files: {', '.join(skip_summary)}")
 
        # Stages that need the EDF: 1 (eog), 2 (gssc), 3 (rems), 5 (em), 6 (eeg)
        needs_edf = (
            not existing["eog"]  or
            not existing["gssc"] or
            not existing["rems"] or
            not existing["em"]   or
            not existing["eeg"]
        )
 
        # ── Load EDF once (only if at least one stage needs it) ─────
        raw = None
        if needs_edf:
            print(f"\n{BOLD}Loading EDF + renaming channels{RESET}")
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            rename_map = build_rename_map(raw.ch_names)
            if rename_map:
                raw.rename_channels(rename_map)
            print(f"    sfreq: {raw.info['sfreq']} Hz  |  channels: {len(raw.ch_names)}")
        else:
            print(f"\n  All intermediate files exist — skipping EDF load.")
 
        # ── Stage 1: EDF → EOG CSV ──────────────────────────────────
        if existing["eog"]:
            print(f"\n{BOLD}[1/7] EDF → EOG CSV — SKIPPED (already exists){RESET}")
        else:
            print(f"\n{BOLD}[1/7] EDF → EOG CSV{RESET}")
            edf_to_csv(edf_path, raw=raw, out_dir=EOG_DIR, lights_path=lights_path)
 
        # ── Stage 2: GSSC sleep staging ─────────────────────────────
        if existing["gssc"]:
            print(f"\n{BOLD}[2/7] GSSC sleep staging — SKIPPED (already exists){RESET}")
            gssc_df = pd.read_csv(GSSC_DIR / f"{session_id}_gssc.csv")
        else:
            print(f"\n{BOLD}[2/7] GSSC sleep staging{RESET}")
            gssc_df = GSSC_to_csv(edf_path, raw=raw, out_dir=GSSC_DIR, lights_path=lights_path)
 
        stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
        hypno_int = gssc_df["stage"].map(stage_map).fillna(0).astype(int).values
 
        # ── Stage 3: Extract REM events ─────────────────────────────
        if existing["rems"]:
            print(f"\n{BOLD}[3/7] Extract REM events — SKIPPED (already exists){RESET}")
            df, loc, roc, loc_clean, roc_clean = None, None, None, None, None
        else:
            print(f"\n{BOLD}[3/7] Extract REM events{RESET}")
            df, loc, roc, loc_clean, roc_clean = extract_rems_from_edf(
                edf_path=edf_path,
                raw=raw,
                out_dir=REMS_DIR,
                lights_path=lights_path,
                gssc_df=gssc_df,
            )
 
        # ── Stage 4: Mask artefacts in EOG CSV ──────────────────────
        print(f"\n{BOLD}[4/7] Mask artefacts in EOG CSV{RESET}")
        eog_csv_path = _wait_for_file(
            EOG_DIR / f"{session_id}_{edf_path.stem}_eog.csv"
        )
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
        if existing["em"]:
            print(f"\n{BOLD}[5/7] Detect & classify eye movements — SKIPPED (already exists){RESET}")
        else:
            print(f"\n{BOLD}[5/7] Detect & classify eye movements{RESET}")
            em_to_csv(
                edf_path=edf_path,
                raw=raw,
                hypno_int=hypno_int,
                out_dir=EM_DIR,
                lights_path=lights_path,
            )
 
        # ── Stage 6: Extract EEG signals ────────────────────────────
        if existing["eeg"]:
            print(f"\n{BOLD}[6/7] Extract EEG signals — SKIPPED (already exists){RESET}")
        else:
            print(f"\n{BOLD}[6/7] Extract EEG signals{RESET}")
            # If stage 3 was skipped, we need to re-run it to get loc/roc/loc_clean/roc_clean
            if loc_clean is None:
                print("    Re-running stage 3 to get filtered signals for EEG extraction...")
                df, loc, roc, loc_clean, roc_clean = extract_rems_from_edf(
                    edf_path=edf_path,
                    raw=raw,
                    out_dir=REMS_DIR,
                    lights_path=lights_path,
                    gssc_df=gssc_df,
                )
            eeg_to_csv(
                edf_path=edf_path,
                loc=loc,
                roc=roc,
                loc_clean=loc_clean,
                roc_clean=roc_clean,
                out_dir=EEG_DIR,
                lights_path=lights_path,
            )
 
        # ── Stage 7: Merge into unified CSV ─────────────────────────
        print(f"\n{BOLD}[7/7] Merge into unified CSV{RESET}")
        eog_file       = _wait_for_file(EOG_DIR  / f"{session_id}_{edf_path.stem}_eog.csv")
        gssc_file      = _wait_for_file(GSSC_DIR / f"{session_id}_gssc.csv")
        events_file    = _wait_for_file(REMS_DIR / f"{session_id}_extracted_rems.csv")
        em_file        = _wait_for_file(EM_DIR   / f"{session_id}_em.csv")
        eeg_file       = _wait_for_file(EEG_DIR  / f"{session_id}_eeg.csv")
        subepochs_file = EM_DIR / f"{session_id}_subepochs.csv"                             # optional, don't wait
        output_file    = MERGED_DIR / f"{session_id}_{edf_path.stem}_eog_merged.csv"
 
 
        merge_all(
            eog_file=eog_file,
            gssc_file=gssc_file,
            events_file=events_file,
            em_file=em_file,
            output_file=output_file,
            subepochs_file=subepochs_file,
            eeg_file=eeg_file,
        )
 
        # ── Cleanup: compress intermediate CSVs to free disk space ────
        if cleanup:
            print(f"\n{BOLD}[Cleanup] Compressing intermediate CSVs{RESET}")
            _compress_intermediates(session_id, edf_path.stem)
        else:
            print(f"\n  Intermediates kept uncompressed (--keep-intermediates was set)")
 
        elapsed = time.perf_counter() - t0
        print(f"\n{GREEN}✓ {session_id} completed in {elapsed:.1f}s ({elapsed/60:.1f} min){RESET}")
        return True
 
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n{RED}✗ {session_id} FAILED after {elapsed:.1f}s: {e}{RESET}")
        traceback.print_exc()
        return False


# =====================================================================
# Step runners — each does one thing
# =====================================================================
def run_process(raw_root: Path, batch_size: int, cleanup: bool = True) -> None:
    """Run the original resume-friendly patient batch processor."""
    if not raw_root.is_dir():
        print(f"Error: '{raw_root}' is not a directory.")
        sys.exit(1)

    sessions = index_sessions(raw_root)                             # find all sessions with EDF+TXT
    todo = [s for s in sessions if not _is_processed(s.patient_id)] # filter out already processed sessions
    n_already = len(sessions) - len(todo)                           # count how many are already done   

    # --- Run summary before starting ---
    print(f"\n{'='*70}")
    print(f"    {BOLD}Pipeline Run{RESET}")
    print(f"    Total sessions found : {len(sessions)}")
    print(f"    Already processed    : {n_already}")
    print(f"    Remaining            : {len(todo)}")
    print(f"    Target successes     : {batch_size}")
    print(f"\n{'='*70}")

    if not todo:
        print(f"\n{GREEN}All patients already processed.{RESET}\n")
        return

    t_start = time.perf_counter()
    ok = fail = 0
    failed_ids = []

    for rec in todo:
        if process_patient(rec, cleanup=cleanup):
            ok += 1
        else:
            fail += 1
            failed_ids.append(rec.patient_id)
        
        if ok >= batch_size:
            break
    
    elapsed = time.perf_counter() - t_start
    remaining = len(todo) - ok - fail

    # --- Run summary after completion ---
    print(f"\n{'='*70}")    
    print(f"    {BOLD}Run Summary{RESET}")
    print(f"\n{'='*70}")
    print(f"    {GREEN}Successful         : {ok}{RESET}")
    print(f"    {RED}FAILED        : {fail}{RESET}")
    print(f"    Attempted     : {ok + fail}")
    print(f"    Still pending : {remaining}")
    print(f"    Total time    : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    
    if ok > 0:
        avg = elapsed / ok
        print(f"    Avg per success: {avg:.1f}s ({avg / 60:.1f} min)")

    if failed_ids:
        print(f"\n    {RED}Failed sessions:{RESET}")
        for sid in failed_ids:
            print(f"    - {sid}")
    
    if remaining > 0:
        print(f"\n  Run again to process next batch.")
    else:
        print(f"\n   {GREEN}All patients processed!{RESET}")
    print(f"\n{'='*70}")

def run_extract(merged_dir: Path, fs: float, pattern: str, csv_path: Path, patient_excel: Path | None = None) -> None:
    """Collect features from merged CSVs and cache to a single feature table."""
    if not merged_dir.is_dir():
        print(f"Error: '{merged_dir}' is not a directory.")
        sys.exit(1)

    combined = collect_features(merged_dir, fs=fs, pattern=pattern)
    if combined.empty:
        print("Error: No features extracted.")
        sys.exit(1)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    print(f"Feature CSV saved -> {csv_path}  ({combined.shape[0]} subjects, {combined.shape[1] - 1} features)")

    # ---- Merge patient info if Excel path provided ----
    if patient_excel is not None and patient_excel.is_file():
        from preprocessing.merge_patient_info import merge_patient_info
        info_csv = csv_path.parent / "features_with_info.csv"
        merge_patient_info(
            feature_csv   = csv_path,
            patient_excel = patient_excel,
            output_csv    = info_csv,
        )
        print(f"Features with patient info saved -> {info_csv}")
    elif patient_excel is not None:
        print(f"Warning: patient Excel not found at {patient_excel} — skipping info merge")


def run_report(csv_path: Path, output_path: Path, title: str | None) -> None:
    """Render the HTML report from a cached feature CSV."""
    if not csv_path.is_file():
        print(f"Error: feature CSV '{csv_path}' not found. Run 'extract' first.")
        sys.exit(1)

    combined = pd.read_csv(csv_path)
    print(f"\nLoaded cached features from {csv_path}  ({combined.shape[0]} subjects, {combined.shape[1] - 1} features)")

    generate_report(combined, output_path, title=title or csv_path.stem)
    print(f"HTML report saved -> {output_path}")
    print(f"\nDone. Open {output_path} in your browser.\n")


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RBD pipeline runner — patient processing + feature extraction + HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Default (no subcommand) preserves the original behavior: process the next N
        unprocessed patients from <raw_root>. Use subcommands to skip straight to
        the parts you want.

        python main.py /data/raw                    # process 10 patients (default)
        python main.py /data/raw --batch-size 5     # process 5 patients
        python main.py process /data/raw            # same as above, explicit
        python main.py extract                      # extract features from merged_csv_eog/
        python main.py report                       # regenerate HTML from cached features (fast)
        python main.py all /data/raw                # process + extract + report in one go
        """,
    )

    sub = parser.add_subparsers(dest="mode")

    # ---- process ----
    p_proc = sub.add_parser("process", help="Run preprocessing stages 1-7 for pending patients.")
    p_proc.add_argument("raw_root", type=str, help="Root directory with raw EDF/TXT recordings")
    p_proc.add_argument("--batch-size", type=int, default=10, help="Patients per batch (default: 10)")
    p_proc.add_argument("--keep-intermediates", action="store_true",
                        help="Keep intermediate CSVs after merging (default: delete to save disk space)")

    # ---- extract ----
    p_ext = sub.add_parser("extract", help="Extract features from merged CSVs into a cached CSV.")
    p_ext.add_argument("merged_dir", type=str, nargs="?", default=str(MERGED_DIR_b),
                       help=f"Directory with merged CSVs (default: {MERGED_DIR_b})")
    p_ext.add_argument("--fs", type=float, default=250.0, help="Sampling frequency [Hz] (default: 250)")
    p_ext.add_argument("--pattern", type=str, default="*_merged.csv", help="Glob pattern (default: *_merged.csv)")
    p_ext.add_argument("--csv", type=str, default=str(DEFAULT_FEATURE_CSV),
                       help=f"Output feature CSV (default: {DEFAULT_FEATURE_CSV})")
    p_ext.add_argument("--patient-excel", type=str, default=None,      
                   help="Path to patient info Excel file to join onto features")

    # ---- report ----
    p_rep = sub.add_parser("report", help="Generate HTML report from cached feature CSV.")
    p_rep.add_argument("--csv", type=str, default=str(DEFAULT_FEATURE_CSV),
                       help=f"Input feature CSV (default: {DEFAULT_FEATURE_CSV})")
    p_rep.add_argument("--output", type=str, default=str(DEFAULT_REPORT_HTML),
                       help=f"Output HTML path (default: {DEFAULT_REPORT_HTML})")
    p_rep.add_argument("--title", type=str, default=None, help="Override report title")

    # ---- all ----
    p_all = sub.add_parser("all", help="process + extract + report in one step.")
    p_all.add_argument("raw_root", type=str, help="Root directory with raw recordings")
    p_all.add_argument("--batch-size", type=int, default=10, help="Patients per batch (default: 10)")
    p_all.add_argument("--keep-intermediates", action="store_true",
                       help="Keep intermediate CSVs after merging (default: delete to save disk space)")
    p_all.add_argument("--merged-dir", type=str, default=str(MERGED_DIR))
    p_all.add_argument("--fs", type=float, default=250.0)
    p_all.add_argument("--pattern", type=str, default="*_merged.csv")
    p_all.add_argument("--csv", type=str, default=str(DEFAULT_FEATURE_CSV))
    p_all.add_argument("--output", type=str, default=str(DEFAULT_REPORT_HTML))
    p_all.add_argument("--title", type=str, default=None)
    p_all.add_argument("--patient-excel", type=str, default=None,
                   help="Path to patient info Excel file to join onto features")

    # ---- cleanup ----
    p_clean = sub.add_parser("cleanup", help="Delete intermediate CSVs for already-merged sessions to free disk space.")
    p_clean.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")

    # ---- Back-compat: allow `python main.py <raw_root>` with no subcommand ----
    argv = sys.argv[1:]
    known_modes = {"process", "extract", "report", "all", "cleanup", "-h", "--help"}
    if argv and argv[0] not in known_modes:
        argv = ["process"] + argv

    args = parser.parse_args(argv)

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    # ---- Dispatch ----
    if args.mode == "process":
        run_process(Path(args.raw_root), args.batch_size, cleanup=not args.keep_intermediates)

    elif args.mode == "extract":
        run_extract(Path(args.merged_dir), args.fs, args.pattern, Path(args.csv), patient_excel=Path(args.patient_excel) if args.patient_excel else None)

    elif args.mode == "report":
        run_report(Path(args.csv), Path(args.output), args.title)

    elif args.mode == "all":
        run_process(Path(args.raw_root), args.batch_size, cleanup=not args.keep_intermediates)
        run_extract(Path(args.merged_dir), args.fs, args.pattern, Path(args.csv), patient_excel=Path(args.patient_excel) if args.patient_excel else None)
        run_report(Path(args.csv), Path(args.output), args.title)

    elif args.mode == "cleanup":
        import gzip
        import shutil

        # Find all sessions that have a merged CSV and compress their intermediates
        merged_files = list(MERGED_DIR.glob("*_merged.csv"))
        if not merged_files:
            print("No merged CSVs found — nothing to clean up.")
            return

        print(f"\n{'='*60}")
        print(f"  Cleanup — {len(merged_files)} merged sessions found")
        print(f"  Compressing intermediate CSVs (keeps .csv.gz for re-inspection)")
        print(f"{'='*60}")

        total_freed = 0.0
        for mf in sorted(merged_files):
            session_id = mf.stem.split("_")[0]
            edf_stem = mf.stem.replace(f"{session_id}_", "").replace("_eog_merged", "")

            intermediates = [
                EOG_DIR  / f"{session_id}_{edf_stem}_eog.csv",
                GSSC_DIR / f"{session_id}_gssc.csv",
                REMS_DIR / f"{session_id}_extracted_rems.csv",
                EM_DIR   / f"{session_id}_em.csv",
                EM_DIR   / f"{session_id}_subepochs.csv",
                EEG_DIR  / f"{session_id}_eeg.csv",
            ]
            existing = [f for f in intermediates if f.exists()]
            if not existing:
                continue

            size_mb = sum(f.stat().st_size for f in existing) / (1024**2)
            print(f"\n  {session_id}: {len(existing)} intermediate files ({size_mb:.1f} MB)")

            if args.dry_run:
                for f in existing:
                    print(f"    [DRY RUN] would compress {f.name}")
            else:
                freed = _compress_intermediates(session_id, edf_stem)
                total_freed += freed

        print(f"\n{'='*60}")
        if args.dry_run:
            print(f"  [DRY RUN] No files were changed.")
        else:
            print(f"  ✓ Total freed: {total_freed:.1f} MB")
        print(f"{'='*60}\n")


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    main()