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
MERGED_DIR_b   = Path("merged_csv_eog_backup")
FEATURES_DIR = Path("features_csv")
REPORTS_DIR  = Path("reports")
EEG_DIR      = Path("eeg_csv")
 
for d in [EOG_DIR, GSSC_DIR, REMS_DIR, EM_DIR, MERGED_DIR, FEATURES_DIR, REPORTS_DIR,EEG_DIR]:
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
# Helper — check if a patient has already been processed
# =====================================================================
def _is_processed(session_id: str) -> bool:
    """
    Check whether the final merged CSV exists for this session.
    If it does, we assume all earlier stages succeeded and skip.
    """
    merged_path = MERGED_DIR / f"{session_id}_contiguous_eog_merged.csv"
    return merged_path.exists()
 
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
 
# =====================================================================
# Core — process one patient through the full pipeline
# =====================================================================
def process_patient(rec) -> bool:
    """
    Run stages 1-7 for a single patient session.

    Returns True if successful, False if an error occurred.
    """
    session_id  = rec.patient_id
    edf_path    = rec.edf_path
    lights_path = rec.txt_path

    print(f"\n{'=' * 70}")
    print(f"  Processing: {BOLD}{session_id}{RESET}")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()

    try:
        # ── Load EDF once ───────────────────────────────────────────
        raw = mne.io.read_raw_edf
        rename_map = build_rename_map(raw.ch_names)
        if rename_map:
            raw.rename_channels(rename_map)

        # ── Stage 1: EDF → EOG CSV ──────────────────────────────────
        print(f"\n{BOLD}[1/7] EDF → EOG CSV{RESET}")
        edf_to_csv(edf_path, raw=raw, out_dir=EOG_DIR, lights_path=lights_path)

        # ── Stage 2: GSSC sleep staging ─────────────────────────────
        print(f"\n{BOLD}[2/7] GSSC sleep staging{RESET}")
        gssc_df = GSSC_to_csv(edf_path, raw=raw, out_dir=GSSC_DIR, lights_path=lights_path)

        stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
        hypno_int = gssc_df["stage"].map(stage_map).fillna(0).astype(int).values

        # ── Stage 3: Extract REM events ─────────────────────────────
        print(f"\n{BOLD}[3/7] Extract REM events{RESET}")
        df, loc, roc, result =extract_rems_from_edf(
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
        print(f"\n{BOLD}[5/7] Detect & classify eye movements{RESET}")
        em_to_csv(
            edf_path=edf_path,
            raw=raw,
            hypno_int=hypno_int,
            out_dir=EM_DIR,
            lights_path=lights_path,
        )

        # ── Stage 6: Extract EEG signals ────────────────────────────────
        print(f"\n{BOLD}[6/7] Extract EEG signals{RESET}")
        loc_clean = result._data_filt[0]
        roc_clean = result._data_filt[1]
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
        eeg_file = _wait_for_file(EEG_DIR / f"{session_id}_eeg.csv")
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
def run_process(raw_root: Path, batch_size: int) -> None:
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
        if process_patient(rec):
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

def run_extract(merged_dir: Path, fs: float, pattern: str, csv_path: Path) -> None:
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

    # ---- extract ----
    p_ext = sub.add_parser("extract", help="Extract features from merged CSVs into a cached CSV.")
    p_ext.add_argument("merged_dir", type=str, nargs="?", default=str(MERGED_DIR_b),
                       help=f"Directory with merged CSVs (default: {MERGED_DIR_b})")
    p_ext.add_argument("--fs", type=float, default=250.0, help="Sampling frequency [Hz] (default: 250)")
    p_ext.add_argument("--pattern", type=str, default="*_merged.csv", help="Glob pattern (default: *_merged.csv)")
    p_ext.add_argument("--csv", type=str, default=str(DEFAULT_FEATURE_CSV),
                       help=f"Output feature CSV (default: {DEFAULT_FEATURE_CSV})")

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
    p_all.add_argument("--merged-dir", type=str, default=str(MERGED_DIR))
    p_all.add_argument("--fs", type=float, default=250.0)
    p_all.add_argument("--pattern", type=str, default="*_merged.csv")
    p_all.add_argument("--csv", type=str, default=str(DEFAULT_FEATURE_CSV))
    p_all.add_argument("--output", type=str, default=str(DEFAULT_REPORT_HTML))
    p_all.add_argument("--title", type=str, default=None)

    # ---- Back-compat: allow `python main.py <raw_root>` with no subcommand ----
    # Argparse can't express "positional that isn't a subcommand", so we detect it manually.
    argv = sys.argv[1:]
    known_modes = {"process", "extract", "report", "all", "-h", "--help"}
    if argv and argv[0] not in known_modes:
        # Treat the old-style invocation as `process`
        argv = ["process"] + argv

    args = parser.parse_args(argv)

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    # ---- Dispatch ----
    if args.mode == "process":
        run_process(Path(args.raw_root), args.batch_size)

    elif args.mode == "extract":
        run_extract(Path(args.merged_dir), args.fs, args.pattern, Path(args.csv))

    elif args.mode == "report":
        run_report(Path(args.csv), Path(args.output), args.title)

    elif args.mode == "all":
        run_process(Path(args.raw_root), args.batch_size)
        run_extract(Path(args.merged_dir), args.fs, args.pattern, Path(args.csv))
        run_report(Path(args.csv), Path(args.output), args.title)


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    main()