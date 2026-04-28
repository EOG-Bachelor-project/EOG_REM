# Filename: main.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Resume-friendly pipeline runner for RBD feature extraction.
#
# Usage:
#   python main.py process /data/raw                            # process 10 patients
#   python main.py process /data/raw --batch-size 5             # process 5
#   python main.py extract patient_info.xlsx                    # extract all feature modules
#   python main.py extract patient_info.xlsx --modules bout     # extract only bout features
#   python main.py extract patient_info.xlsx --force            # re-extract all from scratch
#   python main.py report                                       # generate HTML report
#   python main.py all /data/raw patient_info.xlsx              # full pipeline
#   python main.py all /data/raw patient_info.xlsx --force      # full pipeline, re-extract
#   python main.py cleanup                                      # compress intermediate CSVs
#   python main.py cleanup --dry-run                            # preview cleanup
#
# Feature modules: eog, gssc, eeg, bout, patient
#   Each module saves its own CSV in features_csv/ (e.g. eog_features.csv).
#   After extraction, all module CSVs are outer-joined into features.csv.
#   --force deletes selected module CSVs before re-extracting.
#   Without --force, existing module CSVs are kept (incremental).
#
# Re-running preprocessing stages:
#   The pipeline skips any stage whose output file already exists.
#   To re-run a stage, delete its outputs and the merged CSV, then run again.
#
#   Re-run everything from scratch:
#     rm eog_csv/* gssc_csv/* extracted_rems/* detected_ems/* eeg_csv/* merged_csv_eog/*
#     python main.py process <raw_root> --batch-size 9999

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import argparse
import sys
import time
import traceback
import numpy as np
import pandas as pd
import mne
from pathlib import Path

from preprocessing.index_file import index_sessions
from preprocessing.edf_to_csv import edf_to_csv
from preprocessing.GSSC_to_csv import GSSC_to_csv
from preprocessing.extract_rems_n import extract_rems_from_edf
from preprocessing.em_to_csv import em_to_csv
from preprocessing.merge import merge_all
from preprocessing.channel_standardization import build_rename_map
from preprocessing.eeg_to_csv import eeg_to_csv
from analysis.feat_report import collect_features, generate_report, merge_feature_csvs

# =====================================================================
# Constants
# =====================================================================
EOG_DIR      = Path("eog_csv")
GSSC_DIR     = Path("gssc_csv")
REMS_DIR     = Path("extracted_rems")
EM_DIR       = Path("detected_ems")
MERGED_DIR   = Path("merged_csv_eog")
FEATURES_DIR = Path("features_csv")
REPORTS_DIR  = Path("reports")
EEG_DIR      = Path("eeg_csv")

for d in [EOG_DIR, GSSC_DIR, REMS_DIR, EM_DIR, MERGED_DIR, FEATURES_DIR, REPORTS_DIR, EEG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hardcoded pipeline settings
FS                  = 250.0
PATTERN             = "*_merged.csv*"
DEFAULT_FEATURE_CSV = FEATURES_DIR / "features.csv"
DEFAULT_REPORT_HTML = REPORTS_DIR  / "features_report.html"
AMPLITUDE_THRESH_UV = 300.0

# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"


# =====================================================================
# Helpers
# =====================================================================
def _is_processed(session_id: str) -> bool:
    """Check whether the final merged CSV exists for this session."""
    merged_path = MERGED_DIR / f"{session_id}_contiguous_eog_merged.csv"
    return merged_path.exists() or merged_path.with_suffix(".csv.gz").exists()


def _check_existing_outputs(session_id: str, edf_stem: str) -> dict[str, bool]:
    """Check which intermediate output files already exist for a session."""
    return {
        "eog":       (EOG_DIR  / f"{session_id}_{edf_stem}_eog.csv").exists(),
        "gssc":      (GSSC_DIR / f"{session_id}_gssc.csv").exists(),
        "rems":      (REMS_DIR / f"{session_id}_extracted_rems.csv").exists(),
        "em":        (EM_DIR   / f"{session_id}_em.csv").exists(),
        "subepochs": (EM_DIR   / f"{session_id}_subepochs.csv").exists(),
        "eeg":       (EEG_DIR  / f"{session_id}_eeg.csv").exists(),
    }


def _wait_for_file(path: Path, timeout: float = 10.0, interval: float = 0.5) -> Path:
    """Wait for a file to appear on disk."""
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
    """Gzip intermediate CSV files for a session after a successful merge."""
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
            f.unlink()
            print(f"    {f.name}: {size_before / (1024**2):.1f} MB -> {size_after / (1024**2):.1f} MB ({saved / (1024**2):.1f} MB saved)")

    freed_mb = freed_bytes / (1024 ** 2)
    print(f"    Freed {freed_mb:.1f} MB for {session_id}")
    return freed_mb

def _check_existing_outputs(session_id, edf_stem):
    checks = {
        "eog":       EOG_DIR   / f"{session_id}_{edf_stem}_eog.csv",
        "gssc":      GSSC_DIR  / f"{session_id}_gssc.csv",
        "rems":      REMS_DIR  / f"{session_id}_extracted_rems.csv",
        "em":        EM_DIR    / f"{session_id}_em.csv",
        "subepochs": EM_DIR    / f"{session_id}_subepochs.csv",
        "eeg":       EEG_DIR   / f"{session_id}_eeg.csv"
    }

    # Delete empty files so they are regenerated rather than causing crashes downstream
    for name, path in checks.items():
        if path.exists() and path.stat().st_size == 0:
            print(f"  WARNING: {name} file is empty — deleting so it will be regenerated: {path.name}")
            path.unlink()

    return {name: (path.exists() and path.stat().st_size > 0)
            for name, path in checks.items()}


# =====================================================================
# Core — process one patient through the full pipeline
# =====================================================================
def process_patient(rec) -> bool:
    """Run stages 1-7 for a single patient session, skipping completed stages."""
    session_id  = rec.patient_id
    edf_path    = rec.edf_path
    lights_path = rec.txt_path

    print(f"\n{'=' * 70}")
    print(f"  Processing: {BOLD}{session_id}{RESET}")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()

    try:
        existing = _check_existing_outputs(session_id, edf_path.stem)
        skip_summary = [f"{name}: {'EXISTS' if exists else 'missing'}"
                        for name, exists in existing.items()]
        print(f"  Intermediate files: {', '.join(skip_summary)}")

        needs_edf = (
            not existing["eog"]  or not existing["gssc"] or
            not existing["rems"] or not existing["em"]   or
            not existing["eeg"]
        )

        # ── Load EDF once ───────────────────────────────────────────
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

        # ── Stage 1: EDF → EOG CSV ─────────────────────────────────
        if existing["eog"]:
            print(f"\n{BOLD}[1/7] EDF → EOG CSV — SKIPPED{RESET}")
        else:
            print(f"\n{BOLD}[1/7] EDF → EOG CSV{RESET}")
            edf_to_csv(edf_path, raw=raw, out_dir=EOG_DIR, lights_path=lights_path)

        # ── Stage 2: GSSC sleep staging ─────────────────────────────
        if existing["gssc"]:
            print(f"\n{BOLD}[2/7] GSSC sleep staging — SKIPPED{RESET}")
            gssc_df = pd.read_csv(GSSC_DIR / f"{session_id}_gssc.csv")
        else:
            print(f"\n{BOLD}[2/7] GSSC sleep staging{RESET}")
            gssc_df = GSSC_to_csv(edf_path, raw=raw, out_dir=GSSC_DIR, lights_path=lights_path)

        stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
        hypno_int = gssc_df["stage"].map(stage_map).fillna(0).astype(int).values

        # ── Stage 3: Extract REM events ─────────────────────────────
        if existing["rems"]:
            print(f"\n{BOLD}[3/7] Extract REM events — SKIPPED{RESET}")
            df, loc, roc, loc_clean, roc_clean = None, None, None, None, None
        else:
            result = extract_rems_from_edf(
                edf_path=edf_path, raw=raw, out_dir=REMS_DIR,
                lights_path=lights_path, gssc_df=gssc_df,
            )
            if result is None:
                raise RuntimeError("Signal too short or missing channels — skipping session")
            df, loc, roc, loc_clean, roc_clean = result

        # ── Stage 4: Mask artefacts ─────────────────────────────────
        print(f"\n{BOLD}[4/7] Mask artefacts in EOG CSV{RESET}")
        eog_csv_path = EOG_DIR / f"{session_id}_{edf_path.stem}_eog.csv"
        if not eog_csv_path.exists() or eog_csv_path.stat().st_size < 10:
            print("    EOG CSV missing or empty — regenerating...")
            edf_to_csv(edf_path, raw=raw, out_dir=EOG_DIR, lights_path=lights_path)
        eog_csv_path = _wait_for_file(eog_csv_path)
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
            print(f"\n{BOLD}[5/7] Detect & classify EMs — SKIPPED{RESET}")
        else:
            print(f"\n{BOLD}[5/7] Detect & classify EMs{RESET}")
            em_to_csv(edf_path=edf_path, raw=raw, hypno_int=hypno_int,
                      out_dir=EM_DIR, lights_path=lights_path)

        # ── Stage 6: Extract EEG signals ────────────────────────────
        if existing["eeg"]:
            print(f"\n{BOLD}[6/7] Extract EEG signals — SKIPPED{RESET}")
        else:
            print(f"\n{BOLD}[6/7] Extract EEG signals{RESET}")
            if loc_clean is None:
                print("    Re-running stage 3 to get filtered signals...")
                result = extract_rems_from_edf(
                    edf_path=edf_path, raw=raw, out_dir=REMS_DIR,
                    lights_path=lights_path, gssc_df=gssc_df,
                )
                if result is None:
                    print(f"    {RED}Skipping EEG — extract_rems returned None{RESET}")
                else:
                    df, loc, roc, loc_clean, roc_clean = result

            if loc_clean is not None:
                eeg_to_csv(edf_path=edf_path, loc=loc, roc=roc,
                           loc_clean=loc_clean, roc_clean=roc_clean,
                           out_dir=EEG_DIR, lights_path=lights_path)
            else:
                print(f"    {RED}EEG extraction skipped — no filtered signals{RESET}")

        # ── Stage 7: Merge into unified CSV ─────────────────────────
        print(f"\n{BOLD}[7/7] Merge into unified CSV{RESET}")
        eog_file       = _wait_for_file(EOG_DIR  / f"{session_id}_{edf_path.stem}_eog.csv")
        gssc_file      = _wait_for_file(GSSC_DIR / f"{session_id}_gssc.csv")
        events_file    = _wait_for_file(REMS_DIR / f"{session_id}_extracted_rems.csv")
        em_file        = _wait_for_file(EM_DIR   / f"{session_id}_em.csv")
        eeg_file       = _wait_for_file(EEG_DIR  / f"{session_id}_eeg.csv")
        subepochs_file = EM_DIR / f"{session_id}_subepochs.csv"
        output_file    = MERGED_DIR / f"{session_id}_{edf_path.stem}_eog_merged.csv"

        merge_all(eog_file=eog_file, gssc_file=gssc_file, events_file=events_file,
                  em_file=em_file, output_file=output_file,
                  subepochs_file=subepochs_file, eeg_file=eeg_file)

        # ── Cleanup ─────────────────────────────────────────────────
        print(f"\n{BOLD}[Cleanup] Compressing intermediate CSVs{RESET}")
        _compress_intermediates(session_id, edf_path.stem)

        elapsed = time.perf_counter() - t0
        print(f"\n{GREEN}✓ {session_id} completed in {elapsed:.1f}s ({elapsed/60:.1f} min){RESET}")
        return True

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\n{RED}✗ {session_id} FAILED after {elapsed:.1f}s: {e}{RESET}")
        traceback.print_exc()
        return False


# =====================================================================
# run_process
# =====================================================================
def run_process(raw_root: Path, batch_size: int) -> None:
    """Process the next batch of unprocessed patients."""
    if not raw_root.is_dir():
        print(f"Error: '{raw_root}' is not a directory.")
        sys.exit(1)

    sessions = index_sessions(raw_root)
    todo = [s for s in sessions if not _is_processed(s.patient_id)]
    n_already = len(sessions) - len(todo)

    print(f"\n{'='*70}")
    print(f"    {BOLD}Pipeline Run{RESET}")
    print(f"    Total sessions : {len(sessions)}")
    print(f"    Already done   : {n_already}")
    print(f"    Remaining      : {len(todo)}")
    print(f"    Batch size     : {batch_size}")
    print(f"{'='*70}")

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

    print(f"\n{'='*70}")
    print(f"    {BOLD}Run Summary{RESET}")
    print(f"    {GREEN}Successful : {ok}{RESET}")
    print(f"    {RED}Failed     : {fail}{RESET}")
    print(f"    Pending    : {remaining}")
    print(f"    Time       : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if ok > 0:
        print(f"    Avg/success: {elapsed / ok:.1f}s")
    if failed_ids:
        print(f"\n    {RED}Failed sessions:{RESET}")
        for sid in failed_ids:
            print(f"      - {sid}")
    if remaining > 0:
        print(f"\n  Run again to process next batch.")
    else:
        print(f"\n  {GREEN}All patients processed!{RESET}")
    print(f"{'='*70}")


# =====================================================================
# run_extract
# =====================================================================
def run_extract(
        patient_excel: Path,
        modules:       list[str] | None = None,
        force:         bool = False,
) -> None:
    """Run feature extraction modules and merge into features.csv."""
    if not MERGED_DIR.is_dir():
        print(f"Error: '{MERGED_DIR}' is not a directory.")
        sys.exit(1)

    combined = collect_features(
        merged_dir=MERGED_DIR,
        fs=FS,
        pattern=PATTERN,
        modules=modules,
        force=force,
        patient_excel=patient_excel,
        output_csv=DEFAULT_FEATURE_CSV,
    )

    if combined.empty:
        print("Error: No features extracted.")
        sys.exit(1)

    print(f"Feature CSV saved -> {DEFAULT_FEATURE_CSV}  "
          f"({combined.shape[0]} subjects, {combined.shape[1] - 1} features)")


# =====================================================================
# run_report
# =====================================================================
def run_report() -> None:
    """Generate HTML report from cached features.csv."""
    if not DEFAULT_FEATURE_CSV.is_file():
        print(f"Error: '{DEFAULT_FEATURE_CSV}' not found. Run 'extract' first.")
        sys.exit(1)

    combined = pd.read_csv(DEFAULT_FEATURE_CSV)
    print(f"\nLoaded {DEFAULT_FEATURE_CSV}  ({combined.shape[0]} subjects, {combined.shape[1] - 1} features)")

    generate_report(combined, DEFAULT_REPORT_HTML, title="RBD Features")
    print(f"HTML report saved -> {DEFAULT_REPORT_HTML}")
    print(f"\nDone. Open {DEFAULT_REPORT_HTML} in your browser.\n")


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RBD pipeline — process patients, extract features, generate reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process /data/raw                            # process 10 patients
  python main.py process /data/raw --batch-size 5             # process 5
  python main.py extract patient_info.xlsx                    # all feature modules
  python main.py extract patient_info.xlsx --modules bout     # only bout
  python main.py extract patient_info.xlsx --force            # re-extract from scratch
  python main.py report                                       # HTML report
  python main.py all /data/raw patient_info.xlsx              # full pipeline
  python main.py all /data/raw patient_info.xlsx --force      # full + re-extract
  python main.py cleanup                                      # compress intermediates
        """,
    )
    sub = parser.add_subparsers(dest="mode")

    # ---- process ----
    p_proc = sub.add_parser("process", help="Run preprocessing stages 1-7.")
    p_proc.add_argument("raw_root", type=str, help="Root directory with raw EDF/TXT recordings")
    p_proc.add_argument("--batch-size", type=int, default=10, help="Patients per batch (default: 10)")

    # ---- extract ----
    p_ext = sub.add_parser("extract", help="Extract features into per-module CSVs, then merge.")
    p_ext.add_argument("patient_excel", type=str, help="Path to patient info Excel file")
    p_ext.add_argument("--modules", type=str, nargs="*", default=None,
                       choices=["eog", "gssc", "eeg", "bout", "patient"],
                       help="Which modules to run (default: all)")
    p_ext.add_argument("--force", action="store_true",
                       help="Delete existing module CSVs before re-extracting")

    # ---- report ----
    sub.add_parser("report", help="Generate HTML report from cached features.csv.")

    # ---- all ----
    p_all = sub.add_parser("all", help="process + extract + report in one step.")
    p_all.add_argument("raw_root", type=str, help="Root directory with raw recordings")
    p_all.add_argument("patient_excel", type=str, help="Path to patient info Excel file")
    p_all.add_argument("--batch-size", type=int, default=10, help="Patients per batch (default: 10)")
    p_all.add_argument("--modules", type=str, nargs="*", default=None,
                       choices=["eog", "gssc", "eeg", "bout", "patient"],
                       help="Which feature modules to run (default: all)")
    p_all.add_argument("--force", action="store_true",
                       help="Delete existing module CSVs before re-extracting")

    # ---- cleanup ----
    p_clean = sub.add_parser("cleanup", help="Compress intermediate CSVs to free disk space.")
    p_clean.add_argument("--dry-run", action="store_true", help="Preview without deleting")

    # ---- Back-compat: `python main.py /data/raw` → process ----
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
        run_process(Path(args.raw_root), args.batch_size)

    elif args.mode == "extract":
        run_extract(Path(args.patient_excel), modules=args.modules, force=args.force)

    elif args.mode == "report":
        run_report()

    elif args.mode == "all":
        run_process(Path(args.raw_root), args.batch_size)
        run_extract(Path(args.patient_excel), modules=args.modules, force=args.force)
        run_report()

    elif args.mode == "cleanup":
        import gzip
        import shutil
        import re

        merged_files = list(MERGED_DIR.glob("*_merged.csv*"))
        if not merged_files:
            print("No merged CSVs found — nothing to clean up.")
            return

        print(f"\n{'='*60}")
        print(f"  Cleanup — {len(merged_files)} merged sessions found")
        print(f"{'='*60}")

        total_freed = 0.0
        for mf in sorted(merged_files):
            m = re.match(r"(DCSM_\d+_[a-zA-Z])_(.*?)_eog_merged", mf.stem)
            if not m:
                print(f"  Skipping {mf.name} — could not parse session ID")
                continue
            session_id = m.group(1)
            edf_stem = m.group(2)

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
            print(f"\n  {session_id}: {len(existing)} files ({size_mb:.1f} MB)")

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