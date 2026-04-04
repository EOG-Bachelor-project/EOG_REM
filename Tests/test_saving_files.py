# Filename: test_saving_files.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: File to test the saving of different information from an edf file

# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
import time
import numpy as np
import pandas as pd
 
from preprocessing.extract_rems_n import extract_rems_from_edf
from preprocessing.edf_to_csv import edf_to_csv
from preprocessing.GSSC_to_csv import GSSC_to_csv
from preprocessing.em_to_csv import em_to_csv
from art import * 

# =====================================================================
# Paths
# =====================================================================
edf_path        = Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_3_a\contiguous.edf")
lightstxt_path  = Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_3_a\lights.txt")

# =====================================================================
# Helpers
# =====================================================================
def _section(title: str) -> float:
    """Print a section header and return the start time."""
    print(f"\n{'#' * (len(title) + 6)}")
    print(f"## {title} ##")
    print(f"{'#' * (len(title) + 6)}")
    return time.perf_counter()
 
def _done(t0: float, label: str = ""):
    """Print elapsed time for a section."""
    elapsed = time.perf_counter() - t0
    suffix = f" [{label}]" if label else ""
    print(f"\nTime: {elapsed:.1f}s{suffix}")

# =====================================================================
# TEST
# =====================================================================
total_start = time.perf_counter()
 
t0 = _section("Testing edf_to_csv")
try:
    edf_to_csv(edf_path, lights_path=lightstxt_path)
    print("edf_to_csv SUCCEEDED")
except Exception as e:
    print("edf_to_csv FAILED:", e)
_done(t0, "edf_to_csv")
 
 
t0 = _section("Testing GSSC_to_csv")
gssc_df   = None
hypno_int = None
try:
    gssc_df = GSSC_to_csv(edf_path, lights_path=lightstxt_path)
    stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    hypno_int = gssc_df["stage"].map(stage_map).fillna(0).astype(int).values
    print("GSSC_to_csv SUCCEEDED")
except Exception as e:
    print("GSSC_to_csv FAILED:", e)
_done(t0, "GSSC_to_csv")
 
 
t0 = _section("Testing extract_rems_from_edf (includes remove_artefacts)")
events_df = None
try:
    events_df = extract_rems_from_edf(edf_path, lights_path=lightstxt_path, gssc_df=gssc_df)
    print("extract_rems_from_edf SUCCEEDED")
except Exception as e:
    print("extract_rems_from_edf FAILED:", e)
_done(t0, "extract_rems_from_edf")
 
 
t0 = _section("Verifying artefact removal")
try:
    if events_df is None:
        raise RuntimeError("events_df is None — extract_rems_from_edf must succeed first")
 
    session_id = edf_path.parent.name
    saved_path = Path("extracted_rems") / f"{session_id}_extracted_rems.csv"
    saved_df   = pd.read_csv(saved_path)
 
    n_events = len(saved_df)
    max_loc  = saved_df["LOCAbsValPeak"].max()
    max_roc  = saved_df["ROCAbsValPeak"].max()
 
    print(f"    Events in saved CSV: {n_events}")
    print(f"    Max LOCAbsValPeak:   {max_loc:.1f} µV")
    print(f"    Max ROCAbsValPeak:   {max_roc:.1f} µV")
 
    assert (saved_df["LOCAbsValPeak"] <= 300.0).all(), "FAIL: LOC artefact still present in saved CSV"
    assert (saved_df["ROCAbsValPeak"] <= 300.0).all(), "FAIL: ROC artefact still present in saved CSV"
 
    print("    All assertions passed — saved CSV is artefact-free.")
    print("Artefact removal verification SUCCEEDED")
except Exception as e:
    import traceback
    print("Artefact removal verification FAILED:", e)
    traceback.print_exc()
_done(t0, "artefact verification")
 
 
t0 = _section("Masking artefacts in EOG CSV")
try:
    if events_df is None:
        raise RuntimeError("events_df is None — extract_rems_from_edf must succeed first")
 
    session_id   = edf_path.parent.name
    eog_csv_path = Path("eog_csv") / f"{session_id}_{edf_path.stem}_eog.csv"
    eog_df       = pd.read_csv(eog_csv_path)
 
    print(f"    EOG CSV loaded: {eog_csv_path.name}  ({len(eog_df):,} samples)")
 
    AMPLITUDE_THRESH_UV = 300.0   # µV  (CSV is now stored in µV)

    artefact_mask = (
        (np.abs(eog_df["LOC"].values) > AMPLITUDE_THRESH_UV) |
        (np.abs(eog_df["ROC"].values) > AMPLITUDE_THRESH_UV)
    )
    n_masked = int(artefact_mask.sum())
 
    eog_df.loc[artefact_mask, "LOC"] = np.nan
    eog_df.loc[artefact_mask, "ROC"] = np.nan
 
    eog_df.to_csv(eog_csv_path, index=False)
 
    print(f"    Artefact samples masked: {n_masked:,} / {len(eog_df):,} ({n_masked // len(eog_df):.2%})")
    print(f"    Threshold: {AMPLITUDE_THRESH_UV:.0f} µV (signals stored in µV in EOG CSV)")
    print(f"    Saved masked EOG CSV: {eog_csv_path.name}")
    print("Masking EOG CSV SUCCEEDED")
except Exception as e:
    import traceback
    print("Masking EOG CSV FAILED:", e)
    traceback.print_exc()
_done(t0, "EOG CSV masking")
 
 
t0 = _section("Testing em_to_csv (default classifier)")
try:
    if gssc_df is None or hypno_int is None:
        raise RuntimeError("gssc_df or hypno_int is None — GSSC_to_csv must succeed first")
    em_to_csv(
        edf_path=edf_path,
        gssc_df=gssc_df,
        hypno_int=hypno_int,
        lights_path=lightstxt_path,
        use_Umaer=False,
    )
    print("em_to_csv (default) SUCCEEDED")
except Exception as e:
    print("em_to_csv (default) FAILED:", e)
_done(t0, "em_to_csv default")
 
 
t0 = _section("Testing em_to_csv (Umaer classifier)")
try:
    if gssc_df is None or hypno_int is None:
        raise RuntimeError("gssc_df or hypno_int is None — GSSC_to_csv must succeed first")
    result = em_to_csv(
        edf_path=edf_path,
        gssc_df=gssc_df,
        hypno_int=hypno_int,
        lights_path=lightstxt_path,
        psg_epoch_sec=32,
        use_Umaer=True,
    )
    if result is not None:
        em_df, subepoch = result
        print("em_to_csv (Umaer) SUCCEEDED")
        print(f"    EM events:  {len(em_df)} rows")
        print(f"    Sub-epochs: {len(subepoch)} rows")
    else:
        print("em_to_csv (Umaer) returned None — check signal length")
except Exception as e:
    print("em_to_csv (Umaer) FAILED:", e)
_done(t0, "em_to_csv Umaer")
 
 
# =====================================================================
# Total
# =====================================================================
total_elapsed = time.perf_counter() - total_start
print(f"\n{'=' * 40}")
print(f"Total runtime: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
print(f"{'=' * 40}")
 
tprint("Done")