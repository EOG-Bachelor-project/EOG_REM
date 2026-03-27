# Filename: test_saving_files.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: File to test the saving of different information from an edf file

# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
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
edf_path = Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_1_a\contiguous.edf")
lightstxt_path = Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_1_a\lights.txt")
# =====================================================================
# TEST
# =====================================================================
print("\n####################################")
print("######## Testing edf_to_csv ########")
print("####################################")
try:
    edf_to_csv(edf_path, lights_path=lightstxt_path)
    print("edf_to_csv SUCCEEDED")
except Exception as e:
    print("edf_to_csv FAILED:", e)

print("\n#####################################")
print("######## Testing GSSC_to_csv ########")
print("#####################################")
gssc_df = None
hypno_int = None
try:
    gssc_df = GSSC_to_csv(edf_path, lights_path=lightstxt_path)
    stage_map = {"W":0, "N1":1, "N2":2, "N3":3, "REM":4}
    hypno_int = gssc_df["stage"].map(stage_map).fillna(0).astype(int).values
    print("GSSC_to_csv SUCCEEDED")
except Exception as e:
    print("GSSC_to_csv FAILED:", e)

print("\n###############################################")
print("######## Testing extract_rems_from_edf ########")
print("######## (includes remove_artefacts)    ########")
print("###############################################")
events_df = None
try:
    events_df = extract_rems_from_edf(edf_path, lights_path=lightstxt_path, gssc_df=gssc_df)
    print("extract_rems_from_edf SUCCEEDED")
except Exception as e:
    print("extract_rems_from_edf FAILED:", e)

print("\n#################################################")
print("######## Verifying artefact removal      ########")
print("#################################################")
try:
    if events_df is None:
        raise RuntimeError("events_df is None — extract_rems_from_edf must succeed first")
 
    session_id = edf_path.parent.name
    saved_path = Path("extracted_rems") / f"{session_id}_extracted_rems.csv"
    saved_df   = pd.read_csv(saved_path)
 
    n_events   = len(saved_df)
    max_loc    = saved_df["LOCAbsValPeak"].max()
    max_roc    = saved_df["ROCAbsValPeak"].max()
 
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

print("\n#######################################")
print("######## Testing em_to_csv    #########")
print("######## (default classifier) #########")
print("#######################################")
try:
    if gssc_df is None or hypno_int is None:
        raise RuntimeError("gssc_df or hypno_int is None - GSSC_to_csv must succeed first")
    em_to_csv(
        edf_path=edf_path,
        gssc_df=gssc_df,
        hypno_int=hypno_int,
        lights_path=lightstxt_path,
        use_Umaer=False
    )
    print("em_to_csv (default) SUCCEEDED")
except Exception as e:
    print("em_to_csv (default) FAILED:", e)

print("\n#######################################")
print("######## Testing em_to_csv  ###########")
print("######## (Umaer classifier) ###########")
print("#######################################")
try:
    if gssc_df is None or hypno_int is None:
        raise RuntimeError("gssc_df or hypno_int is None - GSSC_to_csv must succeed first")
    result = em_to_csv(
        edf_path=edf_path,
        gssc_df=gssc_df,
        hypno_int=hypno_int,
        lights_path=lightstxt_path,
        use_Umaer=True
    )
    if result is not None:
        em_df, subepoch = result
        print("em_to_csv (Umaer) SUCCEEDED")
        print(f"    EM events: {len(em_df)} rows")
        print(f"    Sub-epochs: {len(em_df)} rows")
    else:
        print(f"em_to_csv (Umaer) returned None - check signal length")
except Exception as e:
    print("em_to_csv (Umaer) FAILED:", e)


tprint("Done") 