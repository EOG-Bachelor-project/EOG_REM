# Filename: test_saving_files.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: File to test the saving of different information from an edf file

# =====================================================================
# Imports
# =====================================================================

from pathlib import Path
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
print("###############################################")

try:
    extract_rems_from_edf(edf_path, lights_path=lightstxt_path, gssc_df=gssc_df)
    print("extract_rems_from_edf SUCCEEDED")
except Exception as e:
    print("extract_rems_from_edf FAILED:", e)

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