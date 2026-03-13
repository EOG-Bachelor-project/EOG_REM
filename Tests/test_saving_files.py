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
from art import * 
# =====================================================================
# Paths
# =====================================================================
edf_path = Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_1_a\contiguous.edf")

# =====================================================================
# TEST
# =====================================================================
print("\n####################################")
print("######## Testing edf_to_csv ########")
print("####################################")
try:
    edf_to_csv(edf_path)
    print("edf_to_csv succeeded")
except Exception as e:
    print("edf_to_csv failed:", e)

print("\n#####################################")
print("######## Testing GSSC_to_csv ########")
print("#####################################")
try:
    GSSC_to_csv(edf_path)
    print("GSSC_to_csv succeeded")
except Exception as e:
    print("GSSC_to_csv failed:", e)

print("\n###############################################")
print("######## Testing extract_rems_from_edf ########")
print("###############################################")

try:
    extract_rems_from_edf(edf_path)
    print("extract_rems_from_edf succeeded")
except Exception as e:
    print("extract_rems_from_edf failed:", e)

tprint("Done")