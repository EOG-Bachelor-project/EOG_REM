# test_saving_files.py

# =====================================================================
# Imports
# =====================================================================

from pathlib import Path
from preprocessing.extract_rems_n import extract_rems_from_edf
from preprocessing.edf_to_csv import edf_to_csv
from preprocessing.GSSC_to_csv import GSSC_to_csv

# =====================================================================
# Paths
# =====================================================================
edf_path = Path("L:\Auditdata\RBD PD\PD-RBD Glostrup Database_ok\DCSM_1_a\contiguous.edf")

# =====================================================================
# TEST
# =====================================================================
print("\n--- Testing edf_to_csv ---")
try:
    edf_to_csv(edf_path)
    print("edf_to_csv succeeded")
except Exception as e:
    print("edf_to_csv failed:", e)


print("\n--- Testing GSSC_to_csv ---")
try:
    GSSC_to_csv(edf_path)
    print("GSSC_to_csv succeeded")
except Exception as e:
    print("GSSC_to_csv failed:", e)


print("\n--- Testing extract_rems_from_edf ---")
try:
    df = extract_rems_from_edf(edf_path)

    if df is None:
        print("No REMs extracted or required channels missing.")
    else:
        print(df.head())
        print(df.columns)
        print(f"Number of detected events: {len(df)}")

except Exception as e:
    print("extract_rems_from_edf failed:", e)
