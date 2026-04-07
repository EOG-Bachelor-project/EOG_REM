# Filename: test_em_in_stages.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: File to test extraction of EMs in different sleep stages

# =====================================================================
# Imports
# =====================================================================
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.merge import merge_all
import pandas as pd



def number_of_em_in_stages() -> pd.DataFrame: 

    # --- Get data from merge file and keep only necessary ---

    df = merge_all()[['Stage', 'EM_Type']].copy()

    # --- Validate inputs ---

    required_cols = {'Stage','EM_Type'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")
    
    # --- Count number of EMs in eash sleep stage ---

    df = df.copy()
    
    EMN1 = df['Stage'].str.contains('N1').sum()
    if EMN1 > 0:
        print(f"There are a total of {EMN1} EMs in the N1 stage")

    EMN2 = df['Stage'].str.contains('N2').sum()
    if EMN2 > 0:
        print(f"There are a total of {EMN2} EMs in the N2 stage")

    EMN3 = df['Stage'].str.contains('N3').sum()
    if EMN3 > 0:
        print(f"There are a total of {EMN3} EMs in the N3 stage")

    EMREM = df['Stage'].str.contains('REM').sum()
    if EMREM > 0:
        print(f"There are a total of {EMREM} EMs in the REM stage")

    EMW = df['Stage'].str.contains('W').sum()
    if EMW > 0:
        print(f"There are a total of {EMW} EMs in the Wake stage")

    EM_in_stages = EMN1 + EMN2 + EMN3 + EMREM + EMW

    EMtot = df['EM_Type'].count()
    # --- Sanity check validating every EM is in right stage ---
    print(f"Total number of eye movements:{EMtot} \\ Number of EMs indexed in stages:{EM_in_stages}")

    # --- Create new DataFrame to store number of EMs in each stage --- 

    summary_df = pd.DataFrame({'Stage': ['N1', 'N2', 'N3', 'REM', 'W'],
                               'EM_Count':[EMN1, EMN2, EMN3, EMREM, EMW]
                               })

    return summary_df
