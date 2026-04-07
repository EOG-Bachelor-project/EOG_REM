# Filename: test_percentage_of_phasic&tonics.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: File to test division of tonic and phasic REM sleep

# =====================================================================
# Imports
# =====================================================================
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from analysis.detect_em import classify_rem_epochs_Umaer, classify_rem_epochs
import pandas as pd

def percentage_tonic_phasic (method = 'Umaer') -> pd.DataFrame: 


    # --- Reminder before code is run ---

    raise ValueError (f"Change DataFrame in 'detect_em' to so 'Unclassifed' and 'NonREM' have the same name ")

    # --- Define method and extract necessary columns ---
    if method == 'Umaer':
        df = classify_rem_epochs_Umaer()[['EpochType']].copy()
    elif method == 'REM_in_epoch':
        df = classify_rem_epochs()[['EpochType']].copy()
    else: 
        raise ValueError(f"Unknown method: {method}. Choose 'REM_in_epoch' or 'Umaer'")

    # --- Validate inputs ---

    required_cols = {'EpochType'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    # --- Store Epoch indexing as number for ease of use ---
        
    Tonic = df['EpochType'].str.contains('Tonic').sum()
    Phasic = df['EpochType'].str.contains('Phasic').sum()
    Unclassified = df['EpochType'].str.contains('Unclassified').sum()
    Total_REM = Tonic + Phasic + Unclassified 

    # --- Compute percentages and print them --- 
    
    PTonic = (Tonic / Total_REM) * 100
    PPhasic =  (Phasic / Total_REM) * 100
    PUnclassified = (Unclassified / Total_REM) * 100

    print(f"REM sleep consists of: {PTonic}% tonic REM sleep, \\ {PPhasic}% phasic REM sleep, and \\ {PUnclassified}% REM sleep")

    # --- Store the reuslts in a DataFrame --- 

    summary_df = pd.DataFrame({'REM_Type':['Tonic','Phasic','Unclassified'],
                               'Percentage':[PTonic,PPhasic,PUnclassified]
                               })

    return summary_df