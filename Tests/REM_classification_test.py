# Filename: REM_classification_test.py 
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Classify REM sleep stages based on eye movement data and Rosenblum Y, Bogdány T, Nádasy LB, Chen X, Kovács I, Gombos F, Ujma P, Bódizs R, Adelhöfer N, Simor P, Dresler M. Aperiodic neural activity distinguishes between phasic and tonic REM sleep. J Sleep Res. 2025 Aug;34(4):e14439. doi: 10.1111/jsr.14439. Epub 2024 Dec 26. PMID: 39724862; PMCID: PMC12215217. 

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations
import pandas as pd 
#from preprocessing.extract_rems_n import extract_rems_from_edf

df = pd.read_csv(r'C:\Users\rasmu\Desktop\6. Semester\Bachelor Projekt\Git\EOG_REM\extracted_rems\Test edf filer_extracted_rems.csv')

def detect_EM_in_REM(df: pd.DataFrame)-> pd.DataFrame:

    stage_col = 'Stage' if 'Stage' in df.columns else 'stage' if 'stage' in df.columns else None
    if stage_col is None:
        raise ValueError("Dataframe must contain 'stage' column")

    df = df[df[stage_col] == 'REM'].copy()

    df['EM_detected'] = True


    return df 


# test it 
result = detect_EM_in_REM(df)
print(result)