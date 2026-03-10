# test_merge_pipeline.py

# =====================================================================
# Imports
# =====================================================================
from pathlib import Path
import pandas as pd

from preprocessing.edf_to_csv import edf_to_csv, OUT_DIR
from preprocessing.GSSC_to_csv import GSSC_to_csv, GSSC_DIR
from preprocessing.merge import merge_csv_files, merge_edf_rem_events

# =====================================================================
# TEST - edf_rem_events 
# =====================================================================
merged_df = merge_edf_rem_events(
    eog_path="eog_data.csv",
    events_path="rem_events.csv",
    time_col="time_sec",
    loc_col="LOC",
    roc_col="ROC",
    start_col="Start",
    end_col="End",
    peak_col="Peak",
)

print(merged_df.head())
print(merged_df[merged_df["is_rem_event"]].head())

