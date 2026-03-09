# csv_merge.py

# =====================================================================
# Imports
# =====================================================================
import pandas as pd
from pathlib import Path
from preprocessing.upsample import upsample_gssc_to_eog

# =====================================================================
# Functions
# =====================================================================
def merge_csv_files(eog_file: str | Path, gssc_file: str | Path, output_file: str | Path) -> None:
    """
    Merges two CSV files based on a common column and saves the merged result to a new CSV file.\\
    The GSSC staging is upsampled to align with the EOG sample timeline before merging.

    Parameters
    ----------
    eog_file : str | Path
        The path to the EOG CSV file.
    gssc_file : str | Path
        The path to the GSSC CSV file.
    output_file : str | Path
        The path to the output CSV file.
    """

    # 1) Load EOG CSV to check for required columns
    eog_df = pd.read_csv(eog_file)

    if "time_sec" not in eog_df.columns:
        raise ValueError("EOG CSV must contain 'time_sec'.")

    # 2) Get upsampled GSSC staging aligned to EOG timeline
    gssc_df = upsample_gssc_to_eog(eog_file, gssc_file)

    # 3) Merge the original EOG data with the upsampled GSSC staging
    merged_df = pd.concat(
        [
            eog_df.reset_index(drop=True),
            gssc_df.drop(columns=["time_sec"]).reset_index(drop=True),
        ],
        axis=1,
    )
    
    # 4) Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"Merged file saved to: {Path(output_file).name}")