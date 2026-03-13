# Filename: plot.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Plots fixed-length EOG signal epochs for a given sleep stage as enumerated subplots.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from art import *

# =====================================================================
# Functions
# =====================================================================

# —————————————————————————————————————————————————————————————————————
# Function to plot EOG epochs for a given sleep stage
# —————————————————————————————————————————————————————————————————————
def plot_eog_epochs(file: str | Path, 
                    stage: str = "REM",
                    window_sec: float = 30,
                    time_col: str = "time_sec",
                    loc_col: str = "LOC",
                    roc_col: str = "ROC",
                    stage_col: str = "stage",
                    max_epochs: int | None = None,
                    out_dir: Path | None = None
                    ) -> None:
    """
    Plot fixed-length EOG epochs for a given sleep stage as enumerated subplots.\\
    Each epoch gets two subplots: one for the LOC signal and one for the ROC signal, aligned on the same time axis.\\
    
    Epochs are derived from consecutive GSSC stage labels. Each unique run of the target stage is treated as one epoch, then cropped/padded to window_sec.

    Parameters
    ----------
    file : str | Path
        Path to the merged CSV (output of merge_all), containing EOG signals
        and GSSC stage labels aligned per sample.
    stage : str
        Sleep stage to extract epochs from (e.g. 'REM', 'N2', 'W'). Default is 'REM'.
    window_sec : float
        Length of each epoch window in seconds. Default is 30.0.
    time_col : str
        Name of the time column. Default is 'time_sec'.
    loc_col : str
        Name of the LOC channel column. Default is 'LOC'.
    roc_col : str
        Name of the ROC channel column. Default is 'ROC'.
    stage_col : str
        Name of the sleep stage column. Default is 'stage'.
    max_epochs : int | None
        Maximum number of epochs to plot. If None, all epochs are plotted.
    out_dir : Path | None
        If provided, saves each figure as a PNG in this directory instead of showing it.
 
    Returns
    -------
    None
    """

    # 1) Load CSV file
    df = pd.read_csv(file)

    # Validate required columns
    for col in [time_col, loc_col, roc_col, stage_col]:
        if col not in df.columns:
            raise ValueError(f"Merged CSV must contain '{col}' column.")

    # Ensure data is sorted by time
    df = df.sort_values(by=time_col).reset_index(drop=True) 
    
    # 2) Find epoch start times from consecutive stage runs
    # Identify where stage transitions occur and label each run
    df["_stage_block"] = (df[stage_col] != df[stage_col].shift()).cumsum()

    stage_blocks = (
        df[df[stage_col] == stage]
        .groupby("_stage_block")[time_col] # Group by stage runs and get time values
        .first()                           # Get the first time value of each run as the epoch start time
        .reset_index(drop=True)            # Reset index to get a clean list of epoch start times
    )

    if stage_blocks.empty:
        print(f"No epochs found for stage '{stage}' in file {file}.")
        return
    
    # Limit to max_epochs if specified
    if max_epochs is not None:
        stage_blocks = stage_blocks.iloc[:max_epochs]
    print(f"Found {len(stage_blocks)} '{stage}' epoch(s). Plotting {len(stage_blocks)}...")

    # Output directory for saving plots (if specified)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True) 

    # 3) Plot each epoch 
    for i, epoch_start in enumerate(stage_blocks):
        epoch_end = epoch_start + window_sec

        epoch_df = df[(df[time_col] >= epoch_start) & (df[time_col] < epoch_end)]

        if epoch_df.empty:
            print(f"Epoch {i+1}: no data in window [{epoch_start:.1f}, {epoch_end:.1f}] — skipping.")
            continue

        # Relative time for each epoch 
        t = epoch_df[time_col].values - epoch_start

        # 4) Create figure with two subplots
        fig, axs = plt.subplots(3, 1, figsize=(15, 7), sharex=True, gridspec_kw={'hspace': 0.5})

        # Title for the epoch
        fig.suptitle(
            f"Epoch {i+1}  |  Stage: {stage}  |  t = {epoch_start:.1f} - {epoch_end:.1f} s",
            fontsize=12,
            fontweight="bold",
        )

        # --- Subplot 1: LOC ---
        axs[0].plot(t, epoch_df[loc_col].values, color="steelblue", linewidth=0.8) # Plot the LOC signal
        axs[0].set_title("LOC", fontsize=10)                                       # Set title for the subplot
        axs[0].set_ylabel("Amplitude", fontsize=9)                                 # Set y-axis label
        axs[0].axhline(0, color="black", linewidth=0.5)                            # Add a horizontal line at y=0 for reference
        axs[0].grid(alpha=0.5, linestyle="--")
        axs[0].tick_params(labelsize=8)
 
        # --- Subplot 2: ROC ---
        axs[1].plot(t, epoch_df[roc_col].values, color="tomato", linewidth=0.8) # Plot the ROC signal
        axs[1].set_title("ROC", fontsize=10)                                    # Set title for the subplot
        axs[1].set_ylabel("Amplitude", fontsize=9)                              # Set y-axis label
        axs[1].axhline(0, color="black", linewidth=0.5)                         # Add a horizontal line at y=0 for reference
        axs[1].grid(alpha=0.5, linestyle="--")
        axs[1].tick_params(labelsize=8)
 
        # --- Subplot 3: LOC + ROC overlapping ---
        axs[2].plot(t, epoch_df[loc_col].values, color="steelblue", linewidth=0.8, label="LOC") # Plot the LOC signal
        axs[2].plot(t, epoch_df[roc_col].values, color="tomato",    linewidth=0.8, label="ROC") # Plot the ROC signal
        axs[2].set_title("LOC + ROC", fontsize=10)                                              # Set title for the subplot
        axs[2].set_ylabel("Amplitude", fontsize=9)                                              # Set y-axis label
        axs[2].axhline(0, color="black", linewidth=0.5)                                         # Add a horizontal line at y=0 for reference
        axs[2].grid(alpha=0.5, linestyle="--")
        axs[2].tick_params(labelsize=8)
        axs[2].legend(fontsize=8, loc="upper right")                                            # Add legend to the overlapping plot
 
        axs[-1].set_xlabel("Time within epoch (s)", fontsize=10)                                # Set x-axis label on the last subplot
        axs[-1].set_xlim(0, window_sec)

        # 5) Save or show the plot
        if out_dir is not None:
            fname = out_dir / f"epoch_{i+1:03d}_{stage}_t{epoch_start:.1f}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved epoch plot: {fname.name}")
        else:
            plt.show()
            plt.close(fig)
    tprint("DONE")

# TEST
plot_eog_epochs(
    file="C:/Users/AKLO0022/EOG_REM/local_csv_eog/merged_outpu/DCSM_1_a_contiguous_eog_merged.csv",
    stage="REM",
    window_sec=10.0,
    max_epochs=10,
    out_dir=None
)