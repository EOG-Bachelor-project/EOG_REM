# Filename: plot.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Plots fixed-length EOG signal epochs for a given sleep stage as enumerated subplots, with per-stage background shading and a hypnogram subplot.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from art import *

# =====================================================================
# Constants
# =====================================================================
 
# Color map for each sleep stage
STAGE_COLORS = {
    "W":   "#FA8072",  # orange
    "N1":  "#0c0d0d",  # light teal
    "N2":  "#457b9d",  # medium blue
    "N3":  "#1d3557",  # dark navy
    "REM": "#C71585",  # red
}
 
# Numeric mapping for hypnogram y-axis
STAGE_ORDER = {"W": 4, "N1": 3, "N2": 2, "N3": 1, "REM": 0}

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
    Each epoch gets 4 subplots: LOC, ROC, LOC+ROC overlapping (all with stage background shading), and a hypnogram bar at the bottom.
    
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

    # --- 1) Load CSV file ---
    df = pd.read_csv(file)

    # Validate required columns
    for col in [time_col, loc_col, roc_col, stage_col]:
        if col not in df.columns:
            raise ValueError(f"Merged CSV must contain '{col}' column.")

    # Ensure data is sorted by time
    df = df.sort_values(by=time_col).reset_index(drop=True) 
    
    # --- 2) Find epoch start times from consecutive stage runs ---
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
        
    # --- 3) Build shared legend patches ---
    legend_patches = [
        mpatches.Patch(color=color, label=s) 
        for s, color in STAGE_COLORS.items()
    ]

    # --- 4) Plot each epoch --- 
    for i, epoch_start in enumerate(stage_blocks):
        epoch_end = epoch_start + window_sec

        epoch_df = df[(df[time_col] >= epoch_start) & (df[time_col] < epoch_end)]

        if epoch_df.empty:
            print(f"Epoch {i+1}: no data in window [{epoch_start:.1f}, {epoch_end:.1f}] — skipping.")
            continue

        # Relative time for each epoch 
        t = epoch_df[time_col].values - epoch_start
        epoch_df["_t"] = t

        # --- 5) Compute stage shading spans ---
        epoch_df["_span_block"] = (epoch_df[stage_col] != epoch_df[stage_col].shift()).cumsum() 
 
        span_groups = (
            epoch_df.groupby("_span_block") # Group by consecutive runs of the same stage within the epoch
            .agg(                           # Aggregate to get the start and end time of each run, and the stage label
                t_start=("_t", "first"),
                t_end=("_t", "last"),
                stage=(stage_col, "first"),
            )
            .reset_index(drop=True)         # Drop the span block index to get a clean DataFrame of stage spans
        )
 
        # Helper to shade stage regions on an axis
        def add_stage_shading(ax):
            for _, span in span_groups.iterrows():
                color = STAGE_COLORS.get(span["stage"], "#cccccc")
                ax.axvspan(span["t_start"], span["t_end"], color=color, alpha=0.2, linewidth=0)

        # --- 6) Create figure with subplots ---
        fig, axs = plt.subplots(4, 1, figsize=(15, 9), sharex=True, gridspec_kw={'hspace': 0.5, 'height_ratios': [2, 2, 2, 1]})

        # Title for the epoch
        fig.suptitle(
            f"Epoch {i+1}  |  Stage: {stage}  |  t = {epoch_start:.1f} - {epoch_end:.1f} s",
            fontsize=12,
            fontweight="bold",
        )

        # Subplot 1: LOC 
        axs[0].plot(t, epoch_df[loc_col].values, color="steelblue", linewidth=0.8) # Plot the LOC signal
        axs[0].set_title("LOC", fontsize=10)                                       # Set title for the subplot
        axs[0].set_ylabel("Amplitude", fontsize=9)                                 # Set y-axis label
        axs[0].axhline(0, color="black", alpha=0.5, linewidth=0.5)                 # Add a horizontal line at y=0 for reference
        axs[0].grid(alpha=0.3, linestyle="--")
        axs[0].tick_params(labelsize=8)
        add_stage_shading(axs[0])
 
        # Subplot 2: ROC 
        axs[1].plot(t, epoch_df[roc_col].values, color="tomato", linewidth=0.8) # Plot the ROC signal
        axs[1].set_title("ROC", fontsize=10)                                    # Set title for the subplot
        axs[1].set_ylabel("Amplitude", fontsize=9)                              # Set y-axis label
        axs[1].axhline(0, color="black", alpha=0.5, linewidth=0.5)              # Add a horizontal line at y=0 for reference
        axs[1].grid(alpha=0.5, linestyle="--")
        axs[1].tick_params(labelsize=8)
        add_stage_shading(axs[1])
 
        # Subplot 3: LOC + ROC overlapping 
        axs[2].plot(t, epoch_df[loc_col].values, color="steelblue", linewidth=0.8, label="LOC") # Plot the LOC signal
        axs[2].plot(t, epoch_df[roc_col].values, color="tomato",    linewidth=0.8, label="ROC") # Plot the ROC signal
        axs[2].set_title("LOC + ROC", fontsize=10)                                              # Set title for the subplot
        axs[2].set_ylabel("Amplitude", fontsize=9)                                              # Set y-axis label
        axs[2].axhline(0, color="black", alpha=0.5, linewidth=0.5)                              # Add a horizontal line at y=0 for reference
        axs[2].grid(alpha=0.5, linestyle="--")
        axs[2].tick_params(labelsize=8)
        axs[2].legend(fontsize=8, loc="best")                                                   # Add legend to the overlapping plot
        add_stage_shading(axs[2])

        # Subplot 4: Hypnogram bar
        for _, span in span_groups.iterrows():
            color = STAGE_COLORS.get(span["stage"], "#cccccc")
            y_val = STAGE_ORDER.get(span["stage"], -1) # Get the numeric y-value for the stage, default to -1 if stage is unknown
            axs[3].barh(
                y=y_val,                                # Plot a horizontal bar at the corresponding y-value for the stage
                width=span["t_end"] - span["t_start"],  # Width of the bar corresponds to the duration of the stage run
                left=span["t_start"],                   # Left position of the bar corresponds to the start time of the stage run
                color=color,   
                height=0.8,
                align="center",
            )
 
        axs[3].set_title("Hypnogram", fontsize=10)                   # Set title for the hypnogram subplot
        axs[3].set_yticks(list(STAGE_ORDER.values()))               
        axs[3].set_yticklabels(list(STAGE_ORDER.keys()), fontsize=8) # Set y-tick labels to the stage names
        axs[3].set_xlabel("Time within epoch [s]", fontsize=10)      # Set x-axis label for the hypnogram subplot
        axs[3].tick_params(labelsize=8)
        axs[3].set_xlim(0, window_sec)                               # Set x-axis limits to the epoch window

        # Shared stage legend at bottom 
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(STAGE_COLORS),
            fontsize=8,
            title="Sleep Stage",
            title_fontsize=8,
            bbox_to_anchor=(0.5, 0.0),
            frameon=True,
        )

        fig.subplots_adjust(bottom=0.1)

        # --- 7) Save or show the plot ---
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