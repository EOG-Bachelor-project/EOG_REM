# Filename: plot.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Utilities for plotting EOG signals, GSSC sleep staging, and REM event annotations for visualization and analysis.

# ========================================================================
# Imports
# ========================================================================
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========================================================================
# Functions
# ========================================================================

# 1 ————————————————————————————————————————————————————————————————————————
# 1 Function to plot EOG signals
# 1 ————————————————————————————————————————————————————————————————————————
def plot_eog_signals(eog_file: str, 
                     Lcolor: str = "blue", 
                     Rcolor: str = "red",
                     save_path: str = None) -> None:
    """
    Plots EOG signals from a CSV file.

    Parameters
    ----------
    eog_file : str
        The path to the EOG CSV file.
    Lcolor : str
        The color for the LOC signal.
    Rcolor : str
        The color for the ROC signal.
    save_path : str
        Optional path to save the plot image. If None, the plot will be displayed instead.
    """

    # 1) Load EOG CSV and check for required columns
    eog_df = pd.read_csv(eog_file)
    if "time_sec" not in eog_df.columns or "LOC" not in eog_df.columns or "ROC" not in eog_df.columns:
        raise ValueError("EOG CSV must contain 'time_sec', 'LOC', and 'ROC' columns.")
    
    # 2) Plot LOC and ROC signals over time
    plt.figure(figsize=(15, 5), dpi=2000)
    plt.plot(eog_df["time_sec"], eog_df["LOC"], label="LOC", color=Lcolor) # Plot LOC signal
    plt.plot(eog_df["time_sec"], eog_df["ROC"], label="ROC", color=Rcolor) # Plot ROC signal
    plt.xlabel("Time [s]")                                                 # Label for x-axis
    plt.ylabel("Amplitude")                                                # Label for y-axis
    plt.title("EOG Signals")                                               # Title for the plot
    plt.legend()                                                           # Legend
    plt.grid()                                                             # Grid for better visibility
    plt.tight_layout()

    # 3) Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=2000)
    else:
        plt.show()

# 2 ————————————————————————————————————————————————————————————————————————
# 2 Function to plot EOG signals with GSSC staging
# 2 ———————————————————————————————————————————————————————————————————————— 




# 3 ———————————————————————————————————————————————————————————————————————— 
# 3 Function to plot EOG signals with GSSC staging and REM event annotations
# 3 ————————————————————————————————————————————————————————————————————————