# Filename: plot.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Plots fixed-length EOG signal epochs for a given sleep stage as enumerated subplots, with per-stage background shading and a hypnogram subplot.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from art import *

# =====================================================================
# Constants
# =====================================================================
 
# Sleep stage background colours
STAGE_COLORS = {
    "REM": "#DDAA33", 
    "N3": "#114477", 
    "N2": "#4477AA",  
    "N1": "#77AADD",  
    "W": "#BBBBBB", 
}

# Signal colours
SIG_COLORS = {
    "LOC": "#d73027",
    "ROC": "#4575b4"
}

# EM type colours
EM_TYPE_COLORS = {
    "SEM": "#228833",
    "REM": "#EE7733"
}

# Epoch type colours
EPOCH_TYPE_COLORS = {
    "Phasic":  "#AA3377",  
    "Tonic":   "#CCDDAA", 
}

# Default detection thresholds (µV) – shown as ±horizontal lines on signal axes
THRESHOLDS = {
    "Amp_Thresh_SEM": 50.0,   # SEM amplitude threshold
    "amp_thresh_rem": 150.0,  # REM amplitude threshold
}

THRESHOLD_STYLES = {
    "Amp_Thresh_SEM": dict(color="#5C5C5C", linestyle="--",  linewidth=1.0, alpha=0.8, label="SEM thresh (50 µV)"),
    "amp_thresh_rem": dict(color="#242424", linestyle="-.", linewidth=1.0, alpha=0.8, label="REM thresh (150 µV)"),
}

# Numeric mapping for hypnogram y-axis
STAGE_ORDER = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

# =====================================================================
# Helper functions
# =====================================================================
def _get_span_groups(epoch_df: pd.DataFrame, stage_col: str) -> pd.DataFrame:
    """Returns a DataFrame of consecutive stage runs with t_start, t_end, stage."""
    epoch_df = epoch_df.copy()
    epoch_df["_span"] = (epoch_df[stage_col] != epoch_df[stage_col].shift()).cumsum() # Findstage changes occurrences and label each run with a unique span ID.
    return (
        epoch_df.groupby("_span")
        .agg(t_start=("_t", "first"), t_end=("_t", "last"), stage=(stage_col, "first"))
        .reset_index(drop=True)
    )

def _shade_stages(ax, span_groups: pd.DataFrame):
    """Shade background by sleep stage."""
    for _, sp in span_groups.iterrows():
        ax.axvspan(
            sp["t_start"], sp["t_end"],                         # Span start and end times
            color=STAGE_COLORS.get(sp["stage"], "#cccccc"),   # Get color for the stage, default to light grey
            alpha=0.2, linewidth=0
        )

def _shade_epoch_type(ax, epoch_df: pd.DataFrame):
    """Shade background by EpochType (Phasic/Tonic)."""
    epoch_df = epoch_df.copy()

    epoch_df["_epoch_span"] = (
        epoch_df["EpochType"].fillna("None") !=        # Identify where EpochType changes (treat NaN as "None")
        epoch_df["EpochType"].fillna("None").shift()   # Compare to previous row to find changes
    ).cumsum()

    epoch_spans = (
        epoch_df.groupby("_epoch_span")
        .agg(t_start=("_t", "first"), t_end=("_t", "last"), etype=("EpochType", "first"))
        .reset_index(drop=True)
    )

    for _, sp in epoch_spans.iterrows():
        color = EPOCH_TYPE_COLORS.get(sp["etype"], None)
        if color:
            ax.axvspan(sp["t_start"], sp["t_end"], color=color, alpha=0.4, linewidth=0)

def _draw_threshold_lines(ax, thresholds: dict | None = None):
    """
    Draw ±horizontal threshold lines on a signal axis.
 
    Parameters
    ----------
    ax : matplotlib Axes
    thresholds : dict | None
        Mapping of threshold name → µV value.  If None, the module-level
        THRESHOLDS dict is used.  Pass an empty dict {} to suppress lines.
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    added_labels = set()

    for name, val in thresholds.items():
        style = THRESHOLD_STYLES.get(name, dict(color="grey", linestyle="-.", linewidth=0.8, alpha=0.6)) # Default style if not specified
        label = style.get("label", name)                                                                 # Use provided label or fallback to the threshold name
        plot_kw = {k: v for k, v in style.items() if k != "label"}                                       # Extract plotting kwargs, except label

        # positive threshold
        ax.axhline( val, **plot_kw, label=label if label not in added_labels else None, zorder=1)

        # negative threshold (mirror)
        ax.axhline(-val, **plot_kw, label=None, zorder=1)

        added_labels.add(label)

def _plot_signal(ax, t, signal_uv, color, label=None, lw=0.8):
    """Plot a signal in µV."""
    ax.plot(t, signal_uv, color=color, linewidth=lw, label=label, zorder=2)
 
 
def _overlay_segments(ax, t, signal_uv, mask, color, label=None):
    """
    Overlay coloured segments on a signal wherever mask is True.
    Only the first segment gets the label so the legend stays clean.
    """
    runs   = np.where(np.diff(np.concatenate([[False], mask, [False]])))[0]
    first  = True
    for s, e in zip(runs[0::2], runs[1::2]):
        ax.plot(t[s:e], signal_uv[s:e],
                color=color, linewidth=1.8, zorder=3,
                label=label if first else None)
        first = False
 
def _format_signal_ax(ax, title, window_sec, epoch_sec: float = 4.0):
    """Apply common formatting to a signal subplot, including epoch boundary lines."""
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Amplitude [µV]", fontsize=9)
    ax.axhline(0, color="black", alpha=0.4, linewidth=0.5)
    ax.grid(alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, window_sec)

def _epoch_type_legend_patches():
    """Return legend patches for Phasic/Tonic."""
    return [mpatches.Patch(color=c, label=s, alpha=0.5)
            for s, c in EPOCH_TYPE_COLORS.items()]

# =====================================================================
# Functions
# =====================================================================

# 1 —————————————————————————————————————————————————————————————————————
# 1 Function to plot EOG epochs for a given sleep stage
# 1 —————————————————————————————————————————————————————————————————————
def plot_eog_epochs(
        file:       str | Path, 
        stage:      str = "REM",
        window_sec: float = 30,
        epoch_sec:  float = 4.0,
        time_col:   str = "time_sec",
        loc_col:    str = "LOC",
        roc_col:    str = "ROC",
        stage_col:  str = "stage",
        max_epochs: int | None = None,
        out_dir:    Path | None = None,
        show_em:    bool = True,
        thresholds:  dict | None = None,
    ) -> None:
    """
    Plot a single, large LOC + ROC subplot per epoch with a compact hypnogram below it.

    The signal panel shows:
     - Stage background shading
     - Phasic / Tonic background shading (if EpochType column present)
     - SEM and REM overlays (if EM_Type / is_em_event columns present)
     - ±Threshold horizontal lines

    If `show-em` is True amd EM (eye movement) columns are present, detected eye movement periods are overlaid as shaded regions with a peak marker on all signal subplots. \\
    Epochs are derived from consecutive GSSC stage labels. Each unique run of the target stage is treated as one epoch, then cropped/padded to `window_sec`.

    Parameters
    ----------
    file : str | Path
        Path to the merged CSV (output of merge_all), containing EOG signals
        and GSSC stage labels aligned per sample.
    stage : str
        Sleep stage to extract epochs from (e.g. 'REM', 'N2', 'W'). Default is 'REM'.
    window_sec : float
        Length of each epoch window in seconds. Default is **30.0 s**.
    epoch_sec : float
        Length of each epoch in seconds. Default is **4.0 s**. \\
        Vertical dashed lines are drawn at every ``epoch_sec`` boundary within the display window so you can relate the signal to the Phasic/Tonic classification.
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
    show_em : bool
        If true, overlay detected EM events as shaded regions. Default is **True**.
    thresholds : dict | None
        Mapping of threshold name → µV value for horizontal lines.
        Defaults to the module-level THRESHOLDS dict.
        Pass {} to suppress all threshold lines.
 
    Returns
    -------
    None
    """
    
    lprint(length=100, height=1, char="%")
    print("PLOT  EOG  EPOCHS")
    print(f"  display window : {window_sec} [s]")
    print(f"  analysis epoch : {epoch_sec} [s]  ({window_sec/epoch_sec:.1f} epochs per window)")
    lprint(length=100, height=1, char="%")

    # ==== 1) Load CSV file ====
    df = pd.read_csv(file)
    print(f"Columns in merged CSV: \n{df.columns.tolist()}")
    print(f"EpochType values: {df['EpochType'].value_counts().to_dict() if 'EpochType' in df.columns else 'NOT FOUND'}")

    # Validate required columns
    for col in [time_col, loc_col, roc_col, stage_col]:
        if col not in df.columns:
            raise ValueError(f"Merged CSV must contain '{col}' column.")

    # Ensure data is sorted by time
    df = df.sort_values(by=time_col).reset_index(drop=True) 

    # Check which EM columns are available
    has_em_type    = show_em and "EM_Type"     in df.columns
    has_epoch_type = show_em and "EpochType"   in df.columns
    
    # ==== 2) Find epoch start times ====
    # Identify where stage transitions occur and label each run
    df["_block"] = (df[stage_col] != df[stage_col].shift()).cumsum()

    epoch_starts = (
        df[df[stage_col] == stage]
        .groupby("_block")[time_col] # Group by stage runs and get time values
        .first()                     # Get the first time value of each run as the epoch start time
        .reset_index(drop=True)      # Reset index to get a clean list of epoch start times
    )

    if epoch_starts.empty:
        print(f"No epochs found for stage '{stage}' in file {file}.")
        return
    
    # Limit to max_epochs if specified
    if max_epochs is not None:
        epoch_starts = epoch_starts.iloc[:max_epochs]

    print(f"Found {len(epoch_starts)} '{stage}' epoch(s). Plotting {len(epoch_starts)}...")

    # Output directory for saving plots (if specified)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True) 

    # ==== 3) Plot each epoch ====
    for i, epoch_start in enumerate(epoch_starts):
        epoch_end = epoch_start + window_sec
        epoch_df = df[(df[time_col] >= epoch_start) & (df[time_col] < epoch_end)].copy()

        if epoch_df.empty:
            print(f"Epoch {i+1}: no data in window [{epoch_start:.1f}, {epoch_end:.1f}] — skipping.")
            continue

        epoch_df["_t"] = epoch_df[time_col].values - epoch_start # Relative time within the epoch (0 to window_sec)
        t = epoch_df["_t"].values                                # Time vector for plotting
        loc_uv = epoch_df[loc_col].values * 1e6                  # Convert LOC to µV
        roc_uv = epoch_df[roc_col].values * 1e6                  # Convert ROC to µV
 
        # Stage spans for shading
        span_groups = _get_span_groups(epoch_df, stage_col)

        # EM masks
        if has_em_type:
            sem_mask = (epoch_df["EM_Type"].fillna("") == "SEM").values
            rem_mask = (epoch_df["EM_Type"].fillna("") == "REM").values & \
                       epoch_df["is_em_event"].fillna(False).astype(bool).values
        else:
            sem_mask = rem_mask = np.zeros(len(epoch_df), dtype=bool)

        stage_patches = [mpatches.Patch(color=c, label=s) for s, c in STAGE_COLORS.items()]
        _active_thresholds = thresholds if thresholds is not None else THRESHOLDS
        thresh_handles = [
            mpatches.Patch(
                color=THRESHOLD_STYLES.get(n, {}).get("color", "grey"),
                label = THRESHOLD_STYLES.get(n, {}).get("label"),
                alpha=0.8 
            )
            for n in _active_thresholds
        ]

        # ==== 4) Build figure ====
        fig, (ax_sig, ax_hyp) = plt.subplots(
            2, 1, figsize=(15, 10), sharex=False, 
            gridspec_kw={'hspace': 0.5, 'height_ratios': [5, 1]},
        )

        # Title for the epoch
        fig.suptitle(
            f"Epoch {i+1}  |  Stage: {stage}  |  "
            f"t = {epoch_start:.1f} - {epoch_end:.1f} [s]",
            fontsize=13, fontweight="bold",
        )

        # ── Signal panel ──────────────────────────────────────────────
        _shade_stages(ax_sig, span_groups)
        if has_epoch_type:
            _shade_epoch_type(ax_sig, epoch_df)
 
        _plot_signal(ax_sig, t, loc_uv, SIG_COLORS["LOC"], label="LOC", lw=0.9)
        _plot_signal(ax_sig, t, roc_uv, SIG_COLORS["ROC"], label="ROC", lw=0.9)
 
        if has_em_type:
            _overlay_segments(ax_sig, t, loc_uv, sem_mask, EM_TYPE_COLORS["SEM"], label="SEM")
            _overlay_segments(ax_sig, t, roc_uv, sem_mask, EM_TYPE_COLORS["SEM"])
            _overlay_segments(ax_sig, t, loc_uv, rem_mask, EM_TYPE_COLORS["REM"], label="REM event")
            _overlay_segments(ax_sig, t, roc_uv, rem_mask, EM_TYPE_COLORS["REM"])
 
        _draw_threshold_lines(ax_sig, thresholds)
        _format_signal_ax(ax_sig, "LOC + ROC", window_sec, epoch_sec)
 
        sig_handles, _ = ax_sig.get_legend_handles_labels()
        ax_sig.set_xticks(np.arange(0, window_sec +1, 1))
        ax_sig.legend(
            handles=sig_handles + _epoch_type_legend_patches(),
            fontsize=8, loc="center left", ncol=2,
            bbox_to_anchor=(1, 0.5), frameon=True,
        )
 
        # ── Hypnogram panel ───────────────────────────────────────────
        for _, sp in span_groups.iterrows():
            ax_hyp.barh(
                y     = STAGE_ORDER.get(sp["stage"], -1),
                width = sp["t_end"] - sp["t_start"],
                left  = sp["t_start"],
                color = STAGE_COLORS.get(sp["stage"], "#cccccc"),
                height=0.8, align="center",
            )
        ax_hyp.set_yticks(list(STAGE_ORDER.values()))
        ax_hyp.set_yticklabels(list(STAGE_ORDER.keys()), fontsize=8)
        ax_hyp.set_xticks(np.arange(0, window_sec +1, 1))
        ax_hyp.set_xlabel("Time within epoch [s]", fontsize=10)
        ax_hyp.set_xlim(0, window_sec)
        ax_hyp.set_title("Hypnogram", fontsize=9)
        ax_hyp.tick_params(labelsize=8)
 
        fig.legend(
            handles=stage_patches,
            loc="lower center", ncol=len(STAGE_COLORS),
            fontsize=8, title="Sleep stage", title_fontsize=8,
            bbox_to_anchor=(0.5, 0.0), frameon=True,
        )
        fig.subplots_adjust(bottom=0.12, right=0.82)
 

        # --- 5) Save or show the plot ---
        if out_dir is not None:
            fname = out_dir / f"epoch_{i+1:03d}_{stage}_t{epoch_start:.1f}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved epoch plot: {fname.name}")
        else:
            plt.show()
            plt.close(fig)

    tprint("DONE")

# 2 —————————————————————————————————————————————————————————————————————
# 2 Function to plot the full-night overview
# 2 —————————————————————————————————————————————————————————————————————
def plot_fullnight_overview(
        file:       str | Path,
        time_col:   str = "time_sec",
        loc_col:    str = "LOC",
        roc_col:    str = "ROC",
        stage_col:  str = "stage",
        out_dir:    Path | None = None,
        show_em:    bool = True,
        thresholds:  dict | None = None
        ) -> None:
    """
    Plot the full-night EOG recording in a single figure with 4 subplots:
    LOC, ROC, LOC+ROC overlapping (all with stage background shading), and a hypnogram.
 
    Parameters
    ----------
    file : str | Path
        Path to the merged CSV containing EOG signals and GSSC stage labels.
    time_col : str
        Name of the time column. Default is 'time_sec'.
    loc_col : str
        Name of the LOC channel column. Default is 'LOC'.
    roc_col : str
        Name of the ROC channel column. Default is 'ROC'.
    stage_col : str
        Name of the sleep stage column. Default is 'stage'.
    out_dir : Path | None
        If provided, saves the figure as a PNG. Otherwise displays interactively.
    show_em : bool
        If True, overlay SEM/REM segments and Phasic/Tonic shadinbg. Default is **True**
    thresholds : dict | None
        Mapping of threshold name → µV value for horizontal lines.
        Defaults to the module-level THRESHOLDS dict.
        Pass {} to suppress all threshold lines.
 
    Returns
    -------
    None
    """
    
    lprint(length=100, height=1, char="%")
    print("PLOT  FULL-NIGHT")
    lprint(length=100, height=1, char="%")
    
    # --- 1) Load and validate ---
    df = pd.read_csv(file)
    for col in [time_col, loc_col, roc_col, stage_col]:
        if col not in df.columns:
            raise ValueError(f"Merged CSV must contain '{col}' column.")
 
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    has_em_type    = show_em and "EM_Type"     in df.columns
    has_epoch_type = show_em and "EpochType"   in df.columns

    t_min = df[time_col].values / 60
    loc_uv = df[loc_col].values * 1e6
    roc_uv = df[roc_col].values * 1e6
    
    # --- 2) Compute stage shading spans ---
    df["_span_block"] = (df[stage_col] != df[stage_col].shift()).cumsum()
    span_groups = (
        df.groupby("_span_block")
        .agg(
            t_start=(time_col, "first"),
            t_end=(time_col, "last"),
            stage=(stage_col, "first"),
        )
        .reset_index(drop=True)
    )
    span_groups["t_start_min"] = span_groups["t_start"] / 60
    span_groups["t_end_min"]   = span_groups["t_end"]   / 60

    stage_patches = [mpatches.Patch(color=c, label=s) for s, c in STAGE_COLORS.items()]
    _active_thresholds = thresholds if thresholds is not None else THRESHOLDS
    thresh_handles = [
       mpatches.Patch(
            color=THRESHOLD_STYLES.get(n, {}).get("color", "grey"),
            label = THRESHOLD_STYLES.get(n, {}).get("label"),
            alpha=0.8 
        )
        for n in _active_thresholds
    ]
 
    # --- 3) Build figure ---
    fig, (ax_sig, ax_hyp) = plt.subplots(
        2, 1,
        figsize=(20, 9),
        sharex=True,
        gridspec_kw={"hspace": 0.5, "height_ratios": [5, 1]},
    )
 
    fig.suptitle("Full-Night EOG Overview", fontsize=13, fontweight="bold")
 
    # ── Signal panel ──────────────────────────────────────────────────
    # Stage background
    for _, sp in span_groups.iterrows():
        ax_sig.axvspan(
            sp["t_start_min"], sp["t_end_min"],
            color=STAGE_COLORS.get(sp["stage"], "#cccccc"),
            alpha=0.18, linewidth=0,
        )
    
    # Phasic / Tonic shading
    if has_epoch_type:
        df["_ep_span"] = (
            df["EpochType"].fillna("None") != df["EpochType"].fillna("None").shift()
        ).cumsum()

        ep_spans = (
            df.groupby("_ep_span")
            .agg(t_start=(time_col, "first"), t_end=(time_col, "last"), etype=("EpochType", "first"))
            .reset_index(drop=True)
        )
        
        for _, sp in ep_spans.iterrows():
            color = EPOCH_TYPE_COLORS.get(sp["etype"], None)
            if color:
                ax_sig.axvspan(sp["t_start"] / 60, sp["t_end"] / 60, color=color, alpha=0.22, linewidth=0)
        
    # LOC + ROC signals
    ax_sig.plot(t_min, loc_uv, color=SIG_COLORS["LOC"], linewidth=0.35, label="LOC", alpha=0.85, zorder=2)
    ax_sig.plot(t_min, roc_uv, color=SIG_COLORS["ROC"], linewidth=0.35, label="ROC", alpha=0.85, zorder=2)
    
    # SEM / REM overlays
    if has_em_type:
        sem_mask = (df["EM_Type"].fillna("") == "SEM").values
        rem_mask = (df["EM_Type"].fillna("") == "REM").values & df["is_em_event"].fillna(False).astype(bool).values
        
        def _overlay_fullnight(t_arr, sig, mask, color, label):
            runs = np.where(np.diff(np.concatenate([[False], mask, [False]])))[0]
            first = True
            for s, e in zip(runs[0::2], runs[1::2]):
                ax_sig.plot(t_arr[s:e], sig[s:e], color=color, linewidth=1.2, zorder=3,
                            label=label if first else None)
                first = False
        
        _overlay_fullnight(t_min, loc_uv, sem_mask, EM_TYPE_COLORS["SEM"], "SEM")
        _overlay_fullnight(t_min, roc_uv, sem_mask, EM_TYPE_COLORS["SEM"], None)
        _overlay_fullnight(t_min, loc_uv, rem_mask, EM_TYPE_COLORS["REM"], "REM event")
        _overlay_fullnight(t_min, roc_uv, rem_mask, EM_TYPE_COLORS["REM"], None)
        
    _draw_threshold_lines(ax_sig, thresholds)
        
    ax_sig.set_title("LOC + ROC", fontsize=11)
    ax_sig.set_ylabel("Amplitude [µV]", fontsize=10)
    ax_sig.axhline(0, color="black", alpha=0.4, linewidth=0.5)
    ax_sig.set_xticks(np.arange(t_min[0], t_min[-1] + 60, 60))
    ax_sig.grid(alpha=0.25, linestyle="--")
    ax_sig.tick_params(labelsize=8)
    
    sig_handles, _ = ax_sig.get_legend_handles_labels()
    ax_sig.legend(
        handles=sig_handles + _epoch_type_legend_patches(),
        fontsize=8, loc="center left", ncol=2,
        bbox_to_anchor=(1, 0.5), frameon=True,
    )
    
    # ── Hypnogram panel ───────────────────────────────────────────────
    for _, sp in span_groups.iterrows():
        ax_hyp.barh(
            y = STAGE_ORDER.get(sp["stage"], -1),
            width = sp["t_end_min"] - sp["t_start_min"],
            left = sp["t_start_min"],
            color = STAGE_COLORS.get(sp["stage"], "#cccccc"),
            height=0.8, align="center",
            )
    ax_hyp.set_title("Hypnogram", fontsize=9)
    ax_hyp.set_yticks(list(STAGE_ORDER.values()))
    ax_hyp.set_yticklabels(list(STAGE_ORDER.keys()), fontsize=8)
    ax_hyp.set_xticks(np.arange(t_min[0], t_min[-1] + 60, 60))
    ax_hyp.set_xlabel("Time [min]", fontsize=10)
    ax_hyp.tick_params(labelsize=8)
    
    legend_patches = [mpatches.Patch(color=c, label=s) for s, c in STAGE_COLORS.items()]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=len(STAGE_COLORS),
        fontsize=8, title="Sleep Stage", title_fontsize=8,
        bbox_to_anchor=(0.5, 0.0), frameon=True,
    )
    fig.subplots_adjust(bottom=0.10, right=0.82)
    
    # --- 4) Save or show ---
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / "fullnight_overview.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname.name}")
    else:
        plt.show()
        plt.close(fig)
 
    tprint("DONE")
 
 
# 3 —————————————————————————————————————————————————————————————————————
# 3 Function to plot transition epochs
# 3 —————————————————————————————————————————————————————————————————————
def plot_transition_epochs(
        file:           str | Path,
        from_stage:     str | None = None,
        to_stage:       str | None = None,
        window_sec:     float = 60.0,
        epoch_sec:      float = 4.0,
        time_col:       str = "time_sec",
        loc_col:        str = "LOC",
        roc_col:        str = "ROC",
        stage_col:      str = "stage",
        max_epochs:     int | None = None,
        out_dir:        Path | None = None,
        ) -> None:
    """
    Plot EOG epochs centered on stage transitions, showing the signal before
    and after the transition within a single window.
 
    Each epoch is centered on a detected transition point, showing window_sec/2
    of signal before and after. All transitions are plotted unless from_stage
    and/or to_stage are specified to filter to a specific transition type.
 
    Parameters
    ----------
    file : str | Path
        Path to the merged CSV containing EOG signals and GSSC stage labels.
    from_stage : str | None
        The stage transitioning FROM (e.g. 'N2'). If None, all source stages match.
    to_stage : str | None
        The stage transitioning TO (e.g. 'REM'). If None, all target stages match.
    window_sec : float
        Total window length in seconds, centered on the transition. Default is **60.0 s**.
    epoch_sec : float
        Length of each analysis epoch in seconds (e.g. 4 s). Vertical dashed lines are drawn at every epoch_sec boundary within the display window. Default is **4.0 s**.
    time_col : str
        Name of the time column. Default is 'time_sec'.
    loc_col : str
        Name of the LOC channel column. Default is 'LOC'.
    roc_col : str
        Name of the ROC channel column. Default is 'ROC'.
    stage_col : str
        Name of the sleep stage column. Default is 'stage'.
    max_epochs : int | None
        Maximum number of transitions to plot. If None, all are plotted.
    out_dir : Path | None
        If provided, saves each figure as a PNG. Otherwise displays interactively.
 
    Returns
    -------
    None
    """
    lprint(length=100, height=1, char="%")
    print("PLOT  STAGE  TRANSITIONS")
    lprint(length=100, height=1, char="%")
 
    # --- 1) Load and validate ---
    df = pd.read_csv(file)
    for col in [time_col, loc_col, roc_col, stage_col]:
        if col not in df.columns:
            raise ValueError(f"Merged CSV must contain '{col}' column.")
 
    df = df.sort_values(by=time_col).reset_index(drop=True)
    half_win = window_sec / 2
 
    # --- 2) Detect all stage transitions ---
    df["_next_stage"] = df[stage_col].shift(-1)
    transitions = df[df[stage_col] != df["_next_stage"]].copy()
    transitions = transitions.dropna(subset=["_next_stage"])
 
    # Filter by from_stage / to_stage if specified
    if from_stage is not None:
        transitions = transitions[transitions[stage_col] == from_stage]
    if to_stage is not None:
        transitions = transitions[transitions["_next_stage"] == to_stage]
 
    if transitions.empty:
        label = f"{from_stage or '*'} → {to_stage or '*'}"
        print(f"No transitions found for {label}.")
        return
 
    if max_epochs is not None:
        transitions = transitions.iloc[:max_epochs]
 
    label = f"{from_stage or 'any'} -> {to_stage or 'any'}"
    print(f"Found {len(transitions)} transition(s) [{label}]. Plotting...")
 
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
 
    # --- 3) Build shared legend patches ---
    legend_patches = [
        mpatches.Patch(color=color, label=s)
        for s, color in STAGE_COLORS.items()
    ]
 
    # --- 4) Plot each transition epoch ---
    for i, (_, trans_row) in enumerate(transitions.iterrows()):
        trans_time  = trans_row[time_col]
        stage_from  = trans_row[stage_col]
        stage_to    = trans_row["_next_stage"]
 
        win_start = trans_time - half_win
        win_end   = trans_time + half_win
 
        epoch_df = df[(df[time_col] >= win_start) & (df[time_col] < win_end)].copy()
 
        if epoch_df.empty:
            print(f"  Transition {i+1}: no data in window — skipping.")
            continue
 
        # Relative time (0 = transition point)
        t = epoch_df[time_col].values - trans_time
        epoch_df["_t"] = t
 
        # --- 5) Compute stage shading spans ---
        epoch_df["_span_block"] = (epoch_df[stage_col] != epoch_df[stage_col].shift()).cumsum()
        span_groups = (
            epoch_df.groupby("_span_block")
            .agg(
                t_start=("_t", "first"),
                t_end=("_t", "last"),
                stage=(stage_col, "first"),
            )
            .reset_index(drop=True)
        )
 
        def add_stage_shading(ax, span_groups=span_groups):
            for _, span in span_groups.iterrows():
                color = STAGE_COLORS.get(span["stage"], "#cccccc")
                ax.axvspan(span["t_start"], span["t_end"], color=color, alpha=0.2, linewidth=0)
        
        def draw_epoch_lines(ax):
            """Draw epoch boundaries relative to the transition point (t=0)."""
            for xb in np.arange(epoch_sec, half_win + epoch_sec, epoch_sec):
                ax.axvline(xb,  color="#333333", linewidth=1.2, linestyle="--", alpha=0.85, zorder=1)
                ax.axvline(-xb, color="#333333", linewidth=1.2, linestyle="--", alpha=0.85, zorder=1)
 
        # --- 6) Build figure ---
        fig, axs = plt.subplots(
            4, 1,
            figsize=(15, 9),
            sharex=True,
            gridspec_kw={"hspace": 0.5, "height_ratios": [2, 2, 2, 1]},
        )
 
        fig.suptitle(
            f"Transition {i+1}  |  {stage_from} → {stage_to}  |  t = {trans_time:.1f} s",
            fontsize=12,
            fontweight="bold",
        )
 
        # Vertical line marking the transition point
        for ax in axs[:3]:
            ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7, label="Transition")
            draw_epoch_lines(ax)  # Add epoch boundary lines
 
        # Subplot 1: LOC
        axs[0].plot(t, epoch_df[loc_col].values * 1e6, color=SIG_COLORS["LOC"], linewidth=0.8)
        axs[0].set_title("LOC", fontsize=10)
        axs[0].set_ylabel("Amplitude [$\mu$V]", fontsize=9)
        axs[0].axhline(0, color="black", alpha=0.5, linewidth=0.5)
        axs[0].grid(alpha=0.3, linestyle="--")
        axs[0].tick_params(labelsize=8)
        add_stage_shading(axs[0])
 
        # Subplot 2: ROC
        axs[1].plot(t, epoch_df[roc_col].values * 1e6, color=SIG_COLORS["ROC"], linewidth=0.8)
        axs[1].set_title("ROC", fontsize=10)
        axs[1].set_ylabel("Amplitude [$\mu$V]", fontsize=9)
        axs[1].axhline(0, color="black", alpha=0.5, linewidth=0.5)
        axs[1].grid(alpha=0.3, linestyle="--")
        axs[1].tick_params(labelsize=8)
        add_stage_shading(axs[1])
 
        # Subplot 3: LOC + ROC overlapping
        axs[2].plot(t, epoch_df[loc_col].values * 1e6, color=SIG_COLORS["LOC"], linewidth=0.8, label="LOC")
        axs[2].plot(t, epoch_df[roc_col].values * 1e6, color=SIG_COLORS["ROC"], linewidth=0.8, label="ROC")
        axs[2].set_title("LOC + ROC", fontsize=10)
        axs[2].set_ylabel("Amplitude [$\mu$V]", fontsize=9)
        axs[2].axhline(0, color="black", alpha=0.5, linewidth=0.5)
        axs[2].grid(alpha=0.3, linestyle="--")
        axs[2].tick_params(labelsize=8)
        axs[2].legend(fontsize=8, loc="best")
        add_stage_shading(axs[2])
 
        # Subplot 4: Hypnogram
        for _, span in span_groups.iterrows():
            color = STAGE_COLORS.get(span["stage"], "#cccccc")
            y_val = STAGE_ORDER.get(span["stage"], -1)
            axs[3].barh(
                y=y_val,
                width=span["t_end"] - span["t_start"],
                left=span["t_start"],
                color=color,
                height=0.8,
                align="center",
            )
        axs[3].axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        draw_epoch_lines(axs[3])
        axs[3].set_title("Hypnogram", fontsize=10)
        axs[3].set_yticks(list(STAGE_ORDER.values()))
        axs[3].set_yticklabels(list(STAGE_ORDER.keys()), fontsize=8)
        axs[3].set_xlabel("Time relative to transition [s]", fontsize=10)
        axs[3].tick_params(labelsize=8)
        axs[3].set_xlim(-half_win, half_win)
 
        # Shared legend
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
 
        # --- 7) Save or show ---
        if out_dir is not None:
            tag = f"{stage_from}_to_{stage_to}"
            fname = out_dir / f"transition_{i+1:03d}_{tag}_t{trans_time:.1f}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fname.name}")
        else:
            plt.show()
            plt.close(fig)
 
    tprint("DONE")
 
 
# =====================================================================
# Test
# =====================================================================
if __name__ == "__main__":
    plot_eog_epochs(
        file       = "C:/Users/AKLO0022/EOG_REM/merged_csv_eog/DCSM_3_a_contiguous_eog_merged_Umaer.csv",
        stage      = "REM",
        stage_col  = "stage",
        window_sec = 30.0,
        epoch_sec  = 4.0,
        max_epochs = 20,
        out_dir    = None,
    )

    plot_eog_epochs(
        file       = "C:/Users/AKLO0022/EOG_REM/merged_csv_eog/DCSM_3_a_contiguous_eog_merged.csv",
        stage      = "REM",
        stage_col  = "stage",
        window_sec = 30.0,
        epoch_sec  = 4.0,
        max_epochs = 20,
        out_dir    = None,
    )



    plot_fullnight_overview(
        file="C:/Users/AKLO0022/EOG_REM/merged_csv_eog/DCSM_3_a_contiguous_eog_merged_Umaer.csv",
        out_dir=None
    )

    """plot_transition_epochs(
        file        = "C:/Users/AKLO0022/EOG_REM/merged_csv_eog/DCSM_1_a_contiguous_eog_merged.csv", 
        from_stage  = "REM", 
        to_stage    = "W", 
        window_sec  = 60,
        epoch_sec   = 4.0,
        )"""