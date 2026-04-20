# Filename: boxplot_control_vs_rbd.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Plots feature distributions across diagnostic groups (Control, PD, PD+RBD, iRBD) from a feature CSV.


# =====================================================================
# Merge excel file
# =====================================================================
KEEP_COLS = [
    "DCSM_ID", "Sex", "Age at recording", "CPAP", "BMI", "AHI", "DZ768D", "DR298A", "DG209", "DG47B2", "DG4772", "DG473",
    "DG4744", "ZM95300", "Validated", "Other neurological disorders", "Control", "PD(-RBD)", "PD(+RBD)", "iRBD","PLM",]

def merge_patient_info(
        feature_csv:  str | Path,
        patient_excel: str | Path,
        output_csv:   str | Path = "features_csv/features_with_info.csv",
        subject_col:  str = "DCSM_ID",
) -> pd.DataFrame:
    """
    Join patient demographic and diagnostic information from an Excel file
    onto the extracted feature CSV.

    Parameters
    ----------
    feature_csv : str | Path
        Path to the feature CSV (output of collect_features).
    patient_excel : str | Path
        Path to the Excel file containing patient information.
    output_csv : str | Path
        Path to save the merged CSV. Default is 'features_csv/features_with_info.csv'.
    subject_col : str
        Column name used to join the two files. Default is 'DCSM_ID'.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with patient info columns added.
    """
    # --- 1) Load feature CSV ---
    features_df = pd.read_csv(feature_csv)
    print(f"Features:  {features_df.shape[0]} subjects | {features_df.shape[1]} columns")
    print(f"DCSM_ID sample: {features_df[subject_col].head(3).tolist()}")

    # --- 2) Load Excel ---
    patient_df = pd.read_excel(patient_excel)
    print(f"\nExcel:     {patient_df.shape[0]} rows | {patient_df.shape[1]} columns")
    print(f"Excel columns: {patient_df.columns.tolist()}")

    # --- 3) Keep only relevant columns ---
    available = [c for c in KEEP_COLS if c in patient_df.columns]
    missing   = [c for c in KEEP_COLS if c not in patient_df.columns]
    if missing:
        print(f"\nWarning — columns not found in Excel and skipped: {missing}")

    patient_df = patient_df[available]
    print(f"\nKeeping {len(available)} columns from Excel")

    # --- 4) Join ---
    combined = pd.merge(features_df, patient_df, on=subject_col, how="left")
    print(f"\nMerged:    {combined.shape[0]} subjects | {combined.shape[1]} columns")

    n_unmatched = combined[available[1]].isna().sum()  # check first info col for NaNs
    if n_unmatched > 0:
        print(f"Warning — {n_unmatched} subjects had no match in Excel (check DCSM_ID format)")

    # --- 5) Save ---
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    return combined


if __name__ == "__main__":
    merge_patient_info(
        feature_csv   = "features_csv/features.csv",
        patient_excel = "path/to/patient_info.xlsx",  # ← update this
        output_csv    = "features_csv/features_with_info.csv",
    )

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# =====================================================================
# Constants
# =====================================================================
GROUP_COLORS = {
    "Control":  "#4477AA",
    "PD":       "#EE7733",
    "PD+RBD":   "#AA3377",
    "iRBD":     "#228833",
}

# =====================================================================
# Helper
# =====================================================================
def _assign_group(df: pd.DataFrame) -> pd.Series:
    """
    Assign each subject to a diagnostic group based on binary columns.
    Control group is identified by PLM, control, or DG473 in DCSM_ID.

    Priority order: iRBD → PD+RBD → PD → Control
    """
    groups = pd.Series("Control", index=df.index)

    if "PD" in df.columns:
        groups[df["PD"] == 1] = "PD"

    if "PD" in df.columns and "iRBD" in df.columns:
        groups[(df["PD"] == 1) & (df["iRBD"] == 1)] = "PD+RBD"

    if "iRBD" in df.columns:
        groups[(df["iRBD"] == 1) & (df.get("PD", 0) == 0)] = "iRBD"

    # Override with DCSM_ID if control keywords present
    if "DCSM_ID" in df.columns:
        control_mask = df["DCSM_ID"].str.contains(
            "PLM|control|DG473", case=False, na=False
        )
        groups[control_mask] = "Control"

    return groups


# =====================================================================
# Main plot function
# =====================================================================
def plot_group_comparison(
        feature_csv:    str | Path,
        features:       list[str] | None = None,
        out_dir:        Path | None = None,
        n_cols:         int = 4,
) -> None:
    """
    Plot boxplots and violin plots of features across diagnostic groups.

    Parameters
    ----------
    feature_csv : str | Path
        Path to the feature CSV (output of collect_features).
    features : list[str] | None
        List of feature column names to plot. If None, all numeric
        features are plotted.
    out_dir : Path | None
        If provided, saves the figure as PNG. Otherwise shows interactively.
    n_cols : int
        Number of columns in the subplot grid. Default is 4.
    """
    # --- 1) Load ---
    df = pd.read_csv(feature_csv)
    print(f"Loaded: {feature_csv}  ({df.shape[0]} subjects, {df.shape[1]} columns)")

    # --- 2) Assign groups ---
    df["_group"] = _assign_group(df)
    group_counts = df["_group"].value_counts()
    print(f"Group counts:\n{group_counts.to_string()}\n")

    # --- 3) Select features ---
    exclude = {"subject_id", "_group", "PD", "iRBD"}
    if features is None:
        features = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
    print(f"Plotting {len(features)} features...")

    # --- 4) Build figure ---
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        gridspec_kw={"hspace": 0.6, "wspace": 0.4},
    )
    axes = np.array(axes).flatten()

    groups     = list(GROUP_COLORS.keys())
    colors     = list(GROUP_COLORS.values())
    x_positions = np.arange(len(groups))

    for ax, feat in zip(axes, features):
        data_per_group = [
            df.loc[df["_group"] == g, feat].dropna().values
            for g in groups
        ]

        # --- Violin ---
        parts = ax.violinplot(
            [d if len(d) > 1 else [np.nan] for d in data_per_group],
            positions=x_positions,
            showmedians=False,
            showextrema=False,
            widths=0.6,
        )
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor("none")

        # --- Boxplot ---
        bp = ax.boxplot(
            [d if len(d) > 0 else [np.nan] for d in data_per_group],
            positions=x_positions,
            widths=0.25,
            patch_artist=True,
            medianprops=dict(color="#F43F5E", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
            manage_ticks=False,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # --- Jittered points ---
        for xi, (d, color) in enumerate(zip(data_per_group, colors)):
            if len(d) == 0:
                continue
            jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(d))
            ax.scatter(
                xi + jitter, d,
                color=color, s=14, alpha=0.6, zorder=3, edgecolors="none"
            )

        ax.set_title(feat, fontsize=8, fontweight="bold", pad=4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [f"{g}\n(n={len(d)})" for g, d in zip(groups, data_per_group)],
            fontsize=7,
        )
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)

    # --- Hide unused axes ---
    for ax in axes[len(features):]:
        ax.set_visible(False)

    # --- Legend ---
    legend_patches = [
        mpatches.Patch(color=c, label=g, alpha=0.7)
        for g, c in GROUP_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(GROUP_COLORS),
        fontsize=9,
        title="Diagnostic Group",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )
    fig.suptitle("Feature Distributions by Diagnostic Group", fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.06)

    # --- 5) Save or show ---
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / "group_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")
    else:
        plt.show()
        plt.close(fig)


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    plot_group_comparison(
        feature_csv = "features_csv/features.csv",
        features    = None,   # None = all numeric features
        out_dir     = None,   # None = show interactively
        n_cols      = 4,
    )