# Filename: qq-plot.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Generates QQ-plots per feature per class to visually assess whether
#              feature distributions are normal within each class. Also computes
#              a Shapiro-Wilk test summary table. Used to decide between
#              parametric (Welch's t-test) and non-parametric (Mann-Whitney U)
#              testing in the downstream univariate statistics step.

# Usage:
#   python -m statistics.qq-plot --csv features_csv/features.csv --label-col group --out-dir reports/qq_output

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ANSI helpers
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
# =====================================================================
# Constants
# =====================================================================

# Same exclude convention as prepare_data.py — these columns are never features.
EXCLUDE_NON_FEATURE = {
    "subject_id", "DCSM_ID",
    "Control", "PD(-RBD)", "PD(+RBD)", "iRBD", "PLM",
    "group",
}


# =====================================================================
# Data loading
# =====================================================================

def load_data(
        csv_path:  str | Path,
        label_col: str,
        ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the feature CSV and return the feature matrix and label vector.

    Drops all non-feature columns (subject IDs, group indicator columns, and
    the 'group' column itself) so that only numeric features remain in X.

    Parameters
    ----------
    csv_path : str | Path
        Path to the feature CSV (output of the extract / prepare step).
    label_col : str
        Name of the label column (e.g., 'group').

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix with all non-feature columns removed.
    y : pd.Series
        Label vector aligned with X.
    """
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' does not exist in CSV.")
    y = df[label_col]

    # Drop all non-feature columns and keep only numeric ones
    drop_cols = [c for c in df.columns if c in EXCLUDE_NON_FEATURE]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    return X, y
 
 
# =====================================================================
# Binary label assignment
# =====================================================================

def make_binary(
        y:              pd.Series,
        positive_class,
        negative_class = None,
        ) -> pd.Series:
    """
    Convert a (possibly multi-class) label vector to a binary 0/1 Series.

    If `negative_class` is None, all rows that are not in `positive_class`
    become 0 (one-vs-rest). Otherwise only rows matching `positive_class` or
    `negative_class` are kept (everything else gets NaN and should be dropped
    by the caller before further analysis).

    Parameters
    ----------
    y : pd.Series
        Label vector (can contain strings, ints, etc.).
    positive_class : any
        Value in `y` that should map to 1 (e.g., 'iRBD').
    negative_class : any, optional
        Value in `y` that should map to 0 (e.g., 'Control').
        If None, all non-positive rows become 0. **Default is None**.

    Returns
    -------
    pd.Series
        Binary labels (0/1), with NaN for rows that don't fit the split.
    """
    if negative_class is None:
        return (y == positive_class).astype(int)
    out = pd.Series(np.nan, index=y.index)
    out[y == negative_class] = 0
    out[y == positive_class] = 1
    return out
 
 
# =====================================================================
# QQ-plot for a single feature
# =====================================================================

def qq_plot_feature(
        values_pos:   pd.Series,
        values_neg:   pd.Series,
        feature_name: str,
        out_path:     str | Path,
        pos_label:    str = "positive",
        neg_label:    str = "negative",
        ) -> None:
    """
    Build and save a QQ-plot for one feature with both classes side by side.

    Each subplot shows the feature's values for one class plotted against the
    theoretical quantiles of a normal distribution, plus the Shapiro-Wilk
    test result annotated in the title. If a class has fewer than 3 samples
    the subplot is skipped.

    Parameters
    ----------
    values_pos : pd.Series
        Feature values for the positive class.
    values_neg : pd.Series
        Feature values for the negative class.
    feature_name : str
        Name of the feature (used in the figure title).
    out_path : str | Path
        Where to save the pdf.
    pos_label : str
        Label shown in the positive class subplot title. **Default is 'positive'**.
    neg_label : str
        Label shown in the negative class subplot title. **Default is 'negative'**.
    """
    # ---- 1) Set up figure with two subplots, one per class ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, vals, class_label, color in [
        (axes[0], values_pos, f"Positive class ({pos_label})", "tab:blue"),
        (axes[1], values_neg, f"Negative class ({neg_label})", "tab:orange"),
    ]:
        vals = vals.dropna().values
        if len(vals) < 3:
            ax.set_title(f"{class_label}: not enough samples")
            continue

        # ---- 2) QQ-plot against the normal distribution ----
        stats.probplot(vals, dist="norm", plot=ax)

        # Style the points and the reference line
        ax.get_lines()[0].set_markerfacecolor(color)
        ax.get_lines()[0].set_markeredgecolor(color)
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color("black")
        ax.get_lines()[1].set_linewidth(1)

        # ---- 3) Annotate with Shapiro-Wilk result ----
        # Shapiro is only reliable up to ~5000 samples — subsample if larger.
        sample = (vals if len(vals) <= 5000
                  else np.random.default_rng(0).choice(vals, 5000, replace=False))
        try:
            sw_stat, sw_p = stats.shapiro(sample)
            sw_text = f"Shapiro-Wilk: W={sw_stat:.3f}, p={sw_p:.3g}"
            sw_text += "\n(rejects normality)" if sw_p < 0.05 else "\n(normality ok)"
        except Exception as e:
            sw_text = f"Shapiro failed: {e}"

        ax.set_title(f"{class_label}\n{sw_text}", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(f"QQ-plot: {feature_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
# =====================================================================
# Shapiro-Wilk summary table for all features and both classes
# =====================================================================

def shapiro_summary(
        X:     pd.DataFrame,
        y_bin: pd.Series,
        ) -> pd.DataFrame:
    """
    Run Shapiro-Wilk on every (feature, class) pair and return a tidy table.

    For each feature, the test is run separately on the positive-class values
    and the negative-class values. Classes with fewer than 3 samples produce
    NaN entries. Samples are randomly subsampled to 5000 if larger, since
    the Shapiro-Wilk test is only reliable up to ~5000 observations.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows aligned with `y_bin`).
    y_bin : pd.Series
        Binary label vector (0/1).

    Returns
    -------
    pd.DataFrame
        One row per (feature, class) with columns:
          - feature        : feature name
          - class          : 'positive' or 'negative'
          - n              : number of samples used
          - W              : Shapiro-Wilk statistic
          - p              : p-value
          - normal_at_0.05 : True if p >= 0.05 (i.e. normality not rejected)
    """
    rows = []
    rng = np.random.default_rng(0)

    for col in X.columns:
        for class_label, mask in [("positive", y_bin == 1), ("negative", y_bin == 0)]:
            vals = X.loc[mask, col].dropna().values

            # Not enough samples — record NaN entry and continue
            if len(vals) < 3:
                rows.append({
                    "feature":        col,
                    "class":          class_label,
                    "n":              len(vals),
                    "W":              np.nan,
                    "p":              np.nan,
                    "normal_at_0.05": np.nan,
                })
                continue

            # Subsample if too large for Shapiro-Wilk to be reliable
            sample = vals if len(vals) <= 5000 else rng.choice(vals, 5000, replace=False)
            try:
                W, p = stats.shapiro(sample)
                rows.append({
                    "feature":        col,
                    "class":          class_label,
                    "n":              len(vals),
                    "W":              W,
                    "p":              p,
                    "normal_at_0.05": bool(p >= 0.05),
                })
            except Exception:
                rows.append({
                    "feature":        col,
                    "class":          class_label,
                    "n":              len(vals),
                    "W":              np.nan,
                    "p":              np.nan,
                    "normal_at_0.05": np.nan,
                })

    return pd.DataFrame(rows)
 
# =====================================================================
# CLI helpers
# =====================================================================

def _parse_class(val: str):
    """
    Parse a CLI class argument into the most appropriate Python type.

    Tries int first, then float, otherwise keeps it as a string. This lets
    the user pass labels like 'iRBD', '1', or '1.0' without ceremony.
    """
    if val.isdigit():
        return int(val)
    try:
        return float(val)
    except ValueError:
        return val


# =====================================================================
# Main entry point
# =====================================================================

def main() -> None:
    """
    Generate QQ-plots and a Shapiro-Wilk summary table for all binary splits.

    Workflow:
      1. Load the feature CSV and extract numeric features.
      2. For each binary split (Control vs iRBD, Control vs PD(+RBD), Control vs all):
           a. Build a binary 0/1 label vector.
           b. Save one QQ-plot pdf per feature to <out_dir>/<split>/plots/.
           c. Compute a tidy Shapiro-Wilk table and normality overview, saved as CSV.
    """
    # ---- 1) Parse CLI arguments ----
    parser = argparse.ArgumentParser(
        description="QQ-plots and Shapiro-Wilk summary for all binary splits in one command."
    )
    parser.add_argument("--csv",       required=True,  help="Path to the feature CSV.")
    parser.add_argument("--label-col", default="group", help="Name of the label column. Default: 'group'.")
    parser.add_argument("--out-dir",   default="reports/qq_output", help="Root output directory.")
    parser.add_argument("--max-features", type=int, default=None, help="Limit features (debug only).")
    args = parser.parse_args()

    # ---- 2) Define all binary splits to run ----
    splits = [
        {"positive": "iRBD",     "negative": "Control",  "tag": "control_vs_irbd"},
        {"positive": "PD(+RBD)", "negative": "Control",  "tag": "control_vs_pdrbd"},
        {"positive": "iRBD",     "negative": "Control",  "tag": "control_vs_all",
         "extra_positive": "PD(+RBD)"},  # iRBD + PD(+RBD) vs Control
    ]

    # ---- 3) Load data once ----
    print(f"Loading {args.csv} ...")
    X_full, y_full = load_data(args.csv, args.label_col)

    features = list(X_full.columns)
    if args.max_features:
        features = features[:args.max_features]

    # ---- 4) Run each split ----
    for split in splits:
        tag      = split["tag"]
        pos_val  = split["positive"]
        neg_val  = split["negative"]

        print(f"\n{'='*60}")
        print(f"  {BOLD}Split: {tag}{RESET}")
        print(f"{'='*60}")

        # ---- Build binary labels ----
        if "extra_positive" in split:
            # Control vs iRBD + PD(+RBD)
            y_bin = pd.Series(np.nan, index=y_full.index)
            y_bin[y_full == neg_val]                = 0
            y_bin[y_full == pos_val]                = 1
            y_bin[y_full == split["extra_positive"]] = 1
        else:
            y_bin = make_binary(y_full, pos_val, neg_val)

        # Drop rows without valid label
        valid     = y_bin.notna()
        n_dropped = (~valid).sum()
        if n_dropped > 0:
            print(f"  Dropping {n_dropped} rows without a valid label.")
        X     = X_full.loc[valid].reset_index(drop=True)
        y_bin = y_bin.loc[valid].astype(int).reset_index(drop=True)

        n_pos = int((y_bin == 1).sum())
        n_neg = int((y_bin == 0).sum())
        print(f"  Positive: {n_pos} samples | Negative: {n_neg} samples")
        print(f"  Features: {X.shape[1]}")

        # ---- Prepare output dirs per split ----
        out_dir  = Path(args.out_dir) / tag
        plot_dir = out_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(exist_ok=True)

        # ---- Generate QQ-plots ----
        print(f"\nGenerating QQ-plots for {len(features)} features ...")
        for i, col in enumerate(features, 1):
            vals_pos  = X.loc[y_bin == 1, col]
            vals_neg  = X.loc[y_bin == 0, col]
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in col)
            out_path  = plot_dir / f"{i:03d}_{safe_name}.pdf"
            qq_plot_feature(
                vals_pos, vals_neg, col, out_path,
                pos_label = pos_val if "extra_positive" not in split else f"{pos_val} + {split['extra_positive']}",
                neg_label = neg_val,
            )
            if i % 10 == 0:
                print(f"  {i}/{len(features)} ...")

        # ---- Shapiro-Wilk summary ----
        print("\nBuilding Shapiro-Wilk summary table ...")
        summary      = shapiro_summary(X[features], y_bin)
        summary_path = out_dir / "shapiro_summary.csv"
        summary.to_csv(summary_path, index=False)

        pivot = summary.pivot_table(
            index="feature", columns="class",
            values="normal_at_0.05", aggfunc="first",
        )
        pivot["both_normal"] = pivot.get("positive", False) & pivot.get("negative", False)
        overview_path        = out_dir / "normality_overview.csv"
        pivot.to_csv(overview_path)

        n_both_normal = int(pivot["both_normal"].sum())
        print(f"\n{GREEN}{BOLD}Done — {tag}{RESET}")
        print(f"  Plots:              {plot_dir}")
        print(f"  Shapiro-Wilk table: {summary_path}")
        print(f"  Normality overview: {overview_path}")
        print(f"  Both classes normal: {n_both_normal}/{len(features)}")
        print(f"\n{BOLD}Rule of thumb:{RESET}")
        print(f"  - Welch's t-test for features where both classes are normal.")
        print(f"  - Mann-Whitney U for the rest.")


if __name__ == "__main__":
    main()