# Filename: compare_effect_sizes.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Compares univariate effect sizes (Cliff's delta and Cohen's d)
#              across multiple group comparisons (e.g. iRBD vs Control,
#              PD(+RBD) vs Control, PD(-RBD) vs Control).
#
#              Produces a grouped bar chart showing the effect size of each
#              feature across all comparisons, making it easy to see whether
#              a feature is more discriminative for one disease group than another.

# Usage:
#   python -m statistical_analysis.compare_effect_sizes --stats-dir reports/stats --out-dir reports/stats/comparison

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"


# =====================================================================
# Data loading
# =====================================================================

def load_stats(csv_path: str | Path, label: str, metric: str) -> pd.DataFrame:
    """
    Load a univariate_stats.csv and extract the feature name and chosen metric.

    Parameters
    ----------
    csv_path : str | Path
        Path to univariate_stats.csv produced by univariate_stats.py.
    label : str
        Short label for this comparison (e.g. 'iRBD vs HC').
        Used as a column name in the merged output.
    metric : str
        Which effect size column to extract.
        Typically 'abs_cliffs_delta' or 'abs_cohens_d'.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame: [feature, <label>].
    """
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in {csv_path}. "
                         f"Available: {list(df.columns)}")
    return df[["feature", metric]].rename(columns={metric: label})

# ===================================================================
# Helper plot function
# ===================================================================
def plot_top_per_group(
        merged:   pd.DataFrame,
        label:    str,
        metric:   str,
        out_path: Path,
        top_n:    int = 20,
        ) -> None:
    """
    Save a horizontal bar chart of the top-N features for one specific group.

    Unlike plot_grouped_bars (which ranks features by max effect across all
    groups), this ranks features purely by the chosen group's effect size,
    making it easy to see which features matter most for that specific comparison.

    Parameters
    ----------
    merged : pd.DataFrame
        Wide DataFrame with 'feature' column and one column per comparison label.
    label : str
        Which comparison column to rank by (e.g. 'iRBD vs HC').
    metric : str
        Name of the effect size metric (used in axis label and title).
    out_path : Path
        Where to save the pdf.
    top_n : int
        Number of top features to show. **Default is 20**.
    """
    df = (merged.dropna(subset=[label])
                .sort_values(label, ascending=False)
                .head(top_n)
                .sort_values(label, ascending=True)
                .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(df))))
    ax.barh(df["feature"], df[label], color="tab:blue", alpha=0.85)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Top {top_n} features — {label}\n({metric})", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =====================================================================
# Plotting
# =====================================================================

def plot_grouped_bars(
        merged:   pd.DataFrame,
        labels:   list[str],
        metric:   str,
        out_path: Path,
        top_n:    int = 25,
        ) -> None:
    """
    Save a grouped horizontal bar chart comparing effect sizes across groups.

    Features are ranked by their maximum effect size across all comparisons
    and only the top-N are shown. Each group gets its own colour.

    Parameters
    ----------
    merged : pd.DataFrame
        Wide DataFrame with 'feature' column and one column per comparison label.
    labels : list[str]
        Ordered list of comparison labels — must match column names in merged.
    metric : str
        Name of the effect size metric (used in axis label and title).
    out_path : Path
        Where to save the pdf.
    top_n : int
        Number of top features to show. **Default is 25**.
    """
    # ---- 1) Select top-N features by max effect size across all groups ----
    merged = merged.copy()
    merged["max_effect"] = merged[labels].max(axis=1)
    top = (merged.sort_values("max_effect", ascending=False)
                 .head(top_n)
                 .sort_values("max_effect", ascending=True)  # ascending for barh
                 .reset_index(drop=True))

    features  = top["feature"].tolist()
    n_groups  = len(labels)
    n_features = len(features)

    # ---- 2) Set up bar positions ----
    bar_height = 0.8 / n_groups
    y_base     = np.arange(n_features)
    colors     = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * n_features)))

    for i, (label, color) in enumerate(zip(labels, colors)):
        offsets = y_base + (i - (n_groups - 1) / 2) * bar_height
        values  = top[label].fillna(0).values
        ax.barh(offsets, values, height=bar_height * 0.9,
                label=label, color=color, alpha=0.85)

    # ---- 3) Formatting ----
    ax.set_yticks(y_base)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Effect size comparison across groups\n({metric})", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_heatmap(
        merged:   pd.DataFrame,
        labels:   list[str],
        metric:   str,
        out_path: Path,
        top_n:    int = 30,
        ) -> None:
    """
    Save a heatmap of effect sizes — features as rows, comparisons as columns.

    This makes it easy to spot features that are strongly discriminative for
    one group but not another, or features that show a gradient across the
    disease spectrum.

    Parameters
    ----------
    merged : pd.DataFrame
        Wide DataFrame with 'feature' column and one column per comparison label.
    labels : list[str]
        Ordered list of comparison labels.
    metric : str
        Name of the effect size metric (used in the title).
    out_path : Path
        Where to save the pdf.
    top_n : int
        Number of top features to show. **Default is 30**.
    """
    import seaborn as sns

    merged = merged.copy()
    merged["max_effect"] = merged[labels].max(axis=1)
    top = (merged.sort_values("max_effect", ascending=False)
                 .head(top_n)
                 .set_index("feature")[labels])

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 2), max(6, 0.35 * top_n)))
    sns.heatmap(top, annot=True, fmt=".2f", cmap="Blues",
                vmin=0, vmax=1, linewidths=0.3,
                cbar_kws={"label": metric.replace("_", " ").title()},
                ax=ax)
    ax.set_title(f"Effect size heatmap — top {top_n} features\n({metric})", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =====================================================================
# Main entry point
# =====================================================================

def main() -> None:
    """
    Compare effect sizes across multiple univariate_stats.csv files.

    Workflow:
      1. Auto-discover univariate_stats.csv files under --stats-dir, or
         accept manual --csvs and --labels.
      2. Merge all tables on feature name.
      3. Save a grouped bar chart, heatmap, and per-group bar charts.
      4. Save a merged CSV with all effect sizes side by side.
    """
    parser = argparse.ArgumentParser(
        description="Compare effect sizes across group comparisons. "
                    "Pass --stats-dir to auto-discover, or --csvs and --labels manually."
    )
    parser.add_argument("--stats-dir", type=str, default=None,
                        help="Root directory containing per-split subdirs with "
                             "univariate_stats.csv files. Example: reports/stats")
    parser.add_argument("--csvs", nargs="+", default=None,
                        help="Manual paths to univariate_stats.csv files.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Short label for each comparison (same order as --csvs).")
    parser.add_argument("--metric", default="abs_cliffs_delta",
                        choices=["abs_cliffs_delta", "abs_cohens_d", "auc"],
                        help="Effect size metric. Default: abs_cliffs_delta.")
    parser.add_argument("--top-n", type=int, default=25,
                        help="Number of top features to show. Default: 25.")
    parser.add_argument("--out-dir", default="reports/stats/comparison",
                        help="Directory for output files.")
    args = parser.parse_args()

    # ---- Resolve CSV paths and labels ----
    if args.stats_dir is not None:
        stats_dir = Path(args.stats_dir)
        found     = sorted(stats_dir.rglob("univariate_stats.csv"))
        if not found:
            raise FileNotFoundError(f"No univariate_stats.csv found under {stats_dir}")
        csv_paths = [str(p) for p in found]
        labels    = [p.parent.name for p in found]
        print(f"Auto-discovered {len(csv_paths)} comparison(s):")
        for lbl, path in zip(labels, csv_paths):
            print(f"  {lbl}  ←  {path}")
    elif args.csvs is not None and args.labels is not None:
        if len(args.csvs) != len(args.labels):
            raise ValueError("--csvs and --labels must have the same number of entries.")
        csv_paths = args.csvs
        labels    = args.labels
    else:
        raise ValueError("Provide either --stats-dir or both --csvs and --labels.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load and merge ----
    print(f"\nLoading {len(csv_paths)} comparison(s) ...")
    dfs = []
    for csv_path, label in zip(csv_paths, labels):
        print(f"  {label}  ←  {csv_path}")
        dfs.append(load_stats(csv_path, label, args.metric))

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="feature", how="outer")

    # ---- Save merged CSV ----
    merged_csv = out_dir / "effect_size_comparison.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"\n  Merged table → {merged_csv}")

    # ---- Print top features ----
    merged["max_effect"] = merged[labels].max(axis=1)
    top = merged.sort_values("max_effect", ascending=False).head(args.top_n)
    print(f"\n{BOLD}Top {args.top_n} features by max effect size ({args.metric}):{RESET}")
    print(top[["feature"] + labels].to_string(index=False))

    # ---- Save plots ----
    print(f"\nGenerating plots ...")
    plot_grouped_bars(merged, labels, args.metric,
                      out_dir / "grouped_bars.pdf", top_n=args.top_n)
    plot_heatmap(merged, labels, args.metric,
                 out_dir / "heatmap.pdf", top_n=args.top_n)

    for label in labels:
        safe = label.lower().replace(" ", "_")
        plot_top_per_group(merged, label, args.metric, out_dir / f"top20_{safe}.pdf", top_n=20)

    print(f"\n{GREEN}{BOLD}Done. All outputs in: {out_dir}{RESET}")
    print(f"\nInterpretation guide:")
    print(f"  - High effect in iRBD but low in PD → specific to early-stage RBD.")
    print(f"  - Gradient (low iRBD → high PD) → tracks disease progression.")
    print(f"  - Agreement across all groups → general RBD/PD marker.")
    print(f"  - High in PD only → may reflect neurodegeneration rather than RBD.")


if __name__ == "__main__":
    main()