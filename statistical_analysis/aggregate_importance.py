# Filename: aggregate_importance.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Aggregates MDI and permutation feature importance across all
#              sweep runs and computes stability metrics for each feature.
#
#              For each importance type the script computes median rank,
#              rank standard deviation, and the fraction of runs in which
#              each feature appeared in the top-K. Outputs CSV tables and
#              three plots per importance type: a stability bar chart, a
#              rank heatmap, and a top-K consistency bar chart.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"


# =====================================================================
# Importance CSV loading
# =====================================================================
def load_importance_csv(
        path: str | Path,
        kind: str,
        ) -> pd.DataFrame:
    """
    Load an MDI or permutation importance CSV and return a normalised table.

    Both formats are reduced to the same two-column schema [feature, importance]
    so downstream aggregation code is format-agnostic.

    Expected CSV formats (written by evaluate.py):
      mdi         : columns [feature, importance]
      permutation : columns [feature, mean, std]  — 'mean' is used as importance

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    kind : str
        Either 'mdi' or 'permutation'.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with columns [feature, importance].
    """
    df = pd.read_csv(path)
    if kind == "mdi":
        if "importance" not in df.columns:
            raise ValueError(f"MDI CSV is missing the 'importance' column: {path}")
        return df[["feature", "importance"]].copy()
    elif kind == "permutation":
        if "mean" not in df.columns:
            raise ValueError(f"Permutation CSV is missing the 'mean' column: {path}")
        return df.rename(columns={"mean": "importance"})[["feature", "importance"]].copy()
    else:
        raise ValueError(f"Unknown kind '{kind}' — expected 'mdi' or 'permutation'.")
 
 
# =====================================================================
# Aggregation
# =====================================================================

def aggregate(
        sweep_df: pd.DataFrame,
        kind:     str,
        top_k:    int = 20,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate feature importance rankings across all runs in sweep_df.

    For each run, the importance CSV is loaded, features are ranked by
    importance (1 = highest), and the rank is recorded. After processing
    all runs, per-feature statistics (median rank, std, times in top-K)
    are computed.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        sweep_summary.csv loaded as a DataFrame. Must contain a column
        named 'mdi_csv' or 'permutation_csv' (written by the updated
        train.py) and the columns 'seed', 'test_size', 'mode', 'best_model'.
    kind : str
        Either 'mdi' or 'permutation'.
    top_k : int
        Threshold used to compute 'times_top_k' and 'frac_top_k'.
        **Default is 20**.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per feature with columns: feature, n_runs, median_rank,
        mean_rank, std_rank, min_rank, max_rank, mean_importance,
        std_importance, times_top_k, frac_top_k. Sorted by median_rank.
    rank_matrix : pd.DataFrame
        Wide matrix [feature × run_tag] of per-run ranks, used for the
        heatmap. Empty DataFrame if no runs were loaded.
    """
    path_col = "mdi_csv" if kind == "mdi" else "permutation_csv"
 
    if path_col not in sweep_df.columns:
        raise ValueError(
            f"Kolonnen '{path_col}' findes ikke i sweep-summary. "
            f"Kør pipelinen igen med den opdaterede train.py."
        )
 
    rank_records = []      # liste af (feature, run_tag, rank, importance)
    n_runs = 0
 
    for _, row in sweep_df.iterrows():
        path = row.get(path_col, "")
        if not isinstance(path, str) or not path or not Path(path).exists():
            continue
 
        try:
            imp_df = load_importance_csv(path, kind)
        except Exception as e:
            print(f"  Advarsel: kunne ikke læse {path}: {e}")
            continue
 
        # Rangering: 1 = højest importance
        imp_df["rank"] = imp_df["importance"].rank(ascending=False, method="min")
        run_tag = f"seed{row['seed']}_split{str(row['test_size']).replace('.','')}_{row['mode']}_{row['best_model'].replace(' ','_')}"
 
        for _, r in imp_df.iterrows():
            rank_records.append({
                "feature":    r["feature"],
                "run_tag":    run_tag,
                "rank":       r["rank"],
                "importance": r["importance"],
            })
        n_runs += 1
 
    if not rank_records:
        return pd.DataFrame(), pd.DataFrame()
 
    print(f"  Indlæste {n_runs} runs med {kind}-importance.")
 
    long_df = pd.DataFrame(rank_records)
 
    # Aggregér pr. feature
    summary = long_df.groupby("feature").agg(
        n_runs          = ("rank", "count"),
        median_rank     = ("rank", "median"),
        mean_rank       = ("rank", "mean"),
        std_rank        = ("rank", "std"),
        min_rank        = ("rank", "min"),
        max_rank        = ("rank", "max"),
        mean_importance = ("importance", "mean"),
        std_importance  = ("importance", "std"),
    ).reset_index()
 
    # Hvor ofte var feature i top-K?
    in_top_k = long_df.groupby("feature")["rank"].apply(lambda s: (s <= top_k).sum())
    summary["times_top_k"] = summary["feature"].map(in_top_k).fillna(0).astype(int)
    summary["frac_top_k"] = summary["times_top_k"] / n_runs
 
    summary = summary.sort_values("median_rank").reset_index(drop=True)
 
    # Wide matrix til heatmap
    rank_matrix = long_df.pivot(index="feature", columns="run_tag", values="rank")
 
    return summary, rank_matrix
 
 
# =====================================================================
# Plots
# =====================================================================

def plot_stability_bars(
        summary:  pd.DataFrame,
        out_path: Path,
        kind:     str,
        top_k:    int = 20,
        ) -> None:
    """
    Save a horizontal bar chart of the top-K features by median rank.

    Error bars show the standard deviation of the rank across runs —
    a short bar indicates the feature consistently ranks highly.
    The x-axis is inverted so rank 1 (most important) appears on the right.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of `aggregate` — must contain 'feature', 'median_rank', 'std_rank'.
    out_path : Path
        Where to save the pdf.
    kind : str
        Importance type label used in the plot title ('mdi' or 'permutation').
    top_k : int
        Number of top features to show. **Default is 20**.
    """
    top = summary.head(top_k).iloc[::-1]
    if top.empty:
        return
 
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(top))))
    ax.barh(top["feature"], top["median_rank"],
            xerr=top["std_rank"].fillna(0),
            color="tab:blue", alpha=0.8,
            error_kw=dict(elinewidth=0.8, capsize=3))
    ax.set_xlabel("Median rank på tværs af runs  (lavere = vigtigere)")
    ax.set_title(f"Top {top_k} features — {kind} importance stabilitet")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_xaxis()  # 1 i højre side så vigtigste vises længst til højre
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_rank_heatmap(
        rank_matrix: pd.DataFrame,
        summary:     pd.DataFrame,
        out_path:    Path,
        kind:        str,
        top_k:       int = 20,
        ) -> None:
    """
    Save a heatmap of per-run ranks for the top-K features.

    Each cell shows the rank of a feature in one specific run. Low rank
    values (dark cells with viridis_r) indicate high importance. Features
    with stable rankings will show uniform colours across columns.

    Parameters
    ----------
    rank_matrix : pd.DataFrame
        Wide matrix [feature × run_tag] from `aggregate`.
    summary : pd.DataFrame
        Output of `aggregate` — used to select the top-K features by median rank.
    out_path : Path
        Where to save the pdf.
    kind : str
        Importance type label used in the plot title.
    top_k : int
        Number of top features to show. **Default is 20**.
    """
    if rank_matrix.empty:
        return
    top_features = summary.head(top_k)["feature"].tolist()
    sub = rank_matrix.loc[top_features]
 
    # Begræns range så top-features visualiseres tydeligt
    vmax = min(top_k * 2, int(np.nanmax(sub.values)))
 
    fig, ax = plt.subplots(figsize=(max(8, sub.shape[1] * 0.6), max(5, 0.35 * len(sub))))
    sns.heatmap(sub, annot=True, fmt=".0f", cmap="viridis_r",
                vmin=1, vmax=vmax, linewidths=0.3,
                cbar_kws={"label": "Rank (1 = højest)"},
                ax=ax)
    ax.set_title(f"Top {top_k} features — rangering pr. run ({kind})")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_frac_top_k(
        summary:  pd.DataFrame,
        out_path: Path,
        kind:     str,
        top_k:    int = 20,
        ) -> None:
    """
    Save a bar chart showing the fraction of runs each feature was in the top-K.

    A value of 1.0 means the feature appeared in the top-K in every single run.
    Only features with frac_top_k > 0 are shown (up to 30 features).

    Parameters
    ----------
    summary : pd.DataFrame
        Output of `aggregate` — must contain 'feature' and 'frac_top_k'.
    out_path : Path
        Where to save the pdf.
    kind : str
        Importance type label used in the plot title.
    top_k : int
        The K threshold that was used when computing frac_top_k. **Default is 20**.
    """
    top = summary[summary["frac_top_k"] > 0].head(30).iloc[::-1]
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top["frac_top_k"], color="tab:green", alpha=0.8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(f"Andel af runs hvor feature var i top-{top_k}")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Konsistens af top-{top_k} placering ({kind})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
# =====================================================================
# Main entry point
# =====================================================================

def main() -> None:
    """
    Aggregate feature importance rankings across all sweep runs.

    Workflow:
      1. Load sweep_summary.csv and optionally filter by mode or model.
      2. For each importance type (MDI, permutation), load all available
         CSV files and compute per-feature stability metrics.
      3. Save summary CSVs, rank matrices, and three plots per type.
    """
    # ---- 1) Parse CLI arguments ----
    parser = argparse.ArgumentParser(
        description="Aggregate feature importance rankings across sweep runs."
    )
    parser.add_argument("--sweep-summary", required=True,
                        help="Path to sweep_summary.csv from train.py --sweep.")
    parser.add_argument("--out-dir", default="aggregated_importance",
                        help="Directory for output files and plots.")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K threshold for stability metrics and plots. Default 20.")
    parser.add_argument("--mode", default=None, choices=[None, "binary", "multiclass"],
                        help="Filter to a single classification mode.")
    parser.add_argument("--model", default=None,
                        help="Filter to a single model name (e.g., 'Random Forest').")
    args = parser.parse_args()

    # ---- 2) Load and filter sweep summary ----
    sweep_path = Path(args.sweep_summary)
    if not sweep_path.exists():
        raise FileNotFoundError(f"File not found: {sweep_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {sweep_path} ...")
    sweep_df = pd.read_csv(sweep_path)

    # Remove failed runs
    if "status" in sweep_df.columns:
        before   = len(sweep_df)
        sweep_df = sweep_df[sweep_df["status"] == "ok"].reset_index(drop=True)
        if before > len(sweep_df):
            print(f"  Filtered out {before - len(sweep_df)} failed runs.")

    if args.mode:
        sweep_df = sweep_df[sweep_df["mode"] == args.mode].reset_index(drop=True)
        print(f"  Mode filter '{args.mode}': {len(sweep_df)} runs remaining.")
    if args.model:
        sweep_df = sweep_df[sweep_df["best_model"] == args.model].reset_index(drop=True)
        print(f"  Model filter '{args.model}': {len(sweep_df)} runs remaining.")

    if sweep_df.empty:
        print("No runs to aggregate.")
        return

    print(f"  Aggregating over {len(sweep_df)} runs.")
    print(f"  Modes:  {sorted(sweep_df['mode'].unique().tolist())}")
    print(f"  Models: {sorted(sweep_df['best_model'].unique().tolist())}")
    print(f"  Seeds:  {sorted(sweep_df['seed'].unique().tolist())}")

    # ---- 3) Aggregate and plot for each importance type ----
    for kind in ("mdi", "permutation"):
        print(f"\n=== {kind.upper()} ===")
        summary, rank_matrix = aggregate(sweep_df, kind, top_k=args.top_k)

        if summary.empty:
            print(f"  No {kind} CSV files found. "
                  f"(MDI is unavailable for non-tree models.)")
            continue

        # Save tables
        summary_csv = out_dir / f"stable_ranking_{kind}.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"  Stable ranking: {summary_csv}")

        if not rank_matrix.empty:
            matrix_csv = out_dir / f"rank_matrix_{kind}.csv"
            rank_matrix.to_csv(matrix_csv)
            print(f"  Rank matrix:    {matrix_csv}")

        # Save plots
        plot_stability_bars(summary, out_dir / f"top_features_stability_{kind}.pdf",
                            kind, args.top_k)
        plot_rank_heatmap(rank_matrix, summary,
                          out_dir / f"heatmap_top_features_{kind}.pdf",
                          kind, args.top_k)
        plot_frac_top_k(summary, out_dir / f"frac_top_k_{kind}.pdf",
                        kind, args.top_k)

        print(f"\n  Top-{args.top_k} most stable features ({kind}):")
        cols = ["feature", "median_rank", "std_rank", "times_top_k", "n_runs", "mean_importance"]
        print(summary[cols].head(args.top_k).to_string(index=False))

    print(f"\n{GREEN}{BOLD}Done. All outputs in: {out_dir}{RESET}")
    print(f"\nInterpretation guide for the report:")
    print(f"  - Low 'median_rank' + low 'std_rank'  →  feature is consistently important.")
    print(f"  - 'frac_top_k = 1.0'                  →  feature was in top-{args.top_k} in every run.")
    print(f"  - The heatmap shows visually whether a feature is stable or jumps around.")
    print(f"  - Agreement between MDI and permutation rankings = robust result.")


if __name__ == "__main__":
    main()