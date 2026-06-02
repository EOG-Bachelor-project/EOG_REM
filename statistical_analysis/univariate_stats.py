# Filename: univariate_stats.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Performs univariate statistical testing for all features and
#              compares the resulting rankings with machine learning feature
#              importance metrics (MDI and permutation importance).
#
#              The script automatically selects between Welch's t-test and
#              Mann-Whitney U based on Shapiro-Wilk normality testing,
#              computes effect sizes and AUC values, applies FDR correction,
#              and generates summary plots and CSV tables.

# Usage:
#   python -m statistics.univariate_stats --csv features_csv/features.csv --label-col group --out-dir reports/stats

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
from sklearn.metrics import roc_auc_score

# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"

# =====================================================================
# Effect size helpers
# =====================================================================

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d using the pooled standard deviation.

    Parameters
    ----------
    x : np.ndarray
        Values for the positive class.
    y : np.ndarray
        Values for the negative class.

    Returns
    -------
    float
        Cohen's d. Returns NaN if either group has fewer than 2 samples
        or if the pooled standard deviation is zero.
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_sd


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta via the rank-biserial correlation from Mann-Whitney U.

    Values range from -1 to 1. Interpretation thresholds (by absolute value):
    < 0.147 negligible, < 0.33 small, < 0.474 medium, >= 0.474 large.

    Parameters
    ----------
    x : np.ndarray
        Values for the positive class.
    y : np.ndarray
        Values for the negative class.

    Returns
    -------
    float
        Cliff's delta. Returns NaN if either group is empty or the test fails.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    try:
        U, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        return np.nan
    return 2.0 * U / (nx * ny) - 1.0


def feature_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the univariate AUC for a single feature used as a raw classifier.

    Returns max(AUC, 1 - AUC) so that direction does not matter — a value
    of 1.0 means perfect separation, 0.5 means no separation.

    Parameters
    ----------
    values : np.ndarray
        Feature values.
    labels : np.ndarray
        Binary labels (0/1) aligned with `values`.

    Returns
    -------
    float
        AUC in the range [0.5, 1.0]. Returns NaN on failure.
    """
    try:
        auc = roc_auc_score(labels, values)
        return max(auc, 1 - auc)
    except Exception:
        return np.nan


def interpret_d(d: float) -> str:
    """
    Return a verbal interpretation of Cohen's d magnitude.

    Thresholds: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.

    Parameters
    ----------
    d : float
        Cohen's d value.

    Returns
    -------
    str
        One of 'negligible', 'small', 'medium', 'large', or 'n/a'.
    """
    a = abs(d)
    if np.isnan(a): return "n/a"
    if a < 0.2:     return "negligible"
    if a < 0.5:     return "small"
    if a < 0.8:     return "medium"
    return "large"


def interpret_cliffs(delta: float) -> str:
    """
    Return a verbal interpretation of Cliff's delta magnitude.

    Thresholds: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, >= 0.474 large.

    Parameters
    ----------
    delta : float
        Cliff's delta value.

    Returns
    -------
    str
        One of 'negligible', 'small', 'medium', 'large', or 'n/a'.
    """
    a = abs(delta)
    if np.isnan(a): return "n/a"
    if a < 0.147:   return "negligible"
    if a < 0.33:    return "small"
    if a < 0.474:   return "medium"
    return "large"
 
 
# =====================================================================
# Multiple testing correction
# =====================================================================

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to an array of p-values.

    NaN entries are ignored and returned as NaN. The corrected values
    (q-values) are clipped to [0, 1].

    Parameters
    ----------
    pvals : np.ndarray
        Raw p-values, possibly containing NaN.

    Returns
    -------
    np.ndarray
        BH-corrected q-values of the same length as `pvals`.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    valid = ~np.isnan(pvals)
    q = np.full(n, np.nan)
    if valid.sum() == 0:
        return q
    p_valid = pvals[valid]
    order = np.argsort(p_valid)
    ranks = np.arange(1, len(p_valid) + 1)
    q_sorted = p_valid[order] * len(p_valid) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q_valid = np.empty_like(p_valid)
    q_valid[order] = q_sorted
    q[valid] = q_valid
    return q
 
 
# =====================================================================
# Data loading
# =====================================================================

# Same exclude convention as prepare_data.py — these columns are never features.
EXCLUDE_NON_FEATURE = {
    "subject_id", "DCSM_ID",
    "Control", "PD(-RBD)", "PD(+RBD)", "iRBD", "PLM",
    "group",
}


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
    drop_cols = [c for c in df.columns if c in EXCLUDE_NON_FEATURE]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    return X, y


def parse_class(val: str):
    """
    Parse a CLI class argument into the most appropriate Python type.

    Tries int first, then float, otherwise keeps it as a string. This lets
    the user pass labels like 'iRBD', '1', or '1.0' without ceremony.

    Parameters
    ----------
    val : str
        Raw string value from argparse.

    Returns
    -------
    int | float | str
        The parsed value.
    """
    if val.isdigit():
        return int(val)
    try:
        return float(val)
    except ValueError:
        return val


def build_binary_labels(
        y:              pd.Series,
        positive_class,
        negative_class = None,
        ) -> pd.Series:
    """
    Convert a (possibly multi-class) label vector to a binary 0/1 Series.

    If `negative_class` is None, all rows that are not in `positive_class`
    become 0 (one-vs-rest). Otherwise only rows matching `positive_class` or
    `negative_class` are kept — everything else gets NaN and should be
    dropped by the caller before further analysis.

    Parameters
    ----------
    y : pd.Series
        Label vector.
    positive_class : any
        Value in `y` that maps to 1 (e.g., 'iRBD').
    negative_class : any, optional
        Value in `y` that maps to 0 (e.g., 'Control').
        If None, all non-positive rows become 0. **Default is None**.

    Returns
    -------
    pd.Series
        Binary labels (0/1) with NaN for rows that don't fit the split.
    """
    if negative_class is None:
        return (y == positive_class).astype(int)
    out = pd.Series(np.nan, index=y.index)
    out[y == negative_class] = 0
    out[y == positive_class] = 1
    return out
 
 
# =====================================================================
# Statistical testing
# =====================================================================

def choose_test(
        values_pos: np.ndarray,
        values_neg: np.ndarray,
        mode:       str = "auto",
        ) -> str:
    """
    Decide between Welch's t-test ('t') and Mann-Whitney U ('u') for one feature.

    In 'auto' mode, Shapiro-Wilk is run on both groups. If either group
    rejects normality (p < 0.05), Mann-Whitney U is chosen. If the group
    is too small (< 3 samples) or Shapiro-Wilk fails, Mann-Whitney U is
    used as the safe default.

    Parameters
    ----------
    values_pos : np.ndarray
        Feature values for the positive class.
    values_neg : np.ndarray
        Feature values for the negative class.
    mode : str
        'auto' to decide via Shapiro-Wilk, 't' to always use Welch's t-test,
        'u' to always use Mann-Whitney U. **Default is 'auto'**.

    Returns
    -------
    str
        't' for Welch's t-test or 'u' for Mann-Whitney U.
    """
    if mode in ("t", "u"):
        return mode
    rng = np.random.default_rng(0)
    for vals in (values_pos, values_neg):
        v = np.asarray(vals)
        v = v[~np.isnan(v)]
        if len(v) < 3:
            return "u"
        sample = v if len(v) <= 5000 else rng.choice(v, 5000, replace=False)
        try:
            _, p = stats.shapiro(sample)
            if p < 0.05:
                return "u"
        except Exception:
            return "u"
    return "t"


def run_univariate(
        X:         pd.DataFrame,
        y_bin:     pd.Series,
        test_mode: str,
        ) -> pd.DataFrame:
    """
    Run a univariate test and compute effect sizes for every feature in X.

    For each feature:
      1. Choose between Welch's t-test and Mann-Whitney U via `choose_test`.
      2. Run the selected test and record the test statistic and p-value.
      3. Compute Cohen's d, Cliff's delta, and univariate AUC.
      4. Apply Benjamini-Hochberg FDR correction across all p-values.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows aligned with `y_bin`).
    y_bin : pd.Series
        Binary label vector (0/1).
    test_mode : str
        Passed directly to `choose_test` — 'auto', 't', or 'u'.

    Returns
    -------
    pd.DataFrame
        One row per feature with columns: feature, test, stat, p, cohens_d,
        abs_cohens_d, cohens_d_interpretation, cliffs_delta, abs_cliffs_delta,
        cliffs_interpretation, auc, n_pos, n_neg, p_bh_fdr, significant_fdr_0.05.
    """
    rows = []
    for col in X.columns:
        vals_pos = X.loc[y_bin == 1, col].dropna().values
        vals_neg = X.loc[y_bin == 0, col].dropna().values

        # ---- Not enough samples — record empty row ----
        if len(vals_pos) < 2 or len(vals_neg) < 2:
            rows.append({"feature": col, "test": "n/a", "stat": np.nan, "p": np.nan,
                         "cohens_d": np.nan, "cliffs_delta": np.nan, "auc": np.nan,
                         "n_pos": len(vals_pos), "n_neg": len(vals_neg)})
            continue

        # ---- Select and run the appropriate test ----
        chosen = choose_test(vals_pos, vals_neg, test_mode)
        if chosen == "t":
            try:
                stat, p = stats.ttest_ind(vals_pos, vals_neg, equal_var=False, nan_policy="omit")
            except Exception:
                stat, p = np.nan, np.nan
            test_name = "welch_t"
        else:
            try:
                stat, p = stats.mannwhitneyu(vals_pos, vals_neg, alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            test_name = "mann_whitney_u"

        # ---- Compute effect sizes ----
        d     = cohens_d(vals_pos, vals_neg)
        delta = cliffs_delta(vals_pos, vals_neg)
        mask  = X[col].notna()
        auc   = feature_auc(X.loc[mask, col].values, y_bin.loc[mask].values)

        rows.append({
            "feature":              col,
            "test":                 test_name,
            "stat":                 stat,
            "p":                    p,
            "cohens_d":             d,
            "abs_cohens_d":         abs(d) if not np.isnan(d) else np.nan,
            "cohens_d_interpretation": interpret_d(d),
            "cliffs_delta":         delta,
            "abs_cliffs_delta":     abs(delta) if not np.isnan(delta) else np.nan,
            "cliffs_interpretation":   interpret_cliffs(delta),
            "auc":                  auc,
            "n_pos":                len(vals_pos),
            "n_neg":                len(vals_neg),
        })

    # ---- Apply FDR correction across all features ----
    df = pd.DataFrame(rows)
    df["p_bh_fdr"]            = benjamini_hochberg(df["p"].values)
    df["significant_fdr_0.05"] = df["p_bh_fdr"] < 0.05
    return df
 
 
# =====================================================================
# Merge with ML feature importance
# =====================================================================

def merge_importance(
        stats_df: pd.DataFrame,
        mdi_csv:  str | None,
        perm_csv: str | None,
        ) -> pd.DataFrame:
    """
    Join univariate statistics with MDI and permutation importance from the pipeline.

    Reads the CSV files written by evaluate.py and merges them onto the
    stats table by feature name. Also computes rank columns (1 = most
    important / most significant) for every metric, enabling rank
    correlations and scatter plots.

    Expected CSV formats (output of evaluate.py):
      mdi_csv  : columns [feature, importance]
      perm_csv : columns [feature, mean, std]

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of `run_univariate`.
    mdi_csv : str | None
        Path to the MDI importance CSV, or None to skip.
    perm_csv : str | None
        Path to the permutation importance CSV, or None to skip.

    Returns
    -------
    pd.DataFrame
        Merged table with additional columns for importance scores and ranks.
    """
    merged = stats_df.copy()

    # ---- Load and merge MDI ----
    if mdi_csv is not None:
        mdi = pd.read_csv(mdi_csv)
        if "importance" not in mdi.columns:
            raise ValueError(f"MDI CSV is missing the 'importance' column: {mdi_csv}")
        mdi = mdi.rename(columns={"importance": "mdi"})[["feature", "mdi"]]
        merged = merged.merge(mdi, on="feature", how="left")

    # ---- Load and merge permutation importance ----
    if perm_csv is not None:
        perm = pd.read_csv(perm_csv)
        if "mean" not in perm.columns:
            raise ValueError(f"Permutation CSV is missing the 'mean' column: {perm_csv}")
        perm = perm.rename(columns={"mean": "permutation", "std": "permutation_std"})
        keep = ["feature", "permutation"] + (
            ["permutation_std"] if "permutation_std" in perm.columns else []
        )
        merged = merged.merge(perm[keep], on="feature", how="left")

    # ---- Compute rank columns (1 = most important / most significant) ----
    if "mdi" in merged.columns:
        merged["rank_mdi"] = merged["mdi"].rank(ascending=False, method="min")
    if "permutation" in merged.columns:
        merged["rank_permutation"] = merged["permutation"].rank(ascending=False, method="min")
    merged["rank_abs_d"]     = merged["abs_cohens_d"].rank(ascending=False, method="min")
    merged["rank_abs_delta"] = merged["abs_cliffs_delta"].rank(ascending=False, method="min")
    merged["rank_auc"]       = merged["auc"].rank(ascending=False, method="min")
    merged["rank_p"]         = merged["p"].rank(ascending=True, method="min")
    return merged


def rank_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman rank correlations between all ranking columns.

    Parameters
    ----------
    merged : pd.DataFrame
        Output of `merge_importance` — must contain columns starting with 'rank_'.

    Returns
    -------
    pd.DataFrame
        Symmetric Spearman correlation matrix for all rank columns.
        Returns an empty DataFrame if fewer than two rank columns are present.
    """
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    if len(rank_cols) < 2:
        return pd.DataFrame()
    sub = merged[rank_cols].dropna()
    if sub.empty:
        return pd.DataFrame()
    return sub.corr(method="spearman")
 
 
# =====================================================================
# Plots
# =====================================================================

def plot_top_features(
        merged:  pd.DataFrame,
        metric:  str,
        title:   str,
        out_path: Path,
        top_n:   int = 20,
        ) -> None:
    """
    Save a horizontal bar chart of the top-N features ranked by a given metric.

    Bars are coloured blue for positive values and red for negative values.
    Features are sorted by absolute value of the metric.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged stats + importance table.
    metric : str
        Column name to rank and plot (e.g., 'abs_cohens_d', 'mdi').
    title : str
        Plot title.
    out_path : Path
        Where to save the pdf.
    top_n : int
        Number of top features to show. **Default is 20**.
    """
    if metric not in merged.columns:
        return
    df = merged.dropna(subset=[metric]).copy()
    if df.empty:
        return
    df = df.reindex(df[metric].abs().sort_values(ascending=False).index).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in df[metric]]
    ax.barh(df["feature"], df[metric], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rank_scatter(
        merged:    pd.DataFrame,
        x_col:     str,
        y_col:     str,
        out_path:  Path,
        label_top: int = 10,
        ) -> None:
    """
    Save a scatter plot of two rank columns with Spearman correlation in the title.

    The top features (lowest combined rank on both axes) are labelled by name.
    A diagonal reference line is drawn (perfect agreement between rankings).

    Parameters
    ----------
    merged : pd.DataFrame
        Merged stats + importance table with rank columns.
    x_col : str
        Rank column for the x-axis (e.g., 'rank_mdi').
    y_col : str
        Rank column for the y-axis (e.g., 'rank_abs_d').
    out_path : Path
        Where to save the pdf.
    label_top : int
        Number of top features to label. **Default is 10**.
    """
    if x_col not in merged.columns or y_col not in merged.columns:
        return
    df = merged.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return

    rho, p = stats.spearmanr(df[x_col], df[y_col])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=40)

    # Label the features with the lowest combined rank on both axes
    df["combined"] = df[x_col] + df[y_col]
    top = df.nsmallest(label_top, "combined")
    for _, row in top.iterrows():
        ax.annotate(row["feature"], (row[x_col], row[y_col]),
                    fontsize=8, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    lim = max(df[x_col].max(), df[y_col].max()) + 1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel(x_col + "  (1 = most important / most significant)")
    ax.set_ylabel(y_col + "  (1 = most important / most significant)")
    ax.set_title(f"{x_col}  vs  {y_col}\nSpearman ρ = {rho:.3f}  (p = {p:.3g})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_volcano(
        merged:   pd.DataFrame,
        out_path: Path,
        ) -> None:
    """
    Save a volcano plot of Cohen's d (x) vs -log10(p) (y).

    Points are coloured red if they pass the FDR threshold (q < 0.05) and
    grey otherwise. The top 10 most significant features are labelled.
    A horizontal dashed line marks the raw p = 0.05 threshold.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged stats table — must contain 'cohens_d', 'p', and 'p_bh_fdr'.
    out_path : Path
        Where to save the pdf.
    """
    df = merged.dropna(subset=["cohens_d", "p"]).copy()
    if df.empty:
        return
    df["neglog10p"] = -np.log10(df["p"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(8, 6))
    sig = df["p_bh_fdr"] < 0.05
    ax.scatter(df.loc[~sig, "cohens_d"], df.loc[~sig, "neglog10p"],
               alpha=0.5, s=30, color="gray", label="Not significant (FDR)")
    ax.scatter(df.loc[sig, "cohens_d"], df.loc[sig, "neglog10p"],
               alpha=0.8, s=40, color="tab:red", label="Significant (FDR < 0.05)")

    # Label the 10 most significant features
    top = df.reindex(df["neglog10p"].sort_values(ascending=False).index).head(10)
    for _, row in top.iterrows():
        ax.annotate(row["feature"], (row["cohens_d"], row["neglog10p"]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")

    ax.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Volcano plot: effect size vs significance")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
 
 
# =====================================================================
# Main entry point
# =====================================================================

def main() -> None:
    """
    Run univariate statistics for all binary splits in one command.

    Workflow:
      1. Load feature CSV once.
      2. For each binary split (Control vs iRBD, Control vs PD(+RBD), Control vs all):
           a. Build binary labels.
           b. Run per-feature hypothesis tests.
           c. Apply FDR correction.
           d. Merge with MDI/permutation importance (optional).
           e. Save CSV and plots to <out_dir>/<split>/.
    """
    parser = argparse.ArgumentParser(
        description="Univariate statistics for all binary splits in one command."
    )
    parser.add_argument("--csv",       required=True,  help="Path to the feature CSV.")
    parser.add_argument("--label-col", default="group", help="Label column name. Default: 'group'.")
    parser.add_argument("--mdi-csv",         default=None, help="Path to MDI importance CSV.")
    parser.add_argument("--permutation-csv", default=None, help="Path to permutation importance CSV.")
    parser.add_argument("--test", choices=["auto", "t", "u"], default="auto",
                        help="Test selection: auto, t (Welch), u (Mann-Whitney). Default: auto.")
    parser.add_argument("--out-dir", default="reports/stats",
                        help="Root output directory. Default: reports/stats.")
    args = parser.parse_args()

    # ---- Define all binary splits ----
    splits = [
        {"positive": "iRBD",     "negative": "Control",  "tag": "control_vs_irbd"},
        {"positive": "PD(+RBD)", "negative": "Control",  "tag": "control_vs_pdrbd"},
        {"positive": "iRBD",     "negative": "Control",  "tag": "control_vs_all",
         "extra_positive": "PD(+RBD)"},
    ]

    # ---- Load data once ----
    print(f"Loading {args.csv} ...")
    X_full, y_full = load_data(args.csv, args.label_col)

    # ---- Run each split ----
    for split in splits:
        tag     = split["tag"]
        pos_val = split["positive"]
        neg_val = split["negative"]

        print(f"\n{'='*60}")
        print(f"  {BOLD}Split: {tag}{RESET}")
        print(f"{'='*60}")

        # ---- Build binary labels ----
        if "extra_positive" in split:
            y_bin = pd.Series(np.nan, index=y_full.index)
            y_bin[y_full == neg_val]                 = 0
            y_bin[y_full == pos_val]                 = 1
            y_bin[y_full == split["extra_positive"]] = 1
        else:
            y_bin = build_binary_labels(y_full, pos_val, neg_val)

        # Drop rows without valid label
        valid     = y_bin.notna()
        n_dropped = (~valid).sum()
        if n_dropped > 0:
            print(f"  Dropping {n_dropped} rows without a valid label.")
        X     = X_full.loc[valid].reset_index(drop=True)
        y_bin = y_bin.loc[valid].astype(int).reset_index(drop=True)

        print(f"  Positive: {(y_bin == 1).sum()} | Negative: {(y_bin == 0).sum()} | Features: {X.shape[1]}")

        # ---- Run univariate tests ----
        print(f"\nRunning univariate tests (mode='{args.test}') ...")
        stats_df = run_univariate(X, y_bin, args.test)

        n_t = (stats_df["test"] == "welch_t").sum()
        n_u = (stats_df["test"] == "mann_whitney_u").sum()
        print(f"  Welch t-test: {n_t}  |  Mann-Whitney U: {n_u}")
        print(f"  Significant after FDR (q < 0.05): {stats_df['significant_fdr_0.05'].sum()}")

        # ---- Merge with ML importance ----
        merged = merge_importance(stats_df, args.mdi_csv, args.permutation_csv)

        # ---- Save outputs ----
        out_dir  = Path(args.out_dir) / tag
        plot_dir = out_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(exist_ok=True)

        main_csv = out_dir / "univariate_stats.csv"
        merged.to_csv(main_csv, index=False)
        print(f"\n  Main table: {main_csv}")

        rank_corr = rank_correlations(merged)
        if not rank_corr.empty:
            corr_csv = out_dir / "rank_correlations.csv"
            rank_corr.to_csv(corr_csv)
            print(f"  Rank correlations: {corr_csv}")

        # ---- Generate plots ----
        print(f"  Generating plots ...")
        plot_top_features(merged, "abs_cohens_d",    f"Top features — |Cohen's d| ({tag})",
                          plot_dir / "top_cohens_d.pdf")
        plot_top_features(merged, "abs_cliffs_delta", f"Top features — |Cliff's delta| ({tag})",
                          plot_dir / "top_cliffs_delta.pdf")
        plot_top_features(merged, "auc",              f"Top features — AUC ({tag})",
                          plot_dir / "top_auc.pdf")

        if "mdi" in merged.columns:
            plot_top_features(merged, "mdi", f"Top features — MDI ({tag})",
                              plot_dir / "top_mdi.pdf")
        if "permutation" in merged.columns:
            plot_top_features(merged, "permutation", f"Top features — Permutation ({tag})",
                              plot_dir / "top_permutation.pdf")

        plot_volcano(merged, plot_dir / "volcano.pdf")

        if "rank_mdi" in merged.columns:
            plot_rank_scatter(merged, "rank_mdi", "rank_abs_d",
                              plot_dir / "scatter_mdi_vs_cohens_d.pdf")
            plot_rank_scatter(merged, "rank_mdi", "rank_abs_delta",
                              plot_dir / "scatter_mdi_vs_cliffs.pdf")
            plot_rank_scatter(merged, "rank_mdi", "rank_auc",
                              plot_dir / "scatter_mdi_vs_auc.pdf")
        if "rank_permutation" in merged.columns:
            plot_rank_scatter(merged, "rank_permutation", "rank_abs_d",
                              plot_dir / "scatter_permutation_vs_cohens_d.pdf")
            plot_rank_scatter(merged, "rank_permutation", "rank_abs_delta",
                              plot_dir / "scatter_permutation_vs_cliffs.pdf")
            plot_rank_scatter(merged, "rank_permutation", "rank_auc",
                              plot_dir / "scatter_permutation_vs_auc.pdf")
        if "rank_mdi" in merged.columns and "rank_permutation" in merged.columns:
            plot_rank_scatter(merged, "rank_mdi", "rank_permutation",
                              plot_dir / "scatter_mdi_vs_permutation.pdf")

        print(f"  {GREEN}{BOLD}Done — {tag}{RESET}")

    print(f"\n{GREEN}{BOLD}All splits complete.{RESET}")
    print(f"Results saved under: {Path(args.out_dir)}")


if __name__ == "__main__":
    main()