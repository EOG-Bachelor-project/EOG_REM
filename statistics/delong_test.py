# Filename: delong_test.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Compares the AUC of two models evaluated on the same subjects
#              using DeLong's nonparametric test (DeLong et al., 1988).
#
#              Takes the out-of-fold predicted probability CSV files written
#              by train.py and tests whether the difference in AUC between
#              the full model and a baseline model is statistically significant.
#              Also computes 95% confidence intervals for each AUC and saves
#              a ROC curve comparison plot.
#
# Usage:
#   python delong_test.py \
#       --model-a  reports/evaluation/random_forest_oof_probabilities.csv \
#       --model-b  reports/evaluation/random_forest_baseline_oof_probabilities.csv \
#       --name-a  "Full model" \
#       --name-b  "Baseline model" \
#       --out-dir  reports/delong

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc as sklearn_auc

# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"


# =====================================================================
# Data loading
# =====================================================================

def load_oof(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an out-of-fold probability CSV written by train.py.

    The CSV is expected to have columns:
      - y_true          : true binary label (0/1)
      - prob_class_1    : predicted probability for the positive class
      - fold            : fold index (used for verification only)

    For multiclass models the positive class probability is taken from
    the column with the highest index (last class), which corresponds to
    the positive class in binary one-vs-rest setups. If your setup differs
    pass the correct column name via --prob-col.

    Parameters
    ----------
    csv_path : str | Path
        Path to the OOF probability CSV.

    Returns
    -------
    y_true : np.ndarray
        True binary labels (0/1).
    y_score : np.ndarray
        Predicted probabilities for the positive class.
    """
    df = pd.read_csv(csv_path)

    if "y_true" not in df.columns:
        raise ValueError(f"CSV must contain a 'y_true' column: {csv_path}")

    # Find the positive class probability column
    prob_cols = sorted([c for c in df.columns if c.startswith("prob_class_")])
    if not prob_cols:
        raise ValueError(
            f"No 'prob_class_*' columns found in {csv_path}. "
            f"Columns present: {list(df.columns)}"
        )

    # For binary classification prob_class_1 is the positive class
    pos_col = prob_cols[-1]
    print(f"  Using '{pos_col}' as positive class probability.")

    y_true  = df["y_true"].values.astype(int)
    y_score = df[pos_col].values.astype(float)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    n_folds = df["fold"].nunique() if "fold" in df.columns else "unknown"
    print(f"  Loaded {len(y_true)} samples "
          f"({n_pos} positive, {n_neg} negative, {n_folds} folds).")
    return y_true, y_score


# =====================================================================
# DeLong's method — structural components
# =====================================================================

def _structural_components(
        y_true:  np.ndarray,
        y_score: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the structural components V10 and V01 for DeLong variance estimation.

    V10[i] is the probability that positive sample i scores higher than a
    randomly chosen negative sample. V01[j] is the probability that negative
    sample j scores lower than a randomly chosen positive sample. Ties
    contribute 0.5.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    y_score : np.ndarray
        Predicted probability scores for the positive class.

    Returns
    -------
    V10 : np.ndarray
        One value per positive sample.
    V01 : np.ndarray
        One value per negative sample.
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]

    V10 = np.array([
        (np.sum(p > neg) + 0.5 * np.sum(p == neg)) / len(neg)
        for p in pos
    ])
    V01 = np.array([
        (np.sum(n < pos) + 0.5 * np.sum(n == pos)) / len(pos)
        for n in neg
    ])
    return V10, V01


def _auc_and_variance(
        y_true:  np.ndarray,
        y_score: np.ndarray,
        ) -> tuple[float, float]:
    """
    Compute the AUC and its DeLong variance for one model.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    y_score : np.ndarray
        Predicted probability scores.

    Returns
    -------
    auc_val : float
        AUC estimate.
    variance : float
        DeLong variance of the AUC estimate.
    """
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    V10, V01 = _structural_components(y_true, y_score)
    auc_val  = float(V10.mean())
    variance = (np.var(V10, ddof=1) / n_pos +
                np.var(V01, ddof=1) / n_neg)
    return auc_val, variance


def _covariance(
        y_true:    np.ndarray,
        y_score_a: np.ndarray,
        y_score_b: np.ndarray,
        ) -> float:
    """
    Compute the DeLong covariance between two AUC estimates.

    This accounts for the fact that both models were evaluated on the
    same test subjects, making the two AUC estimates correlated.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    y_score_a : np.ndarray
        Predicted probabilities from model A.
    y_score_b : np.ndarray
        Predicted probabilities from model B.

    Returns
    -------
    float
        Covariance between AUC_A and AUC_B.
    """
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    V10_a, V01_a = _structural_components(y_true, y_score_a)
    V10_b, V01_b = _structural_components(y_true, y_score_b)
    cov = (np.cov(V10_a, V10_b, ddof=1)[0, 1] / n_pos +
           np.cov(V01_a, V01_b, ddof=1)[0, 1] / n_neg)
    return float(cov)


# =====================================================================
# DeLong test
# =====================================================================

def delong_test(
        y_true:    np.ndarray,
        y_score_a: np.ndarray,
        y_score_b: np.ndarray,
        alpha:     float = 0.05,
        ) -> dict:
    """
    Compare two AUC values from the same test set using DeLong's test.

    The test statistic is

        z = (AUC_A - AUC_B) / sqrt(Var_A + Var_B - 2 * Cov_AB)

    where Var_A, Var_B are the DeLong variance estimates and Cov_AB is
    the DeLong covariance. Under the null hypothesis (AUC_A = AUC_B)
    z follows a standard normal distribution.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1). Must be the same for both models.
    y_score_a : np.ndarray
        Predicted probabilities from model A (positive class).
    y_score_b : np.ndarray
        Predicted probabilities from model B (positive class).
    alpha : float
        Significance level for the confidence intervals and significance
        flag. **Default is 0.05**.

    Returns
    -------
    dict with keys:
        auc_a       : float  — AUC of model A
        auc_b       : float  — AUC of model B
        auc_diff    : float  — AUC_A - AUC_B
        var_a       : float  — DeLong variance of AUC_A
        var_b       : float  — DeLong variance of AUC_B
        cov_ab      : float  — DeLong covariance
        z           : float  — z-statistic
        p_value     : float  — two-sided p-value
        ci_a        : tuple  — (1-alpha) CI for AUC_A
        ci_b        : tuple  — (1-alpha) CI for AUC_B
        significant : bool   — True if p_value < alpha
    """
    if len(y_true) != len(y_score_a) or len(y_true) != len(y_score_b):
        raise ValueError("y_true, y_score_a, and y_score_b must have the same length.")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("y_true must be binary (0/1).")

    # ---- Compute AUC, variance, covariance ----
    auc_a, var_a = _auc_and_variance(y_true, y_score_a)
    auc_b, var_b = _auc_and_variance(y_true, y_score_b)
    cov_ab       = _covariance(y_true, y_score_a, y_score_b)

    # ---- z-statistic for the difference ----
    se_diff = np.sqrt(max(var_a + var_b - 2 * cov_ab, 0))
    z       = (auc_a - auc_b) / se_diff if se_diff > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))

    # ---- Confidence intervals for individual AUCs ----
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_a   = (auc_a - z_crit * np.sqrt(var_a),
              auc_a + z_crit * np.sqrt(var_a))
    ci_b   = (auc_b - z_crit * np.sqrt(var_b),
              auc_b + z_crit * np.sqrt(var_b))

    return {
        "auc_a":       round(auc_a,           4),
        "auc_b":       round(auc_b,           4),
        "auc_diff":    round(auc_a - auc_b,   4),
        "var_a":       round(var_a,           6),
        "var_b":       round(var_b,           6),
        "cov_ab":      round(cov_ab,          6),
        "z":           round(z,               4),
        "p_value":     round(p_value,         6),
        "ci_a":        (round(ci_a[0], 4), round(ci_a[1], 4)),
        "ci_b":        (round(ci_b[0], 4), round(ci_b[1], 4)),
        "significant": p_value < alpha,
    }


# =====================================================================
# Results printing
# =====================================================================

def print_results(result: dict, name_a: str, name_b: str) -> None:
    """
    Print a formatted summary of the DeLong test results to the console.

    Parameters
    ----------
    result : dict
        Output of `delong_test`.
    name_a : str
        Display name for model A.
    name_b : str
        Display name for model B.
    """
    sig_str = f"{GREEN}significant{RESET}" if result["significant"] else "not significant"

    print(f"\n{BOLD}DeLong test results{RESET}")
    print(f"{'─' * 55}")
    print(f"  {name_a:<30s}  AUC = {result['auc_a']:.4f}  "
          f"95% CI [{result['ci_a'][0]:.4f}, {result['ci_a'][1]:.4f}]")
    print(f"  {name_b:<30s}  AUC = {result['auc_b']:.4f}  "
          f"95% CI [{result['ci_b'][0]:.4f}, {result['ci_b'][1]:.4f}]")
    print(f"{'─' * 55}")
    print(f"  ΔAUC (A - B)  : {result['auc_diff']:+.4f}")
    print(f"  z-statistic   : {result['z']:.4f}")
    print(f"  p-value       : {result['p_value']:.6f}")
    print(f"  Conclusion    : {sig_str}")
    print(f"{'─' * 55}")

    if result["significant"]:
        if result["auc_diff"] > 0:
            print(f"\n  {name_a} has a significantly higher AUC than {name_b}.")
        else:
            print(f"\n  {name_b} has a significantly higher AUC than {name_a}.")
    else:
        print(f"\n  No significant difference in AUC between the two models.")


# =====================================================================
# ROC plot
# =====================================================================

def plot_roc_comparison(
        y_true:    np.ndarray,
        y_score_a: np.ndarray,
        y_score_b: np.ndarray,
        name_a:    str,
        name_b:    str,
        result:    dict,
        out_path:  Path,
        ) -> None:
    """
    Save a ROC curve plot comparing the two models.

    The DeLong test result (z, p, ΔAUC) is annotated directly on the plot.
    95% confidence intervals are shown in the legend.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1).
    y_score_a : np.ndarray
        Predicted probabilities from model A.
    y_score_b : np.ndarray
        Predicted probabilities from model B.
    name_a : str
        Display name for model A.
    name_b : str
        Display name for model B.
    result : dict
        Output of `delong_test`.
    out_path : Path
        Where to save the PNG.
    """
    fpr_a, tpr_a, _ = roc_curve(y_true, y_score_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, y_score_b)

    fig, ax = plt.subplots(figsize=(7, 6))

    # ---- Plot ROC curves ----
    ax.plot(fpr_a, tpr_a, color="tab:blue", linewidth=2,
            label=(f"{name_a}\n"
                   f"AUC = {result['auc_a']:.3f}  "
                   f"[{result['ci_a'][0]:.3f}, {result['ci_a'][1]:.3f}]"))
    ax.plot(fpr_b, tpr_b, color="tab:orange", linewidth=2,
            label=(f"{name_b}\n"
                   f"AUC = {result['auc_b']:.3f}  "
                   f"[{result['ci_b'][0]:.3f}, {result['ci_b'][1]:.3f}]"))
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8,
            linestyle="--", label="Random")

    # ---- Annotate DeLong result ----
    sig_str = "significant" if result["significant"] else "not significant"
    ax.text(0.37, 0.08,
            f"DeLong test:  z = {result['z']:.3f},  "
            f"p = {result['p_value']:.4f}\n"
            f"ΔAUC = {result['auc_diff']:+.3f}  ({sig_str})",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(f"ROC curve comparison\n{name_a}  vs  {name_b}", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC plot → {out_path}")


# =====================================================================
# Main entry point
# =====================================================================

def main() -> None:
    """
    Run DeLong's test to compare AUC between a full model and a baseline.

    Workflow:
      1. Load out-of-fold probability CSVs for both models.
      2. Verify both files have the same subjects in the same order.
      3. Run DeLong's test and print the results.
      4. Save a ROC comparison plot and a results CSV.
    """
    # ---- 1) Parse CLI arguments ----
    parser = argparse.ArgumentParser(
        description="Compare AUC between two models using DeLong's test."
    )
    parser.add_argument("--model-a", required=True,
                        help="Path to OOF probability CSV for model A (full model).")
    parser.add_argument("--model-b", required=True,
                        help="Path to OOF probability CSV for model B (baseline).")
    parser.add_argument("--name-a", default="Full model",
                        help="Display name for model A. Default: 'Full model'.")
    parser.add_argument("--name-b", default="Baseline model",
                        help="Display name for model B. Default: 'Baseline model'.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for CI and significance flag. Default: 0.05.")
    parser.add_argument("--out-dir", default="reports/delong",
                        help="Directory for output files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2) Load OOF probabilities ----
    print(f"\nLoading model A: {args.model_a}")
    y_true_a, y_score_a = load_oof(args.model_a)

    print(f"\nLoading model B: {args.model_b}")
    y_true_b, y_score_b = load_oof(args.model_b)

    # ---- 3) Verify alignment ----
    if len(y_true_a) != len(y_true_b):
        raise ValueError(
            f"The two CSV files have different numbers of rows "
            f"({len(y_true_a)} vs {len(y_true_b)}). "
            f"Both models must be evaluated on exactly the same subjects."
        )
    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError(
            "The y_true columns do not match between the two files. "
            "Both models must be evaluated on the same subjects in the same order."
        )

    y_true = y_true_a  # same for both

    # ---- 4) Run DeLong test ----
    result = delong_test(y_true, y_score_a, y_score_b, alpha=args.alpha)
    print_results(result, args.name_a, args.name_b)

    # ---- 5) Save results CSV ----
    results_df = pd.DataFrame([{
        "model_a":     args.name_a,
        "model_b":     args.name_b,
        "auc_a":       result["auc_a"],
        "auc_b":       result["auc_b"],
        "auc_diff":    result["auc_diff"],
        "ci_a_lower":  result["ci_a"][0],
        "ci_a_upper":  result["ci_a"][1],
        "ci_b_lower":  result["ci_b"][0],
        "ci_b_upper":  result["ci_b"][1],
        "z":           result["z"],
        "p_value":     result["p_value"],
        "significant": result["significant"],
        "alpha":       args.alpha,
    }])
    csv_path = out_dir / "delong_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results CSV → {csv_path}")

    # ---- 6) Save ROC comparison plot ----
    plot_roc_comparison(
        y_true, y_score_a, y_score_b,
        args.name_a, args.name_b,
        result,
        out_dir / "roc_comparison.png",
    )

    print(f"\n{GREEN}{BOLD}Done. All outputs in: {out_dir}{RESET}")


if __name__ == "__main__":
    main()