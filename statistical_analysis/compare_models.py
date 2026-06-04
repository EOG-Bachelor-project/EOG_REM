# Filename: compare_models.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Compares Model 1 (all features) vs Model 2 (macrostructure +
#              EEG spectral) using the same K-fold CV setup. AUCs are compared
#              with a bootstrap test on the difference in per-fold AUCs.
#
# Pipeline overview:
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Load features and assign labels (same as train.py)
# 2. Run K-fold CV for Model 1 (all features) — calls run_training
# 3. Run K-fold CV for Model 2 (macrostructure + spectral) — calls run_training
#    with feature columns restricted to MODEL2_FEATURES
# 4. For each classifier, bootstrap the AUC difference between Model 1 and 2:
#       a. Resample predictions with replacement (n_bootstrap iterations)
#       b. Compute AUC for both models on each bootstrap sample
#       c. p-value = fraction of samples where ΔAUC ≤ 0
#       d. 95% CI = 2.5th and 97.5th percentile of bootstrapped ΔAUC
# 5. Save comparison summary CSV and PDF report
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Usage:
#   python -m statistical_analysis.compare_models features_csv/features_with_group.csv --method kfold --mode binary --binary-mode control_vs_all control_vs_irbd control_vs_pd --evaluate
#   python -m statistical_analysis.compare_models features_csv/features_with_group.csv --method nested --mode binary --binary-mode control_vs_all control_vs_irbd control_vs_pd --k1 5 --k2 5 --n_iter 20 --evaluate

# ================================================================================
# Imports
# ================================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import datetime
from pathlib import Path

from sklearn.metrics import roc_auc_score

from ml.prepare_data import prepare, load_features
from ml.train import run_training, DEFAULT_SEED
from ml.model2 import get_model2_features

# ================================================================================
# Constants
# ================================================================================

BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"

# DTU colors
DTU_RED         = (0.6, 0, 0)
DTUBLUE         = (0.1843, 0.2431, 0.9176)
DTUBRIGHTGREEN  = (0.1216, 0.8157, 0.5098)
DTUNAVY         = (0.0118, 0.0588, 0.3098)
DTUYELLOW       = (0.9647, 0.8157, 0.3019)
DTUORANGE       = (0.9882, 0.4627, 0.2039)
DTUPINK         = (0.9686, 0.7333, 0.6941)
DTUGREY         = (0.8549, 0.8549, 0.8549)
DTURED          = (0.9098, 0.2471, 0.2824)
DTUGREEN        = (0, 0.5333, 0.2078)
DTUPURPLE       = (0.4745, 0.1373, 0.5569)

# ================================================================================
# Bootstrap AUC comparison
# ================================================================================
def bootstrap_auc_comparison(
        y_true_1:     np.ndarray,
        y_prob_1:     np.ndarray,
        y_true_2:     np.ndarray,
        y_prob_2:     np.ndarray,
        n_bootstrap:  int = 10_000,
        seed:         int = DEFAULT_SEED,
        mode:         str = "binary",
        ) -> dict:
    """
    Bootstrap test for the difference in AUC between two models.

    Both models are evaluated on the same subjects (aggregated CV fold
    predictions), so the test is paired. For binary classification the
    AUC of the positive class is used. For multiclass, macro-averaged
    OvR AUC is used.

    Parameters
    ----------
    y_true_1, y_true_2 : np.ndarray
        True labels for each model (should be identical if same CV split).
    y_prob_1, y_prob_2 : np.ndarray
        Predicted probabilities, shape (N,) for binary or (N, C) for multiclass.
    n_bootstrap : int
        Number of bootstrap iterations.\\
        **Default:** 10 000.
    seed : int
        Random seed.\\
        **Default:** 42.
    mode : str
        Mode of classification task. Determines how AUC is computed.\\
        Options: 'binary' or 'multiclass'.\\
        **Default:** 'binary'.

    Returns
    -------
    dict with keys:
        auc_1, auc_2       — observed AUCs
        delta_auc          — auc_1 - auc_2
        p_value            — fraction of bootstrap samples where delta ≤ 0
        ci_lower, ci_upper — 95% bootstrap CI on delta_auc
        bootstrap_deltas   — full array of bootstrapped deltas
    """
    # Set up random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # Define a helper function to compute AUC based on mode
    def _auc(y_true, y_prob):
        if mode == "binary":
            prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob                      # Handle both (N,) and (N, C) shapes for binary
            return roc_auc_score(y_true, prob)
        else:
            return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") # Multiclass macro-averaged OvR AUC

    # --- Observed AUCs ---
    auc_1 = _auc(y_true_1, y_prob_1) # Compute AUC for Model 1
    auc_2 = _auc(y_true_2, y_prob_2) # Compute AUC for Model 2
    delta_obs = auc_1 - auc_2        # Observed difference in AUC between the two models
    print(f"Observed AUCs: Model 1 = {auc_1:.3f}, Model 2 = {auc_2:.3f}, Delta AUC = {delta_obs:.3f}")

    # --- Bootstrap ---
    n = len(y_true_1)               # Number of subjects (should be same for both models)
    deltas = np.empty(n_bootstrap)  # Array to store bootstrapped Delta AUC values

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)                    # resample with replacement
        d1 = _auc(y_true_1[idx], y_prob_1[idx])             # AUC for Model 1 on bootstrap sample
        d2 = _auc(y_true_2[idx], y_prob_2[idx])             # AUC for Model 2 on bootstrap sample
        deltas[i] = d1 - d2                                 # Store the difference in AUC for this bootstrap sample    

    # p-value + confidence intervals
    p_value  = np.mean(deltas <= 0)                         # p-value = fraction of bootstrap samples where Model 1 is not better than Model 2
    ci_lower = np.percentile(deltas, 2.5)                   # 2.5th percentile of bootstrapped Delta AUC for lower bound of 95% CI 
    ci_upper = np.percentile(deltas, 97.5)                  # 97.5th percentile of bootstrapped Delta AUC for upper bound of 95% CI
    print(fr"Bootstrap results: p-value = {p_value:.4f}, 95% CI for Delta AUC = [{ci_lower:.3f}, {ci_upper:.3f}]")

    return {
        "auc_1":             auc_1,
        "auc_2":             auc_2,
        "delta_auc":         delta_obs,
        "p_value":           p_value,
        "ci_lower":          ci_lower,
        "ci_upper":          ci_upper,
        "bootstrap_deltas":  deltas,
    }


# ================================================================================
# Plots
# ================================================================================
def _plot_bootstrap_delta(
        bootstrap_result: dict,
        model_name:       str,
        title:            str = "",
        ) -> plt.Figure:
    """
    Histogram of bootstrapped ΔAUC with observed value and 95% CI marked.

    Parameters
    ----------
    bootstrap_result : dict
        Output from bootstrap_auc_comparison.
    model_name : str
        Name of the classifier/model being compared (for title).
    title : str
        Additional title text (e.g. mode info).\\
        **Default:** "".
    """
    # --- Extract values from bootstrap result ---
    deltas    = bootstrap_result["bootstrap_deltas"]    # Array of bootstrapped Delta AUC values
    delta_obs = bootstrap_result["delta_auc"]           # Observed Delta AUC from the original data
    ci_lower  = bootstrap_result["ci_lower"]            # Lower bound of 95% CI for Delta AUC
    ci_upper  = bootstrap_result["ci_upper"]            # Upper bound of 95% CI for Delta AUC
    p_value   = bootstrap_result["p_value"]             # p-value for the test of whether Model 1 is better than Model 2

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deltas, bins=60, color=DTUNAVY, alpha=0.7, edgecolor="white", linewidth=0.3)                                # Histogram of bootstrapped Delta AUC values
    ax.axvline(delta_obs, color=DTURED,    lw=2,   label=f"Observed Delta AUC = {delta_obs:.3f}")                       # Vertical line for observed Delta AUC
    ax.axvline(ci_lower,  color="grey",    lw=1.5, linestyle="--", label=f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]")    # Lower bound of 95% CI 
    ax.axvline(ci_upper,  color="grey",    lw=1.5, linestyle="--")                                                      # Upper bound of 95% CI 
    ax.axvline(0,         color="black",   lw=1,   linestyle=":", alpha=0.5, label="No difference (Delta AUC=0)")       # Vertical line at 0 for reference
    ax.set_xlabel("Delta AUC  (Model 1 - Model 2)", fontsize=11)                                                        # X-axis label
    ax.set_ylabel("Bootstrap frequency", fontsize=11)                                                                   # Y-axis label
    ax.set_title(f"{model_name}  —  {title}\np = {p_value:.4f}", fontsize=12, fontweight="bold", color=DTUNAVY)         # Title
    ax.legend(fontsize=9)                                                                                               # Legend
    ax.grid(alpha=0.3)                                                                                                  # Grid
    plt.tight_layout()
    return fig


def _plot_auc_comparison_bar(
        comparison_rows: list[dict],
        title:           str = "AUC comparison: Model 1 vs Model 2",
        ) -> plt.Figure:
    """
    Grouped bar chart of AUC per classifier for both models.

    Parameters
    ----------
    comparison_rows : list of dicts
        Each dict should have keys 'classifier', 'auc_1', 'auc_2', and 'delta_auc'.
    title : str
        Title for the plot. Default "AUC comparison: Model 1 vs Model 2".
    """
    df      = pd.DataFrame(comparison_rows) # Take the list of comparison results and convert it to a DataFrame for plotting
    models  = df["classifier"].tolist()     # List of classifier names 
    x       = np.arange(len(models))        # X positions for the bars
    width   = 0.35                          # Width of the bars

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, df["auc_1"], width, label="Model 1 (all features)", color=DTUNAVY)              # Bar for Model 1 AUC
    ax.bar(x + width/2, df["auc_2"], width, label="Model 2 (macrostructure + spectral)", color=DTURED)  # Bar for Model 2 AUC

    ax.set_xticks(x)                                                                                    # X-axis ticks
    ax.set_xticklabels(models, fontsize=10)                                                             # X-axis tick labels (classifier names)
    ax.set_ylabel("AUC", fontsize=11)                                                                   # Y-axis label
    ax.set_ylim(0, 1.05)                                                                                # Y-axis limits
    ax.set_title(title, fontsize=12, fontweight="bold", color=DTUNAVY)                                  # Title
    ax.legend(fontsize=9)                                                                               # Legend
    ax.grid(axis="y", alpha=0.3)                                                                        # Grid
    plt.tight_layout()
    return fig


# ================================================================================
# Main comparison
# ================================================================================
def compare_models(
        feature_csv:      str | Path,
        method:           str = "kfold",
        mode:             str = "binary",
        binary_mode:      str = "control_vs_all",
        k1:               int = 5,
        k2:               int = 5,
        n_iter:           int = 20,
        k_folds:          int = 5,
        seed:             int = DEFAULT_SEED,
        imputer_strategy: str = "knn",
        n_bootstrap:      int = 10_000,
        save_dir:         str | Path = "reports/comparison",
        evaluate:         bool = False,
        ) -> dict:
    """
    Run Model 1 vs Model 2 comparison.

    Parameters
    ----------
    feature_csv : str | Path
        Path to the features CSV file.
    method : str
        CV method to use for training.\\
        Options: 'kfold' or 'nested'.\\
        **Default:** 'kfold'.
    mode : str
        Classification mode.\\
        Options: 'binary' or 'multiclass'.\\
        **Default:** 'binary'.
    binary_mode : str
        !!! Only used when mode='binary' !!!\\
        Binary classification task, determines how labels are assigned.\\
        Options: 'control_vs_all', 'control_vs_irbd', 'control_vs_pd'.\\
        **Default:** 'control_vs_all'.
    k1 : int
        K outer folds for nested CV (ignored if method='kfold').\\
        **Default:** 5.
    k2 : int
        K inner folds for nested CV (ignored if method='kfold').\\
        **Default:** 5.
    n_iter : int
        Number of iterations for nested CV (ignored if method='kfold').\\
        **Default:** 20.
    k_folds : int
        Number of CV folds.\\
        **Default:** 5.
    seed : int
        Random seed.\\
        **Default:** 42.
    imputer_strategy : str
        Imputation strategy.\\
        **Default:** 'knn'.
    n_bootstrap : int
        Bootstrap iterations for AUC comparison.\\
        **Default:** 10 000.
    save_dir : str | Path
        Output directory for CSVs and PDF.\\
        **Default:** "reports/comparison".
    evaluate : bool
        If True, also generate per-model evaluation PDFs via ``evaluate.py``.\\
        **Default:** False.


    Returns
    -------
    dict
        result_m1, result_m2, comparison_df, bootstrap_results.
    """
    # Ensure save directory exists
    save_dir = Path(save_dir)                   
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{mode}" + (f"_{binary_mode}" if mode == "binary" else "")

    print(f"\n{'='*60}")
    print(f"  {BOLD}Model comparison — {method.upper()} CV{RESET}")
    print(f"  Mode    : {mode}" + (f"  ({binary_mode})" if mode == "binary" else ""))
    print(f"  Folds   : {k_folds}  |  Seed: {seed}")
    print(f"{'='*60}")

    # ---- 1) Model 1: all features ----
    print(f"\n{BOLD}── Model 1: all features ──{RESET}")
    result_m1 = run_training(
        feature_csv      = feature_csv,
        method           = method,
        mode             = mode,
        binary_mode      = binary_mode,
        k_1              = k1,
        k_2              = k2,
        n_iter           = n_iter,
        k_folds          = k_folds,
        seed             = seed,
        imputer_strategy = imputer_strategy,
        save_dir         = save_dir / "model1",
    )

    # ---- 2) Model 2: macrostructure + spectral ----
    # Filter feature columns before running — load df once to get Model 2 cols
    print(f"\n{BOLD}── Model 2: macrostructure + EEG spectral ──{RESET}")
    df_full      = load_features(feature_csv)
    m2_feat_cols = get_model2_features(df_full)

    # Temporarily monkeypatch prepare to use only Model 2 features by writing a filtered CSV and passing it to run_training
    import tempfile, os
    id_and_group_cols = ["subject_id", "DCSM_ID", "Control", "PD(-RBD)", "PD(+RBD)", "iRBD", "PLM"] 
    keep_cols = [c for c in id_and_group_cols if c in df_full.columns] + m2_feat_cols
    df_m2 = df_full[keep_cols]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df_m2.to_csv(f, index=False)
        m2_csv = f.name

    try:
        result_m2 = run_training(
            feature_csv      = m2_csv,
            method           = method,
            mode             = mode,
            binary_mode      = binary_mode,
            k_1              = k1,
            k_2              = k2,
            n_iter           = n_iter,
            k_folds          = k_folds,
            seed             = seed,
            imputer_strategy = imputer_strategy,
            save_dir         = save_dir / "model2",
        )
    finally:
        os.unlink(m2_csv)  # clean up temp file

    # ---- 3) Bootstrap AUC comparison per classifier ----
    print(f"\n{BOLD}── Bootstrap AUC comparison (n={n_bootstrap}) ──{RESET}")
    
    preds_m1 = result_m1["predictions"] # Dict of predictions per classifier for Model 1. Each value is a dict with 'y_true' and 'y_prob'.
    preds_m2 = result_m2["predictions"] # Dict of predictions per classifier for Model 2.

    comparison_rows     = [] 
    bootstrap_results   = {}
    pdf_figs            = []

    for clf_name in preds_m1:
        if clf_name not in preds_m2:
            continue

        p1 = preds_m1[clf_name]
        p2 = preds_m2[clf_name]

        if p1["y_prob"] is None or p2["y_prob"] is None:
            print(f"  [SKIP] {clf_name} — no probabilities available")
            continue

        boot = bootstrap_auc_comparison(
            y_true_1    = p1["y_true"],
            y_prob_1    = p1["y_prob"],
            y_true_2    = p2["y_true"],
            y_prob_2    = p2["y_prob"],
            n_bootstrap = n_bootstrap,
            seed        = seed,
            mode        = mode,
        )
        bootstrap_results[clf_name] = boot

        sig = "✓" if boot["p_value"] < 0.05 else "✗"
        print(f"  {clf_name:<25s}  "
              f"AUC1={boot['auc_1']:.3f}  AUC2={boot['auc_2']:.3f}  "
              f"$\Delta$ AUC={boot['delta_auc']:.3f}  "
              f"p={boot['p_value']:.4f}  {sig}")

        comparison_rows.append({
            "classifier":  clf_name,
            "auc_1":       boot["auc_1"],
            "auc_2":       boot["auc_2"],
            "delta_auc":   boot["delta_auc"],
            "ci_lower":    boot["ci_lower"],
            "ci_upper":    boot["ci_upper"],
            "p_value":     boot["p_value"],
            "significant": boot["p_value"] < 0.05,
        })

        pdf_figs.append(_plot_bootstrap_delta(
            bootstrap_result = boot,
            model_name       = clf_name,
            title            = f"Bootstrap ΔAUC  ({tag})",
        ))

    comparison_df = pd.DataFrame(comparison_rows)

    # ---- 4) Bar chart ----
    if comparison_rows:
        pdf_figs.insert(0, _plot_auc_comparison_bar(
            comparison_rows = comparison_rows,
            title           = f"AUC comparison: Model 1 vs Model 2  ({tag})",
        ))

    # ---- 5) Save ----
    csv_path = save_dir / f"comparison_{tag}.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n  Saved -> {csv_path}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path  = save_dir / f"comparison_{tag}_{timestamp}.pdf"
    with pdf_backend.PdfPages(pdf_path) as pdf:
        for fig in pdf_figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        meta           = pdf.infodict()
        meta["Title"]  = f"Model comparison — {tag}"
        meta["Author"] = "EOG_REM pipeline"
    print(f"  Saved -> {pdf_path}")
    print(f"{'='*60}\n")

    # ---- 6) Optional evaluate ----
    if evaluate:
        from ml.evaluate import evaluate_all
        from ml.train import _label_map
        lm          = _label_map(mode, binary_mode)
        class_names = [lm[k] for k in sorted(lm)]
        run_cfg     = {
            "mode":             mode, 
            "binary_mode":      binary_mode or "—",
            "method":           method,
            "k_folds":          k_folds if method == "kfold" else f"{k1}/{k2}", 
            "seed":             seed,
            "imputer_strategy": imputer_strategy,
        }
        print(f"\n{BOLD}── Evaluation PDFs: Model 1 ──{RESET}")
        evaluate_all(
            predictions         = result_m1["predictions"], 
            summary             = result_m1["summary"],
            X                   = result_m1["X"], y = result_m1["y"], 
            class_names         = class_names,
            mode                = mode, 
            imputer_strategy    = imputer_strategy, 
            seed                = seed,
            run_config          = {**run_cfg, "n_subjects": len(result_m1["y"]), "n_features": result_m1["X"].shape[1]},
            save_dir            = save_dir / "model1_eval",
        )
        print(f"\n{BOLD}── Evaluation PDFs: Model 2 ──{RESET}")
        evaluate_all(
            predictions         = result_m2["predictions"], 
            summary             = result_m2["summary"],
            X                   = result_m2["X"], y = result_m2["y"],
            class_names         = class_names,
            mode                = mode, 
            imputer_strategy    = imputer_strategy, 
            seed                = seed,
            run_config          = {**run_cfg, "n_subjects": len(result_m2["y"]), "n_features": result_m2["X"].shape[1]},
            save_dir            = save_dir / "model2_eval",
        )

    return {
        "result_m1":          result_m1,
        "result_m2":          result_m2,
        "comparison_df":      comparison_df,
        "bootstrap_results":  bootstrap_results,
    }

# ================================================================================
# CLI
# ================================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Model 1 (all features) vs Model 2 (macrostructure + spectral) "
                    "using K-fold CV and bootstrap AUC test. "
                    "Pass multiple --binary-mode values to run all combinations."
                    )
    parser.add_argument("feature_csv", type=str)
    parser.add_argument("--method", type=str, default="kfold", 
                        choices=["kfold", "nested"],
                        help="CV method to use for training. Default 'kfold'.")
    parser.add_argument("--mode", type=str, nargs="+", default=["binary"],
                        choices=["binary", "multiclass"])
    parser.add_argument("--binary-mode", type=str, nargs="+", default=["control_vs_all"],
                        choices=["control_vs_all", "control_vs_irbd", "control_vs_pd"])
    parser.add_argument("--k1", type=int, default=5, 
                        help="K outer folds for nested CV (ignored if --method=kfold). Default 5.")
    parser.add_argument("--k2", type=int, default=5,
                        help="K inner folds for nested CV (ignored if --method=kfold). Default 5.")
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Number of iterations for nested CV (ignored if --method=kfold). Default 20.")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--imputer", type=str, default="knn",
                        choices=["knn", "median", "mean", "most_frequent"])
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--save-dir", type=str, default="reports/comparison")
    parser.add_argument("--evaluate", action="store_true",
                        help="Also generate per-model evaluation PDFs.")
    args = parser.parse_args()

    # Build list of (mode, binary_mode) runs
    runs = []
    for mode in args.mode:
        if mode == "multiclass":
            runs.append((mode, None))
        else:
            for bm in args.binary_mode:
                runs.append((mode, bm))

    for mode, binary_mode in runs:
        compare_models(
            feature_csv      = args.feature_csv,
            mode             = mode,
            binary_mode      = binary_mode or "control_vs_all",
            method           = args.method,
            k1               = args.k1,
            k2               = args.k2,
            n_iter           = args.n_iter,
            k_folds          = args.k_folds,
            seed             = args.seed,
            imputer_strategy = args.imputer,
            n_bootstrap      = args.n_bootstrap,
            save_dir         = args.save_dir,
            evaluate         = args.evaluate,
        )