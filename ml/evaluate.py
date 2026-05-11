# Filename: evaluate.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Evaluates the trained model on the test set and computes performance metrics.
#              Computes confusion matrix, ROC curve (per class for multiclass),
#              feature importance bar chart and saves all plots into a single PDF per model.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations
 
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
 
# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Colors
dtu_red = (0.6, 0, 0)
dtublue = (0.1843, 0.2431, 0.9176)
dtubrightgreen = (0.1216, 0.8157, 0.5098)
dtunavy = (0.0118, 0.0588, 0.3098)
dtuyellow = (0.9647, 0.8157, 0.3019)
dtuorange = (0.9882, 0.4627, 0.2039)
dtupink = (0.9686, 0.7333, 0.6941)
dtugrey = (0.8549, 0.8549, 0.8549)
dtured = (0.9098, 0.2471, 0.2824)
dtugreen = (0, 0.5333, 0.2078)
dtupurple = (0.4745, 0.1373, 0.5569)


# =====================================================================
# Report info page  — always the first page of the PDF
# =====================================================================
 
def _plot_run_info(
        model_name:  str,
        class_names: list[str],
        y_test,
        y_pred,
        run_config:  dict | None = None,
        cv_results:  pd.DataFrame | None = None,
        best_params: dict | None = None,
        ) -> plt.Figure:
    """
    Build and return a summary info page for the PDF.
 
    Shows:
      - Run configuration (seed, test_size, mode, n_outer, n_inner, n_iter, timestamp)
      - Test set metrics (accuracy, balanced accuracy, F1, precision, recall)
      - Class distribution of the test set
      - CV results table (mean ± std per model) if provided
      - Best hyperparameters found for this model
 
    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "Random Forest") — used in the title and PDF filename.
    class_names : list[str]
        List of class names corresponding to the label encoding (e.g., ["Wake", "NREM", "REM"]).
    y_test : array-like      
        True test labels.
    y_pred : array-like      
        Predicted test labels.
    run_config  : dict | None
        Keys (all optional): seed, test_size, mode, binary_mode,
        n_outer, n_inner, n_iter, feature_csv, n_train, n_test, n_features.
    cv_results  : pd.DataFrame | None
        Output of two_layer_cross_validate - shown as a table.
    best_params : dict | None
        Best hyperparameters chosen by inner CV for this model.
    """
    fig = plt.figure(figsize=(12, 15))
    fig.patch.set_facecolor("white")
 
    # ── title ───────────────────────────────────────────────────────
    fig.text(0.5, 0.97, f"Evaluation Report  for  {model_name}",
             ha="center", va="top", fontsize=16, fontweight="bold", color=dtunavy)
    fig.text(0.5, 0.945,
             datetime.datetime.now().strftime("Generated %Y-%m-%d  %H:%M"),
             ha="center", va="top", fontsize=9, color=dtugrey)
 
    y_cursor = 0.91
 
    def _section(title: str) -> None:
        nonlocal y_cursor
        fig.text(0.05, y_cursor, title, fontsize=11, fontweight="bold", color=dtunavy, transform=fig.transFigure)

        y_cursor -= 0.022
        fig.add_artist(
            plt.Line2D([0.05, 0.95], 
                       [y_cursor, y_cursor],
                       transform=fig.transFigure,
                       color=dtunavy, 
                       linewidth=0.8, 
                       alpha=0.5)
                       )
        y_cursor -= 0.010
 
    def _row(label: str, value: str, indent: float = 0.07, bold_val: bool = False, color: str = "black") -> None:
        nonlocal y_cursor
        fig.text(indent, y_cursor, label, fontsize=9, color="#444444", transform=fig.transFigure)
        fig.text(0.42, y_cursor, value, fontsize=9, color=color, 
                 fontweight="bold" if bold_val else "normal",
                 transform=fig.transFigure)
        y_cursor -= 0.018
 
    # ── 1. Run configuration ─────────────────────────────────────────
    _section("1  Run configuration")
    if run_config:
        mode_str = run_config.get("mode", "—")
        if mode_str == "binary":
            mode_str += f"  ({run_config.get('binary_mode', '')})"
        cfg_items = [
            ("Feature CSV",       str(run_config.get("feature_csv", "—"))),
            ("Mode",              mode_str),
            ("Seed",              str(run_config.get("seed", "—"))),
            ("Test size",         f"{run_config.get('test_size', '—')}"
                                  f"  ({run_config.get('n_train','?')} train / "
                                  f"{run_config.get('n_test','?')} test)"),
            ("Features",          str(run_config.get("n_features", "—"))),
            ("Imputer strategy",  str(run_config.get("imputer_strategy", "—"))),
            ("Outer CV folds",    str(run_config.get("n_outer", "—"))),
            ("Inner CV folds",    str(run_config.get("n_inner", "—"))),
            ("Search iterations", str(run_config.get("n_iter",  "—"))),
            ("Classes",           ", ".join(class_names)),
        ]
        for label, val in cfg_items:
            _row(label, val)
    else:
        _row("(no run config provided)", "—")
    y_cursor -= 0.008
 
    # ── 2. Test set metrics ──────────────────────────────────────────
    _section("2  Test set metrics")

    # Prediction metrics (weighted average for multiclass)
    acc  = accuracy_score(y_test, y_pred)                                       # Accuracy          = (TP + TN) / (TP + TN + FP + FN)
    bal  = balanced_accuracy_score(y_test, y_pred)                              # Balanced accuracy = (TPR + TNR) / 2
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)        # f1                = 2 * (Precision * Recall) / (Precision + Recall)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) # Precision         = TP / (TP + FP)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)    # Recall            = TP / (TP + FN)
    for label, val in [
        ("Accuracy",          f"{acc:.4f}"),
        ("Balanced accuracy", f"{bal:.4f}"),
        ("F1 (weighted)",     f"{f1:.4f}"),
        ("Precision",         f"{prec:.4f}"),
        ("Recall",            f"{rec:.4f}"),
    ]:
        _row(label, val, bold_val=True, color=dtunavy)
    y_cursor -= 0.008
 
    # ── 3. Class distribution ────────────────────────────────────────
    _section("3  Test set class distribution")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, cnt in zip(unique, counts):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        _row(f"  {name}", f"{cnt}  ({cnt / len(y_test) * 100:.1f}%)")
    y_cursor -= 0.008
 
    # ── 4. CV summary table ──────────────────────────────────────────
    if cv_results is not None and not cv_results.empty:
        _section("4  Two-layer CV summary  (balanced accuracy)")
        col_x   = [0.07, 0.42, 0.56, 0.66, 0.76]
        headers = ["Model", "Mean", "Std", "Min", "Max"]
        for hx, h in zip(col_x, headers):
            fig.text(hx, y_cursor, h, fontsize=8, fontweight="bold",
                     color="#444444", transform=fig.transFigure)
        y_cursor -= 0.016
 
        best_mean = cv_results[
            cv_results["model"] != "Baseline (majority)"
        ]["bal_acc_mean"].max()
 
        for _, r in cv_results.iterrows():
            is_best = (r["model"] != "Baseline (majority)"
                       and r.get("bal_acc_mean", 0) == best_mean)
            marker = " ←" if is_best else ""
            vals = [
                r["model"] + marker,
                f"{r.get('bal_acc_mean', 0):.3f}",
                f"{r.get('bal_acc_std',  0):.3f}",
                f"{r.get('bal_acc_min',  0):.3f}",
                f"{r.get('bal_acc_max',  0):.3f}",
            ]
            color = dtugreen if is_best else "black"
            for hx, v in zip(col_x, vals):
                fig.text(hx, y_cursor, v, fontsize=8, color=color,
                         transform=fig.transFigure)
            y_cursor -= 0.016
        y_cursor -= 0.008
 
    # ── 5. Best hyperparameters ──────────────────────────────────────
    if best_params:
        _section("5  Best hyperparameters")
        for k, v in sorted(best_params.items()):
            _row(f"  {k}", str(v))
 
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# =====================================================================
# Internal helper — figures are returned, NOT saved individually
# =====================================================================

def _plot_confusion_matrix(
        y_true,
        y_pred,
        class_names: list[str],
        title:       str = "Confusion Matrix",
        ) -> plt.Figure:
    """
    Build and return a confusion matrix figure (counts + normalised).

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    class_names : list of str
        Class names for axis labels.
    title : str
        Figure title. **Default is "Confusion Matrix"**.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The confusion matrix figure (with counts and normalised).
    """
    # ---- 1) Compute confusion matrix and normalised version ----
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ---- 2) Plot side-by-side heatmaps for counts and normalised ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", color=dtunavy)

    for ax, data, fmt, subtitle in zip(
        axes,                           
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Normalised (row%)"],
    ):
        sns.heatmap(                                 # heatmap with annotations
            data,
            annot=True,
            fmt=fmt,                                 # String format for annotations
            cmap="Blues",                            # colormap  
            xticklabels=class_names,                 # x-axis tick labels
            yticklabels=class_names,                 # y-axis tick labels
            linewidths=0.5,                          # line width for cells
            ax=ax,                                   # axes to plot on
        )
        ax.set_xlabel("Predicted label", fontsize=10) # x-axis label
        ax.set_ylabel("True label",      fontsize=10) # y-axis label
        ax.set_title(subtitle,           fontsize=10) # subtitle
        ax.tick_params(labelsize=9)                   # tick label size

    plt.tight_layout()

    # Print classification report to console
    print(f"\n{BOLD}Classification report{RESET}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return fig


def _plot_roc_curves(
        y_true,
        y_prob,
        class_names: list[str],
        title:       str = "ROC Curves",
        ) -> plt.Figure:
    """
    Build and return a one-vs-rest ROC curve figure for every class.
    Works for binary (2-class) and multi-class problems.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_prob : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class (from predict_proba).
    class_names : list of str
        Class names for axis labels.
    title : str
        Figure title. **Default is "ROC Curves"**.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The ROC curve figure (one-vs-rest for each class).
    """
    # ---- 1) Binarize labels for one-vs-rest ROC computation ---
    n_classes  = len(class_names)
    y_bin      = label_binarize(y_true, classes=list(range(n_classes)))

    # Binary case: label_binarize returns shape (n, 1) — expand to (n, 2)
    if n_classes == 2:
        y_bin  = np.hstack([1 - y_bin, y_bin])

    # Get distinct colors for each class
    colors = (dtunavy, dtublue, dtured, dtupink, dtupurple, dtubrightgreen, dtuorange, dtuyellow)

    # ---- 2) Compute and plot ROC curve for each class ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])                                             # x-axis limit           
    ax.set_ylim([0.0, 1.02])                                            # y-axis limit
    ax.set_xlabel("False Positive Rate", fontsize=10)                   # x-axis label
    ax.set_ylabel("True Positive Rate",  fontsize=10)                   # y-axis label
    ax.set_title(title, fontsize=12, fontweight="bold", color=dtunavy)  # title
    ax.legend(loc="lower right", fontsize=9)                            # legend
    ax.grid(alpha=0.3, linestyle="--")                                  # grid 
    plt.tight_layout()

    return fig


def _plot_feature_importance_MDI(
        model,
        feature_names: list[str],
        top_n:         int = 20,
        title:         str = "Feature Importance (MDI)",
        ) -> tuple[plt.Figure | None, pd.DataFrame]:
    """
    Build and return an MDI feature importance figure + DataFrame.
    Returns (None, empty_df) if the model does not expose feature_importances_.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
        The model to extract feature importances from. If a Pipeline is passed,
        the final estimator is used.
    feature_names : list of str
        Names of the features corresponding to the columns in X_test.
    top_n : int
        Number of top features to show in the plot. **Default is 20**.
    title : str
        Title for the feature importance plot. **Default is "Feature Importance (MDI)"**.
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The feature importance bar chart figure, or None if not available.
    importance_df : pd.DataFrame
        DataFrame containing features and their importance scores, sorted by importance.
    """
    # If model is a Pipeline, get the final estimator
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    # ---- 1) Check if the model exposes feature_importances_ (e.g., tree-based models) ----
    if not hasattr(clf, "feature_importances_"):
        print(f"  [SKIP] MDI importance not available for {type(clf).__name__}")
        return None, pd.DataFrame(columns=["feature", "importance"])

    # ---- 2) Build importance DataFrame ----
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    top = importance_df.head(top_n) # get top-N features for plotting

    # ---- 3) Plot horizontal bar chart of top-N features ----
    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=dtublue, alpha=0.8)     # horizontal bars
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=10)                              # x-axis label
    ax.set_title(title, fontsize=12, fontweight="bold", color=dtunavy)                   # title              
    ax.tick_params(labelsize=9)                                                          # tick label size                        
    ax.grid(axis="x", alpha=0.3, linestyle="--")                                         # vertical grid lines            
    plt.tight_layout()

    return fig, importance_df


def _plot_feature_importance_permutation(
        model,
        X_test:    pd.DataFrame,
        y_test,
        top_n:     int = 20,
        n_repeats: int = 10,
        scoring:   str = "balanced_accuracy",
        title:     str = "Feature Importance (Permutation)",
        seed:      int = 42,
    ) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Build and return a permutation importance figure + DataFrame.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
        The model to evaluate. If a Pipeline is passed, the final estimator is used.
    X_test : pd.DataFrame
        Test features (used for permutation).
    y_test : array-like
        Test labels (used for scoring).
    top_n : int
        Number of top features to show in the plot. **Default is 20**.
    n_repeats : int
        Number of times to permute each feature. **Default is 10**.
    scoring : str
        Scoring metric to evaluate the model performance after permutation. 
        **Default is "balanced_accuracy"**.
    title : str
        Title for the feature importance plot. 
        **Default is "Feature Importance (Permutation)"**.
    seed : int
        Random seed for reproducibility of permutations. 
        **Default is 42**.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The permutation importance bar chart figure.
    importance_df : pd.DataFrame
        DataFrame containing features and their permutation importance scores (mean and std), 
        sorted by mean importance.
    """
    # ---- 1) Compute permutation importance using sklearn's built-in function ----
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats    = n_repeats,
        random_state = seed,
        scoring      = scoring,
        n_jobs       = -1,
    )

    # ---- 2) Build importance DataFrame with mean and std of importance scores ----
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean":    result.importances_mean,
        "std":     result.importances_std,
          }).sort_values("mean", ascending=False).reset_index(drop=True)

    top = importance_df.head(top_n).iloc[::-1].reset_index(drop=True)
    
    # ---- 3) Plot horizontal bar chart of top-N features with error bars ----
    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1))
    ax.barh(                                                                # horizontal bars with error bars
        top["feature"], top["mean"],
        xerr=top["std"],
        color=dtu_red, alpha=0.8,
        error_kw=dict(elinewidth=0.8, capsize=3),
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)  # vertical line at zero
    ax.set_xlabel(f"Permutation importance ({scoring})", fontsize=10)       # x-axis label
    ax.set_title(title, fontsize=12, fontweight="bold", color=dtunavy)      # title
    ax.tick_params(labelsize=9)                                             # tick label size
    ax.grid(axis="x", alpha=0.3, linestyle="--")                            # vertical grid lines
    plt.tight_layout()

    return fig, importance_df


# =====================================================================
# Public single-plot functions (kept for standalone / interactive use)
# =====================================================================

def plot_confusion_matrix(
        y_true, 
        y_pred, 
        class_names, 
        title="Confusion Matrix",
        save_path=None
        ):
    """
    Plot and optionally save a confusion matrix figure (counts + normalised).

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    class_names : list of str
        Class names for axis labels.
    title : str
        Figure title. **Default is "Confusion Matrix"**.
    save_path : str or Path, optional
        If provided, the figure is saved to this path instead of shown interactively.
    """

    fig = _plot_confusion_matrix(y_true, y_pred, class_names, title)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_roc_curves(
        y_true, 
        y_prob, 
        class_names, 
        title="ROC Curves", 
        save_path=None
        ):
    """
    Plot and optionally save a one-vs-rest ROC curve figure for every class.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_prob : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class (from predict_proba).
    class_names : list of str
        Class names for axis labels.
    title : str
        Figure title. **Default is "ROC Curves"**.
    save_path : str or Path, optional
        If provided, the figure is saved to this path instead of shown interactively.
    """

    fig = _plot_roc_curves(y_true, y_prob, class_names, title)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_feature_importance_MDI(
        model, 
        feature_names, 
        top_n=20,
        title="Feature Importance (MDI)", 
        save_path=None
        ):
    """
    Plot and optionally save an MDI feature importance figure.
    
    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
        The model to extract feature importances from. If a Pipeline is passed,
        the final estimator is used.
    feature_names : list of str
        Names of the features corresponding to the columns in X_test.
    top_n : int
        Number of top features to show in the plot. **Default is 20**.
    title : str
        Title for the feature importance plot. **Default is "Feature Importance (MDI)"**.
    save_path : str or Path, optional
        If provided, the figure is saved to this path instead of shown interactively.
    """
    
    fig, df = _plot_feature_importance_MDI(model, feature_names, top_n, title)
    if fig is not None:
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)
    return df


def plot_feature_importance_permutation(
        model, 
        X_test, 
        y_test, 
        top_n=20,
        n_repeats=10, 
        scoring="balanced_accuracy",
        title="Feature Importance (Permutation)",
        save_path=None, 
        seed=42
        ):
    """
    Plot and optionally save a permutation importance figure.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
        The model to evaluate. If a Pipeline is passed, the final estimator is used.
    X_test : pd.DataFrame
        Test features (used for permutation).
    y_test : array-like
        Test labels (used for scoring).
    top_n : int
        Number of top features to show in the plot. **Default is 20**.
    n_repeats : int
        Number of times to permute each feature. **Default is 10**.
    scoring : str
        Scoring metric to evaluate the model performance after permutation. 
        **Default is "balanced_accuracy"**.
    title : str
        Title for the feature importance plot. 
        **Default is "Feature Importance (Permutation)"**.
    save_path : str or Path, optional
        If provided, the figure is saved to this path instead of shown interactively.
    seed : int
        Random seed for reproducibility of permutations. 
        **Default is 42**.
    """
    
    fig, df = _plot_feature_importance_permutation(
        model, X_test, y_test, top_n, n_repeats, scoring, title, seed)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return df

# =====================================================================
# Full model evaluation  — all plots → one PDF
# =====================================================================

def evaluate_model(
        model,  
        X_test: pd.DataFrame,  
        y_test,  
        class_names: list[str],  
        model_name: str = "Model",  
        top_n: int = 20,  
        save_dir: str | Path | None = None,  
        seed: int = 42,  
        run_config: dict | None = None,  
        cv_results: pd.DataFrame | None = None,  
        best_params: dict | None = None,
        ) -> dict:
    """
    Run all evaluation plots for one fitted model and collect them into a
    single PDF file  ``<save_dir>/<model_name>.pdf``.

    If ``save_dir`` is None every figure is shown interactively instead.

    Parameters
    ----------
    model : fitted sklearn Pipeline or estimator
        Model to evaluate. If a Pipeline is passed, the final estimator must expose
        either `feature_importances_` (for MDI) or support `permutation_importance`.
    X_test : pd.DataFrame
        Test features.
    y_test : array-like
        Test labels.
    class_names : list of str
        Class names for axis labels.
    model_name : str
        Used as the PDF filename and in every plot title.
    top_n : int
        Top-N features shown in importance plots. Default 20.
    save_dir : str | Path | None
        Directory for the output PDF. Created if it does not exist.
    seed : int
        Random seed for permutation importance.
    run_config : dict | None
        Optional run configuration info to include in the first page of the PDF.
    cv_results : pd.DataFrame | None
        Optional CV results summary to include in the first page of the PDF.
    best_params : dict | None
        Optional best hyperparameters to include in the first page of the PDF.
    
    Returns
    -------
    """
    y_pred = model.predict(X_test)

    # Collect probabilities if the model supports predict_proba
    has_proba = hasattr(model, "predict_proba")
    y_prob    = model.predict_proba(X_test) if has_proba else None

    tag = model_name.lower().replace(" ", "_")

    # ── build all figures ────────────────────────────────────────────
    figs: list[plt.Figure] = []

    # Page 1 — info sheet
    figs.append(_plot_run_info(
        model_name  = model_name,
        class_names = class_names,
        y_test      = y_test,
        y_pred      = y_pred,
        run_config  = run_config,
        cv_results  = cv_results,
        best_params = best_params,
    ))

    # Page 2 — confusion matrix
    figs.append(_plot_confusion_matrix(
        y_true=y_test, y_pred=y_pred, class_names=class_names,
        title=f"{model_name} — Confusion Matrix",
    ))

    # Page 3 — ROC curves
    if y_prob is not None:
        figs.append(_plot_roc_curves(
            y_true=y_test, y_prob=y_prob, class_names=class_names,
            title=f"{model_name} — ROC Curves (one-vs-rest)",
        ))
    else:
        print("  [SKIP] ROC curve — model has no predict_proba")

    # Page 4 — MDI importance
    mdi_fig, mdi_df = _plot_feature_importance_MDI(
        model=model, feature_names=list(X_test.columns),
        top_n=top_n, title=f"{model_name} — Feature Importance (MDI)",
    )
    if mdi_fig is not None:
        figs.append(mdi_fig)

    # Page 5 — permutation importance
    perm_fig, perm_df = _plot_feature_importance_permutation(
        model=model, X_test=X_test, y_test=y_test,
        top_n=top_n, title=f"{model_name} — Feature Importance (Permutation)",
        seed=seed,
    )
    figs.append(perm_fig)

    # ── save or show ─────────────────────────────────────────────────
    pdf_path = None 
    mdi_csv_path = None  
    permutation_csv_path = None 

    if save_dir is not None:  
        out_dir = Path(save_dir)  
        out_dir.mkdir(parents=True, exist_ok=True)  
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
        mode_tag        = run_config.get("mode", "") if run_config else ""  
        seed_tag        = run_config.get("seed", "") if run_config else ""  
        imputer_tag     = run_config.get("imputer_strategy", "") if run_config else "" 
        
        # Fælles stem så PDF og CSV'er får samme navn  
        stem = f"{tag}_{mode_tag}_seed{seed_tag}_{imputer_tag}_{timestamp}"  
        pdf_path = out_dir / f"{stem}.pdf" 
        #--- Write PDF --- 
        with pdf_backend.PdfPages(pdf_path) as pdf: 
            for fig in figs:  
                pdf.savefig(fig, bbox_inches="tight")  
                plt.close(fig)  
            meta = pdf.infodict()  
            meta["Title"]   = f"{model_name} — evaluation report" 
            meta["Author"]  = "EOG_REM pipeline"  
            meta["Subject"] = "Model evaluation plots" 
        print(f"\n{GREEN}{BOLD}Saved → {pdf_path}{RESET}") 

        # --- Write MDI CSV ---
        if not mdi_df.empty:  
            mdi_csv_path = out_dir / f"{stem}_mdi.csv"  
            mdi_df.to_csv(mdi_csv_path, index=False) 
            print(f"{GREEN}Saved -> {mdi_csv_path}{RESET}") 
            
            print(f"\n{BOLD}Top-{top_n} MDI features:{RESET}") 
            print(mdi_df.head(top_n).to_string(index=False)) 
        
        # --- Write permutation CSV ---  
        permutation_csv_path = out_dir / f"{stem}_permutation.csv" 
        perm_df.to_csv(permutation_csv_path, index=False) 
        print(f"{GREEN}Saved -> {permutation_csv_path}{RESET}") 
        
        print(f"\n{BOLD}Top-{top_n} permutation features:{RESET}") 
        print(perm_df.head(top_n).to_string(index=False)) 
    
    else: 
        for fig in figs:  
            plt.show()  
            plt.close(fig) 
    
    return {
        "mdi_df":               mdi_df, 
        "perm_df":              perm_df, 
        "pdf_path":             pdf_path, 
        "mdi_csv_path":         mdi_csv_path, 
        "permutation_csv_path": permutation_csv_path, 
        }