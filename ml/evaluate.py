# Filename: evaluate.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Evaluates the trained model on the test set and computes performance metrics.
#              Computes confusion matrix, ROC curve (per class for multiclass),
#              feature importance bar chart and saves all plots into a single PDF per model.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

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
)

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize

# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"


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
    fig.suptitle(title, fontsize=13, fontweight="bold")

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
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_classes))

    # ---- 2) Compute and plot ROC curve for each class ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])                             # x-axis limit           
    ax.set_ylim([0.0, 1.02])                            # y-axis limit
    ax.set_xlabel("False Positive Rate", fontsize=10)   # x-axis label
    ax.set_ylabel("True Positive Rate",  fontsize=10)   # y-axis label
    ax.set_title(title, fontsize=12, fontweight="bold") # title
    ax.legend(loc="lower right", fontsize=9)            # legend
    ax.grid(alpha=0.3, linestyle="--")                  # grid 
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
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#0E7490", alpha=0.8) # horizontal bars
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=10)                              # x-axis label
    ax.set_title(title, fontsize=12, fontweight="bold")                                  # title              
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
        color="#AA3377", alpha=0.8,
        error_kw=dict(elinewidth=0.8, capsize=3),
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)  # vertical line at zero
    ax.set_xlabel(f"Permutation importance ({scoring})", fontsize=10)       # x-axis label
    ax.set_title(title, fontsize=12, fontweight="bold")                     # title
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
        X_test:      pd.DataFrame,
        y_test,
        class_names: list[str],
        model_name:  str = "Model",
        top_n:       int = 20,
        save_dir:    str | Path | None = None,
        seed:        int = 42,
        ) -> None:
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
    """
    y_pred = model.predict(X_test)

    # Collect probabilities if the model supports predict_proba
    has_proba = hasattr(model, "predict_proba")
    y_prob    = model.predict_proba(X_test) if has_proba else None

    tag = model_name.lower().replace(" ", "_")

    # ── build all figures ────────────────────────────────────────────
    figs: list[plt.Figure] = []

    figs.append(_plot_confusion_matrix(
        y_true      = y_test,
        y_pred      = y_pred,
        class_names = class_names,
        title       = f"{model_name} — Confusion Matrix",
    ))

    if y_prob is not None:
        figs.append(_plot_roc_curves(
            y_true      = y_test,
            y_prob      = y_prob,
            class_names = class_names,
            title       = f"{model_name} — ROC Curves (one-vs-rest)",
        ))
    else:
        print("  [SKIP] ROC curve — model has no predict_proba")

    mdi_fig, mdi_df = _plot_feature_importance_MDI(
        model         = model,
        feature_names = list(X_test.columns),
        top_n         = top_n,
        title         = f"{model_name} — Feature Importance (MDI)",
    )
    if mdi_fig is not None:
        figs.append(mdi_fig)

    perm_fig, perm_df = _plot_feature_importance_permutation(
        model     = model,
        X_test    = X_test,
        y_test    = y_test,
        top_n     = top_n,
        title     = f"{model_name} — Feature Importance (Permutation)",
        seed      = seed,
    )
    figs.append(perm_fig)

    # ── save or show ────────────────────────────────────────────────
    if save_dir is not None:
        out_dir  = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"{tag}.pdf"

        with pdf_backend.PdfPages(pdf_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # PDF metadata
            meta = pdf.infodict()
            meta["Title"]   = f"{model_name} — evaluation report"
            meta["Author"]  = "EOG_REM pipeline"
            meta["Subject"] = "Model evaluation plots"

        print(f"\n{GREEN}{BOLD}Saved evaluation report → {pdf_path}{RESET}")

        # Also print importance tables to console
        if not mdi_df.empty:
            print(f"\n{BOLD}Top-{top_n} MDI features:{RESET}")
            print(mdi_df.head(top_n).to_string(index=False))

        print(f"\n{BOLD}Top-{top_n} permutation features:{RESET}")
        print(perm_df.head(top_n).to_string(index=False))

    else:
        for fig in figs:
            plt.show()
            plt.close(fig)