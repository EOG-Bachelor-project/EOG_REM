# Filename: evaluate.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Evaluates the trained model on the test set and computes performance metrics.
#              Computes confusion matrix, ROC curve (per class for multiclass), 
#              feature importance bar chart and saves plots.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations
from xml.parsers.expat import model  # for Python 3.10+ type hinting features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path  

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from ml.prepare_data import X_train

# ANSI helpers
BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"

# =====================================================================
# Confusion Matrix 
# =====================================================================
def plot_confusion_matrix(
        y_true, 
        y_pred,
        class_names:    list[str],
        title:          str = "Confussion Matrix",
        save_path:      str | Path | None = None,
        ):
    """
    Computes and plots the confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels by the model.
    class_names : list of str
        Names of the classes in the same order as the labels.
    title : str
        Plot title. Default is 'Confusion Matrix'.
    save_path : str or Path, optional
        If provided, saves the plot to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplot(1, 2, figsize=(14,5))
    fig.suptitle(title, fontsize=13, fontweight ="bold")

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", "2.f"]
        ["Counts", "Normalised (row%)"]
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xlabel("Predicted labbel", fontsize = 10)
        ax.set_ylabel("True label", fontsize = 10)
        ax.set_title(subtitle, fontsize = 10)
        ax.tick_params(labelsize=9)
    
    plt.tight_layout

    print(f"\n {BOLD}Classification report {RESET}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    if save_path is not None:
        Path(save_path).parent.mkdir(Parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()
        plt.close(fig)


# =====================================================================
# ROC Curve
# =====================================================================

def plot_roc_curve(X_train, y_train, X_test, y_test):
    """ 
    Computes and plots the ROC curve for each class.
    """

    model = None 
    model.fit(X_train, y_train)

# Predict targets and their probabilities for the test set

    predictions = model.predict(X_test)
    y_hat_prob = model.predict_proba(X_test)[:, 1]

    # Compute and print performance

    print(classification_report(y_test, predictions))

# Compute ROC curve and AUC

    fpr, tpr, _ = roc_curve(y_test, y_hat_prob)
    roc_auc = auc(fpr, tpr)

# Plot ROC curves

    plt.figure(figsize=(8, 6))
    plt.plot(fpr,tpr, label = f'Roc curve (area = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],color = 'red', linestyle = '--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()

# =====================================================================
# Feature Importance - MDI
# =====================================================================

def plot_feature_importance_MDI(
        model,
        feature_names: list[str],
        top_n:         int = 20,
        title:         str = "Feature Importance (MDI)",
        save_path:     str | Path | None = None,
) -> pd.DataFrame:
    """
    Plot Mean Decrease in Impurity (MDI) feature importance from a fitted
    Random Forest or XGBoost model.

    Parameters
    ----------
    model : fitted sklearn Pipeline or estimator
        Must have a ``feature_importances_`` attribute, or be a Pipeline
        whose last step has one.
    feature_names : list of str
        Feature names matching the training columns.
    top_n : int
        Number of top features to show. Default is 20.
    title : str
        Plot title.
    save_path : str | Path | None
        If provided, saves the figure. Otherwise shows interactively.

    Returns
    -------
    pd.DataFrame
        Feature importance table sorted descending.
    """
    # Handle Pipeline vs raw estimator 
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    importances = clf.features_importances_

    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top = importance_df.head(top_n)

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#0E7490", alpha=0.8)
    ax.set_xlabel("Mean Decrease in Impurity", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()
        plt.close(fig)

    return importance_df

# =====================================================================
# Feature Importance - Permutation
# =====================================================================

def plot_feature_importance_permutation(
    model, 
    X_test:       pd.DataFrame,
    y_test,
    top_n:        int = 20,
    n_repeats:    int = 10,
    scoring:      str = "balanced_accuracy",
    title:        str = " Feature importance (permutation)",
    save_path:    str | Path | None = None,
)-> pd.DataFrame:

    """
    Compute and plot permutation feature importance on the test set.

    Parameters
    ----------
    model : fitted sklearn Pipeline or estimator
    X_test : pd.DataFrame
        Test features.
    y_test : array-like
        Test labels.
    top_n : int
        Number of top features to show. Default is 20.
    n_repeats : int
        Number of permutation repeats. Default is 10.
    scoring : str
        Scoring metric. Default is 'balanced_accuracy'.
    title : str
        Plot title.
    save_path : str | Path | None
        If provided, saves the figure. Otherwise shows interactively.

    Returns
    -------
    pd.DataFrame
        Permutation importance table sorted descending by mean importance.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats = n_repeats,
        random_state = 42,
        scoring=scoring,
        n_jobs =-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean":    result.importances_mean,
        "std":     result.importances_std,
    }).sort_values("mean", ascending = False).reset_index(drop=True)

    top = importance_df.head(top_n).iloc[::-1].reset_index(drop=True)

    # Plot feature importance

    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1))
    ax.barh(
        top["feature"], top["mean"],
        xerr=top["std"],
        color="#AA3377", alpha=0.8,
        error_kw=dict(elinewidth=0.8, capsize=3),
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Permutation importance ({scoring})", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()
        plt.close(fig)

    return importance_df

# =====================================================================
# Full model evaluation
# =====================================================================

def evaluate_model(
        model,
        X_test:         pd.DataFrame,
        y_test,
        class_names:    list[str],
        model_name:     str = "Model",
        top_n:          int = 20,
        save_dir:       str | Path | None = None,
) -> None:
    """
    Run all evaluation plots for one fitted model.

    Parameters
    ----------
    model : fitted sklearn Pipeline
    X_test : pd.DataFrame
    y_test : array-like
    class_names : list of str
    model_name : str
        Used in plot titles and filenames.
    top_n : int
        Top features to show. Default is 20.
    save_dir : str | Path | None
        If provided, saves all figures here. Otherwise shows interactively.
    """
    y_pred = model.predict(X_test)
    tag    =model_name.lower().replace(" ","_")

    def _path(name):
        return Path(save_dir) / f"{tag}_{name}.png" if save_dir else None
    
    plot_confusion_matrix(
        y_true      = y_test,
        y_pred      = y_pred,
        class_names = class_names,
        title       = f"{model_name} - Confusion Matrix",
        save_path   = _path("confusion_matrix"),
    )

    plot_feature_importance_MDI(
        model         = model,
        feature_names = list(X_test.colums), 
        top_n         = top_n,
        title         = f"{model_name} - Feature Importance (MDI)",
        save_path     = _path(f"feature_importance_mdi"),
    )

    plot_feature_importance_permutation(
        model = model,
        X_test = X_test,
        y_test = y_test,
        top_n = top_n,
        title = f"{model_name} - Feature Importance (Permutation)",
        save_path = _path("plot_feature_importance_permutation"),
    )