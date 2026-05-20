# Filename: evaluate.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Generates evaluation plots from K-fold CV results produced by train.py.
#              Saves one PDF per model containing a summary page, confusion matrix,
#              ROC curves, and MDI feature importance (RF and XGBoost only).

# Pipeline overview:
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Receive fold_results, predictions, summary, X, y from train.py
# 2. For each model:
#       a. Summary page  — mean ± std of all metrics across folds
#       b. Confusion matrix — aggregated y_true / y_pred across all folds
#       c. ROC curves — aggregated probabilities across all folds
#       d. MDI feature importance — refit model on full X to get importances
#          (RF and XGBoost only; skipped for LR and SVM)
# 3. Save one PDF per model to save_dir
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# ================================================================================
# Imports
# ================================================================================
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
)
from sklearn.preprocessing import label_binarize
from sklearn.impute import KNNImputer, SimpleImputer

# ================================================================================
# Constants
# ================================================================================

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

COLORs = [DTUNAVY, DTURED, DTUBRIGHTGREEN, DTUBLUE, DTUPURPLE, DTU_RED, DTUORANGE, DTUYELLOW, DTUGREEN, DTUPINK, DTUGREY]

BOLD  = "\033[1m"
GREEN = "\033[92m"
RESET = "\033[0m"


# ================================================================================
# Page 1 — Summary
# ================================================================================
def _plot_summary(
        model_name:   str,
        summary:      pd.DataFrame,
        class_names:  list[str],
        run_config:   dict | None = None,
        ) -> plt.Figure:
    """
    Summary page: run config and mean ± std metrics table across all models.

    Parameters
    ----------
    model_name : str
        Name of the model to highlight in the summary table.
    summary : pd.DataFrame
        Output of summarise() — mean ± std per model across folds.
    class_names : list[str]
        Human-readable class labels in label order.
    run_config : dict | None
        Optional config info to show on the summary page (e.g. mode, seed, imputer).
        Keys can include: 'mode', 'binary_mode', 'k_folds', 'seed', 'n_subjects',
        'n_features', 'imputer_strategy', 'classes'.
    """
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"{model_name}  —  K-fold CV evaluation",
                 fontsize=14, fontweight="bold", color=DTUNAVY, y=0.97)
 
    y_cursor = 0.88
    line_h   = 0.038
 
    def _section(title):
        nonlocal y_cursor
        fig.text(0.07, y_cursor, title, fontsize=10, fontweight="bold",
                 color=DTUNAVY, transform=fig.transFigure)
        y_cursor -= line_h * 0.6
        fig.add_artist(plt.Line2D([0.07, 0.93], [y_cursor, y_cursor],
                                  color=DTUNAVY, linewidth=0.5,
                                  transform=fig.transFigure))
        y_cursor -= line_h * 0.5
 
    def _row(label, val, bold_val=False):
        nonlocal y_cursor
        fig.text(0.08, y_cursor, label, fontsize=8, color="#444444",
                 transform=fig.transFigure)
        fig.text(0.42, y_cursor, val, fontsize=8,
                 fontweight="bold" if bold_val else "normal",
                 color=DTUNAVY if bold_val else "black",
                 transform=fig.transFigure)
        y_cursor -= line_h
 
    # ── 1. Run config ────────────────────────────────────────────────
    _section("1  Run configuration")
    if run_config:
        for label, val in [
            ("Mode",             run_config.get("mode", "—")),
            ("Binary mode",      run_config.get("binary_mode", "—")),
            ("Folds (K)",        str(run_config.get("k_folds", "—"))),
            ("Seed",             str(run_config.get("seed", "—"))),
            ("Subjects",         str(run_config.get("n_subjects", "—"))),
            ("Features",         str(run_config.get("n_features", "—"))),
            ("Imputer",          run_config.get("imputer_strategy", "—")),
            ("Classes",          ", ".join(class_names)),
        ]:
            _row(label, val)
    y_cursor -= 0.01
 
    # ── 2. CV summary table ──────────────────────────────────────────
    _section("2  CV summary  (mean ± std across folds)")
    col_x   = [0.08, 0.32, 0.46, 0.58, 0.70, 0.82]
    headers = ["Model", "Bal. accuracy", "Accuracy", "F1", "Precision", "Recall"]
    for hx, h in zip(col_x, headers):
        fig.text(hx, y_cursor, h, fontsize=8, fontweight="bold",
                 color="#444444", transform=fig.transFigure)
    y_cursor -= line_h
 
    for _, row in summary.iterrows():
        is_current = row["model"] == model_name
        marker = " ←" if is_current else ""
        vals = [
            row["model"] + marker,
            f"{row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f}",
            f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}",
            f"{row['f1_mean']:.3f} ± {row['f1_std']:.3f}",
            f"{row['precision_mean']:.3f} ± {row['precision_std']:.3f}",
            f"{row['recall_mean']:.3f} ± {row['recall_std']:.3f}",
        ]
        for hx, v in zip(col_x, vals):
            fig.text(hx, y_cursor, v, fontsize=7.5, color=DTUNAVY if is_current else "black",
                     transform=fig.transFigure)
        y_cursor -= line_h
 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
 
 
# ================================================================================
# Page 2 — Confusion matrix
# ================================================================================
def _plot_confusion_matrix(
        y_true,
        y_pred,
        class_names: list[str],
        title:       str = "Confusion matrix",
        ) -> plt.Figure:
    """
    Counts and normalised confusion matrix side by side.
    y_true / y_pred are aggregated across all CV folds.

    Parameters
    ----------
    y_true : array-like
        True labels (aggregated across all CV folds).
    y_pred : array-like
        Predicted labels (aggregated across all CV folds).
    class_names : list[str]
        Human-readable class labels in label order.
    title : str
        Title for the confusion matrix page.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", color=DTUNAVY)
 
    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Normalised (row %)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    linewidths=0.5, 
                    ax=ax)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.set_title(subtitle, fontsize=10)
        ax.tick_params(labelsize=9)
 
    plt.tight_layout()
 
    print(f"\n{BOLD}Classification report — {title}{RESET}")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, zero_division=0))
    return fig
 
 
# ================================================================================
# Page 3 — ROC curves
# ================================================================================
def _plot_roc_curves(
        y_true,
        y_prob,
        class_names: list[str],
        title:       str = "ROC curves",
        ) -> plt.Figure:
    """
    One-vs-rest ROC curve per class, aggregated across all CV folds.
    Works for binary and multiclass.

    Parameters
    ----------
    y_true : array-like
        True labels (aggregated across all CV folds).
    y_prob : array-like
        Predicted probabilities (aggregated across all CV folds). Shape (n_samples, n_classes).
    class_names : list[str]
        Human-readable class labels in label order.
    title : str
        Title for the ROC curves page.
    """
     # ---- 1) Binarize labels for one-vs-rest ROC computation ---
    n_classes  = len(class_names)
    y_bin      = label_binarize(y_true, classes=list(range(n_classes)))

    # Binary case: label_binarize returns shape (n, 1) — expand to (n, 2)
    if n_classes == 2:
        y_bin  = np.hstack([1 - y_bin, y_bin])


    # ---- 2) Compute and plot ROC curve for each class ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip(class_names, COLORs)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])                                             # x-axis limit           
    ax.set_ylim([0.0, 1.02])                                            # y-axis limit
    ax.set_xlabel("False Positive Rate", fontsize=10)                   # x-axis label
    ax.set_ylabel("True Positive Rate",  fontsize=10)                   # y-axis label
    ax.set_title(title, fontsize=12, fontweight="bold", color=DTUNAVY)  # title
    ax.legend(loc="lower right", fontsize=9)                            # legend
    ax.grid(alpha=0.3, linestyle="--")                                  # grid 
    plt.tight_layout()

    return fig
 
 
# ================================================================================
# Page 4 — MDI feature importance
# ================================================================================
def _plot_feature_importance_mdi(
        model,
        feature_names: list[str],
        top_n:         int = 20,
        title:         str = "Feature importance (MDI)",
        ) -> plt.Figure | None:
    """
    MDI (mean decrease in impurity) feature importance bar chart.
    Returns None if the model doesn't support feature_importances_
    (e.g. Logistic Regression, SVM).
    """
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "feature_importances_"):
        return None
 
    importance_df = (
        pd.DataFrame({"feature":    feature_names,
                      "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
 
    top = importance_df.head(top_n).iloc[::-1]
 
    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1))
    ax.barh(top["feature"], top["importance"], color=DTURED, alpha=0.8)
    ax.set_xlabel("MDI importance", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=DTUNAVY)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig
 
 
# ================================================================================
# Public entry point
# ================================================================================
def evaluate_model(
        model_name:       str,
        predictions:      dict,
        summary:          pd.DataFrame,
        X:                pd.DataFrame,
        y:                pd.Series,
        class_names:      list[str],
        mode:             str = "binary",
        top_n:            int = 20,
        imputer_strategy: str = "knn",
        seed:             int = 42,
        run_config:       dict | None = None,
        save_dir:         str | Path = "reports/evaluation",
        ) -> Path | None:
    """
    Generate and save a PDF evaluation report for one model.
 
    Parameters
    ----------
    model_name : str
        Name of the model to evaluate (must be a key in predictions).
    predictions : dict
        Output of cross_validate() — per-model dict with 'y_true', 'y_pred', 'y_prob'.
    summary : pd.DataFrame
        Output of summarise() — mean ± std per model across folds.
    X : pd.DataFrame
        Full feature matrix (used to refit model for MDI).
    y : pd.Series
        Full label vector (used to refit model for MDI).
    class_names : list[str]
        Human-readable class labels in label order.
    mode : str
        'binary' or 'multiclass'.\\
        Used to determine which models support MDI importance (RF and XGBoost only in binary).\\
        Default 'binary'. 
    top_n : int
        Number of top features for MDI plot.\\
        Default 20.
    imputer_strategy : str
        Imputation strategy used during CV (replicated here for MDI refit).
    seed : int
        Random seed.\\
        Default 42.
    run_config : dict | None
        Config info shown on the summary page.
    save_dir : str | Path
        Output directory for the PDF.\\
        Default 'reports/evaluation'.
 
    Returns
    -------
    Path | None
        Path to the saved PDF, or None if save_dir is None.
    """
    from ml.train import get_models  # local import to avoid circular dependency
 
    if model_name not in predictions:
        print(f"  [SKIP] '{model_name}' not found in predictions.")
        return None
 
    preds      = predictions[model_name]
    y_true     = preds["y_true"]
    y_pred     = preds["y_pred"]
    y_prob     = preds["y_prob"]   # None for models without predict_proba
 
    print(f"\n{BOLD}Evaluating: {model_name}{RESET}")
 
    figs: list[plt.Figure] = []
 
    # ---- Page 1: summary ----
    figs.append(_plot_summary(
        model_name  = model_name,
        summary     = summary,
        class_names = class_names,
        run_config  = run_config,
    ))
 
    # ---- Page 2: confusion matrix ----
    figs.append(_plot_confusion_matrix(
        y_true      = y_true,
        y_pred      = y_pred,
        class_names = class_names,
        title       = f"{model_name}  —  Confusion matrix (aggregated CV folds)",
    ))
 
    # ---- Page 3: ROC curves ----
    if y_prob is not None:
        figs.append(_plot_roc_curves(
            y_true      = y_true,
            y_prob      = y_prob,
            class_names = class_names,
            title       = f"{model_name}  —  ROC curves (aggregated CV folds)",
        ))
    else:
        print(f"  [SKIP] ROC curves — {model_name} has no predict_proba")
 
    # ---- Page 4: MDI feature importance (refit on full data) ----
    model_specs = get_models(seed=seed, mode=mode)
    if model_name in model_specs:
        pipeline = model_specs[model_name]
 
        # Impute full dataset before refit
        if imputer_strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=imputer_strategy)
 
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        pipeline.fit(X_imp, y)
 
        mdi_fig = _plot_feature_importance_mdi(
            model         = pipeline,
            feature_names = list(X.columns),
            top_n         = top_n,
            title         = f"{model_name}  —  Feature importance MDI (refit on all data)",
        )
        if mdi_fig is not None:
            figs.append(mdi_fig)
        else:
            print(f"  [SKIP] MDI importance — not available for {model_name}")
 
    # ---- Save PDF ----
    if save_dir is None:
        for fig in figs:
            plt.show()
            plt.close(fig)
        return None
 
    out_dir   = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag       = model_name.lower().replace(" ", "_")
    pdf_path  = out_dir / f"{tag}_{timestamp}.pdf"
 
    with pdf_backend.PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        meta            = pdf.infodict()
        meta["Title"]   = f"{model_name} — evaluation report"
        meta["Author"]  = "EOG_REM pipeline"
 
    print(f"{GREEN}{BOLD}Saved → {pdf_path}{RESET}")
    return pdf_path
 
 
# ================================================================================
# Comparison PDF — all binary modes × all models
# ================================================================================
def _plot_comparison_roc_per_model(
        model_name:      str,
        binary_results:  dict,
        ax:              plt.Axes,
        ) -> None:
    """
    Plot ROC curves for all binary modes for a single model onto ax.
    binary_results: {binary_mode: result_dict from run_training}

    Parameters
    ----------
    model_name : str
        Name of the model to plot (must be a key in each result_dict["predictions"]).
    binary_results : dict
        {binary_mode: result_dict} where each result_dict is the output of run_training() for that binary mode.
    ax : plt.Axes
        Matplotlib Axes to plot on.
    """
    for color, (bm, result) in zip(COLORs, binary_results.items()):
        preds  = result["predictions"].get(model_name)
        if preds is None or preds["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_prob"][:, 1])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{bm}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_title(model_name, fontsize=10, fontweight="bold", color=DTUNAVY)
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate",  fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
 
 
def _plot_comparison_roc_per_task(
        binary_mode:     str,
        binary_results:  dict,
        ax:              plt.Axes,
        ) -> None:
    """
    Plot ROC curves for all models for a single binary mode onto ax.

    Parameters
    ----------
    binary_mode : str
        Binary mode to plot (must be a key in binary_results).
    binary_results : dict
        {binary_mode: result_dict} where each result_dict is the output of run_training() for that binary mode.
    ax : plt.Axes
        Matplotlib Axes to plot on.
    """
    result = binary_results[binary_mode]
    for color, (model_name, preds) in zip(COLORs, result["predictions"].items()):
        if preds["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_prob"][:, 1])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{model_name}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_title(binary_mode, fontsize=10, fontweight="bold", color=DTUNAVY)
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate",  fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
 
 
def _plot_comparison_bar(binary_results: dict) -> plt.Figure:
    """
    Grouped bar chart: balanced accuracy mean ± std for all models * all binary modes.
    """
    model_names  = list(next(iter(binary_results.values()))["summary"]["model"])
    binary_modes = list(binary_results.keys())
    n_models     = len(model_names)
    n_modes      = len(binary_modes)
 
    x      = np.arange(n_models)
    width  = 0.8 / n_modes
 
    fig, ax = plt.subplots(figsize=(12, 5))
 
    for i, (bm, color) in enumerate(zip(binary_modes, COLORs)):
        summary = binary_results[bm]["summary"]
        means   = []
        stds    = []
        for m in model_names:
            row = summary[summary["model"] == m]
            means.append(row["balanced_accuracy_mean"].values[0] if len(row) else 0)
            stds.append(row["balanced_accuracy_std"].values[0]   if len(row) else 0)
        offset = (i - n_modes / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=bm, color=color, capsize=4)
 
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Balanced accuracy (mean ± std)", fontsize=10)
    ax.set_title("Model comparison across binary tasks", fontsize=13, fontweight="bold", color=DTUNAVY)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig
 
 
def evaluate_binary_comparison(
        binary_results: dict,
        save_dir:       str | Path = "reports/evaluation",
        ) -> Path:
    """
    Generate a comparison PDF across all binary modes.
 
    Parameters
    ----------
    binary_results : dict
        {binary_mode: result_dict} where each result_dict is the output
        of run_training() for that binary mode.
    save_dir : str | Path
        Output directory.
 
    Returns
    -------
    Path
        Path to the saved PDF.
    """
    model_names  = list(next(iter(binary_results.values()))["predictions"].keys())
    binary_modes = list(binary_results.keys())
    figs         = []
 
    # ---- Page 1: ROC per model (all binary tasks) ----
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    fig.suptitle("ROC curves per model — all binary tasks", fontsize=13, fontweight="bold", color=DTUNAVY)
    for ax, model_name in zip(axes, model_names):
        _plot_comparison_roc_per_model(model_name, binary_results, ax)
    plt.tight_layout()
    figs.append(fig)
 
    # ---- Page 2: ROC per binary task (all models) ----
    n_modes = len(binary_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5))
    if n_modes == 1:
        axes = [axes]
    fig.suptitle("ROC curves per binary task — all models", fontsize=13, fontweight="bold", color=DTUNAVY)
    for ax, bm in zip(axes, binary_modes):
        _plot_comparison_roc_per_task(bm, binary_results, ax)
    plt.tight_layout()
    figs.append(fig)
 
    # ---- Page 3: bar chart ----
    figs.append(_plot_comparison_bar(binary_results))
 
    # ---- Save ----
    out_dir   = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path  = out_dir / f"binary_comparison_{timestamp}.pdf"
 
    with pdf_backend.PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        meta          = pdf.infodict()
        meta["Title"] = "Binary mode comparison"
 
    print(f"{GREEN}{BOLD}Saved → {pdf_path}{RESET}")
    return pdf_path
 
 
# ================================================================================
# Convenience — evaluate all models at once
# ================================================================================
def evaluate_all(
        predictions:      dict,
        summary:          pd.DataFrame,
        X:                pd.DataFrame,
        y:                pd.Series,
        class_names:      list[str],
        mode:             str = "binary",
        top_n:            int = 20,
        imputer_strategy: str = "knn",
        seed:             int = 42,
        run_config:       dict | None = None,
        save_dir:         str | Path = "reports/evaluation",
        binary_results:   dict | None = None,
        ) -> dict[str, Path | None]:
    """
    Run evaluate_model for every model in predictions, then generate a
    binary comparison PDF if mode='binary' and binary_results is provided.
 
    Parameters
    ----------
    binary_results : dict | None
        {binary_mode: run_training result} for all binary modes run.
        If provided and mode='binary', generates an additional comparison PDF.
 
    Returns
    -------
    dict[str, Path | None]
        Mapping of model name to saved PDF path.
    """
    results = {}
    for model_name in predictions:
        results[model_name] = evaluate_model(
            model_name       = model_name,
            predictions      = predictions,
            summary          = summary,
            X                = X,
            y                = y,
            class_names      = class_names,
            mode             = mode,
            top_n            = top_n,
            imputer_strategy = imputer_strategy,
            seed             = seed,
            run_config       = run_config,
            save_dir         = save_dir,
        )
 
    return results
 