# Filename: train.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Trains a machine learning model on the extracted features and patient labels.
#              Supports binary and multi-class classification with cross-validation.

# Usage:
#       python -m ml.train features_csv/features.csv --mode binary
#       python -m ml.train features_csv/features.csv --mode multiclass

# ================================================================================
# Imports
# ================================================================================
from __future__ import annotations

import numpy as np          # for numerical operations
import pandas as pd         # for data manipulation
from pathlib import Path    # for handling file paths

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

#XGBoost imports 
import xgboost as xgb
from xgboost import XGBClassifier


from ml.prepare_data import prepare
from ml.evaluate import evaluate_model


# ================================================================================
# Constants
# ================================================================================
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
 
DEFAULT_SEED = 42

# ================================================================================
# Model definitions
# ================================================================================
def get_models(seed: int = DEFAULT_SEED, mode: str = "binary") -> dict[str, Pipeline]:
    """
    Return a dictionary of named model pipelines.
 
    Each pipeline includes StandardScaler followed by the classifier.
    This ensures models that need scaled features (LogReg, SVM)
    get them automatically.
 
    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
 
    Returns
    -------
    dict[str, Pipeline]
        Mapping of model name to sklearn Pipeline. 
    """

    # XGBoost params depend on mode
    if mode == "binary":
        xgb_params = dict(
            objective    = "binary:logistic",
            eval_metric  = "logloss",
            n_estimators = 100,
            max_depth    = 6,
            learning_rate= 0.1,
            random_state = seed,
        )
    else:
        xgb_params = dict(
            objective    = "multi:softprob",
            eval_metric  = "mlogloss",
            n_estimators = 100,
            max_depth    = 6,
            learning_rate= 0.1,
            random_state = seed,
        )

    models = {
        # --- Random Forest ---
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
        # --- Logistic Regression ---
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=seed,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        # --- Support Vector Machine ---
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                random_state=seed,
                class_weight="balanced",
                probability=True,
            )),
        ]),
        # --- XGBoost ---
        # NOTE: XGBoost does not support class_weight="balanced".
        # Class imbalance is handled via scale_pos_weight (binary)
        # or compute_sample_weight (multiclass) passed to fit().
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(**xgb_params)),
        ]),
    }
    
    return models
 
 
# ================================================================================
# Cross-validation
# ================================================================================
def cross_validate_models(
        X_train:    pd.DataFrame,
        y_train:    pd.Series,
        n_folds:    int = 5,
        seed:       int = DEFAULT_SEED,
        mode:       str = "binary"
        ) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation on all models.
 
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    n_folds : int
        Number of CV folds.\\
        Default is 5.
    seed : int
        Random seed.\\
        Default is 42.
 
    Returns
    -------
    pd.DataFrame
        One row per model with mean and std of each metric across folds.
    """
    # ---- 1) Get models and CV splitter ----
    models = get_models(seed, mode=mode)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # ---- 2) Run cross-validation for each model ----
    scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
 
    print(f"\n{BOLD}Cross-validation ({n_folds}-fold){RESET}")
    print(f"{'='*60}")
    
    results = []
    for name, pipeline in models.items():
        print(f"\n  Training: {BOLD}{name}{RESET} ...", end=" ", flush=True)
 
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
 
        row = {"model": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"{metric}_mean"] = np.mean(cv_results[key])
            row[f"{metric}_std"]  = np.std(cv_results[key])
 
        results.append(row)
 
        acc  = row["accuracy_mean"]
        f1   = row["f1_weighted_mean"]
        print(f"{GREEN}done{RESET}  |  accuracy: {acc:.3f}  |  F1: {f1:.3f}")
 
    results_df = pd.DataFrame(results)
 
    # ---- 3) Print summary table ----
    print(f"\n{BOLD}Cross-validation summary{RESET}")
    print(f"{'-'*60}")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'F1':>10s} {'Precision':>10s} {'Recall':>10s}")
    print(f"  {'-'*65}")
    for _, r in results_df.iterrows():
        print(f"  {r['model']:<25s} "
              f"{r['accuracy_mean']:>7.3f}±{r['accuracy_std']:.3f} "
              f"{r['f1_weighted_mean']:>5.3f}±{r['f1_weighted_std']:.3f} "
              f"{r['precision_weighted_mean']:>5.3f}±{r['precision_weighted_std']:.3f} "
              f"{r['recall_weighted_mean']:>5.3f}±{r['recall_weighted_std']:.3f}")
 
    return results_df
 
 
# ================================================================================
# Test set evaluation
# ================================================================================
def evaluate_on_test(
        X_train:    pd.DataFrame,
        y_train:    pd.Series,
        X_test:     pd.DataFrame,
        y_test:     pd.Series,
        seed:       int = DEFAULT_SEED,
        mode:       str = "binary"
        ) -> pd.DataFrame:
    """
    Train all models on the full training set and evaluate on the test set.
 
    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and test features.
    y_train, y_test : pd.Series
        Training and test labels.
    seed : int
        Random seed.
 
    Returns
    -------
    pd.DataFrame
        One row per model with test set metrics.
    """
    # ---- 1) Get models ----
    models = get_models(seed, mode=mode)
 
    print(f"\n{BOLD}Test set evaluation{RESET}")
    print(f"{'='*60}")

    # ---- 2) Train each model on full training set and evaluate on test set ----
    results = []
    for name, pipeline in models.items():
        print(f"\n  {BOLD}{name}{RESET}")
 
        # XGBoost needs sample_weight for class imbalance instead of class_weight
        if name == "XGBoost":
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight("balanced", y_train)
            pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
 
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
 
        results.append({
            "model": name,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
        })
 
        print(f"    Accuracy:  {acc:.3f}")
        print(f"    F1:        {f1:.3f}")
        print(f"    Precision: {prec:.3f}")
        print(f"    Recall:    {rec:.3f}")
 
        # Classification report
        print(f"\n    Classification report:")
        report = classification_report(y_test, y_pred, zero_division=0)
        for line in report.split("\n"):
            print(f"    {line}")
 
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"    Confusion matrix:")
        print(f"    {cm}")
 
    return pd.DataFrame(results)
 
 
# ================================================================================
# Feature importance (for Random Forest)
# ================================================================================
def get_feature_importance(
        X_train:    pd.DataFrame,
        y_train:    pd.Series,
        top_n:      int = 20,
        seed:       int = DEFAULT_SEED,
        ) -> pd.DataFrame:
    """
    Train a Random Forest and return feature importances.
 
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    top_n : int
        Number of top features to show. Default is 20.
    seed : int
        Random seed.
 
    Returns
    -------
    pd.DataFrame
        Feature importance table sorted by importance (descending).
    """
    # ---- 1) Train Random Forest ----
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    
    # ---- 2) Fit model ----
    rf.fit(X_train, y_train)
    
    # --- 3) Extract feature importances ----
    importances = rf.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
 
    print(f"\n{BOLD}Top {top_n} features (Random Forest importance){RESET}")
    print(f"{'-'*50}")
    for _, r in importance_df.head(top_n).iterrows():
        bar = "#" * int(r["importance"] * 200)
        print(f"  {r['feature']:<40s}  {r['importance']:.4f}  {bar}")
 
    return importance_df

# ================================================================================
# Feature importance (for XGBoost)
# ================================================================================
 
def get_xgb_feature_importance(
        X_train:    pd.DataFrame,
        y_train:    pd.Series,
        top_n:      int = 20,
        seed:       int = DEFAULT_SEED,
        mode:       str = "binary",
) -> pd.DataFrame:
    """
    Train an XGBoost model and return feature importances.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    top_n : int
        Number of top features to show. Default is 20.
    seed : int
        Random seed. Default is 42.
    mode : str
        'binary' or 'multiclass'. Default is 'binary'.

    Returns
    -------
    pd.DataFrame
        Feature importance table sorted by importance (descending).
    """
    # ---- 1) Build XGBoost pipeline ----
    if mode == "binary":
        xgb_params = dict(
            objective     = "binary:logistic",
            eval_metric   = "logloss",
            n_estimators  = 100,
            max_depth     = 6,
            learning_rate = 0.1,
            random_state  = seed,
        )
    else:
        xgb_params = dict(
            objective     = "multi:softprob",
            eval_metric   = "mlogloss",
            n_estimators  = 100,
            max_depth     = 6,
            learning_rate = 0.1,
            random_state  = seed,
        )

    xgb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(**xgb_params)),
    ])

    # ---- 2) Fit with sample weights to handle class imbalance ----
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weight = compute_sample_weight("balanced", y_train)
    xgb_pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight)

    # ---- 3) Extract feature importances ----
    importances  = xgb_pipeline.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n{BOLD}Top {top_n} features (XGBoost importance){RESET}")
    print(f"{'-'*50}")
    for _, r in importance_df.head(top_n).iterrows():
        bar = "#" * int(r["importance"] * 200)
        print(f"  {r['feature']:<40s}  {r['importance']:.4f}  {bar}")

    return importance_df
 
# ================================================================================
# Full training pipeline
# ================================================================================
def run_training(
        feature_csv:    str | Path,
        mode:           str = "binary",
        binary_mode:    str = "control_vs_all",
        test_size:      float = 0.2,
        n_folds:        int = 5,
        seed:           int = DEFAULT_SEED,
        drop_nan:       bool = True,
        ) -> dict:
    
    """
    Full training pipeline: prepare data, cross-validate, evaluate on test set,
    and extract feature importances.
 
    Parameters
    ----------
    feature_csv : str | Path
        Path to features.csv.
    mode : str
        'binary' or 'multiclass'.
    binary_mode : str
        'control_vs_all' or 'control_vs_irbd'. Only used if mode='binary'.
    test_size : float
        Fraction for test set. Default 0.2.
    n_folds : int
        Number of CV folds. Default 5.
    seed : int
        Random seed. Default 42.
    drop_nan : bool
        Drop subjects with NaN features. Default True.
 
    Returns
    -------
    dict
        Dictionary with keys: 'cv_results', 'test_results', 'feature_importance',
        'X_train', 'X_test', 'y_train', 'y_test'.
    """
    print(f"\n{'='*60}")
    print(f"  {BOLD}RBD Classification Pipeline{RESET}")
    print(f"  Mode: {mode}" + (f" ({binary_mode})" if mode == "binary" else ""))
    print(f"{'='*60}")
 
    # ---- 1) Prepare data ----
    X_train, X_test, y_train, y_test = prepare(
        feature_csv=feature_csv,
        mode=mode,
        binary_mode=binary_mode,
        test_size=test_size,
        seed=seed,
        drop_nan=drop_nan,
    )
 
    # ---- 2) Cross-validation ----
    cv_results = cross_validate_models(X_train, y_train, n_folds=n_folds, seed=seed, mode=mode)
 
    # ---- 3) Test set evaluation ----
    test_results = evaluate_on_test(X_train, y_train, X_test, y_test, seed=seed, mode=mode)
 
    # ---- 4) Feature importance ----
    importance_df = get_feature_importance(X_train, y_train, seed=seed)
    xgb_importance_df = get_xgb_feature_importance(X_train, y_train, seed=seed, mode=mode)
 
    print(f"\n{'='*60}")
    print(f"  {GREEN}Training complete{RESET}")
    print(f"{'='*60}\n")
 
    # ---- 5) Evaluation plots ----
    
    print(f"\n{BOLD}Generating evaluation plots{RESET}")
    print(f"{'='*60}")

    # Determine class names from y_test
    class_names = sorted(y_test.unique().tolist())

    # Re-fit each model on full training set and evaluate
    for name, pipeline in get_models(seed=seed, mode=mode).itmes():
        print(f"\n Evaluating: {BOLD}{name}{RESET}")
        pipeline.fit(X_train, y_train)
        evaluate_model(
            model        = pipeline,
            X_test       = X_test,
            y_test       = y_test,
            class_names  = class_names,
            model_name   = name,
            top_n        = 20,
            save_dir     = "reports/evaluation"      
        )

    return {
        "cv_results": cv_results,
        "test_results": test_results,
        "feature_importance": importance_df,
        "xgb_feature_importance": xgb_importance_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "evaluation_dir": Path("reports/evaluation"),
    }
 
 
# ================================================================================
# CLI - Command line interface
# ================================================================================
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Train classification models for RBD detection.")
    parser.add_argument("feature_csv", type=str, help="Path to features.csv")
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "multiclass"],
                        help="Classification mode (default: binary)")
    parser.add_argument("--binary-mode", type=str, default="control_vs_all",
                        choices=["control_vs_all", "control_vs_irbd"],
                        help="Binary split mode (default: control_vs_all)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default: 0.2)")
    parser.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
 
    args = parser.parse_args()
 
    run_training(
        feature_csv=args.feature_csv,
        mode=args.mode,
        binary_mode=args.binary_mode,
        test_size=args.test_size,
        n_folds=args.folds,
        seed=args.seed,
    )