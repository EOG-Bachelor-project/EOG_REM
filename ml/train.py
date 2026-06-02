# Filename: train.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: K-fold and nested (2-layer) cross-validation of RBD classifiers on
#              EOG-derived features. Imputation is fit on the training fold and applied
#              to the validation fold to avoid data leakage. Reports mean ± std of
#              metrics across folds.
#
# Usage:
#   K-fold CV only:
#   python -m ml.train features_csv/features.csv --method kfold --mode binary --binary-mode control_vs_irbd
#
#   Nested CV only:
#   python -m ml.train features_csv/features.csv --method nested --mode binary --binary-mode control_vs_irbd
#
#   CV + evaluation PDFs:
#   python -m ml.train features_csv/features.csv --method kfold --mode binary --binary-mode control_vs_irbd --evaluate
#
#   Multiple modes with evaluation:
#   python -m ml.train features_csv/features.csv --method kfold --mode binary multiclass --binary-mode control_vs_all control_vs_irbd control_vs_pd --k-folds 5 --evaluate
#   python -m ml.train features_csv/features.csv --method nested --mode binary multiclass --binary-mode control_vs_all control_vs_irbd control_vs_pd --k-outer 5 --k-inner 5 --evaluate
#
#   Custom output dirs:
#   python -m ml.train features_csv/features.csv --method nested --evaluate --save-dir reports/cv --eval-save-dir reports/evaluation

# Pipeline overview:
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Load features from CSV and assign diagnostic group labels
# 2. Filter to subjects valid for the chosen classification task
# 3a. K-fold CV (--method kfold):
#       For each fold (K=5):
#         a. Split data into training (4/5) and validation (1/5) folds
#         b. Fit imputer on training fold, transform both folds
#         c. Train each model on the training fold (default hyperparameters)
#         d. Evaluate on the validation fold
# 3b. Nested CV (--method nested):
#       For each outer fold (K1=5):
#         a. Split data into outer training (4/5) and validation (1/5) folds
#         b. Fit imputer on outer training fold, transform both folds
#         c. For each model, run RandomizedSearchCV on outer training fold (K2=5 inner folds)
#         d. Evaluate best model from inner CV on outer validation fold
# 4. Aggregate results: mean ± std per model across folds
# 5. Save fold-level results and summary to CSV
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# ================================================================================
# Imports
# ================================================================================
from __future__ import annotations                      # for Python 3.10+ type hinting (e.g. list[str])
 
import numpy as np                                      # for any numeric operations if needed in the future
import pandas as pd                                     # for data manipulation and loading CSVs
from pathlib import Path                                # for handling file paths in a platform-independent way
 
from sklearn.ensemble import RandomForestClassifier     # for the Random Forest model
from sklearn.linear_model import LogisticRegression     # for the Logistic Regression model
from sklearn.svm import SVC                             # for the Support Vector Machine model  
from sklearn.impute import KNNImputer, SimpleImputer    # for imputation strategies
from sklearn.preprocessing import StandardScaler        # for feature scaling 
from sklearn.pipeline import Pipeline                   # for creating model pipelines
from sklearn.model_selection import (
    RandomizedSearchCV,                                 # for hyperparameter tuning
    StratifiedKFold,                                    # for stratified K-fold cross-validation
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier                       # for the XGBoost model
 
from ml.prepare_data import prepare                     # for loading features and assigning labels
 
# ================================================================================
# Constants
# ================================================================================
 
# ANSI escape codes for colored terminal output
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RESET  = "\033[0m"
 
DEFAULT_SEED = 42
 
# ================================================================================
# Label map
# ================================================================================
def _label_map(mode: str, binary_mode: str | None = None) -> dict[int, str]:
    if mode == "multiclass":
        return {0: "Control", 1: "iRBD", 2: "PD(-RBD)", 3: "PD(+RBD)"}
    # binary
    if binary_mode == "control_vs_all":
        return {0: "Control", 1: "iRBD + PD(+RBD)"}
    if binary_mode == "control_vs_irbd":
        return {0: "Control", 1: "iRBD"}
    if binary_mode == "control_vs_pd":
        return {0: "Control", 1: "PD(+RBD)"}
    return {0: "Control", 1: "Disease"}
 
 
# ================================================================================
# Model definitions (default hyperparameters)
# ================================================================================
def get_models(
        seed: int = DEFAULT_SEED, 
        mode: str = "binary"
        ) -> dict[str, Pipeline]:
    """
    Return a dict of named sklearn Pipelines with default hyperparameters. \\
    Each pipeline: StandardScaler -> classifier.
 
    Parameters
    ----------
    seed : int
        Random seed for reproducibility.\\
        *Default 42*
    mode : str
        'binary' or 'multiclass'. Determines the XGBoost objective and metric. \\
        *Default 'binary'*
 
    Returns
    -------
    dict[str, Pipeline]
        Dictionary mapping model names to sklearn Pipelines.
    """
    xgb_objective = "binary:logistic" if mode == "binary" else "multi:softprob"
    xgb_metric    = "logloss"
 
    return {
        # ——— Logistic Regression ————————————————————————
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),   # z-score normalization: zero mean, unit variance
            ("clf", LogisticRegression( 
                C=1.0,                      # hyperparameter: regularization strength (smaller = stronger regularization)
                max_iter=2000,              # maximum iterations for convergence
                random_state=seed,          # random seed for reproducibility
                class_weight="balanced",    # upweight minority classes inversely proportional to frequency
                solver="lbfgs",             # optimization algorithm
            )),
        ]),
        # ——— SVM ————————————————————————————————————————
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=1.0,                     # hyperparameter: regularization strength
                kernel="rbf",              # hyperparameter: kernel type
                random_state=seed,         # random seed for reproducibility
                class_weight="balanced",   # upweight minority classes inversely proportional to frequency
                probability=True,          # enable probability estimates (required for some metrics e.g. AUC)
            )),
        ]),
        # ——— Random Forest ——————————————————————————————
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,           # hyperparameter: number of trees in the forest
                max_depth=None,             # hyperparameter: maximum depth of each tree (None = unlimited)
                random_state=seed,          # random seed for reproducibility
                class_weight="balanced",    # upweight minority classes inversely proportional to frequency
            )),
        ]),
        # ——— XGBoost ————————————————————————————————————
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=100,           # hyperparameter: number of boosting rounds
                learning_rate=0.1,          # hyperparameter: step size shrinkage to prevent overfitting
                max_depth=6,                # hyperparameter: maximum tree depth
                objective=xgb_objective,    # objective function (binary logistic or softmax for multiclass)
                eval_metric=xgb_metric,     # evaluation metric (log loss)
                random_state=seed,          # random seed for reproducibility
                verbosity=0,                # silent
            )),
        ]),
    }
# ================================================================================
# Hyperparameter search spaces (for future use with RandomizedSearchCV)
# ================================================================================
def get_search_spaces(
        seed: int = DEFAULT_SEED, 
        mode: str = "binary"
    ) -> dict:
    """
    Return model pipelines with hyperparameter search spaces for 2-layer (nested) CV.

    Each entry contains a pipeline (StandardScaler -> classifier) and a
    param_dist dict sampled by RandomizedSearchCV in the inner CV loop.

    Parameters
    ----------
    seed : int
        Random seed. *Default 42*.
    mode : str
        'binary' or 'multiclass'. Determines XGBoost objective. *Default 'binary'*.

    Returns
    -------
    dict
        {model_name: {"pipeline": Pipeline, "param_dist": dict}}
    """
    # --- Set the loss function ---
    xgb_objective = "binary:logistic" if mode == "binary" else "multi:softprob"
    # NOTE: Binary classification uses logistic regression output, multiclass uses softmax probabilities.

    return {
        # ——— Logistic Regression ————————————————————————
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),                   # z-score normalization: zero mean, unit variance
                ("clf", LogisticRegression(
                    max_iter=2000,                              # maximum iterations for convergence
                    random_state=seed,                          # random seed for reproducibility
                    class_weight="balanced",                    # upweight minority classes inversely proportional to frequency
                    solver="lbfgs",                             # optimization algorithm
                )),
            ]),
            "param_dist": {
                "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], # regularization strength (smaller = stronger regularization)
            },
        },
        # ——— SVM ————————————————————————————————————————
        "SVM": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),                   # z-score normalization: zero mean, unit variance
                ("clf", SVC(
                    random_state=seed,                          # random seed for reproducibility
                    class_weight="balanced",                    # upweight minority classes inversely proportional to frequency
                    probability=True,                           # enable probability estimates (required for AUC)
                )),
            ]),
            "param_dist": {
                "clf__C":      [0.1, 1.0, 10.0, 100.0],         # regularization strength
                "clf__gamma":  ["scale", "auto", 0.01, 0.001],  # kernel coefficient for RBF
                "clf__kernel": ["rbf", "linear"],               # kernel type
            },
        },
        # ——— Random Forest ——————————————————————————————
        "Random Forest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),                   # z-score normalization: zero mean, unit variance
                ("clf", RandomForestClassifier(
                    random_state=seed,                          # random seed for reproducibility
                    class_weight="balanced",                    # upweight minority classes inversely proportional to frequency
                )),
            ]),
            "param_dist": {
                "clf__n_estimators": [50, 100, 200],            # number of trees in the forest
                "clf__max_depth":    [None, 3, 5, 10],          # maximum depth of each tree (None = unlimited)
            },
        },
        # ——— XGBoost ————————————————————————————————————
        "XGBoost": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),                   # z-score normalization: zero mean, unit variance
                ("clf", XGBClassifier(
                    objective=xgb_objective,                    # loss function (binary logistic or softmax)
                    eval_metric="logloss",                      # evaluation metric
                    random_state=seed,                          # random seed for reproducibility
                    verbosity=0,                                # silent
                )),
            ]),
            "param_dist": {
                "clf__n_estimators":  [50, 100, 200],           # number of boosting rounds
                "clf__max_depth":     [3, 5, 6, 8],             # maximum tree depth
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],   # step size shrinkage to prevent overfitting
                "clf__subsample":     [0.7, 0.8, 1.0],          # subsample ratio of training instances
            },
        },
    }
 
# ================================================================================
# K-fold cross-validation
# ================================================================================
def cross_validate(
        X:                  pd.DataFrame,
        y:                  pd.Series,
        feature_cols:       list[str],
        k_folds:            int = 5,
        seed:               int = DEFAULT_SEED,
        mode:               str = "binary",
        imputer_strategy:   str = "knn",
        ) -> pd.DataFrame:
    """
    Stratified K-fold cross-validation.
 
    For each fold:
      1. Fit imputer on training fold, transform both train and val.
      2. Train each model on the training fold.
      3. Evaluate on the validation fold.
 
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (all subjects).
    y : pd.Series
        Label vector.
    feature_cols : list[str]
        Feature column names.
    k_folds : int
        Number of CV folds. *Default 5*.
    seed : int
        Random seed. *Default 42*.
    mode : str
        'binary' or 'multiclass'. *Default 'binary'*.
    imputer_strategy : str
        'knn', 'median', 'mean', or 'most_frequent'. *Default 'knn'*.
 
    Returns
    -------
    fold_results : pd.DataFrame
        One row per (model, fold) with columns: model, fold, accuracy,
        balanced_accuracy, f1, precision, recall.
    predictions : dict[str, dict]
        Per-model predictions aggregated across all folds:
        {'Model Name': {'y_true': array, 'y_pred': array, 'y_prob': array | None}}
    """
    # ---- Setup ----
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed) # class-wise stratified K-fold
    models = get_models(seed=seed, mode=mode)                               # model pipelines
 
    avg = "binary" if mode == "binary" else "weighted"                      # averaging method for multi-class metrics
    rows = []                                                               # results storage
 
    # Storage for aggregated predictions across folds (for evaluate.py)
    predictions: dict[str, dict] = {
        name: {"y_true": [], "y_pred": [], "y_prob": []}
        for name in models
    }
 
    print(f"\n{BOLD}K-fold cross-validation  (K={k_folds}, seed={seed}){RESET}")
    print(f"  Mode     : {mode}")
    print(f"  Imputer  : {imputer_strategy}")
    print(f"  Models   : {', '.join(models)}")
    print("=" * 60)
 
    # ---- K-fold CV loop ----
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):     # enumerate from 1 for nicer printing
        X_tr  = X.iloc[train_idx].reset_index(drop=True)                    # training fold features
        X_val = X.iloc[val_idx].reset_index(drop=True)                      # validation fold features
        y_tr  = y.iloc[train_idx].reset_index(drop=True)                    # training fold labels
        y_val = y.iloc[val_idx].reset_index(drop=True)                      # validation fold labels
 
        # ---- Impute inside fold ----
        if imputer_strategy == "knn":             
            imputer = KNNImputer(n_neighbors=5)                                      # KNN imputation with 5 neighbors
        else:                           
            imputer = SimpleImputer(strategy=imputer_strategy)                       # mean, median, or most_frequent
 
        X_tr  = pd.DataFrame(imputer.fit_transform(X_tr),  columns=feature_cols)     # fit on train, transform train
        X_val = pd.DataFrame(imputer.transform(X_val),     columns=feature_cols)     # transform val with same imputer
 
        print(f"\n  Fold {fold_idx}/{k_folds}  "
              f"(train={len(y_tr)}, val={len(y_val)})")
 
        # ---- Majority baseline ----
        majority = y_tr.value_counts().idxmax()                                 # most common class in training fold
        y_base   = np.full(len(y_val), majority)                                # predict majority class for all val subjects
        base_bal = balanced_accuracy_score(y_val, y_base)                       # balanced accuracy of the baseline
        print(f"    Baseline (majority={majority})  bal_acc={base_bal:.3f}")    
 
        rows.append({
            "model":             "Baseline (majority)",
            "fold":              fold_idx,
            "accuracy":          accuracy_score(y_val, y_base),
            "balanced_accuracy": base_bal,
            "f1":                f1_score(y_val, y_base, average=avg, zero_division=0),
            "precision":         precision_score(y_val, y_base, average=avg, zero_division=0),
            "recall":            recall_score(y_val, y_base, average=avg, zero_division=0),
        })
 
        # ---- Train and evaluate each model ----
        for name, pipeline in models.items():
            pipeline.fit(X_tr, y_tr)                     # fit pipeline (scaling + classifier) on training fold
            y_pred = pipeline.predict(X_val)             # predict on validation fold
            y_prob = pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None
 
            bal = balanced_accuracy_score(y_val, y_pred)
            print(f"    {name:<25s}  bal_acc={bal:.3f}")
 
            rows.append({
                "model":             name,
                "fold":              fold_idx,
                "accuracy":          accuracy_score(y_val, y_pred),                                 # (TP + TN) / (TP + TN + FP + FN)
                "balanced_accuracy": bal,                                                           # average recall per class — handles class imbalance
                "f1":                f1_score(y_val, y_pred, average=avg, zero_division=0),         # 2 * (precision * recall) / (precision + recall)
                "precision":         precision_score(y_val, y_pred, average=avg, zero_division=0),  # TP / (TP + FP)
                "recall":            recall_score(y_val, y_pred, average=avg, zero_division=0),     # TP / (TP + FN)
            })
 
            # Accumulate predictions for evaluate.py
            predictions[name]["y_true"].extend(y_val.tolist())
            predictions[name]["y_pred"].extend(y_pred.tolist())
            if y_prob is not None:
                predictions[name]["y_prob"].extend(y_prob.tolist())
 
    # Convert lists to arrays
    for name in predictions:
        predictions[name]["y_true"] = np.array(predictions[name]["y_true"])
        predictions[name]["y_pred"] = np.array(predictions[name]["y_pred"])
        predictions[name]["y_prob"] = np.array(predictions[name]["y_prob"]) if predictions[name]["y_prob"] else None
 
    return pd.DataFrame(rows), predictions

# ================================================================================
# Nested cross-validation (2-layer)
# ================================================================================
def nested_cross_validate(
        X:                pd.DataFrame,
        y:                pd.Series,
        feature_cols:     list[str],
        k_1:              int = 5,
        k_2:              int = 5,
        n_iter:           int = 20,
        seed:             int = DEFAULT_SEED,
        mode:             str = "binary",
        imputer_strategy: str = "knn",
        ) -> tuple[pd.DataFrame, dict]:
    """
    Nested (2-layer) stratified K-fold cross-validation.

    Outer loop (K1): estimates unbiased generalization error.
    Inner loop (K2): selects best hyperparameters per model per fold using RandomizedSearchCV.

    For each outer fold:
      1. Fit imputer on outer training fold, transform both outer train and val.
      2. For each model, run RandomizedSearchCV on outer training fold (inner CV).
      3. Evaluate best model from inner CV on outer validation fold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (all subjects).
    y : pd.Series
        Label vector.
    feature_cols : list[str]
        Feature column names.
    k_1 : int
        Number of outer CV folds. *Default 5*.
    k_2 : int
        Number of inner CV folds. *Default 5*.
    n_iter : int
        Number of RandomizedSearchCV iterations per model per fold. *Default 20*.
    seed : int
        Random seed. *Default 42*.
    mode : str
        'binary' or 'multiclass'. *Default 'binary'*.
    imputer_strategy : str
        'knn', 'median', 'mean', or 'most_frequent'. *Default 'knn'*.

    Returns
    -------
    fold_results : pd.DataFrame
        One row per (model, fold) with columns: model, fold, accuracy,
        balanced_accuracy, f1, precision, recall.
    predictions : dict[str, dict]
        Per-model predictions aggregated across all outer folds:
        {'Model Name': {'y_true': array, 'y_pred': array, 'y_prob': array | None}}
    """
    # ---- Setup ----
    outer_cv   = StratifiedKFold(n_splits=k_1, shuffle=True, random_state=seed) # outer loop
    inner_cv   = StratifiedKFold(n_splits=k_2, shuffle=True, random_state=seed) # inner loop
    search_spaces = get_search_spaces(seed=seed, mode=mode)                     # model pipelines + search spaces

    avg  = "binary" if mode == "binary" else "weighted"  # averaging method for metrics
    rows = []                                            # results storage

    # Storage for aggregated predictions across outer folds (for evaluate.py)
    predictions: dict[str, dict] = {
        name: {"y_true": [], "y_pred": [], "y_prob": []}
        for name in search_spaces
    }

    print(f"\n{BOLD}Nested CV  (K1={k_1}, K2={k_2}, n_iter={n_iter}, seed={seed}){RESET}")
    print(f"  Mode     : {mode}")
    print(f"  Imputer  : {imputer_strategy}")
    print(f"  Models   : {', '.join(search_spaces)}")
    print("=" * 60)

    # ---- Outer CV loop ----
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
        X_tr  = X.iloc[train_idx].reset_index(drop=True)   # outer training fold features
        X_val = X.iloc[val_idx].reset_index(drop=True)     # outer validation fold features
        y_tr  = y.iloc[train_idx].reset_index(drop=True)   # outer training fold labels
        y_val = y.iloc[val_idx].reset_index(drop=True)     # outer validation fold labels

        # ---- Impute inside outer fold ----
        if imputer_strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=imputer_strategy)

        X_tr  = pd.DataFrame(imputer.fit_transform(X_tr),  columns=feature_cols)  # fit on outer train only
        X_val = pd.DataFrame(imputer.transform(X_val),     columns=feature_cols)  # transform val with same imputer

        print(f"\n  Outer fold {fold_idx}/{k_1}  "
              f"(train={len(y_tr)}, val={len(y_val)})")

        # ---- Majority baseline ----
        # NOTE: The majority class is determined from the outer training fold to avoid data leakage. 
        #       The baseline is then evaluated on the outer validation fold.
        majority = y_tr.value_counts().idxmax()
        y_base   = np.full(len(y_val), majority)
        base_bal = balanced_accuracy_score(y_val, y_base)
        print(f"    Baseline (majority={majority})  bal_acc={base_bal:.3f}")

        rows.append({
            "model":             "Baseline (majority)",
            "fold":              fold_idx,
            "accuracy":          accuracy_score(y_val, y_base),
            "balanced_accuracy": base_bal,
            "f1":                f1_score(y_val, y_base, average=avg, zero_division=0),
            "precision":         precision_score(y_val, y_base, average=avg, zero_division=0),
            "recall":            recall_score(y_val, y_base, average=avg, zero_division=0),
        })

        # ---- Inner CV: hyperparameter tuning per model ----
        for name, spec in search_spaces.items():
            search = RandomizedSearchCV(
                estimator            = spec["pipeline"],        # model pipeline
                param_distributions  = spec["param_dist"],     # hyperparameter search space
                n_iter               = n_iter,                  # number of random combinations to try
                cv                   = inner_cv,                # inner CV splits
                scoring              = "balanced_accuracy",     # optimise for balanced accuracy
                refit                = True,                    # refit best model on full outer training fold
                random_state         = seed,
                n_jobs               = -1,                      # use all available cores
            )
            search.fit(X_tr, y_tr)                             # run inner CV on outer training fold

            best_model = search.best_estimator_                 # best model from inner CV
            y_pred     = best_model.predict(X_val)             # predict on outer validation fold
            y_prob     = best_model.predict_proba(X_val) if hasattr(best_model, "predict_proba") else None

            bal = balanced_accuracy_score(y_val, y_pred)
            print(f"    {name:<25s}  bal_acc={bal:.3f}  best_params={search.best_params_}")

            rows.append({
                "model":             name,
                "fold":              fold_idx,
                "accuracy":          accuracy_score(y_val, y_pred),
                "balanced_accuracy": bal,
                "f1":                f1_score(y_val, y_pred, average=avg, zero_division=0),
                "precision":         precision_score(y_val, y_pred, average=avg, zero_division=0),
                "recall":            recall_score(y_val, y_pred, average=avg, zero_division=0),
            })

            # Accumulate predictions for evaluate.py
            predictions[name]["y_true"].extend(y_val.tolist())
            predictions[name]["y_pred"].extend(y_pred.tolist())
            if y_prob is not None:
                predictions[name]["y_prob"].extend(y_prob.tolist())

    # Convert lists to arrays
    for name in predictions:
        predictions[name]["y_true"] = np.array(predictions[name]["y_true"])
        predictions[name]["y_pred"] = np.array(predictions[name]["y_pred"])
        predictions[name]["y_prob"] = np.array(predictions[name]["y_prob"]) if predictions[name]["y_prob"] else None

    return pd.DataFrame(rows), predictions

# ================================================================================
# Summary
# ================================================================================
def summarise(fold_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fold results into mean ± std per model.
 
    Returns
    -------
    pd.DataFrame
        One row per model, columns: model, <metric>_mean, <metric>_std.
    """
    metrics = ["accuracy", "balanced_accuracy", "f1", "precision", "recall"]   # metrics to summarise
 
    agg = fold_results.groupby("model")[metrics].agg(["mean", "std"])          # mean and std per model
 
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]                         # flatten MultiIndex → accuracy_mean, accuracy_std, ...
 
    return agg.reset_index().sort_values("balanced_accuracy_mean", ascending=False)
 
 
# ================================================================================
# Full training pipeline
# ================================================================================
def run_training(
        feature_csv:      str | Path,
        mode:             str = "binary",
        binary_mode:      str = "control_vs_all",
        k_folds:          int = 5,
        k_1:              int = 5,
        k_2:              int = 5,
        n_iter:           int = 20,
        seed:             int = DEFAULT_SEED,
        imputer_strategy: str = "knn",
        method:           str = "kfold",
        save_dir:         str | Path = "reports/cv",
        ) -> dict:
    """
    Full CV pipeline — K-fold or nested CV.

    Parameters
    ----------
    feature_csv : str | Path
        Path to the feature CSV file.
    mode : str
        'binary' or 'multiclass'. *Default 'binary'*.
    binary_mode : str
        Only used when mode='binary'. *Default 'control_vs_all'*.
    k_folds : int
        Number of folds for K-fold CV. *Default 5*.
    k_1 : int
        Number of outer folds for nested CV. *Default 5*.
    k_2 : int
        Number of inner folds for nested CV. *Default 5*.
    n_iter : int
        RandomizedSearchCV iterations for nested CV. *Default 20*.
    seed : int
        Random seed. *Default 42*.
    imputer_strategy : str
        Imputation strategy. *Default 'knn'*.
    method : str
        'kfold' or 'nested'. *Default 'kfold'*.
    save_dir : str | Path
        Directory for output CSVs. *Default 'reports/cv'*.

    Returns
    -------
    dict
        fold_results, summary, predictions, X, y.
    """
    print(f"\n{'='*60}")
    print(f"  {BOLD}RBD Classification — {method.upper()} CV{RESET}")
    print(f"  Mode    : {mode}" + (f"  ({binary_mode})" if mode == "binary" else ""))
    if method == "kfold":
        print(f"  Folds   : {k_folds}  |  Seed: {seed}")
    else:
        print(f"  Outer   : {k_1}  |  Inner: {k_2}  |  Seed: {seed}")
    print(f"{'='*60}")

    # ---- 1) Load ----
    X, y, feature_cols = prepare(
        feature_csv=feature_csv,
        mode=mode,
        binary_mode=binary_mode,
    )

    # ---- 2) CV ----
    if method == "kfold":
        fold_results, predictions = cross_validate(
            X=X, y=y, feature_cols=feature_cols,
            k_folds=k_folds, seed=seed, mode=mode,
            imputer_strategy=imputer_strategy,
        )
    else:
        fold_results, predictions = nested_cross_validate(
            X=X, y=y, feature_cols=feature_cols,
            k_1=k_1, k_2=k_2, n_iter=n_iter,
            seed=seed, mode=mode, imputer_strategy=imputer_strategy,
        )

    # ---- 3) Summary ----
    summary = summarise(fold_results)

    print(f"\n{BOLD}Summary (mean ± std across folds){RESET}")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"  {row['model']:<25s}  "
              f"bal_acc={row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f}  "
              f"f1={row['f1_mean']:.3f} ± {row['f1_std']:.3f}")

    # ---- 4) Save ----
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{method}_{mode}" + (f"_{binary_mode}" if mode == "binary" else "")
    fold_results.to_csv(save_dir / f"fold_results_{tag}.csv", index=False)
    summary.to_csv(save_dir / f"summary_{tag}.csv", index=False)
    print(f"\n  Saved -> {save_dir / f'fold_results_{tag}.csv'}")
    print(f"  Saved -> {save_dir / f'summary_{tag}.csv'}")
    print(f"{'='*60}\n")

    return {
        "fold_results": fold_results,
        "summary":      summary,
        "predictions":  predictions,
        "X":            X,
        "y":            y,
    }


# ================================================================================
# CLI
# ================================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CV for RBD classification. K-fold or nested CV."
    )
    parser.add_argument("feature_csv", type=str)
    parser.add_argument("--method", type=str, default="kfold",
                        choices=["kfold", "nested"],
                        help="CV method. Default: 'kfold'")
    parser.add_argument("--mode", type=str, nargs="+", default=["binary"],
                        choices=["binary", "multiclass"])
    parser.add_argument("--binary-mode", type=str, nargs="+", default=["control_vs_all"],
                        choices=["control_vs_all", "control_vs_irbd", "control_vs_pd"])
    parser.add_argument("--k-folds", type=int, default=5,
                        help="Folds for K-fold CV. Default: 5")
    parser.add_argument("--k1", type=int, default=5,
                        help="Outer folds for nested CV. Default: 5")
    parser.add_argument("--k2", type=int, default=5,
                        help="Inner folds for nested CV. Default: 5")
    parser.add_argument("--n-iter",  type=int, default=20,
                        help="RandomizedSearchCV iterations for nested CV. Default: 20")
    parser.add_argument("--seed",    type=int, default=DEFAULT_SEED)
    parser.add_argument("--imputer", type=str, default="knn",
                        choices=["knn", "median", "mean", "most_frequent"])
    parser.add_argument("--save-dir", type=str, default="reports/cv")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-save-dir", type=str, default="reports/evaluation")

    args = parser.parse_args()

    # Build list of (mode, binary_mode) runs
    runs = []
    for mode in args.mode:
        if mode == "multiclass":
            runs.append((mode, None))
        else:
            for bm in args.binary_mode:
                runs.append((mode, bm))

    all_results = []

    for mode, binary_mode in runs:
        result = run_training(
            feature_csv      = args.feature_csv,
            mode             = mode,
            binary_mode      = binary_mode or "control_vs_all",
            k_folds          = args.k_folds,
            k_1              = args.k1,
            k_2              = args.k2,
            n_iter           = args.n_iter,
            seed             = args.seed,
            imputer_strategy = args.imputer,
            method           = args.method,
            save_dir         = args.save_dir,
        )
        all_results.append((mode, binary_mode, result))

    if args.evaluate:
        from ml.evaluate import evaluate_all, evaluate_binary_comparison

        binary_results = {
            bm: r for m, bm, r in all_results if m == "binary" and bm is not None
        }

        for mode, binary_mode, result in all_results:
            lm          = _label_map(mode, binary_mode)
            class_names = [lm[k] for k in sorted(lm)]

            evaluate_all(
                predictions      = result["predictions"],
                summary          = result["summary"],
                X                = result["X"],
                y                = result["y"],
                class_names      = class_names,
                mode             = mode,
                imputer_strategy = args.imputer,
                seed             = args.seed,
                run_config       = {
                    "mode":             mode,
                    "binary_mode":      binary_mode or "—",
                    "method":           args.method,
                    "k_folds":          args.k_folds if args.method == "kfold" else f"{args.k1}/{args.k2}",
                    "seed":             args.seed,
                    "imputer_strategy": args.imputer,
                    "n_subjects":       len(result["y"]),
                    "n_features":       result["X"].shape[1],
                },
                save_dir         = args.eval_save_dir,
                binary_results   = binary_results if mode == "binary" else None,
            )

        if len(binary_results) > 1:
            evaluate_binary_comparison(
                binary_results = binary_results,
                save_dir       = args.eval_save_dir,
            )