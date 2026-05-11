# Filename: train.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Trains a machine learning model on the extracted features and patient labels.
#              Supports binary and multi-class classification with two-layer cross-validation
#              for unbiased generalization error estimation and hyperparameter selection.
#              Also supports sweeping over multiple seeds, test sizes and classification modes.
 
# Usage:
#       python -m ml.train features_csv/features.csv --mode binary
#       python -m ml.train features_csv/features.csv --mode multiclass
#       python -m ml.train features_csv/features.csv --sweep

# ================================================================================
# Imports
# ================================================================================
from __future__ import annotations

import itertools
import numpy as np              # for numerical operations
import pandas as pd             # for data manipulation
from pathlib import Path        # for handling file paths
import scipy.stats as stats     # for statistical tests (e.g. McNemar's test)

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost imports 
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
# Label map helper
# ================================================================================
def _label_map(mode: str) -> dict[int, str]:
    if mode == "binary":
        return {0: "Control", 1: "Disease"}
    return {0: "Control", 1: "iRBD", 2: "PD(-RBD)", 3: "PD(+RBD)"}

# ================================================================================
# McNemar helper
# ================================================================================
def _mcnemar_test(y_true, y_pred_a, y_pred_b) -> tuple[float, float, float, tuple]:
    """
    McNemar's test comparing two classifiers.
    
    Parameters  
    ----------
    y_true : array-like
        True labels.
    y_pred_a : array-like
        Predictions from classifier A.
    y_pred_b : array-like
        Predictions from classifier B.
    Returns
    -------
    tuple[float, float, int, int, tuple[float, float]]
        - theta_hat: McNemar's statistic (difference in error rates).
        - p_value: Two-sided p-value for the null hypothesis of equal error rates.
        - n12: Number of samples misclassified by A but not B.
        - n21: Number of samples misclassified by B but not A.
        - ci: 95% confidence interval for theta_hat (lower, upper), or (None, None) if s < 5.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    n12 = int(( correct_a & ~correct_b).sum())
    n21 = int((~correct_a &  correct_b).sum())
    n   = len(y_true)

    theta_hat = (n12 - n21) / n
    s = n12 + n21
    m = min(n12, n21)
    p_value = float(2 * stats.binom.cdf(m, s, 0.5)) if s > 0 else 1.0

    # CI via Beta approximation (eq. 11.33), requires s >= 5
    ci = (None, None)
    if s >= 5:
        E_theta = theta_hat
        denom   = n * s - (n12 - n21) ** 2
        Q       = (n ** 2 * (n + 1) * (E_theta + 1) * (1 - E_theta)) / denom if denom != 0 else 1
        f = (E_theta + 1) / 2 * (Q - 1)
        g = (1 - E_theta) / 2 * (Q - 1)
        ci = (
            2 * stats.beta.ppf(0.025, f, g) - 1,
            2 * stats.beta.ppf(0.975, f, g) - 1,
        )

    return theta_hat, p_value, n12, n21, ci

# ================================================================================
# Model + hyperparameter search space definitions
# ================================================================================
def get_model_search_spaces(
        seed: int = DEFAULT_SEED, 
        mode: str = "binary"
        ) -> dict:
    """
    Define the models and their hyperparameter search spaces for RandomizedSearchCV.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. 
        **Default is 42**.
    mode : str
        'binary' or 'multiclass'. Affects model definitions and label mapping. 
        **Default is 'binary'**.
    
    Returns
    -------
    dict
        Dictionary of model specifications, where each key is a model name and each value is a dict containing:
        - "pipeline": sklearn Pipeline with preprocessing and classifier.
        - "param_dist": dict of hyperparameter distributions for RandomizedSearchCV.
    """
    xgb_objective  = "binary:logistic" if mode == "binary" else "multi:softprob"
    xgb_metric     = "logloss"         if mode == "binary" else "mlogloss"
 
    return {
        "Random Forest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=-1,
                )),
            ]),
            "param_dist": {
                "clf__n_estimators": [50, 100, 200, 300],
                "clf__max_depth":    [None, 5, 10, 20],
                "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        },
 
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    class_weight="balanced",
                    solver="lbfgs",
                )),
            ]),
            "param_dist": {
                "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
 
        "SVM": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    random_state=seed,
                    class_weight="balanced",
                    probability=True,
                )),
            ]),
            "param_dist": {
                "clf__C":     [0.1, 1.0, 10.0, 100.0],
                "clf__gamma": ["scale", "auto", 0.01, 0.001],
                "clf__kernel": ["rbf", "linear"],
            },
        },
 
        "XGBoost": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    objective    = xgb_objective,
                    eval_metric  = xgb_metric,
                    random_state = seed,
                    verbosity    = 0,
                )),
            ]),
            "param_dist": {
                "clf__n_estimators":  [50, 100, 200],
                "clf__max_depth":     [3, 5, 6, 8],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample":     [0.7, 0.8, 1.0],
            },
        },
    }

# ================================================================================
# two-Layer Cross-validation
# ================================================================================
def two_layer_cross_validate(
        X_train:    pd.DataFrame,
        y_train:    pd.Series,
        n_outer:    int = 5,
        n_inner:    int = 5,
        n_iter:     int = 20,
        seed:       int = DEFAULT_SEED,
        mode:       str = "binary",
        ) -> pd.DataFrame:
    """
    Two-layer (nested) cross-validation.
 
    Outer loop  (K1 = n_outer folds): estimates unbiased generalization error.
    Inner loop  (K2 = n_inner folds): selects best hyperparameters per model per fold.
 
    For each outer fold and each model:
      1. RandomizedSearchCV on D_train (inner CV) finds best hyperparams.
      2. Best model is re-fitted on full D_train and evaluated on D_test.
 
    Also computes a majority-class baseline per outer fold.
 
    Parameters
    ----------
    X_train : pd.DataFrame
        X training features. 
    y_train : pd.Series
        y training labels.
    n_outer : int
        Outer CV folds for unbiased error estimation. **Default is 5**.
    n_inner : int
        Inner CV folds for hyperparameter tuning. **Default is 5**.
    n_iter : int
        Number of hyperparameter settings to try in RandomizedSearchCV. **Default is 20**.
    seed : int
        Random seed for reproducibility. **Default is 42**.
    mode : str
        'binary' or 'multiclass'. Affects model definitions and label mapping. **Default is 'binary'**.
 
    Returns
    -------
    pd.DataFrame
        Mean ± std of metrics across outer folds, one row per model + baseline.
    """
    model_specs = get_model_search_spaces(seed=seed, mode=mode)
    outer_cv    = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner_cv    = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    scoring     = "balanced_accuracy"
 
    print(f"\n{BOLD}Two-layer cross-validation  "
          f"(outer={n_outer}, inner={n_inner}, n_iter={n_iter}){RESET}")
    print(f"{'='*60}")
 
    # Storage: fold_scores[model_name] = list of per-fold balanced_accuracy
    fold_scores: dict[str, list[float]] = {name: [] for name in model_specs}
    fold_scores["Baseline (majority)"] = []
 
    best_params_per_model: dict[str, list[dict]] = {name: [] for name in model_specs}
 
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        X_tr = X_train.iloc[train_idx].reset_index(drop=True)
        X_val= X_train.iloc[val_idx].reset_index(drop=True)
        y_tr = y_train.iloc[train_idx].reset_index(drop=True)
        y_val= y_train.iloc[val_idx].reset_index(drop=True)
 
        print(f"\n  Outer fold {fold_idx}/{n_outer}  "
              f"(train={len(y_tr)}, val={len(y_val)})")
 
        # ---- Baseline ----
        majority = y_tr.value_counts().idxmax()
        y_base   = np.full(len(y_val), majority)
        fold_scores["Baseline (majority)"].append(
            balanced_accuracy_score(y_val, y_base)
        )
 
        # ---- Each model ----
        for name, spec in model_specs.items():
            print(f"    {name:<22s} ...", end=" ", flush=True)
 
            # XGBoost: pass sample_weight via fit_params
            fit_params = {}
            if name == "XGBoost":
                sw = compute_sample_weight("balanced", y_tr)
                fit_params = {"clf__sample_weight": sw}
 
            search = RandomizedSearchCV(
                estimator  = spec["pipeline"],
                param_distributions = spec["param_dist"],
                n_iter     = n_iter,
                cv         = inner_cv,
                scoring    = scoring,
                refit      = True,
                random_state = seed,
                n_jobs     = -1,
                error_score= 0.0,
            )
 
            search.fit(X_tr, y_tr, **fit_params)
            best_params_per_model[name].append(search.best_params_)
 
            y_pred = search.best_estimator_.predict(X_val)
            score  = balanced_accuracy_score(y_val, y_pred)
            fold_scores[name].append(score)
 
            print(f"bal_acc = {score:.3f}  (best params: {search.best_params_})")
 
    # ---- Summarise across folds ----
    print(f"\n{BOLD}Two-layer CV summary  (metric: balanced accuracy){RESET}")
    print(f"  {'-'*55}")
    print(f"  {'Model':<25s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*55}")
 
    rows = []
    for name, scores in fold_scores.items():
        arr = np.array(scores)
        row = {
            "model": name,
            "bal_acc_mean": arr.mean(),
            "bal_acc_std":  arr.std(),
            "bal_acc_min":  arr.min(),
            "bal_acc_max":  arr.max(),
            "fold_scores":  scores,
        }
        rows.append(row)
        marker = f"  {GREEN}←{RESET}" if name != "Baseline (majority)" and arr.mean() == max(
            np.mean(v) for k, v in fold_scores.items() if k != "Baseline (majority)"
        ) else ""
        print(f"  {name:<25s}  {arr.mean():>8.3f}  {arr.std():>8.3f}  "
              f"{arr.min():>8.3f}  {arr.max():>8.3f}{marker}")
 
    results_df = pd.DataFrame(rows)
 
    # ---- Best model ----
    model_rows  = results_df[results_df["model"] != "Baseline (majority)"]
    best_name   = model_rows.loc[model_rows["bal_acc_mean"].idxmax(), "model"]
    best_mean   = model_rows["bal_acc_mean"].max()
    base_mean   = results_df.loc[
        results_df["model"] == "Baseline (majority)", "bal_acc_mean"
    ].values[0]
 
    print(f"\n  {GREEN}{BOLD}Best model: {best_name}  "
          f"(bal_acc = {best_mean:.3f}  vs  baseline = {base_mean:.3f}){RESET}")
 
    # Most common best params across folds
    print(f"\n  Most frequent best hyperparameters for {best_name}:")
    param_records = best_params_per_model[best_name]
    all_keys = set(k for d in param_records for k in d)
    for key in sorted(all_keys):
        values   = [d.get(key) for d in param_records if key in d]
        from collections import Counter
        most_common = Counter(values).most_common(1)[0]
        print(f"    {key:<35s}  {most_common[0]}  (chosen {most_common[1]}/{n_outer} folds)")
 
    return results_df, best_name
 
# ================================================================================
# Final model: fit on full train, evaluate once on held-out test
# ================================================================================
def fit_and_evaluate_best(
        best_name:  str,
        X_train:    pd.DataFrame,
        y_train:    pd.Series,
        X_test:     pd.DataFrame,
        y_test:     pd.Series,
        n_inner:    int = 5,
        n_iter:     int = 20,
        seed:       int = DEFAULT_SEED,
        mode:       str = "binary",
        save_dir:   str | Path | None = "reports/evaluation",
        is_best:    bool = False,
        run_config: dict | None = None,  
        cv_results: pd.DataFrame | None = None,
        ) -> None:
    """
    Fit the best model (chosen by outer CV) on the full training set using one
    final inner RandomizedSearchCV pass, then evaluate once on X_test.
 
    This is the only time X_test is touched.
 
    Parameters
    ----------
    best_name : str         
        Model name returned by two_layer_cross_validate.
    X_train : pd.DataFrame
        Training features (full training set).
    y_train : pd.Series
        Training labels (full training set).
    X_test : pd.DataFrame
        Test features (held-out test set).
    y_test : pd.Series
        Test labels (held-out test set).
    n_inner : int
        Number of folds for inner CV.
    n_iter : int
        Number of iterations for randomized search.
        **Default is 20**.
    seed : int
        Random seed.
        **Default is 42**.
    mode : str
        'binary' or 'multiclass'.
        **Default is 'binary'**.
    save_dir : str | Path | None
        Where to write the evaluation PDF.
    is_best : bool
        Whether this is the best model (for display purposes).
    run_config : dict | None
        Configuration for the run.
    cv_results : pd.DataFrame | None
        Cross-validation results.
    """
    marker = f"  {GREEN} <- CV best{RESET}" if is_best else ""
    print(f"\n{BOLD}Final evaluation: {best_name}{RESET}{marker}")
    print(f"{'='*60}")
    print(f"  Fitting on full training set with inner CV "
          f"(n_inner={n_inner}, n_iter={n_iter}) ...")
 
    spec     = get_model_search_spaces(seed=seed, mode=mode)[best_name]
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
 
    fit_params = {}
    if best_name == "XGBoost":
        fit_params = {"clf__sample_weight": compute_sample_weight("balanced", y_train)}
 
    search = RandomizedSearchCV(
        estimator   = spec["pipeline"],
        param_distributions = spec["param_dist"],
        n_iter      = n_iter,
        cv          = inner_cv,
        scoring     = "balanced_accuracy",
        refit       = True,
        random_state= seed,
        n_jobs      = -1,
    )
    search.fit(X_train, y_train, **fit_params)
    best_model = search.best_estimator_
 
    print(f"  Best params: {search.best_params_}")
 
    # ---- Test set metrics ----
    y_pred = best_model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    bal    = balanced_accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
 
    lm           = _label_map(mode)
    class_names  = [lm[c] for c in sorted(y_test.unique().tolist())]
 
    print(f"\n  {BOLD}Test set results ({best_name}){RESET}")
    print(f"  {'─'*40}")
    print(f"  Accuracy:          {acc:.3f}")
    print(f"  Balanced accuracy: {bal:.3f}")
    print(f"  F1 (weighted):     {f1:.3f}")
    print(f"  Precision:         {prec:.3f}")
    print(f"  Recall:            {rec:.3f}")
    print(f"\n  Classification report:")
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    for line in report.split("\n"):
        print(f"    {line}")
 
    # ---- Evaluation PDF ----
    evaluate_model(
        model       = best_model,
        X_test      = X_test,
        y_test      = y_test,
        class_names = class_names,
        model_name  = best_name,
        top_n       = 20,
        save_dir    = save_dir,
        seed        = seed,
        run_config  = run_config,
        cv_results  = cv_results,
        best_params = search.best_params_,
    )
 
    return  best_model, {
        "test_acc":     round(acc, 4),  # test set accuracy
        "test_bal_acc": round(bal, 4),  # test set balanced accuracy
        "test_f1":      round(f1,  4),  # test set F1 score (weighted)
        "test_prec":    round(prec, 4), # test set precision (weighted)
        "test_rec":     round(rec, 4),  # test set recall (weighted)
        "y_pred":       y_pred,         # test set predictions
        }

# ================================================================================
# Feature importance summary (uses full training set, for reporting)
# ================================================================================
def print_feature_importance(
        model,
        X_train:    pd.DataFrame,
        y_train:    pd.Series,
        top_n:      int = 20,
        model_name: str = "Model",
        ) -> pd.DataFrame:
    """
    Print MDI feature importance if available (RF / XGBoost). 

    Parameters
    ----------
    model : sklearn Pipeline or estimator
        The trained model. Should have a "clf" step if it's a Pipeline. \\
        *clf = the final estimator (e.g. RandomForestClassifier or XGBClassifier)*
    X_train : pd.DataFrame
        Training features (used to get feature names).
    y_train : pd.Series
        Training labels (used to fit the model if not already fitted).
    top_n : int 
        Number of top features to print. 
        **Default is 20**.
    model_name : str
        Name of the model for display purposes. 
        **Default is "Model"**.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["feature", "importance"] sorted by importance (descending).
    """
    # Define clf as the final estimator (e.g. RandomForestClassifier or XGBClassifier)
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    # If model is not fitted, fit it on the full training set
    if not hasattr(clf, "feature_importances_"):
        print(f"  [SKIP] MDI importance not available for {type(clf).__name__}")
        return pd.DataFrame(columns=["feature", "importance"])
    
    # Extract feature importances and create a DataFrame
    importance_df = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    # Print top features with a simple text-based bar chart
    print(f"\n{BOLD}Top {top_n} features — {model_name} (MDI){RESET}")
    print(f"{'─'*60}")
    for _, r in importance_df.head(top_n).iterrows():
        bar = "#" * int(r["importance"] * 200)
        print(f"  {r['feature']:<45s}  {r['importance']:.4f}  {bar}")
 
    return importance_df

# ================================================================================
# Full training pipeline
# ================================================================================
def run_training(
        feature_csv:    str | Path,
        mode:           str = "binary",
        binary_mode:    str = "control_vs_all",
        test_size:      float = 0.2,
        n_outer:        int = 5,
        n_inner:        int = 5,
        n_iter:         int = 20,
        seed:           int = DEFAULT_SEED,
        drop_nan:       bool = False,
        save_dir:       str | Path = "reports/evaluation",
        imputer_strategy: str = "median",
        ) -> dict:
    """
    Full two-layer cross-validation pipeline:
 
        1. Prepare data (train/test split).
        2. Two-layer CV on X_train:
              outer loop -> unbiased generalization error per model.
              inner loop -> hyperparameter selection per fold.
        3. Select best model from outer CV results.
        4. Fit best model on full X_train (with one final inner CV pass).
        5. Evaluate ONCE on X_test (held-out).
        6. Save evaluation PDF.
 
    Parameters
    ----------
    feature_csv : str | Path
        Path to features.csv file containing extracted features and labels.
    mode : str
        Options: 'binary' or 'multiclass'. \n
        Determines whether to do binary or multi-class.
        Classification mode. Affects label mapping and model definitions.
        **Default is 'binary'**.
    binary_mode : str
        Options: 'control_vs_all' or 'control_vs_irbd'. \n
        Only used if mode='binary'. Determines how to binarize the labels:
        - 'control_vs_all': Control (0) vs all disease (1)
        - 'control_vs_irbd': Control (0) vs iRBD (1), excluding PD patients. \\
        **Default is 'control_vs_all'**.
    test_size : float
        Fraction of the dataset to include in the test split. 
        **Default is 0.2**.
    n_outer : int
        Number of outer CV folds. 
        **Default is 5**.
    n_inner : int
        Number of inner CV folds. 
        **Default is 5**.
    n_iter : int
        Number of RandomizedSearch iterations per model per fold. 
        **Default is 20**.
    seed : int
        Random seed. 
        **Default is 42**.
    drop_nan : bool
        Whether to drop subjects with NaN features. Default False (median imputation).
    save_dir : str | Path
        Directory where evaluation reports will be saved.
        **Default is "reports/evaluation"**.

    Returns
    -------
    dict
        Dictionary containing:
        - "cv_results": DataFrame with CV results per model.
        - "best_name": Name of the best model from CV.
        - "best_model": The fitted best model on full training set.
        - "X_train": Training features.
        - "X_test": Test features.
        - "y_train": Training labels.
        - "y_test": Test labels.
        - "importance_df": DataFrame of feature importances for the best model.
        - "evaluation_dir": Path to the directory where evaluation report is saved.
    """
    print(f"\n{'='*60}")
    print(f"  {BOLD}RBD Classification Pipeline{RESET}")
    print(f"  Mode: {mode}" + (f"  ({binary_mode})" if mode == "binary" else ""))
    print(f"  Two-layer CV  (outer={n_outer}, inner={n_inner}, n_iter={n_iter})")
    print(f"{'='*60}")
 
    # ---- 1) Prepare data ----
    X_train, X_test, y_train, y_test = prepare(
        feature_csv      = feature_csv,
        mode             = mode,
        binary_mode      = binary_mode,
        test_size        = test_size,
        seed             = seed,
        drop_nan         = drop_nan,
        imputer_strategy = imputer_strategy,
    )

    # Build the run_config dict that will appear on every PDF info page
    run_config = {
        "feature_csv": str(feature_csv),
        "mode":        mode,
        "binary_mode": binary_mode if mode == "binary" else "—",
        "seed":        seed,
        "test_size":   test_size,
        "n_train":     len(X_train),
        "n_test":      len(X_test),
        "n_features":  X_train.shape[1],
        "n_outer":     n_outer,
        "n_inner":     n_inner,
        "n_iter":      n_iter,
        "imputer_strategy":  imputer_strategy,
    }
 
    # ---- 2) Two-layer CV — never touches X_test ----
    cv_results, best_name = two_layer_cross_validate(
        X_train = X_train,
        y_train = y_train,
        n_outer = n_outer,
        n_inner = n_inner,
        n_iter  = n_iter,
        seed    = seed,
        mode    = mode,
    )
 
    # ---- 3) Fit all models on full train, evaluate each on test ----
    best_model = None                                                   # will be set in loop when we get to the best model
    model_specs = get_model_search_spaces(seed=seed, mode=mode)         # for fitting and evaluation
    lm          = _label_map(mode)                                      # label map for class names (for evaluation report)
    class_names = [lm[c] for c in sorted(y_test.unique().tolist())]     # for evaluation report
    all_preds    = {}                                                   # store test set predictions for all models (for later analysis)

    for name in model_specs:
        model, test_metrics = fit_and_evaluate_best(
            best_name = name,
            X_train   = X_train,
            y_train   = y_train,
            X_test    = X_test,
            y_test    = y_test,
            n_inner   = n_inner,
            n_iter    = n_iter,
            seed      = seed,
            mode      = mode,
            save_dir  = save_dir,
            is_best   = (name == best_name), 
            run_config= run_config,
            cv_results= cv_results,
        )
        all_preds[name] = test_metrics.pop("y_pred")  
        if name == best_name:
            best_model   = model
            best_metrics = test_metrics
 
    # ---- 4) Feature importance (informational, on training set) ----
    importance_df = print_feature_importance(
        model      = best_model,
        X_train    = X_train,
        y_train    = y_train,
        top_n      = 20,
        model_name = best_name,
    )
 
    print(f"\n{'='*60}")
    print(f"  {GREEN}{BOLD}Pipeline complete.{RESET}")
    print(f"  Best model : {best_name}")
    print(f"  Report     : {save_dir}/{best_name.lower().replace(' ', '_')}_*.pdf")
    print(f"{'='*60}\n")
 
    return {
        "cv_results":       cv_results,
        "best_name":        best_name,
        "best_model":       best_model,
        "best_metrics":     best_metrics,
        "all_predictions":  all_preds,
        "X_train":          X_train,
        "X_test":           X_test,
        "y_train":          y_train,
        "y_test":           y_test,
        "importance_df":    importance_df,
        "evaluation_dir":   Path(save_dir),
    }

# ================================================================================
# Sweep — run over multiple seeds / test sizes / modes automatically
# ================================================================================
def sweep_training(
        feature_csv:        str | Path,
        seeds:              list[int]   = (42, 0, 1),
        test_sizes:         list[float] = (0.2, 0.25),
        modes:              list[str]   = ("binary", "multiclass"),
        binary_mode:        str         = "control_vs_all",
        n_outer:            int         = 5,
        n_inner:            int         = 5,
        n_iter:             int         = 20,
        drop_nan:           bool        = False,
        imputer_strategy:   str       = "knn",
        save_base_dir:      str | Path  = "reports/sweep",
        ) -> pd.DataFrame:
    """
    Run run_training for every combination of seed x test_size x mode.
 
    Each combination gets its own subdirectory:
        ``<save_base_dir>/seed{seed}_split{test_size}_{mode}/``
 
    A summary CSV is written to <save_base_dir>/sweep_summary.csv.
 
    Parameters
    ----------
    feature_csv : str | Path
        Path to features.csv file containing extracted features and labels.
    seeds : list[int]
        List of random seeds to use for data splitting and model training.
        **Default is [42, 0, 1]**.
    test_sizes : list[float]
        List of test set sizes (fractions) to use for train/test splitting.
        **Default is [0.2, 0.25]**.
    modes : list[str]
        List of classification modes to run. Options: 'binary', 'multiclass'.
        **Default is ['binary', 'multiclass']**.
    binary_mode : str
        Only used if mode='binary'. Determines how to binarize the labels:
        - 'control_vs_all': Control (0) vs all disease (1)
        - 'control_vs_irbd': Control (0) vs iRBD (1), excluding PD patients. \\
        **Default is 'control_vs_all'**.        
    n_outer : int
        Number of outer CV folds for each run.
        **Default is 5**.
    n_inner : int
        Number of inner CV folds for each run.
        **Default is 5**.
    n_iter : int
        Number of RandomizedSearch iterations for each run.
        **Default is 20**.
    drop_nan : bool
        Whether to drop subjects with NaN features. Default False (median imputation).
    save_base_dir : str | Path
        Base directory where all run subdirectories and summary CSV will be saved.
        Each run gets its own subdirectory named:
        ``<save_base_dir>/seed{seed}_split{test_size}_{mode}/``
        **Default is 'reports/sweep'**. 
 
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per run, containing:
        - run: Run index (1-based)
        - seed: Random seed used
        - test_size: Test set size used
        - mode: Classification mode used
        - best_model: Best model name from CV
        - cv_bal_acc: Mean balanced accuracy from CV for the best model
        - cv_bal_std: Std of balanced accuracy from CV for the best model
        - status: "ok" or error message if the run failed
    """

    combos   = list(itertools.product(seeds, test_sizes, modes))    # All combinations of parameters to run
    n_combos = len(combos)                                          # Total number of runs for progress tracking
    base_dir = Path(save_base_dir)                                  # Base directory for all run subdirectories             
    base_dir.mkdir(parents=True, exist_ok=True)                     # Ensure base directory exists
 
    print(f"\n{'='*60}")
    print(f"  {BOLD}Sweep — {n_combos} configurations{RESET}")
    print(f"  Seeds      : {list(seeds)}")
    print(f"  Test sizes : {list(test_sizes)}")
    print(f"  Imputer    : {imputer_strategy}")
    print(f"  Modes      : {list(modes)}")
    print(f"  Output     : {base_dir}")
    print(f"{'='*60}")
 
    summary_rows = []
 
    for run_idx, (seed, test_size, mode) in enumerate(combos, 1):
        tag     = f"seed{seed}_split{str(test_size).replace('.','')}_{mode}"
        run_dir = base_dir / tag
 
        print(f"\n{'─'*60}")
        print(f"  Run {run_idx}/{n_combos}:  seed={seed}  test_size={test_size}  mode={mode}")
        print(f"  Output → {run_dir}")
        print(f"{'─'*60}")
 
        try:
            result = run_training(
                feature_csv  = feature_csv,
                mode         = mode,
                binary_mode  = binary_mode,
                test_size    = test_size,
                n_outer      = n_outer,
                n_inner      = n_inner,
                n_iter       = n_iter,
                seed         = seed,
                drop_nan     = drop_nan,
                imputer_strategy = imputer_strategy,
                save_dir     = run_dir,
            )
 
            cv_df    = result["cv_results"]
            best_row     = cv_df[cv_df["model"] == result["best_name"]].iloc[0]
            base_row     = cv_df[cv_df["model"] == "Baseline (majority)"].iloc[0]
            metrics      = result["best_metrics"]
            cv_test_diff = round(best_row["bal_acc_mean"] - metrics["test_bal_acc"], 4)

            # ---- McNemar: best vs all others ----
            preds  = result["all_predictions"]              # Test set predictions for all models
            y_test = result["y_test"]                       # True test labels          
            mcnemar = {}                                    # Intialise dict to hold McNemar p-values for best vs each other model
            for name, y_pred_other in preds.items():
                if name == result["best_name"]:
                    continue
                theta_hat, p, n12, n21, ci = _mcnemar_test(
                    y_test.values,
                    preds[result["best_name"]],
                    y_pred_other,
                )
                key = f"mcnemar_p_{name.lower().replace(' ','_')}_vs_best"
                mcnemar[key] = p
                mcnemar[key.replace("_p_", "_theta_")] = round(theta_hat, 4) # Store p-value for best vs this other model
 
            summary_rows.append({
                "run":            run_idx,                                              # Run index (1-based)
                "seed":           seed,                                                 # Seed used for this run
                "test_size":      test_size,                                            # Test set size used for this run
                "mode":           mode,                                                 # Classification mode used for this run
                "best_model":     result["best_name"],                                  # Best model name from CV for this run
                "cv_bal_acc":     round(best_row["bal_acc_mean"], 4),                   # Mean balanced accuracy from CV for the best model
                "cv_bal_std":     round(best_row["bal_acc_std"],  4),                   # Std of balanced accuracy from CV for the best model
                "test_bal_acc":   metrics["test_bal_acc"],                              # Test set balanced accuracy for the best model
                "test_acc":       metrics["test_acc"],                                  # Test set accuracy for the best model
                "test_f1":        metrics["test_f1"],                                   # Test set F1 score (weighted) for the best model
                "test_prec":      metrics["test_prec"],                                 # Test set precision (weighted) for the best model
                "test_rec":       metrics["test_rec"],                                  # Test set recall (weighted) for the best model
                "cv_test_diff":   cv_test_diff,                                         # Difference between CV balanced accuracy and test set balanced accuracy for the best model. Flag for potential overfitting if large negative.
                "beats_baseline": best_row["bal_acc_mean"] > base_row["bal_acc_mean"],  # Whether the best model's CV balanced accuracy beats the baseline's CV balanced accuracy
                "baseline_acc":   round(base_row["bal_acc_mean"], 4),                   # Baseline (majority class) balanced accuracy from CV
                **mcnemar,                                                              # Add McNemar p-values for best vs each other model
                "status":         "ok",
                })
 
        except Exception as e:
            print(f"\n  {RED}ERROR in run {run_idx}: {e}{RESET}")
            summary_rows.append({
                "run": run_idx, "seed": seed, "test_size": test_size,
                "mode": mode, "best_model": "—", "cv_bal_acc": float("nan"),
                "cv_bal_std": float("nan"), "status": f"ERROR: {e}",
            })
 
    # ── print and save summary ────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = base_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
 
    print(f"\n{'='*60}")
    print(f"  {GREEN}{BOLD}Sweep complete.{RESET}")
    print(f"\n  Summary:")
    print(summary_df.to_string(index=False))
    print(f"\n  Saved summary → {summary_csv}")
    print(f"{'='*60}\n")
 
    return summary_df
 
 
# ================================================================================
# CLI
# ================================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train RBD classifiers with two-layer CV. Use --sweep for multi-config runs."
    )
    parser.add_argument("feature_csv", type=str)
    parser.add_argument("--mode",        type=str,   default="binary",
                        choices=["binary", "multiclass"])
    parser.add_argument("--binary-mode", type=str,   default="control_vs_all",
                        choices=["control_vs_all", "control_vs_irbd"])
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--n-outer",     type=int,   default=5)
    parser.add_argument("--n-inner",     type=int,   default=5)
    parser.add_argument("--n-iter",      type=int,   default=20)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--drop-nan",    action="store_true", default=False)
    parser.add_argument("--save-dir",    type=str,   default="reports/evaluation")
    parser.add_argument("--imputer",     type=str,   default="knn",
                        choices=["median", "mean", "most_frequent", "knn"],
                        help="Imputation strategy for NaN values (default: knn). If --drop-nan is set, this is ignored.")

    # Sweep options
    parser.add_argument("--sweep",       action="store_true",
                        help="Run over multiple seeds/splits/modes")
    parser.add_argument("--seeds",       type=int,   nargs="+", default=[42, 0, 1])
    parser.add_argument("--test-sizes",  type=float, nargs="+", default=[0.2, 0.25])
    parser.add_argument("--modes",       type=str,   nargs="+", default=["binary", "multiclass"])
    parser.add_argument("--sweep-dir",   type=str,   default="reports/sweep")

    args = parser.parse_args()

    if args.sweep:
        sweep_training(
            feature_csv      = args.feature_csv,
            seeds            = args.seeds,
            test_sizes       = args.test_sizes,
            modes            = args.modes,
            binary_mode      = args.binary_mode,
            n_outer          = args.n_outer,
            n_inner          = args.n_inner,
            n_iter           = args.n_iter,
            drop_nan         = args.drop_nan,
            imputer_strategy = args.imputer,
            save_base_dir    = args.sweep_dir,
        )
    else:
        run_training(
            feature_csv      = args.feature_csv,
            mode             = args.mode,
            binary_mode      = args.binary_mode,
            test_size        = args.test_size,
            n_outer          = args.n_outer,
            n_inner          = args.n_inner,
            n_iter           = args.n_iter,
            seed             = args.seed,
            drop_nan         = args.drop_nan,
            save_dir         = args.save_dir,
            imputer_strategy = args.imputer,
        )