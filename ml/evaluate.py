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

# =====================================================================
# Confusion Matrix 
# =====================================================================
def plot_confusion_matrix(
        y_true, 
        y_pred,
        class_names:    list[str],
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
    save_path : str or Path, optional
        If provided, saves the plot to this path.
    """


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
# Feature Importance
# =====================================================================

def plot_feature_importance_MDI(X_train, model: None):
    """
    Plots feture importance based on Mean Decrease in Impurity (MDI) from the model. 
    """
    feature_name = X_train.columns
    MDI_importances = model.feature_importances_
    idx_MDI_sorted = np.argsort(MDI_importances)

    # Plot feature importance

    plt.figure(figsize=(10, 6))
    plt.barh(feature_name[idx_MDI_sorted], MDI_importances[idx_MDI_sorted])
    plt.xlabel('Mean Decrease in Impurity (MDI)', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.title('Feature Importance based on Mean Decrease in Impurity', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_feature_importance_permutation_importance(X_train, X_test, y_test, model: None):
    """
    Plots feature importance based on permutation importance from model.
    """

    feature_name = X_train.columns

    result = permutation_importance(model, X_test, y_test, 
    n_repeats=10, random_state=26, scoring='roc_auc')

    idx_P_sorted = result.importances_mean.argsort()
    P_importance = pd.DataFrame(result.importances[idx_P_sorted].T, 
    columns = feature_name[idx_P_sorted])

    # Plot feature importance

    plt.figure(figsize=(14, 6))
    plt.boxplot(P_importance.values, vert=False, 
    tick_labels = P_importance.columns, widths=0.3) 
    plt.xlabel('Permutation Importance (ROC AUC)', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.title('Feature Importance based on Permutation Importance', fontweight='bold')
    plt.tight_layout()
    plt.show()