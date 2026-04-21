# Filename: evaluate.py
# Authors: Adam Klovborg & Rasmus Kleffel
# Description: Evaluates the trained model on the test set and computes performance metrics.
#              Computes confusion matrix, ROC curve (per class for multiclass), 
#              feature importance bar chart and saves plots.

# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations  # for Python 3.10+ type hinting features

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


# =====================================================================
# Feature Importance
# =====================================================================


