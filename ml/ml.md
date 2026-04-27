# Machine learning (ML) theories

- [Supervised learning](#supervised-learning)
- [Models and the prediction function](#models-and-the-prediction-function)
- [Training error vs. generalization error](#training-error-vs-generalization-error)
- [Overfitting and underfitting](#overfitting-and-underfitting)
- [Cross-validation](#cross-validation)
- [Two-layer cross-validation](#two-layer-cross-validation)
- [Regularization](#regularization)
- [Bias-variance decomposition](#bias-variance-decomposition)
- [Feature selection](#feature-selection)
- [Performance metrics and class imbalance](#performance-metrics-and-class-imbalance)
- [Statistical evaluation of models](#statistical-evaluation-of-models)
- [References](#references)

### Supervised learning
In supervised machine learning the task is to predict a quantity based on other quantities. It is useful to distinguish between classification and regression (Ch. 1.2.1, Ch. 8).
- **Classification:** In classification we are given observed values $x$ and have to predict a discrete response $y$. I.e., we are given discrete observations of some object and have to determine what class the object belongs to. In our case, $x$ are EOG-derived features from a patient's polysomnography (PSG) recording and $y$ is the diagnostic label (RBD, PD, or control). This is a multi-class classification problem.
- **Regression:** In regression we are given observed values $x$ and have to predict a continuous response $y$. For example, predicting $P(\text{RBD})$ — the probability a patient has RBD — is a regression-like output even though the underlying task is classification.

---

### Models and the prediction function
In supervised learning we are given a training set ($\mathcal{D}^{\text{train}}$) comprised of $N$ observations $x_1, \ldots, x_N$ and $N$ targets $y_1, \ldots, y_N$ and we wish to predict $y$ from $x$ (Ch. 8):
$$
y = f(x, w) + \varepsilon,
$$
where $w$ is a vector of tunable parameters and $\varepsilon$ represents a noise term. Learning consists of selecting the parameters $w$ based on the training data $X, y$. A computer program which carries out this process — i.e., based on a training set $\mathcal{D}^{\text{train}}$ it constructs a function $f$ — is known as a **model** (Ch. 1.3.1). Different models are denoted $\mathcal{M}_1, \ldots, \mathcal{M}_S$.

Once the function $f$ is learned during the **training phase**, it can be used to predict targets for new, unobserved data known as the **test set** $\mathcal{D}^{\text{test}}$ (Ch. 1.3.1).

In our project, a model learns which combinations of EOG features (e.g., eye movement counts per sleep stage, EOG amplitude statistics, phasic/tonic epoch ratios) best distinguish RBD patients from controls using PSG recordings from the Danish Center for Sleep Medicine (DCSM).

---

### Training error vs. generalization error
For a model $\mathcal{M}_s$, the **training error** is the average error on the training set (Equation 10.1):
$$
E^{\text{train}}_{\mathcal{M}_s} = \frac{1}{N^{\text{train}}} \sum_{i \in \mathcal{D}^{\text{train}}} L(y_i, f_{\mathcal{M}_s}(x_i, w)),
$$
where $L$ is a loss function (e.g., squared error for regression, misclassification for classification).

The **generalization error** is the expected error on all future, unseen data drawn from the true distribution (Equation 10.2):
$$
E^{\text{gen}}_{\mathcal{M}} = \mathbb{E}_{(x,y)}[L(y, f_{\mathcal{M}}(x))].
$$

This is the quantity that ultimately decides which model is better. Since the training set was used to train the model, we should expect $E^{\text{train}}_{\mathcal{M}}$ to be lower than $E^{\text{gen}}_{\mathcal{M}}$. The problem is that we cannot compute $E^{\text{gen}}_{\mathcal{M}}$ directly since we don't know the true distribution of the data — this is why we need cross-validation.

In our project, if we train a classifier on all DCSM patients and report accuracy on those same patients, that number is misleadingly optimistic. The model may have learned patient-specific quirks rather than genuine RBD biomarkers, and its performance on new patients would likely be worse.

---

### Overfitting and underfitting
A model that is too complex can **overfit** the data: it fits the training set very well (low training error) but generalizes poorly to new data (high generalization error). It has learned noise or patient-specific quirks rather than genuine patterns (Ch. 10). Conversely, a model that is too simple will **underfit**: it cannot capture the real relationships in the data (Ch. 14).

In our project, our feature pipeline extracts ~170 EOG-derived features per patient (eye movement counts, amplitude statistics, spectral features, phasic/tonic ratios, etc.) from a cohort of DCSM patients. A very complex model could memorize idiosyncrasies of specific recordings — for example, learning artifact patterns unique to certain patients rather than genuine RBD signatures in EOG microstructure. On the other hand, using only a single feature (e.g., total eye movement count) with a simple threshold would underfit, missing the multivariate patterns across sleep stages that distinguish RBD.

---

### Cross-validation
Cross-validation is the principal way of estimating the generalization error when we cannot compute it directly (Ch. 10.1.3). Three common approaches:

- **Hold-out method:** Split the full dataset $\mathcal{D}$ into $\mathcal{D}^{\text{train}} \cup \mathcal{D}^{\text{test}}$, train on $\mathcal{D}^{\text{train}}$, compute the test error on $\mathcal{D}^{\text{test}}$, and use $E^{\text{gen}}_{\mathcal{M}} \approx E^{\text{test}}_{\mathcal{M}}$. Simple, but the estimate depends on the particular split.

- **K-fold cross-validation:** Split the full dataset into $K$ pieces $\mathcal{D}_1, \ldots, \mathcal{D}_K$. For each $k$, treat $\mathcal{D}_k$ as the test set and the remaining $K-1$ pieces as the training set. The generalization error is estimated as the weighted average of the $K$ test errors. Since each data point is used once in the test set, this method is generally more precise than the hold-out method, but requires $K$ times more computation (Ch. 10.1.3).

- **Leave-one-out cross-validation:** K-fold CV with $K = N$. Each model is trained on all data except a single observation and tested on that observation. Uses as much data as possible for training, but requires $N$ models to be trained. Overall, 10-fold cross-validation is recommended (Ch. 10.1.3).

Cross-validation can also be used for **model selection**: we estimate the generalization error ($E^{\text{gen}}_{\mathcal{M}}$) for each model $\mathcal{M}_1, \ldots, \mathcal{M}_S$ and select the one with the lowest estimated generalization error (Ch. 10.1.4).

In our project, with a limited DCSM cohort split across RBD, PD, and control groups, K-fold CV (e.g., $K = 10$) is essential. A single hold-out split could, by chance, put most RBD patients in the training set and leave too few for a meaningful test, making the generalization estimate unreliable.

---

### Two-layer cross-validation
If we use cross-validation to both select the best model and estimate its generalization error, the estimate will be too optimistic: the selected model always looks better than it truly is because it was chosen as the minimum among several noisy estimates (Ch. 10.1.5).

Two-layer (nested) cross-validation solves this:
- **Outer loop:** Splits data into train/test for estimating the true generalization error.
- **Inner loop:** Within each outer training fold, performs cross-validation to select the best model or hyperparameters.

In our project, the inner loop might determine the optimal hyperparameters (e.g., regularization strength $\lambda$, number of selected features, or tree depth), while the outer loop provides an honest estimate of how well that optimally-tuned model will perform on truly unseen patients.

---

### Regularization
Regularization is a general technique for controlling model complexity to prevent overfitting (Ch. 14.1). For the linear regression model, regularization adds a penalty term to the error function:

$$
E_{\text{reg}}(w) = \left|\left| y - \tilde{X}w \right|\right|^2 + \lambda \cdot w^\top w
$$

The parameter $\lambda$ controls the strength of regularization: a large $\lambda$ forces $w$ toward zero (simpler model, higher bias), while $\lambda = 0$ gives the unregularized solution (more flexible, higher variance). The optimal $\lambda$ is selected using cross-validation (Ch. 14.1).

In our project, with ~170 EOG features and a limited patient cohort, regularization is essential. Without it, a model could find spurious correlations — for instance, fitting to noise in rarely-occurring eye movement types that happen to correlate with RBD labels in the training set but carry no real diagnostic value. Feature selection (reducing to a smaller optimal feature subset) also acts as an implicit form of regularization, constraining which features the model can use.

---

### Bias-variance decomposition
The generalization error can be decomposed into three terms (Equation 14.9):
$$
\mathbb{E}_{\mathcal{D}}[E^{\text{gen}}] = \mathbb{E}_x\left[ \underbrace{\text{Var}_{y|x}[y]}_{\text{irreducible noise}} + \underbrace{\text{Bias}^2}_{\text{systematic error}} + \underbrace{\text{Variance}}_{\text{model sensitivity}} \right]
$$

$$
\begin{align*}
\text{Bias} = \bar{y}(x) - \bar{f}(x) 
\\
\text{Variance} = \text{Var}_{\mathcal{D}}[f_{\mathcal{D}}(x)] 
\end{align*}
$$

- **Irreducible noise** ($\text{Var}_{y|x}[y]$): The intrinsic difficulty of the problem. It does not depend on the model and cannot be reduced (Ch. 14.2). In our project, this reflects inherent biological variability — two patients with very similar EOG features may still have different diagnoses due to factors not captured by EOG alone (e.g., chin EMG activity, genetic factors, disease stage).
- **Bias:** How far the model's average prediction is from the true value. A model that is too simple (e.g., a linear classifier applied to non-linear EOG patterns) has high bias.
- **Variance:** How much the model's predictions vary depending on the specific training set. A model that is too complex relative to the available data has high variance — swapping a few patients in or out of the training set would drastically change its predictions.

Reducing one term typically increases the other — this is the **bias-variance tradeoff**. The purpose of regularization in this context is to substantially reduce the variance without introducing too much bias (Ch. 14.2).

---

### Feature selection
With many features and few observations, selecting the most informative features improves generalization. The book describes two sequential approaches (Ch. 10.2):

- **Forward selection:** Start with no features. At each step, add the feature that most reduces the estimated generalization error.
- **Backward selection:** Start with all features. At each step, remove the feature whose removal most reduces the estimated generalization error.

These are not guaranteed to find the same subset or the globally optimal subset, but they are far more efficient than exhaustive search ($2^M$ models). For $M = 20$, sequential selection evaluates at most $\frac{M(M+1)}{2} + 1 = 211$ models versus over a million for exhaustive search (Ch. 10.2).

In our project, our pipeline extracts ~170 features spanning eye movement counts per stage, EOG amplitude statistics, phasic/tonic ratios, REM epoch durations, and spectral properties. Not all of these will be informative for distinguishing RBD — some may be redundant or noisy. Feature selection (via forward/backward selection or feature importance ranking) will identify which EOG markers carry the strongest diagnostic signal, both improving model performance and providing clinical insight into which aspects of EOG microstructure are most altered in RBD.

---

### Performance metrics and class imbalance
The **confusion matrix** summarizes classification performance (Ch. 8.2.1):

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

From this we compute (Ch. 8.2.1):
- **Accuracy** $= \frac{TP + TN}{N}$ and **Error rate** $= \frac{FP + FN}{N} = 1 - \text{Accuracy}$

However, accuracy can be misleading when classes are imbalanced (Ch. 16). In our project, if the DCSM cohort contains substantially more controls than RBD patients, a model that always predicts "no RBD" could still achieve high accuracy. Therefore we also use:
- **Sensitivity (recall)** $= \frac{TP}{TP + FN}$: Of all true RBD patients, how many were correctly identified? A missed RBD patient means missing early Parkinson's disease detection.
- **Specificity** $= \frac{TN}{TN + FP}$: Of all non-RBD patients, how many were correctly ruled out? False positives lead to unnecessary clinical follow-up.
- **AUC (Area Under ROC Curve):** Measures discriminative ability across all classification thresholds, providing a single number that captures the tradeoff between sensitivity and specificity.

---

### Statistical evaluation of models
To determine whether one model is significantly better than another, we need statistical tests beyond simply comparing estimated generalization errors (Ch. 11). Because cross-validation folds share training data, the estimates are correlated, making standard tests unreliable. The book discusses approaches based on the correlated $t$-test and credibility intervals that account for this dependency when comparing models (Ch. 11.4).

In our project, this is relevant when comparing different classifiers (e.g., Random Forest vs. logistic regression vs. decision tree) on our EOG features — we need to determine if any observed difference in AUC or accuracy is statistically significant or simply due to the particular data split.

---

### References
* *Introduction to Machine Learning and Data Mining* by Tue Herlau, Mikkel N. Schmidt and Morten Mørup (2023) — https://www.polyteknisk.dk/home/Detaljer/9788771252866