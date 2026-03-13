"""
================================================================================
core/metrics.py — Evaluation Metrics
================================================================================

WHAT ARE METRICS?
    After training is done, metrics answer: "How GOOD is my model?"

    IMPORTANT DISTINCTION:
    - Loss function: used DURING training to guide learning (gradient descent uses it)
    - Metric:        used AFTER training to evaluate the final model (humans read this)

    Example: MSE loss might be 245.7 → hard to interpret
             R² score might be 0.94   → easy: "94% of variance explained, good model!"

TYPES OF METRICS:
    - Regression metrics   → how close are predicted numbers to actual numbers?
    - Classification metrics → how often does the model pick the right category?

GLOBAL in this file:
    numpy as np — used by all functions
================================================================================
"""

import numpy as np


# ==============================================================================
# REGRESSION METRICS
# ==============================================================================

def r2_score(y_true, y_pred):
    """
    THEORY:
        R² (R-squared) — the most common metric for regression.
        Answers: "What fraction of the variation in y does my model explain?"

        Range: -∞ to 1.0
            R² = 1.0  → perfect model (every prediction is exact)
            R² = 0.0  → model is no better than just predicting the mean of y
            R² < 0.0  → model is WORSE than predicting the mean (very bad model)

        Formula:
            SS_res = Σ (y_true - y_pred)²     ← residual sum of squares (error of model)
            SS_tot = Σ (y_true - mean(y))²    ← total sum of squares (variation in data)
            R² = 1 - SS_res / SS_tot

        INTUITION:
            SS_tot is "how much variation exists in the data"
            SS_res is "how much variation the model failed to explain"
            R² = "fraction the model DID explain"

    EXAMPLE:
        y_true = [3, 5, 2, 8, 6]
        y_pred = [3.1, 4.9, 2.2, 7.8, 6.1]
        R² ≈ 0.99  → excellent model, explains 99% of variance

    PARAMETERS:
        y_true (local) — actual values, shape: (n,)
        y_pred (local) — predicted values, shape: (n,)

    RETURNS:
        float between -∞ and 1.0
    """
    ss_res = np.sum((y_true - y_pred) ** 2)          # LOCAL: error of the model
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) # LOCAL: total variation in data
    return 1 - (ss_res / ss_tot)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error — same as MSE but in the original units.

    Why use RMSE over MSE?
        If predicting house prices in dollars, MSE is in dollars² (hard to interpret).
        RMSE is in dollars (same unit as y), much easier to understand.

        RMSE = 15,000 means: "on average, predictions are off by $15,000"

    Formula: sqrt(mean((y_pred - y_true)²))
    """
    return np.sqrt(np.mean((y_pred - y_true) ** 2))  # sqrt brings back original units


# ==============================================================================
# CLASSIFICATION METRICS
# ==============================================================================

def accuracy(y_true, y_pred):
    """
    THEORY:
        The simplest classification metric.
        Fraction of predictions that were correct.

        Formula: accuracy = number_correct / total_predictions

        Range: 0.0 to 1.0 (or 0% to 100%)

    WHEN IT'S MISLEADING:
        Imagine 99% of emails are NOT spam.
        A model that always predicts "not spam" gets 99% accuracy.
        But it never catches any spam! Accuracy alone is not enough.
        → That's why we need Precision, Recall, and F1.

    PARAMETERS:
        y_true (local) — actual class labels, e.g. [0, 1, 0, 1, 1]
        y_pred (local) — predicted labels,   e.g. [0, 1, 1, 1, 0]

    RETURNS:
        float: 0.0 to 1.0
    """
    correct = np.sum(y_true == y_pred)   # LOCAL: count of correct predictions
    total = len(y_true)                  # LOCAL: total number of predictions
    return correct / total


def confusion_matrix_values(y_true, y_pred):
    """
    THEORY:
        For binary classification (0 or 1), there are 4 possible outcomes:

        ┌──────────────────┬────────────────────┬────────────────────┐
        │                  │  Predicted: YES(1)  │  Predicted: NO(0)  │
        ├──────────────────┼────────────────────┼────────────────────┤
        │ Actual: YES (1)  │  True Positive (TP)  │ False Negative (FN) │
        │ Actual: NO  (0)  │  False Positive (FP) │ True Negative (TN)  │
        └──────────────────┴────────────────────┴────────────────────┘

        TP = model said YES, and it was correct
        TN = model said NO, and it was correct
        FP = model said YES, but it was wrong  (false alarm)
        FN = model said NO, but it was wrong   (missed it!)

    RETURNS:
        dict with TP, TN, FP, FN counts
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))   # predicted positive, actually positive
    tn = np.sum((y_pred == 0) & (y_true == 0))   # predicted negative, actually negative
    fp = np.sum((y_pred == 1) & (y_true == 0))   # predicted positive, actually negative
    fn = np.sum((y_pred == 0) & (y_true == 1))   # predicted negative, actually positive
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def precision(y_true, y_pred):
    """
    THEORY:
        Of all the times the model said YES, how often was it right?

        Formula: TP / (TP + FP)

        WHEN IT MATTERS:
        When False Positives are costly.
        Example: spam filter — you don't want to mark real emails as spam (FP = bad)
        → want HIGH precision: when we say "spam", we're almost always right

        Range: 0.0 to 1.0
    """
    cm = confusion_matrix_values(y_true, y_pred)   # LOCAL: get all 4 values
    tp = cm['TP']
    fp = cm['FP']
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true, y_pred):
    """
    THEORY:
        Of all the actual YES cases, how many did the model catch?

        Formula: TP / (TP + FN)

        Also called: Sensitivity, True Positive Rate

        WHEN IT MATTERS:
        When False Negatives are costly.
        Example: cancer detection — you don't want to miss a real cancer (FN = deadly)
        → want HIGH recall: catch as many real cases as possible, even if some are false alarms

        Range: 0.0 to 1.0

    PRECISION vs RECALL TRADEOFF:
        Usually, increasing precision decreases recall and vice versa.
        You can't always have both high. Choose based on your problem.
    """
    cm = confusion_matrix_values(y_true, y_pred)
    tp = cm['TP']
    fn = cm['FN']
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """
    THEORY:
        Combines Precision and Recall into a single number.
        The harmonic mean of precision and recall.

        Formula: F1 = 2 * (precision * recall) / (precision + recall)

        A high F1 score requires BOTH precision and recall to be high.
        If either one is low, F1 is pulled down.

        Range: 0.0 to 1.0

        USE WHEN:
        You care about both precision and recall equally,
        or when class imbalance exists (one class is much more common).

    EXAMPLE:
        precision = 0.9, recall = 0.9  →  F1 = 0.9  (both high, F1 high)
        precision = 0.9, recall = 0.1  →  F1 = 0.18 (one low, F1 low)
    """
    p = precision(y_true, y_pred)   # LOCAL
    r = recall(y_true, y_pred)      # LOCAL
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def print_classification_report(y_true, y_pred):
    """
    Print a full summary of classification performance.
    Useful to call after model.predict() to see all metrics at once.
    """
    cm = confusion_matrix_values(y_true, y_pred)
    print("=" * 40)
    print("CLASSIFICATION REPORT")
    print("=" * 40)
    print(f"Accuracy  : {accuracy(y_true, y_pred):.4f}")
    print(f"Precision : {precision(y_true, y_pred):.4f}")
    print(f"Recall    : {recall(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print("-" * 40)
    print(f"TP={cm['TP']}  TN={cm['TN']}  FP={cm['FP']}  FN={cm['FN']}")
    print("=" * 40)
