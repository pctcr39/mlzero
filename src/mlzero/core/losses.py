"""
================================================================================
core/losses.py — Loss Functions
================================================================================

WHAT IS A LOSS FUNCTION?
    A loss function answers ONE question: "How wrong is my model right now?"

    Think of it like a score on a test — but instead of higher = better,
    LOWER loss = better model.

    Every ML training loop looks like this:
        1. Model makes a prediction
        2. Loss function measures the error
        3. Optimizer uses that error to improve the model
        4. Repeat until loss is small enough

WHY DO WE HAVE MULTIPLE LOSS FUNCTIONS?
    Different problems need different ways to measure "wrong":
    - Predicting a house price (number)?       → MSE or MAE
    - Predicting spam or not spam (yes/no)?    → Binary Cross-Entropy
    - Predicting which of 10 classes?          → Cross-Entropy

VARIABLES EXPLANATION (used in all functions below):
    y_true  — (LOCAL) The real/actual answers from your dataset
              Example: actual house prices [200000, 350000, 150000]
    y_pred  — (LOCAL) What your model predicted
              Example: predicted prices    [210000, 320000, 170000]
    n       — (LOCAL, implicit via np.mean) number of data points
================================================================================
"""

import numpy as np  # numpy is imported at MODULE level (global to this file)
                    # Every function below can use np without re-importing


# ==============================================================================
# 1. MEAN SQUARED ERROR (MSE)
# ==============================================================================

def mse(y_true, y_pred):
    """
    THEORY:
        Used for REGRESSION problems (predicting a continuous number).
        Measures the average of squared differences between prediction and truth.

        Formula:  MSE = (1/n) * Σ (y_pred - y_true)²

        WHY SQUARE?
        - Without squaring: positive errors cancel negative errors → misleading
        - Squaring makes ALL errors positive
        - Squaring also PUNISHES large errors more than small ones
          Example: error of 10 → 100 (10x worse than error of 1 → 1)

        GOAL: minimize this number toward 0

    PARAMETERS:
        y_true  (local) — numpy array of actual values,   shape: (n,)
        y_pred  (local) — numpy array of predicted values, shape: (n,)

    RETURNS:
        A single float number. Example: 245.7
        Interpretation: on average, predictions are off by sqrt(245.7) ≈ 15.7 units

    EXAMPLE:
        y_true = [3, 5, 2]
        y_pred = [4, 4, 3]
        errors =  [1, 1, 1]      ← (y_pred - y_true)
        squared = [1, 1, 1]      ← errors squared
        MSE = mean([1, 1, 1]) = 1.0
    """
    errors = y_pred - y_true      # LOCAL: difference at each data point
    squared = errors ** 2         # LOCAL: square each error (all become positive)
    return np.mean(squared)       # return average of all squared errors


# ==============================================================================
# 2. MEAN ABSOLUTE ERROR (MAE)
# ==============================================================================

def mae(y_true, y_pred):
    """
    THEORY:
        Also used for REGRESSION. Similar to MSE but uses absolute value instead of squaring.

        Formula:  MAE = (1/n) * Σ |y_pred - y_true|

        DIFFERENCE FROM MSE:
        - MAE treats all errors equally (an error of 10 is just 10x worse than error of 1)
        - MSE punishes large errors exponentially (10² = 100x worse)
        - Use MAE when your data has outliers (extreme values) you want to ignore
        - Use MSE when large errors are especially bad and you want to penalize them

    PARAMETERS:
        y_true  (local) — actual values
        y_pred  (local) — predicted values

    RETURNS:
        A single float. Interpretation: average error in original units.

    EXAMPLE:
        y_true = [100, 200, 300]
        y_pred = [110, 190, 310]
        errors =  [10, 10, 10]
        MAE = mean([10, 10, 10]) = 10.0
        → On average, predictions are off by 10 units
    """
    errors = y_pred - y_true          # LOCAL: raw error at each point
    abs_errors = np.abs(errors)       # LOCAL: absolute value (remove negatives)
    return np.mean(abs_errors)        # average of absolute errors


# ==============================================================================
# 3. BINARY CROSS-ENTROPY
# ==============================================================================

def binary_cross_entropy(y_true, y_pred):
    """
    THEORY:
        Used for BINARY CLASSIFICATION (output is 0 or 1, yes or no).
        Examples: spam detection, disease diagnosis, fraud detection.

        Your model outputs a PROBABILITY between 0 and 1.
        Example: 0.85 means "85% confident this is spam"

        Formula:  Loss = -(1/n) * Σ [y * log(p) + (1-y) * log(1-p)]

        HOW TO READ THIS:
        - When y_true = 1 (positive case):  loss = -log(p)
            → if model predicted p=0.9  → loss = -log(0.9) = 0.105  (small, good!)
            → if model predicted p=0.1  → loss = -log(0.1) = 2.303  (large, bad!)
        - When y_true = 0 (negative case):  loss = -log(1-p)
            → if model predicted p=0.1  → loss = -log(0.9) = 0.105  (small, good!)
            → if model predicted p=0.9  → loss = -log(0.1) = 2.303  (large, bad!)

        The log function naturally creates very large penalties when the model
        is very confident but very wrong.

    PARAMETERS:
        y_true  (local) — actual labels, values must be 0 or 1.  shape: (n,)
        y_pred  (local) — predicted probabilities, values 0.0 to 1.0.  shape: (n,)

    INTERNAL VARIABLE:
        y_pred (clipped) — WHY DO WE CLIP?
            log(0) = -infinity → would crash the program
            We clip to a tiny value (0.000000000000001) instead of 0
            This is a numerical stability trick used in ALL ML frameworks

    RETURNS:
        A single float. Goal is to minimize toward 0.
    """
    # CLIP: keep y_pred between 1e-15 and (1 - 1e-15) to avoid log(0)
    # This is a LOCAL operation — doesn't modify the original array
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Compute both terms of the formula:
    positive_term = y_true * np.log(y_pred)          # LOCAL: term when y=1
    negative_term = (1 - y_true) * np.log(1 - y_pred)  # LOCAL: term when y=0

    return -np.mean(positive_term + negative_term)    # negative because log gives negatives


# ==============================================================================
# 4. CROSS-ENTROPY (Multi-class)
# ==============================================================================

def cross_entropy(y_true_onehot, y_pred_probs):
    """
    THEORY:
        Used for MULTI-CLASS CLASSIFICATION (output is one of N categories).
        Examples: digit recognition (0-9), image classification (cat/dog/bird).

        WHAT IS ONE-HOT ENCODING?
        Instead of label = 2, we represent it as:
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  ← only the correct class is 1
        This is called "one-hot" — one position is "hot" (1), rest are 0.

        Formula:  Loss = -(1/n) * Σ Σ y * log(p)
        (sum over all classes, then average over all examples)

        INTUITION: Same as binary cross-entropy but for N classes instead of 2.

    PARAMETERS:
        y_true_onehot  (local) — one-hot labels.    shape: (n, num_classes)
                                 Example for 3 classes: [[0,1,0], [1,0,0], [0,0,1]]
        y_pred_probs   (local) — predicted probabilities per class. shape: (n, num_classes)
                                 Example: [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]]
                                 Each row must sum to 1.0 (that's what softmax ensures)

    RETURNS:
        A single float loss value.
    """
    # Clip to avoid log(0) — same reason as binary cross-entropy
    y_pred_probs = np.clip(y_pred_probs, 1e-15, 1 - 1e-15)

    # For each example, only the correct class contributes (because y=0 for others)
    per_class_loss = y_true_onehot * np.log(y_pred_probs)  # LOCAL: shape (n, num_classes)
    per_example_loss = np.sum(per_class_loss, axis=1)      # LOCAL: sum over classes → shape (n,)

    return -np.mean(per_example_loss)                      # average over all examples
