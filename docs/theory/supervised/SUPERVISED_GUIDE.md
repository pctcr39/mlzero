# Supervised Learning — Complete Guide
> Read this before opening any file in `01_supervised/`

---

## What is Supervised Learning?

**Supervised = learning with a teacher.**

You give the model:
- `X` = inputs (features) — what the model sees
- `y` = outputs (labels) — the correct answer

The model learns the relationship: `X → y`

```
Training data:
  X (features)          y (label)
  ─────────────────────────────────
  [size=80m², rooms=3]  → price=250,000
  [size=60m², rooms=2]  → price=180,000
  [size=120m², rooms=4] → price=380,000

Model learns: "bigger house → higher price"

New house: [size=95m², rooms=3]
Model predicts: price ≈ 295,000   ← no label needed, model guesses
```

---

## Two Types of Supervised Learning

### 1. Regression — predict a NUMBER

```
Input → continuous output

Examples:
  House features → price (e.g. 245,000)
  Weather data   → temperature (e.g. 23.5°C)
  Study hours    → exam score (e.g. 87.4)
```

### 2. Classification — predict a CATEGORY

```
Input → discrete output (one of N classes)

Binary (2 classes):
  Email   → spam or not spam (0 or 1)
  Patient → sick or healthy (0 or 1)

Multi-class (N classes):
  Image   → cat, dog, or bird (0, 1, or 2)
  Digit   → 0, 1, 2, ..., 9
```

---

## The Universal Training Pipeline

This same pipeline runs for EVERY supervised algorithm:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  RAW DATA                                                       │
│  ─────────                                                      │
│  CSV file, database, API, sensor                                │
│       ↓                                                         │
│  PREPROCESSING                                                  │
│  ─────────────                                                  │
│  • Handle missing values (NaN)                                  │
│  • Normalize / scale features                                   │
│  • Encode categorical variables                                 │
│       ↓                                                         │
│  SPLIT DATA                                                     │
│  ──────────                                                     │
│  X_train, y_train (80%)  ← model learns from this              │
│  X_test,  y_test  (20%)  ← model evaluated on this ONLY        │
│       ↓                                                         │
│  TRAIN MODEL                                                    │
│  ───────────                                                    │
│  for each epoch:                                                │
│    y_pred = model.predict(X_train)      ← forward pass         │
│    loss   = loss_fn(y_train, y_pred)    ← measure error        │
│    grads  = compute_gradients(loss)     ← which direction?     │
│    params = optimizer.step(params, grads) ← update weights     │
│       ↓                                                         │
│  EVALUATE                                                       │
│  ────────                                                       │
│  y_pred_test = model.predict(X_test)                           │
│  score = metrics(y_test, y_pred_test)  ← final report card     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Split Data Into Train and Test?

This is one of the most important concepts in ML.

```
WRONG approach (no split):
  Train on 100% of data → test on same 100%
  Model just memorizes the answers → 100% accuracy
  Give it new data → fails completely
  This is called OVERFITTING.

CORRECT approach (train/test split):
  Train on 80% → test on the 20% it never saw
  If it performs well on test → it actually LEARNED
  Not just memorized
```

Think of it like school:
- Training data = practice problems with answers
- Test data = the actual exam (new problems, no answers given)

---

## Overfitting vs Underfitting

```
                        Model Complexity
    Low ────────────────────────────────────── High
     │                                          │
     │  UNDERFITTING          GOOD FIT     OVERFITTING
     │  ─────────────         ────────     ───────────
     │  Model too simple      Balance      Model too complex
     │  Can't learn pattern   ✓            Memorizes training data
     │  Low train accuracy                 High train accuracy
     │  Low test accuracy                  Low test accuracy
     │
```

**Fix underfitting:** more complex model, more features, more epochs
**Fix overfitting:** regularization (L1/L2), more data, simpler model, dropout

---

## Algorithms in `01_supervised/`

### Regression (predicting numbers)
```
regression/
├── linear_scratch.py    ← y = wX + b, gradient descent from scratch
├── linear_sklearn.py    ← same using sklearn (5 lines)
├── polynomial.py        ← y = w₁X + w₂X² + b (curved lines)
└── regularized.py       ← Ridge (L2) and Lasso (L1) to prevent overfitting
```

### Classification (predicting categories)
```
classification/
├── logistic_scratch.py  ← sigmoid + cross-entropy, from scratch
├── decision_tree.py     ← tree of if/else rules, from scratch
├── random_forest.py     ← ensemble of decision trees
├── svm.py               ← maximum margin classifier
└── knn.py               ← k-nearest neighbors (no training!)
```

### Evaluation
```
evaluation/
├── metrics.py           ← accuracy, precision, recall, F1, R²
└── cross_validation.py  ← k-fold, proper train/val/test setup
```

---

## Comparison: When to Use Each Algorithm

| Algorithm | Speed | Accuracy | Interpretable | Notes |
|---|---|---|---|---|
| Linear Regression | Fast | Medium | Yes | Start here for regression |
| Logistic Regression | Fast | Medium | Yes | Start here for classification |
| Decision Tree | Fast | Medium | Yes | Easy to visualize and explain |
| Random Forest | Medium | High | No | Often beats individual trees |
| SVM | Slow | High | No | Works well on small datasets |
| KNN | No training | Medium | Yes | Slow at prediction time |

---

## How to Read Each Algorithm File

Every algorithm file has this structure:

```
"""
THEORY: What this algorithm does conceptually
MATH:   The formula being implemented
"""

import numpy as np
from core.base import BaseModel    ← inherits the interface

class AlgorithmName(BaseModel):
    def __init__(self, hyperparameters):
        """Initialize with hyperparameters."""

    def fit(self, X, y):
        """Learn from training data. Implements gradient descent loop."""

    def predict(self, X):
        """Apply learned parameters to new data."""

    def score(self, X, y):
        """Evaluate model performance."""


# ---- EXERCISES AT BOTTOM ----
# Run this file directly to see it in action
if __name__ == "__main__":
    # demo code
```

---

## Variable Naming Convention (used in all files)

| Variable | Meaning | Shape |
|---|---|---|
| `X` | Input features (capital = matrix) | (n_samples, n_features) |
| `y` | Labels/targets | (n_samples,) |
| `w` | Weights (slope) | (n_features,) |
| `b` | Bias (intercept) | scalar |
| `y_pred` | Model predictions | (n_samples,) |
| `lr` | Learning rate | scalar |
| `n` | Number of samples | int |
| `m` | Number of features | int |

---

## Start Here

```
1. Open regression/linear_scratch.py
2. Read top docstring (theory)
3. Read __init__: what hyperparameters?
4. Read fit(): what is the training loop?
5. Read predict(): how does it use what it learned?
6. Run the file: python 01_supervised/regression/linear_scratch.py
7. Do the exercises at the bottom
```

---

*After regression → classification → evaluation → Phase 2 complete.*
