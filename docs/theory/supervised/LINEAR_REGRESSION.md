# Linear Regression — Complete Theory Guide
> From first principles to production-ready understanding.
> Read this BEFORE touching any code.

---

## 1. What Problem Does It Solve?

Given a set of inputs `X`, predict a **continuous number** `y`.

```
EXAMPLES:
  Input (X)                          Output (y)
  ─────────────────────────────────────────────────
  [house size, rooms, location]   →  price ($)
  [hours studied, sleep hours]    →  exam score
  [temperature, humidity]         →  energy usage (kWh)
  [engine RPM, load]              →  fuel consumption
```

The word "linear" means the relationship is a **straight line** (or flat plane in higher dimensions).

---

## 2. The Model Formula

### Single feature (1 input):
```
y_pred = w × x + b

Example: price = 3000 × size + 50000
                  ↑              ↑
              weight (slope)   bias (intercept)
```

### Multiple features (n inputs):
```
y_pred = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Example: price = 3000×size + 15000×rooms + 500×age + 50000

In matrix form (compact notation):
y_pred = X @ w + b
         ↑   ↑   ↑
         data weights offset
```

**Matrix form is the same formula — just written compactly so it works for any number of features.**

### Variable definitions:
| Symbol | Name | Shape | What it means |
|---|---|---|---|
| `X` | Feature matrix | (n, m) | n samples, m features each |
| `w` | Weight vector | (m,) | importance of each feature |
| `b` | Bias scalar | scalar | baseline prediction |
| `y_pred` | Predictions | (n,) | model's guesses |
| `y` | True labels | (n,) | actual values from data |
| `n` | Sample count | int | number of data points |
| `m` | Feature count | int | number of input features |

---

## 3. What Does the Model Learn?

The model learns the values of `w` and `b`.

**Before training:** `w = 0`, `b = 0` → predicts everything as 0 (useless)

**After training:** `w ≈ 3000`, `b ≈ 50000` → predicts house prices well

The training process finds `w` and `b` that minimize prediction errors.

---

## 4. How Do We Measure Error? (Loss Function)

**Mean Squared Error (MSE):**
```
MSE = (1/n) × Σᵢ (y_pred[i] - y[i])²
```

Step-by-step with 3 samples:
```
Actual:    y     = [200000, 350000, 150000]
Predicted: y_pred = [210000, 330000, 160000]

Errors:           = [10000,  -20000,  10000]   ← can be +/-
Squared:          = [1e8,     4e8,    1e8]      ← always positive
MSE = mean         = (1e8 + 4e8 + 1e8) / 3 = 2e8
```

**Why square the errors?**
- Positive and negative errors would cancel without squaring
- Squaring penalizes LARGE errors more (10000² = 10x worse than 1000² relative)
- Makes the math smooth for gradient computation (differentiable everywhere)

---

## 5. How Does the Model Improve? (Gradient Descent)

### The Big Idea

Imagine the loss as a bowl-shaped valley. You're blindfolded at a random point.
How do you reach the bottom (minimum loss)?
→ Feel which direction goes downhill. Take a small step. Repeat.

```
Loss
  │   *                   ← start (w=0, high loss)
  │     *
  │       *
  │         *  *
  │              * * *    ← converging to minimum
  │                   ───  minimum (best w and b)
  └──────────────────────── epochs
```

### The Math

The gradient tells us: "if I increase `w` by a tiny amount, how much does loss change?"

**Gradient for w:**
```
dLoss/dw = (2/n) × X^T @ (y_pred - y)
         = (2/n) × X^T @ error
```

**Gradient for b:**
```
dLoss/db = (2/n) × Σ (y_pred - y)
         = (2/n) × sum(error)
```

**Update rule (gradient descent step):**
```
w = w - lr × dLoss/dw
b = b - lr × dLoss/db
```

### Why subtract the gradient?

- Gradient points "uphill" (direction of increasing loss)
- We want to go "downhill" (decreasing loss)
- So we SUBTRACT the gradient
- `lr` (learning rate) controls step size

### Deriving the gradient (chain rule)

```
Loss = (1/n) × Σ (y_pred - y)²

Let error = y_pred - y = (X @ w + b) - y

dLoss/dw = (1/n) × Σ 2 × error × d(error)/dw
         = (1/n) × Σ 2 × error × X        ← d(Xw+b)/dw = X
         = (2/n) × X^T @ error             ← vectorized form
```

---

## 6. Learning Rate — The Most Important Hyperparameter

```
lr too HIGH (e.g. 1.0):
  Step is too large → overshoots minimum → loss bounces or explodes
  Loss:  500 → 600 → 400 → 700 → 300 → NaN   ← explodes

lr too LOW (e.g. 0.0000001):
  Step is too small → takes forever → still far from minimum after 1000 epochs
  Loss:  500 → 499 → 498 → 497 → ...   ← too slow

lr just right (e.g. 0.0001):
  Converges smoothly to minimum
  Loss:  500 → 200 → 80 → 30 → 10 → 5 → 4.9 → 4.9   ← converged
```

**Rule of thumb:** Start with `lr = 0.001`. If loss explodes → divide by 10. If too slow → multiply by 10.

---

## 7. Feature Normalization (Critical for Multi-Feature)

### The Problem

Imagine predicting house price from size (m²) and age (years):
```
Size:  50 to 200 (range: 150)
Age:   1 to 50   (range: 49)
```

The weight for size needs to be tiny (e.g. 3000).
The weight for age is much larger (e.g. -500 per year).

Gradient descent struggles when features have very different scales because:
- Large-scale features dominate gradients
- Small-scale features are learned very slowly
- Need different learning rates for different features (hard to tune)

### The Solution: Standardization (Z-score normalization)

```
X_normalized = (X - mean(X)) / std(X)

After normalization:
  mean = 0, std = 1  (for every feature)
  All features are on the same scale
```

**Important:** Compute mean and std on TRAINING data only. Apply same values to test data.

```python
mean = X_train.mean(axis=0)   # shape: (m,) — one mean per feature
std  = X_train.std(axis=0)    # shape: (m,) — one std per feature

X_train_norm = (X_train - mean) / std
X_test_norm  = (X_test - mean) / std   # use TRAIN mean and std!
```

---

## 8. Evaluation Metrics

### R² (R-squared)
```
R² = 1 - SS_res / SS_tot

SS_res = Σ (y - y_pred)²    ← error of the model
SS_tot = Σ (y - mean(y))²   ← total variation in data

R² = 1.0  →  perfect model
R² = 0.0  →  model is no better than predicting the mean
R² < 0.0  →  model is WORSE than predicting the mean (bad!)
```

### RMSE (Root Mean Squared Error)
```
RMSE = sqrt(MSE)
     = sqrt(mean((y_pred - y)²))

Benefit: same unit as y
  If predicting house prices in $, RMSE is in $ (interpretable!)
  RMSE = 15000 means "average prediction error is $15,000"
```

---

## 9. Overfitting vs Underfitting

```
UNDERFITTING                GOOD FIT              OVERFITTING
──────────────────────────────────────────────────────────────
Model too simple            Just right            Model too complex
Can't learn the pattern     Generalizes well      Memorizes training data

Train R² = 0.50             Train R² = 0.92       Train R² = 0.99
Test  R² = 0.48             Test  R² = 0.89       Test  R² = 0.45
```

**How to detect:**
- Large gap between train R² and test R² → overfitting
- Both train and test R² are low → underfitting

**How to fix overfitting in linear regression:**
- L1 regularization (Lasso): adds λ × Σ|w| to loss → drives some weights to 0
- L2 regularization (Ridge): adds λ × Σw² to loss → shrinks all weights toward 0

---

## 10. When to Use Linear Regression

| Condition | Use Linear Regression |
|---|---|
| Output is continuous number | ✅ |
| Relationship is approximately linear | ✅ |
| Need interpretable model (explain weights) | ✅ |
| Fast training and prediction needed | ✅ |
| Output is a category (yes/no) | ❌ → use Logistic Regression |
| Relationship is highly nonlinear | ❌ → use Random Forest or NN |
| Very few features with complex interactions | ❌ → use Decision Tree |

---

## 11. Files for This Algorithm

| File | Purpose | Run with |
|---|---|---|
| `src/mlzero/supervised/regression/linear.py` | Implementation | `from mlzero...` |
| `scripts/supervised/linear_regression_demo.py` | Full demo + experiments | `python scripts/...` |
| `scripts/supervised/linear_regression_sklearn.py` | sklearn comparison | `python scripts/...` |
| `tests/test_supervised/test_linear.py` | Unit tests | `pytest tests/...` |
| `configs/supervised/linear_regression.yaml` | Hyperparameters | loaded in scripts |
| `notebooks/experiments/01_linear_regression.ipynb` | Interactive exploration | `jupyter notebook` |

---

## 12. Progress Checklist

- [ ] Read this guide fully
- [ ] Understand MSE formula (can compute by hand with 3 samples)
- [ ] Understand gradient descent (can explain the bowl analogy)
- [ ] Understand learning rate effect (can predict: what happens if lr=0.1?)
- [ ] Run demo: `python scripts/supervised/linear_regression_demo.py`
- [ ] Run experiments: change lr, epochs, observe results
- [ ] Run tests: `pytest tests/test_supervised/ -v` — all pass
- [ ] Compare to sklearn: weights match within rounding
- [ ] Implement multi-feature version (Exercise 4 in demo)
- [ ] Understand normalization: why is it needed?
- [ ] Update progress in `docs/guides/LEARNING_GUIDE.md`

---

## 13. Key Formulas — Quick Reference

```
Model:        y_pred = X @ w + b
Loss (MSE):   L = (1/n) × ||y_pred - y||²
Gradient w:   dL/dw = (2/n) × X.T @ (y_pred - y)
Gradient b:   dL/db = (2/n) × sum(y_pred - y)
Update w:     w = w - lr × dL/dw
Update b:     b = b - lr × dL/db
Normalize:    X_norm = (X - mean) / std
R²:           1 - SS_res / SS_tot
RMSE:         sqrt(mean((y_pred - y)²))
```

---

*Next: Logistic Regression — same idea, but for classification (yes/no output).*
