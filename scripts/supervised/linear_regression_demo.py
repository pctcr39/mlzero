"""
================================================================================
scripts/supervised/linear_regression_demo.py
Linear Regression — Complete Learning Demo
================================================================================

PURPOSE:
    Walk through Linear Regression in 4 parts.
    Run each part, read the output, understand what is happening.

    Part 1: Single Feature  — simplest case, 1 input -> 1 output
    Part 2: Multi-Feature   — real-world case, many inputs -> 1 output
    Part 3: Normalization   — why scaling features matters
    Part 4: sklearn Compare — verify our scratch impl matches industry library

HOW TO RUN:
    source .venv/bin/activate
    python scripts/supervised/linear_regression_demo.py

PREREQUISITES:
    pip install -e .    <- must be done once to install mlzero package
================================================================================
"""

import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml

from mlzero.supervised.regression.linear import LinearRegression
from mlzero.core.metrics import r2_score, rmse


# ==============================================================================
# LOAD CONFIG
# ==============================================================================

with open("configs/supervised/linear_regression.yaml") as f:
    cfg = yaml.safe_load(f)

LR          = cfg["lr"]
EPOCHS      = cfg["epochs"]
N_SAMPLES   = cfg["dataset"]["n_samples"]
NOISE_STD   = cfg["dataset"]["noise_std"]
TRUE_W      = cfg["dataset"]["true_w"]
TRUE_B      = cfg["dataset"]["true_b"]
TRAIN_RATIO = cfg["dataset"]["train_ratio"]
SEED        = cfg["dataset"]["random_seed"]

np.random.seed(SEED)


# ==============================================================================
# HELPER
# ==============================================================================

def train_test_split(X, y, train_ratio=0.8):
    """
    Split arrays into train and test sets.

    WHY SPLIT? The model TRAINS on train set, EVALUATES on test set.
    Test set = data the model never saw during training.
    This checks if the model actually learned vs just memorized.
    """
    n     = len(X)
    split = int(n * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


def silent_fit(model, X, y):
    """Fit model without printing epoch logs (used for batch experiments)."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    model.fit(X, y)
    sys.stdout = old_stdout


# ==============================================================================
# PART 1: Single Feature
# ==============================================================================

def part1_single_feature():
    print("\n" + "="*65)
    print("PART 1: Single Feature  House Size -> Price")
    print("="*65)
    print("""
WHAT: 1 input (house size m2) -> 1 output (price $)
TRUE: price = 3000 x size + 50000  (model must discover these numbers)
GOAL: see gradient descent find w~3000, b~50000 from data alone
    """)

    X = np.random.rand(N_SAMPLES, 1) * 100
    y = TRUE_W * X.flatten() + TRUE_B + np.random.randn(N_SAMPLES) * NOISE_STD

    X_train, X_test, y_train, y_test = train_test_split(X, y, TRAIN_RATIO)

    print(f"Dataset : {N_SAMPLES} samples  ->  {len(X_train)} train / {len(X_test)} test")
    print(f"Training: lr={LR}, epochs={EPOCHS}\n")

    model = LinearRegression(lr=LR, epochs=EPOCHS)
    model.fit(X_train, y_train)

    train_r2  = model.score(X_train, y_train)
    test_r2   = model.score(X_test,  y_test)
    test_rmse = rmse(y_test, model.predict(X_test))

    print(f"\n{'Learned':<12}: w = {model.w[0]:,.1f}   b = {model.b:,.1f}")
    print(f"{'Expected':<12}: w ~ {TRUE_W:,.0f}   b ~ {TRUE_B:,.0f}")
    print(f"\nTrain R2 : {train_r2:.4f}")
    print(f"Test  R2 : {test_r2:.4f}  <- key metric (never seen in training)")
    print(f"Test RMSE: ${test_rmse:,.0f}  <- average prediction error")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Part 1: Single Feature Linear Regression", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(X_train.flatten(), y_train, alpha=0.5, s=30, label="Train", color="steelblue")
    ax.scatter(X_test.flatten(),  y_test,  alpha=0.7, s=40, label="Test",  color="orange", marker="^")
    x_line = np.linspace(0, 100, 200).reshape(-1, 1)
    ax.plot(x_line, model.predict(x_line), color="red", linewidth=2.5, label=f"Model R2={test_r2:.3f}")
    ax.set_xlabel("House Size (m2)"); ax.set_ylabel("Price ($)")
    ax.set_title("Data + Fitted Line"); ax.legend(); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax = axes[1]
    ax.plot(model.loss_history, color="crimson", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Training Loss Curve\n(drops = model is learning)"); ax.grid(alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig("outputs/plots/part1_single_feature.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Plot saved: outputs/plots/part1_single_feature.png")

    return model


# ==============================================================================
# PART 2: Multi-Feature
# ==============================================================================

def part2_multi_feature():
    print("\n" + "="*65)
    print("PART 2: Multi-Feature  Size + Rooms + Age -> Price")
    print("="*65)
    print("""
WHAT: 3 inputs [size, rooms, age] -> price
TRUE: price = 3000*size + 15000*rooms - 500*age + 40000

Matrix form:  y_pred = X @ w + b
  X shape: (n, 3)   n samples, 3 features each
  w shape: (3,)     3 weights to learn (one per feature)
  b shape: scalar   1 bias
    """)

    n = 200
    size  = np.random.rand(n) * 100 + 30
    rooms = np.random.randint(1, 6, n).astype(float)
    age   = np.random.rand(n) * 40
    X = np.column_stack([size, rooms, age])

    true_w = np.array([3000.0, 15000.0, -500.0])
    true_b = 40000.0
    y = X @ true_w + true_b + np.random.randn(n) * 8000

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Normalize: all features to mean=0, std=1
    # CRITICAL: compute statistics on TRAIN only, apply same to TEST
    feat_mean = X_train.mean(axis=0)   # shape (3,) - one mean per feature
    feat_std  = X_train.std(axis=0)    # shape (3,) - one std per feature

    X_train_n = (X_train - feat_mean) / feat_std
    X_test_n  = (X_test  - feat_mean) / feat_std   # use TRAIN stats!

    print(f"Dataset: {n} samples, 3 features -> shape {X.shape}")
    print(f"True weights: {true_w}  bias: {true_b}")
    print(f"\nNormalization: features rescaled to mean=0, std=1")
    print(f"  Before: size range 30-130, rooms 1-5, age 0-40")
    print(f"  After:  all features centered around 0 with std=1")
    print(f"\nTraining...\n")

    model = LinearRegression(lr=0.01, epochs=1000)
    silent_fit(model, X_train_n, y_train)

    train_r2  = model.score(X_train_n, y_train)
    test_r2   = model.score(X_test_n,  y_test)
    test_rmse = rmse(y_test, model.predict(X_test_n))

    print(f"Train R2  : {train_r2:.4f}")
    print(f"Test  R2  : {test_r2:.4f}")
    print(f"Test  RMSE: ${test_rmse:,.0f}")
    print(f"\nLearned weights (in normalized space): {model.w}")
    print(f"NOTE: weights look different from true_w because features are normalized.")
    print(f"      Denormalized weight for size: {model.w[0] / feat_std[0]:.0f}  (expected ~3000)")

    # Predicted vs Actual plot
    y_pred_test = model.predict(X_test_n)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Part 2: Multi-Feature Linear Regression", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_test, y_pred_test, alpha=0.5, s=30, color="steelblue")
    min_v, max_v = y_test.min(), y_test.max()
    ax.plot([min_v, max_v], [min_v, max_v], "r-", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($)"); ax.set_ylabel("Predicted Price ($)")
    ax.set_title(f"Predicted vs Actual (R2={test_r2:.3f})\nClose to diagonal = good")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    residuals = y_test - y_pred_test
    ax.scatter(y_pred_test, residuals, alpha=0.5, s=30, color="darkorange")
    ax.axhline(0, color="red", linewidth=2, linestyle="--")
    ax.set_xlabel("Predicted Price ($)"); ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot\nRandom scatter = no hidden patterns"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/part2_multi_feature.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Plot saved: outputs/plots/part2_multi_feature.png")


# ==============================================================================
# PART 3: Learning Rate Experiment
# ==============================================================================

def part3_learning_rate_experiment():
    print("\n" + "="*65)
    print("PART 3: Learning Rate Experiments")
    print("="*65)
    print("""
HYPOTHESIS: Different learning rates lead to different convergence behaviors.
  lr too HIGH -> steps too large -> loss oscillates or explodes
  lr too LOW  -> steps too tiny  -> converges very slowly
  lr just right -> smooth, fast convergence

We train the SAME data with 4 different lr values.
Observe loss curves to understand the effect.
    """)

    X = np.random.rand(100, 1) * 100
    y = 3000 * X.flatten() + 50000 + np.random.randn(100) * 5000
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Normalize for fair comparison
    mean, std = X_train.mean(), X_train.std()
    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    lr_values = [0.5, 0.1, 0.01, 0.001]
    results   = {}

    print(f"\n{'lr':<10} {'Final Loss':>15} {'Test R2':>10} {'Result':>15}")
    print("-"*55)

    for lr in lr_values:
        model = LinearRegression(lr=lr, epochs=200)
        silent_fit(model, X_train_n, y_train)

        final_loss = model.loss_history[-1]
        test_r2    = model.score(X_test_n, y_test)

        if np.isnan(final_loss) or np.isinf(final_loss) or final_loss > 1e12:
            status = "EXPLODED"
        elif test_r2 > 0.85:
            status = "CONVERGED"
        elif final_loss > model.loss_history[0] * 0.95:
            status = "TOO SLOW"
        else:
            status = "PARTIAL"

        results[lr] = (model.loss_history, test_r2)
        print(f"{lr:<10} {min(final_loss, 1e12):>15.2f} {test_r2:>10.4f} {status:>15}")

    print("""
CONCLUSIONS:
  lr=0.5   -> explodes (gradient step is too big, overshoots the minimum)
  lr=0.1   -> converges but slowly or oscillates
  lr=0.01  -> converges smoothly in ~100 epochs (sweet spot here)
  lr=0.001 -> converges but needs more epochs to reach the same loss

RULE: Start with lr=0.01 on normalized data. Watch the loss curve.
      If it oscillates: halve the lr. If too slow: double the lr.
    """)

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["red", "orange", "green", "blue"]

    for (lr, (loss_hist, r2)), color in zip(results.items(), colors):
        safe_hist = [min(v, 1e8) if not (np.isnan(v) or np.isinf(v)) else 1e8
                     for v in loss_hist]
        ax.plot(safe_hist, label=f"lr={lr}  (R2={r2:.3f})", color=color, linewidth=2)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss (log scale)", fontsize=11)
    ax.set_title("Effect of Learning Rate on Training\nLower final loss = better", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/part3_learning_rate.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Plot saved: outputs/plots/part3_learning_rate.png")


# ==============================================================================
# PART 4: Compare vs sklearn
# ==============================================================================

def part4_sklearn_comparison():
    print("\n" + "="*65)
    print("PART 4: Our Implementation vs sklearn")
    print("="*65)
    print("""
WHAT: Train same dataset with our scratch LinearRegression AND sklearn.
GOAL: If weights match -> our implementation is mathematically correct.

sklearn uses closed-form solution: w = (X^T X)^-1 X^T y  (exact answer)
Our version uses gradient descent: iteratively approaches the answer
They should converge to the same result if we use enough epochs + good lr.
    """)

    from sklearn.linear_model import LinearRegression as SklearnLR

    X = np.random.rand(150, 1) * 100
    y = 3000 * X.flatten() + 50000 + np.random.randn(150) * 5000
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    our_model = LinearRegression(lr=0.0001, epochs=2000)
    silent_fit(our_model, X_train, y_train)
    our_r2 = our_model.score(X_test, y_test)

    sk_model = SklearnLR()
    sk_model.fit(X_train, y_train)
    sk_r2 = sk_model.score(X_test, y_test)

    our_w, our_b   = our_model.w[0], our_model.b
    sk_w,  sk_b    = sk_model.coef_[0], sk_model.intercept_

    w_match = abs(our_w - sk_w)  / abs(sk_w)  < 0.03   # within 3%
    b_match = abs(our_b - sk_b)  / abs(sk_b)  < 0.03
    r2_match = abs(our_r2 - sk_r2)              < 0.01

    print(f"\n{'Metric':<15} {'Ours':>12} {'sklearn':>12} {'Match':>8}")
    print("-"*50)
    print(f"{'w (slope)':<15} {our_w:>12.2f} {sk_w:>12.2f} {'YES' if w_match else 'CLOSE':>8}")
    print(f"{'b (intercept)':<15} {our_b:>12.2f} {sk_b:>12.2f} {'YES' if b_match else 'CLOSE':>8}")
    print(f"{'Test R2':<15} {our_r2:>12.4f} {sk_r2:>12.4f} {'YES' if r2_match else 'CLOSE':>8}")

    if w_match and b_match:
        print("\nCONCLUSION: Our implementation is CORRECT.")
        print("Gradient descent found the same weights as sklearn's exact solver.")
    else:
        print("\nCONCLUSION: Close but not exact (gradient descent needs more epochs).")
        print("Try increasing epochs to 5000 to get closer to sklearn.")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(X_test.flatten(), y_test, alpha=0.4, s=30, label="Actual data", color="gray")
    x_line = np.linspace(0, 100, 200).reshape(-1, 1)
    ax.plot(x_line, our_model.predict(x_line), color="red",  linewidth=2.5,
            label=f"Ours  (R2={our_r2:.3f})", linestyle="-")
    ax.plot(x_line, sk_model.predict(x_line),  color="blue", linewidth=2.0,
            label=f"sklearn (R2={sk_r2:.3f})", linestyle="--")
    ax.set_xlabel("House Size (m2)"); ax.set_ylabel("Price ($)")
    ax.set_title("Our Model vs sklearn\nOverlapping lines = correct implementation", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    plt.savefig("outputs/plots/part4_sklearn_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Plot saved: outputs/plots/part4_sklearn_comparison.png")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*65)
    print("LINEAR REGRESSION - COMPLETE LEARNING DEMO")
    print("="*65)
    print(f"Config loaded: lr={LR}, epochs={EPOCHS}, n_samples={N_SAMPLES}")

    part1_single_feature()
    part2_multi_feature()
    part3_learning_rate_experiment()
    part4_sklearn_comparison()

    print("\n" + "="*65)
    print("ALL PARTS COMPLETE")
    print("="*65)
    print("""
EXERCISES - edit this file to try:

1. EASY   - Change lr in configs/supervised/linear_regression.yaml
            Rerun. How does the convergence change?

2. MEDIUM - In Part 1, change n_samples from 100 to 10.
            Does the model still find w~3000, b~50000?
            What does this tell you about needing enough data?

3. MEDIUM - Remove normalization in Part 2 (comment out feat_mean/std lines).
            Does training still converge? Why or why not?

4. HARD   - Add a 4th feature: distance_to_center
            true weight = -2000 (farther = cheaper)
            X = np.column_stack([size, rooms, age, dist])
            Can the model learn the negative weight correctly?

5. HARD   - In Part 3: after 200 epochs, which lr reaches lowest loss?
            Run 2000 epochs instead. Do results change?
    """)
