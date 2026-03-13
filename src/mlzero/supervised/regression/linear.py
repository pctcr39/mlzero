"""
================================================================================
src/mlzero/supervised/regression/linear.py
Linear Regression — Implemented From Scratch
================================================================================

THEORY:
    Linear Regression finds the best straight line through your data.
    It learns to predict a continuous number (price, temperature, score, etc.)

    THE MODEL:
        y_pred = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
        or in matrix form:
        y_pred = X @ w + b

    WHERE:
        X  = input features (matrix of data)
        w  = weights (what the model learns — how important each feature is)
        b  = bias (the baseline prediction when all features are 0)
        @  = matrix multiplication

    ANALOGY:
        Predicting house price from size:
        price = 3000 × size + 50000
        w = 3000  (each extra m² adds $3000)
        b = 50000 (base price even for size=0)
        The model LEARNS these numbers from data.

WHAT WE MINIMIZE (Loss Function):
    MSE = (1/n) × Σ (y_pred - y_true)²

    This measures the average squared error across all training samples.
    We want this as small as possible.

HOW WE LEARN (Gradient Descent):
    We compute how much changing w and b affects the loss (gradient),
    then nudge w and b in the direction that reduces loss.

    dLoss/dw = (2/n) × X^T @ (y_pred - y_true)   ← gradient for weights
    dLoss/db = (2/n) × mean(y_pred - y_true)       ← gradient for bias

    w = w - lr × dLoss/dw
    b = b - lr × dLoss/db

GLOBAL IMPORTS:
    numpy  — matrix math
    matplotlib — plotting
    sys    — to import from parent directories
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm          # GLOBAL (optional): live progress bar during training
    TQDM_AVAILABLE = True          # GLOBAL: flag — True if tqdm is installed
except ImportError:
    TQDM_AVAILABLE = False         # GLOBAL: graceful fallback if tqdm not installed

# ── Package imports ────────────────────────────────────────────────────────────
from mlzero.core.base import BaseModel
from mlzero.core.losses import mse
from mlzero.core.metrics import r2_score, rmse


# ==============================================================================
# THE MODEL CLASS
# ==============================================================================

class LinearRegression(BaseModel):
    """
    Linear Regression from scratch.

    Inherits from BaseModel → guarantees fit(), predict(), score() exist.

    HYPERPARAMETERS (set by user, not learned):
        lr     — learning rate: controls how big each gradient descent step is
                 Too high → overshoots minimum (loss explodes)
                 Too low  → takes forever to converge
                 Sweet spot: 0.001 to 0.01 (depends on data scale)

        epochs — number of training iterations
                 More epochs → more chances to improve
                 Too many → wastes time after convergence
                 Too few  → model hasn't learned enough

    LEARNED PARAMETERS (set by fit(), used by predict()):
        self.w — weight vector, shape: (n_features,)
                 After training: how much each feature contributes to prediction
        self.b — bias scalar
                 After training: the baseline prediction
    """

    def __init__(self, lr=0.01, epochs=1000, verbose=True, stream=True):
        # INSTANCE variables — belong to this model object
        self.lr      = lr       # learning rate (hyperparameter)
        self.epochs  = epochs   # training iterations (hyperparameter)
        self.verbose = verbose  # whether to print progress at all
        self.stream  = stream   # True = live tqdm bar, False = print every 100 epochs
        self.w = None           # weights — not set until fit() is called
        self.b = None           # bias   — not set until fit() is called
        self.loss_history = []  # track loss at each epoch to plot later

    def fit(self, X, y, callback=None):
        """
        Learn weights (w) and bias (b) from training data.

        PARAMETERS:
            X (local)        — feature matrix, shape: (n_samples, n_features)
                               Example: [[80, 3], [60, 2], [120, 4]]
                                         size rooms
            y (local)        — target vector, shape: (n_samples,)
                               Example: [250000, 180000, 380000]
            callback (local) — optional function called every N epochs for real-time UI updates
                               Signature: callback(epoch, loss, w, b)
                               Example: used by Streamlit to stream live loss charts
                               WHY: decouples training logic from UI — the model doesn't
                                    know or care about the UI; it just calls the callback
                                    and the caller decides what to do with the data

        This method runs gradient descent: it loops `epochs` times,
        each time making predictions, measuring loss, computing gradients,
        and updating w and b.

        SETS:
            self.w — learned weights after training
            self.b — learned bias after training
        """
        # ── Get dimensions ────────────────────────────────────────────────────
        n, m = X.shape   # n = number of samples, m = number of features
                         # LOCAL: these are just convenient shortcuts

        # ── Initialize parameters ─────────────────────────────────────────────
        # We start with all zeros. Gradient descent will improve them.
        # Could also use random initialization: np.random.randn(m) * 0.01
        self.w = np.zeros(m)   # shape: (m,) — one weight per feature
        self.b = 0.0           # scalar — single bias value

        # ── Training loop ─────────────────────────────────────────────────────
        # STREAMING: tqdm wraps range(epochs) to show a live progress bar.
        # It streams updates to the terminal in real-time as each epoch completes.
        # Without streaming: you see nothing until all epochs are done.
        # With streaming: you see live loss, speed (epochs/sec), and ETA.

        use_bar = self.verbose and self.stream and TQDM_AVAILABLE  # LOCAL

        epoch_iter = tqdm(
            range(self.epochs),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,       # adapts bar width to terminal
            colour="green",           # green bar
            disable=not use_bar,      # if False: tqdm is invisible (no output)
        ) if TQDM_AVAILABLE else range(self.epochs)

        for epoch in epoch_iter:

            # STEP 1: Forward pass — compute predictions
            y_pred = X @ self.w + self.b   # LOCAL: shape (n,)

            # STEP 2: Compute loss
            current_loss = mse(y, y_pred)           # LOCAL: scalar
            self.loss_history.append(current_loss)   # save for plotting

            # STEP 3: Compute gradients
            error = y_pred - y                        # LOCAL: shape (n,)
            dw    = (2 / n) * X.T @ error             # LOCAL: shape (m,)
            db    = (2 / n) * np.sum(error)           # LOCAL: scalar

            # STEP 4: Update parameters
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

            # STEP 5: Stream live metrics into the progress bar
            if use_bar and TQDM_AVAILABLE:
                # tqdm.set_postfix() updates the right side of the bar in real-time
                # This is what "streaming" means: metrics appear AS training runs
                epoch_iter.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "w[0]": f"{self.w[0]:.3f}",
                    "b":    f"{self.b:.3f}",
                })
            elif self.verbose and not self.stream and epoch % 100 == 0:
                print(f"Epoch {epoch:5d} | Loss: {current_loss:.4f}")

            # CALLBACK: notify the caller (e.g., a Streamlit UI) every N epochs
            # LOCAL: callback_interval — how often to fire (every 1% of epochs, min 1)
            # WHY every N instead of every epoch? Updating a UI 1000x/sec is too fast
            # and causes flickering. Firing every 1% = smooth 100 updates total.
            if callback is not None:
                callback_interval = max(1, self.epochs // 100)   # LOCAL
                if epoch % callback_interval == 0 or epoch == self.epochs - 1:
                    callback(epoch, current_loss, self.w.copy(), float(self.b))

        if self.verbose:
            print(f"\nTraining complete! Final loss: {self.loss_history[-1]:.6f}")

    def predict(self, X):
        """
        Use learned parameters to make predictions on new data.

        PARAMETERS:
            X (local) — feature matrix, shape: (n_samples, n_features)
                        Can be training data or new unseen data.

        RETURNS:
            y_pred (local) — predicted values, shape: (n_samples,)

        NOTE: predict() does NOT update w or b. It only reads them.
        """
        if self.w is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        return X @ self.w + self.b   # same formula as in fit(), but no learning

    def score(self, X, y):
        """
        Evaluate model using R² score.

        RETURNS:
            float: 0.0 to 1.0 (higher is better)
        """
        y_pred = self.predict(X)    # LOCAL
        return r2_score(y, y_pred)

    def plot_loss(self):
        """Plot how loss decreased during training."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history)
        plt.title("Training Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ==============================================================================
# DEMO & EXERCISES
# ==============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("LINEAR REGRESSION FROM SCRATCH — DEMO")
    print("=" * 55)

    # ── Generate fake dataset ─────────────────────────────────────────────────
    np.random.seed(42)   # seed makes random numbers reproducible across runs
    n_samples = 100      # LOCAL: number of data points

    # Single feature: house size (m²)
    X_single = np.random.rand(n_samples, 1) * 100   # shape: (100, 1)

    # True relationship: price = 3000 * size + 50000 + noise
    # We want the model to discover w=3000, b=50000 on its own
    y = (3000 * X_single.flatten()          # LOCAL: flatten (100,1) → (100,)
         + 50000
         + np.random.randn(n_samples) * 5000)  # noise

    # ── Split into train / test ───────────────────────────────────────────────
    split_idx = int(0.8 * n_samples)  # LOCAL: 80% for training
    X_train, X_test = X_single[:split_idx], X_single[split_idx:]
    y_train, y_test = y[:split_idx],        y[split_idx:]

    print(f"\nDataset: {n_samples} samples")
    print(f"Training: {len(X_train)} samples")
    print(f"Testing:  {len(X_test)} samples\n")

    # ── Train model ───────────────────────────────────────────────────────────
    model = LinearRegression(lr=0.0001, epochs=1000)
    model.fit(X_train, y_train)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\nLearned:  w = {model.w[0]:.1f}, b = {model.b:.1f}")
    print(f"Expected: w ≈ 3000, b ≈ 50000")

    train_r2 = model.score(X_train, y_train)
    test_r2  = model.score(X_test, y_test)
    print(f"\nTrain R² : {train_r2:.4f}")
    print(f"Test  R² : {test_r2:.4f}")
    print("(R² = 1.0 is perfect, 0.0 = no better than guessing the mean)")

    test_rmse = rmse(y_test, model.predict(X_test))
    print(f"Test RMSE: {test_rmse:.1f} (average prediction error in dollars)")

    # ── Plot 1: Data + fitted line ────────────────────────────────────────────
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X_train.flatten(), y_train, alpha=0.5, label="Train data")
    plt.scatter(X_test.flatten(),  y_test,  alpha=0.5, label="Test data", color="orange")
    x_line = np.linspace(0, 100, 100).reshape(-1, 1)   # LOCAL: 100 points for drawing line
    plt.plot(x_line, model.predict(x_line), color="red", linewidth=2, label="Model")
    plt.xlabel("House Size (m²)")
    plt.ylabel("Price ($)")
    plt.title("Linear Regression Fit")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history)
    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ── EXERCISES ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("EXERCISES (edit this file to try them):")
    print("=" * 55)
    print("1. Change lr=0.1  → what happens to loss?")
    print("2. Change epochs=50 → how does R² change?")
    print("3. Change epochs=5000 → does R² keep improving?")
    print("4. Add a second feature: X = [size, rooms]")
    print("   Hint: X = np.column_stack([size_array, rooms_array])")
    print("5. Try to get R² > 0.95 by tuning lr and epochs")
