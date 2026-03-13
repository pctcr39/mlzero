"""
================================================================================
core/optimizers.py — Optimizers (How the Model Learns)
================================================================================

WHAT IS AN OPTIMIZER?
    After a loss function tells us "you're this wrong",
    the optimizer decides HOW to adjust the model to be less wrong.

    In other words: the optimizer is the "learning" in machine learning.

ANALOGY:
    Imagine you're blindfolded on a hilly landscape.
    Your goal: reach the lowest point (minimum loss).
    The optimizer says: "take a step in this direction."

    - Gradient = which direction is uphill
    - We go the OPPOSITE direction (downhill)
    - Learning rate = how big each step is

THE CORE FORMULA (used in ALL neural networks and ML models):
    new_weight = old_weight - learning_rate × gradient

    If gradient is positive  → weight was too high  → subtract to reduce it
    If gradient is negative  → weight was too low   → subtract negative = increase it

GLOBAL VARIABLES in this file:
    None — all optimizers are classes or functions with local state only

LOCAL VARIABLES (appear inside each optimizer):
    lr          — learning rate, controls step size (e.g. 0.001)
    gradients   — direction and magnitude to update each weight
    params      — the model's weights and biases being updated
================================================================================
"""

import numpy as np


# ==============================================================================
# 1. GRADIENT DESCENT (Batch)
# ==============================================================================

class GradientDescent:
    """
    THEORY:
        The simplest optimizer. Uses ALL training data to compute gradients,
        then takes ONE step to update weights.

        Called "Batch" Gradient Descent because it uses the whole batch of data.

        PROS: Stable, smooth convergence
        CONS: Slow for large datasets (must process all data before each update)

    WHEN TO USE:
        Small datasets (a few hundred to few thousand samples).
        When stability matters more than speed.

    PARAMETERS:
        lr (local, set at init) — Learning Rate
            Too high (e.g. 0.1):   model overshoots the minimum, loss explodes
            Too low  (e.g. 0.00001): model learns very slowly
            Just right (e.g. 0.001): loss decreases smoothly
            Typical range: 0.0001 to 0.01
    """

    def __init__(self, lr=0.001):
        # lr is stored as an INSTANCE variable (self.lr)
        # This makes it accessible in all methods of this class
        self.lr = lr   # learning rate — controls step size

    def step(self, params, gradients):
        """
        Take one optimization step: update all parameters using their gradients.

        PARAMETERS:
            params     (local) — dictionary of model parameters
                                 Example: {'w': array([1.2, 3.4]), 'b': array([0.5])}
            gradients  (local) — dictionary of gradients for each parameter
                                 Example: {'w': array([0.3, 0.1]), 'b': array([0.02])}

        RETURNS:
            updated params dictionary

        EXAMPLE:
            Before: w = 2.0, gradient = 0.5, lr = 0.1
            After:  w = 2.0 - (0.1 × 0.5) = 2.0 - 0.05 = 1.95
        """
        updated = {}   # LOCAL: will hold the updated parameters

        for key in params:
            # Core update rule: new_param = old_param - lr * gradient
            updated[key] = params[key] - self.lr * gradients[key]

        return updated


# ==============================================================================
# 2. STOCHASTIC GRADIENT DESCENT (SGD)
# ==============================================================================

class SGD:
    """
    THEORY:
        Instead of using ALL data to compute gradients,
        use ONE random sample per update.

        "Stochastic" means "random" — we randomly pick one data point each step.

        PROS: Very fast updates, can escape local minima (due to noise)
        CONS: Noisy — loss bounces around instead of smoothly decreasing

        Despite being noisy, SGD often works well in practice because
        the noise actually helps the model find better solutions.

    USED IN:
        Most neural networks use a variant of SGD (usually Mini-batch SGD below).

    PARAMETERS:
        lr (instance) — learning rate, same as GradientDescent
    """

    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, gradients):
        """Same update rule as GradientDescent — the difference is OUTSIDE this function.
        When calling this, you pass gradients computed from just 1 sample."""
        updated = {}
        for key in params:
            updated[key] = params[key] - self.lr * gradients[key]
        return updated


# ==============================================================================
# 3. MINI-BATCH GRADIENT DESCENT
# ==============================================================================

class MiniBatchGD:
    """
    THEORY:
        The best of both worlds between Batch GD and SGD.
        Uses a small "batch" (e.g. 32 samples) per update instead of all data or just 1.

        PROS:
        - Faster than Batch GD (don't need all data per step)
        - More stable than SGD (averages over a few samples)
        - Works well with GPU acceleration (processes batches in parallel)

        CONS: One more hyperparameter to tune (batch_size)

        THIS IS THE DEFAULT OPTIMIZER USED IN PRACTICE.
        When people say "SGD", they usually mean mini-batch SGD.

    PARAMETERS:
        lr         (instance) — learning rate
        batch_size (instance) — how many samples per update (common: 32, 64, 128)

    USED IN:
        PyTorch and TensorFlow both default to mini-batch training.
        Almost all modern neural networks are trained with mini-batch GD.
    """

    def __init__(self, lr=0.001, batch_size=32):
        self.lr = lr
        self.batch_size = batch_size   # how many samples per update step

    def get_batches(self, X, y):
        """
        Split the dataset into mini-batches.

        PARAMETERS:
            X (local) — input features, shape: (n_samples, n_features)
            y (local) — labels,         shape: (n_samples,)

        YIELDS (returns one at a time):
            X_batch, y_batch — one mini-batch at a time

        EXAMPLE:
            Dataset: 1000 samples, batch_size=32
            → yields 31 batches of 32, then 1 batch of 8  (1000 / 32)
        """
        n = len(X)                                      # LOCAL: total number of samples
        indices = np.random.permutation(n)             # LOCAL: shuffle indices randomly
        X_shuffled = X[indices]                        # LOCAL: shuffled X
        y_shuffled = y[indices]                        # LOCAL: shuffled y (same shuffle)

        for start in range(0, n, self.batch_size):     # step through data by batch_size
            end = start + self.batch_size              # LOCAL: end index of this batch
            yield X_shuffled[start:end], y_shuffled[start:end]

    def step(self, params, gradients):
        """Same update rule — difference is batching handled by get_batches()."""
        updated = {}
        for key in params:
            updated[key] = params[key] - self.lr * gradients[key]
        return updated


# ==============================================================================
# 4. ADAM (Advanced — used in deep learning)
# ==============================================================================

class Adam:
    """
    THEORY:
        Adam = Adaptive Moment Estimation.
        The most popular optimizer in deep learning (PyTorch default).

        KEY IDEA: instead of using the same learning rate for all parameters,
        Adam gives each parameter its OWN adaptive learning rate.

        Parameters that rarely get large gradients → get a bigger lr
        Parameters that often get large gradients → get a smaller lr

        Uses two "moments":
        - m (1st moment) = running average of gradients (direction)
        - v (2nd moment) = running average of SQUARED gradients (magnitude)

        Update rule:
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient²
            param = param - lr * m / (sqrt(v) + epsilon)

    PARAMETERS:
        lr      (instance) — learning rate (default: 0.001, works well for most cases)
        beta1   (instance) — decay for 1st moment (default 0.9 = "remember 90% of past")
        beta2   (instance) — decay for 2nd moment (default 0.999)
        epsilon (instance) — tiny number to avoid division by zero (e.g. 1e-8)

    INSTANCE VARIABLES (state that persists between steps):
        self.m  — 1st moment (mean of gradients)
        self.v  — 2nd moment (variance of gradients)
        self.t  — time step counter

    USED IN:
        Almost all modern deep learning. If you don't know which optimizer to use,
        start with Adam(lr=0.001).
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1     # how much to "remember" past gradient direction
        self.beta2 = beta2     # how much to "remember" past gradient magnitude
        self.epsilon = epsilon # prevents division by zero

        # These are the "memory" of the optimizer
        # They start empty and get populated during the first call to step()
        self.m = {}  # INSTANCE: 1st moment estimates (initialized lazily)
        self.v = {}  # INSTANCE: 2nd moment estimates (initialized lazily)
        self.t = 0   # INSTANCE: time step counter

    def step(self, params, gradients):
        """
        Update parameters using Adam's adaptive learning rates.

        The bias correction (m_hat, v_hat) fixes the fact that m and v
        start at 0 and are biased toward 0 in early steps.
        """
        self.t += 1        # increment time step
        updated = {}       # LOCAL: will hold updated parameters

        for key in params:
            # Initialize moment vectors to zeros on first call
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])  # same shape as param, all 0s
                self.v[key] = np.zeros_like(params[key])

            g = gradients[key]   # LOCAL: current gradient for this parameter

            # Update moments (running averages)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g ** 2

            # Bias correction (important in early steps when m and v are near 0)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)  # LOCAL
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)  # LOCAL

            # Adaptive update
            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated


# ==============================================================================
# SUMMARY: Which optimizer to use?
# ==============================================================================
#
#  Learning Stage     → Use This
#  ─────────────────────────────────────────────────────────
#  Phase 2 (scratch)  → GradientDescent (simplest, easiest to understand)
#  Phase 5 (NN)       → MiniBatchGD or Adam
#  Phase 6+ (DL)      → Adam (almost always)
#
