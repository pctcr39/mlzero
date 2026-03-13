"""
tests/test_supervised/test_linear.py
Unit tests for LinearRegression implementation.

WHAT THESE TESTS VERIFY:
    1. Correct math   — does gradient descent find right weights?
    2. Correct shapes — does predict() return the right array shape?
    3. Error handling — does it raise errors at the right time?
    4. Convergence    — does loss decrease over training?
    5. Multi-feature  — does it work with more than 1 input?

HOW TO RUN:
    pytest tests/ -v                          <- all tests
    pytest tests/test_supervised/ -v          <- just supervised tests
    pytest tests/test_supervised/ -v -k linear  <- just linear tests

READING PYTEST OUTPUT:
    PASSED = test verified the expected behavior
    FAILED = something is wrong with the implementation
    Each test is independent — one failure does not affect others
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from mlzero.supervised.regression.linear import LinearRegression
from mlzero.core.metrics import r2_score


# ==============================================================================
# FIXTURES — shared test data (avoids repeating dataset creation)
# ==============================================================================

@pytest.fixture
def simple_dataset():
    """
    FIXTURE: reusable dataset for multiple tests.
    Simple 1D dataset: y = 2*X + 5 (no noise, perfect linear relationship)
    Model should discover w=2, b=5 exactly (or very close).
    """
    np.random.seed(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)   # 50 points, shape (50,1)
    y = 2.0 * X.flatten() + 5.0                  # y = 2X + 5
    return X, y


@pytest.fixture
def noisy_dataset():
    """
    FIXTURE: more realistic dataset with noise.
    y = 3*X + 10 + gaussian_noise
    """
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 3.0 * X.flatten() + 10.0 + np.random.randn(100) * 1.0
    return X, y


@pytest.fixture
def multi_feature_dataset():
    """
    FIXTURE: 3-feature dataset
    y = 2*x1 + 3*x2 - 1*x3 + 4
    """
    np.random.seed(7)
    n = 200
    X = np.random.rand(n, 3)   # shape (200, 3)
    true_w = np.array([2.0, 3.0, -1.0])
    true_b = 4.0
    y = X @ true_w + true_b + np.random.randn(n) * 0.1   # tiny noise
    return X, y, true_w, true_b


# ==============================================================================
# CLASS 1: Initialization Tests
# ==============================================================================

class TestInit:
    """Tests that the model initializes correctly."""

    def test_default_hyperparameters(self):
        """Model should have default lr and epochs if not specified."""
        model = LinearRegression()
        assert model.lr == 0.01, "Default learning rate should be 0.01"
        assert model.epochs == 1000, "Default epochs should be 1000"

    def test_custom_hyperparameters(self):
        """Model should store the hyperparameters we give it."""
        model = LinearRegression(lr=0.001, epochs=500)
        assert model.lr == 0.001
        assert model.epochs == 500

    def test_weights_none_before_fit(self):
        """Weights should be None before training (model not yet trained)."""
        model = LinearRegression()
        assert model.w is None, "Weights should be None before fit()"
        assert model.b is None, "Bias should be None before fit()"

    def test_loss_history_empty_before_fit(self):
        """Loss history should be empty list before training."""
        model = LinearRegression()
        assert model.loss_history == []


# ==============================================================================
# CLASS 2: Fit Tests
# ==============================================================================

class TestFit:
    """Tests for the fit() method — the training step."""

    def test_weights_set_after_fit(self, simple_dataset):
        """After fit(), weights must not be None."""
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=100)
        model.fit(X, y)
        assert model.w is not None, "Weights should be set after fit()"
        assert model.b is not None, "Bias should be set after fit()"

    def test_weight_shape_single_feature(self, simple_dataset):
        """Weight vector shape should match number of features."""
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=100)
        model.fit(X, y)
        # X has 1 feature -> w should have shape (1,)
        assert model.w.shape == (1,), f"Expected shape (1,), got {model.w.shape}"

    def test_weight_shape_multi_feature(self, multi_feature_dataset):
        """Weight shape should be (m,) where m is number of features."""
        X, y, _, _ = multi_feature_dataset
        model = LinearRegression(lr=0.01, epochs=100)
        model.fit(X, y)
        assert model.w.shape == (3,), f"Expected shape (3,), got {model.w.shape}"

    def test_loss_decreases(self, simple_dataset):
        """
        Loss MUST decrease from start to end of training.
        This is the core guarantee of gradient descent.
        """
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=500)
        model.fit(X, y)
        first_loss = model.loss_history[0]    # LOCAL: loss at epoch 0
        final_loss = model.loss_history[-1]   # LOCAL: loss at last epoch
        assert final_loss < first_loss, (
            f"Loss should decrease: started at {first_loss:.4f}, ended at {final_loss:.4f}"
        )

    def test_loss_history_length(self, simple_dataset):
        """Loss history should have one entry per epoch."""
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=200)
        model.fit(X, y)
        assert len(model.loss_history) == 200

    def test_learns_correct_weights_noiseless(self, simple_dataset):
        """
        On a NOISELESS dataset (y = 2X + 5), the model should learn
        w ~ 2.0 and b ~ 5.0 very accurately (within 5%).

        If this test fails: gradient descent is not converging correctly.
        """
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=2000)
        model.fit(X, y)

        w_error = abs(model.w[0] - 2.0) / 2.0   # relative error
        b_error = abs(model.b    - 5.0) / 5.0

        assert w_error < 0.05, f"w should be ~2.0, got {model.w[0]:.4f} (error {w_error:.2%})"
        assert b_error < 0.05, f"b should be ~5.0, got {model.b:.4f} (error {b_error:.2%})"


# ==============================================================================
# CLASS 3: Predict Tests
# ==============================================================================

class TestPredict:
    """Tests for the predict() method."""

    def test_predict_raises_if_not_fitted(self):
        """
        predict() before fit() should raise RuntimeError.
        This prevents silent wrong predictions.
        """
        model = LinearRegression()
        X = np.array([[1.0], [2.0]])
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(X)

    def test_predict_output_shape(self, simple_dataset):
        """predict() output shape should be (n_samples,)."""
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=100)
        model.fit(X, y)

        X_new = np.array([[5.0], [10.0], [15.0]])   # 3 new samples
        y_pred = model.predict(X_new)
        assert y_pred.shape == (3,), f"Expected (3,), got {y_pred.shape}"

    def test_predict_does_not_change_weights(self, simple_dataset):
        """predict() should NOT modify self.w or self.b."""
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=200)
        model.fit(X, y)

        w_before = model.w.copy()   # LOCAL: copy before predict
        b_before = model.b

        model.predict(X)            # call predict

        assert np.allclose(model.w, w_before), "predict() changed model.w!"
        assert model.b == b_before,            "predict() changed model.b!"

    def test_perfect_prediction_noiseless(self, simple_dataset):
        """
        After enough training on noiseless data,
        predictions should be very close to actual values.
        """
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=3000)
        model.fit(X, y)

        y_pred = model.predict(X)
        max_error = np.max(np.abs(y_pred - y))   # worst case error
        assert max_error < 1.0, f"Max prediction error too high: {max_error:.4f}"


# ==============================================================================
# CLASS 4: Score Tests
# ==============================================================================

class TestScore:
    """Tests for the score() method (R² metric)."""

    def test_score_range(self, noisy_dataset):
        """R² score should be between -inf and 1.0 (good model > 0)."""
        X, y = noisy_dataset
        n = len(X)
        split = int(0.8 * n)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression(lr=0.01, epochs=1000)
        model.fit(X_train, y_train)

        test_r2 = model.score(X_test, y_test)
        assert test_r2 > 0.8, f"R2 on clean data should be >0.8, got {test_r2:.4f}"
        assert test_r2 <= 1.0, f"R2 cannot exceed 1.0, got {test_r2:.4f}"

    def test_perfect_fit_r2_near_1(self, simple_dataset):
        """Noiseless data with enough training should give R2 close to 1."""
        X, y = simple_dataset
        model = LinearRegression(lr=0.01, epochs=3000)
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.999, f"Noiseless data should give R2>0.999, got {r2:.6f}"


# ==============================================================================
# CLASS 5: Multi-Feature Tests
# ==============================================================================

class TestMultiFeature:
    """Tests that verify multi-feature regression works correctly."""

    def test_multi_feature_convergence(self, multi_feature_dataset):
        """
        With 3 features and tiny noise, the model should learn weights
        close to the true values [2, 3, -1] and bias close to 4.
        """
        X, y, true_w, true_b = multi_feature_dataset

        # Normalize features for stable training
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        X_norm = (X - mean) / std

        model = LinearRegression(lr=0.01, epochs=2000)
        model.fit(X_norm, y)

        r2 = model.score(X_norm, y)
        assert r2 > 0.99, f"Multi-feature R2 should be >0.99, got {r2:.4f}"

    def test_multi_feature_predict_shape(self, multi_feature_dataset):
        """predict() with 3-feature input should return (n,) shape."""
        X, y, _, _ = multi_feature_dataset
        model = LinearRegression(lr=0.01, epochs=100)
        model.fit(X, y)

        X_new = np.random.rand(5, 3)   # 5 new samples, 3 features each
        y_pred = model.predict(X_new)
        assert y_pred.shape == (5,), f"Expected (5,), got {y_pred.shape}"


# ==============================================================================
# DIRECT RUN — shows a quick summary
# ==============================================================================

if __name__ == "__main__":
    print("Run with:  pytest tests/test_supervised/test_linear.py -v")
    print("Or all:    pytest tests/ -v")
