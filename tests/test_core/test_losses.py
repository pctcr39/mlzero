"""
tests/test_core/test_losses.py
Unit tests for core loss functions.

WHAT IS A UNIT TEST?
    A unit test checks that one small piece of code (one function) works correctly.
    It's like verifying your calculator gives 4 when you press 2+2.

    WHY DO WE TEST?
    - Catch bugs early (before they hide in complex code)
    - Confidence to change code without breaking things
    - Serves as documentation of expected behavior

HOW TO RUN:
    pytest tests/                   ← run all tests
    pytest tests/test_core/ -v      ← run this folder, verbose output
"""

import numpy as np
import pytest
import sys, os

# ── Allow importing mlzero without pip install (for quick test runs) ──────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from mlzero.core.losses import mse, mae, binary_cross_entropy


class TestMSE:
    """Tests for Mean Squared Error loss."""

    def test_perfect_prediction(self):
        """MSE should be 0 when predictions exactly match targets."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mse(y_true, y_pred) == 0.0

    def test_known_value(self):
        """MSE with known values: errors=[1,1,1] → squared=[1,1,1] → mean=1.0"""
        y_true = np.array([3.0, 5.0, 2.0])
        y_pred = np.array([4.0, 4.0, 3.0])
        assert mse(y_true, y_pred) == pytest.approx(1.0)

    def test_always_non_negative(self):
        """MSE must always be >= 0 (squaring makes all errors positive)."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([5.0, 25.0, 35.0])
        assert mse(y_true, y_pred) >= 0.0


class TestMAE:
    """Tests for Mean Absolute Error loss."""

    def test_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mae(y_true, y_pred) == 0.0

    def test_known_value(self):
        """MAE with errors=[10,10,10] → mean=10.0"""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        assert mae(y_true, y_pred) == pytest.approx(10.0)

    def test_mse_vs_mae_outlier(self):
        """MSE punishes outliers more than MAE (because of squaring)."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 100.0])   # last one is an outlier
        assert mse(y_true, y_pred) > mae(y_true, y_pred)   # MSE larger
