"""
================================================================================
app/streamlit_app.py
mlzero — Streaming AI Training Dashboard
================================================================================

PURPOSE:
    A web UI that lets you:
      Tab 1 — TRAIN:   watch the model learn in real-time (streaming loss curve)
      Tab 2 — PREDICT: input house features → get an instant price prediction
      Tab 3 — LOGS:    view stored CI/CD build logs from GitHub Actions

THEORY (why a UI for ML?):
    ML is not magic — it's math running in a loop.
    Watching the loss curve drop in real-time builds an INTUITION for:
      - How fast the model converges
      - What "exploding gradients" looks like visually
      - Why learning rate matters

HOW IT WORKS (streaming):
    1. User clicks "Train" button
    2. We create empty placeholders on the page (st.empty())
    3. We call model.fit(..., callback=update_ui)
    4. Every ~1% of training, callback() fires and UPDATES the placeholders
    5. The chart grows live — you see the loss drop in real-time
    → This is the same idea as streaming tokens in ChatGPT: push partial results
      to the UI as they're computed, don't wait for everything to finish.

HOW TO RUN:
    # Option A: directly
    source .venv/bin/activate
    streamlit run app/streamlit_app.py

    # Option B: via Docker
    docker-compose up
    → open http://localhost:8501

GLOBAL IMPORTS:
    streamlit  — web UI framework (Python-only, no JavaScript needed)
    numpy      — math
    pandas     — for displaying tables
    mlzero     — our ML package
================================================================================
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── mlzero imports ─────────────────────────────────────────────────────────────
from mlzero.supervised.regression.linear import LinearRegression
from mlzero.core.metrics import r2_score, rmse


# ==============================================================================
# PAGE CONFIG (must be first Streamlit call)
# ==============================================================================

st.set_page_config(
    page_title="mlzero — ML Training Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# SIDEBAR — Hyperparameter Controls
# ==============================================================================

st.sidebar.title("🧠 mlzero")
st.sidebar.caption("Machine Learning from Zero")
st.sidebar.markdown("---")

st.sidebar.subheader("Hyperparameters")

# GLOBAL: sidebar widgets — values are shared across all tabs
LR     = st.sidebar.select_slider(
    "Learning Rate (lr)",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    value=0.01,
    help="Controls the step size each gradient descent update. Too high → explodes. Too low → slow.",
)
EPOCHS = st.sidebar.slider(
    "Epochs",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Number of times the model sees the training data. More = more learning (up to a point).",
)
N_SAMPLES = st.sidebar.slider(
    "Dataset Size",
    min_value=50,
    max_value=500,
    value=150,
    step=50,
    help="Number of synthetic house price data points to generate.",
)
NOISE_STD = st.sidebar.slider(
    "Noise Level ($)",
    min_value=0,
    max_value=20000,
    value=8000,
    step=1000,
    help="How much random noise to add to prices. High noise = harder to learn.",
)
SEED = st.sidebar.number_input("Random Seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Theory:**
- Model: `y = X @ w + b`
- Loss: `MSE = mean((y_pred - y)²)`
- Gradient: `dw = (2/n) * X.T @ error`
- Update: `w = w - lr * dw`
""")


# ==============================================================================
# DATA GENERATION HELPER
# ==============================================================================

def make_dataset(n_samples, noise_std, seed, n_features=1):
    """
    Generate a synthetic house price dataset.

    THEORY:
        We CREATE a dataset where we KNOW the true answer:
            price = 3000 × size + 15000 × rooms - 500 × age + 40000 + noise
        This lets us verify: did the model DISCOVER the true weights?

    PARAMETERS:
        n_samples  (LOCAL) — number of data points
        noise_std  (LOCAL) — standard deviation of Gaussian noise added to prices
        seed       (LOCAL) — random seed for reproducibility
        n_features (LOCAL) — 1 = single feature (size only), 3 = multi-feature

    RETURNS:
        X_train, X_test, y_train, y_test, true_w, true_b, feat_mean, feat_std
    """
    rng = np.random.default_rng(seed)   # LOCAL: seeded random generator

    if n_features == 1:
        X = rng.uniform(20, 200, (n_samples, 1))          # LOCAL: house sizes
        true_w = np.array([3000.0])                        # LOCAL: price per m²
        true_b = 50000.0
    else:
        size  = rng.uniform(30, 150, n_samples)            # LOCAL
        rooms = rng.integers(1, 6, n_samples).astype(float)
        age   = rng.uniform(0, 40, n_samples)
        X = np.column_stack([size, rooms, age])            # LOCAL: shape (n, 3)
        true_w = np.array([3000.0, 15000.0, -500.0])
        true_b = 40000.0

    noise = rng.normal(0, noise_std, n_samples)            # LOCAL
    y = X @ true_w + true_b + noise                        # LOCAL: true prices

    # Train/test split (80/20)
    split     = int(0.8 * n_samples)                       # LOCAL
    X_train   = X[:split]
    X_test    = X[split:]
    y_train   = y[:split]
    y_test    = y[split:]

    # Normalize using TRAIN statistics only
    # CRITICAL: compute on train, apply to both — prevents data leakage
    feat_mean = X_train.mean(axis=0)                       # LOCAL: shape (m,)
    feat_std  = X_train.std(axis=0) + 1e-8                # LOCAL: +eps avoids /0

    X_train_n = (X_train - feat_mean) / feat_std          # LOCAL: normalized
    X_test_n  = (X_test  - feat_mean) / feat_std          # LOCAL: same stats!

    return X_train_n, X_test_n, y_train, y_test, true_w, true_b, feat_mean, feat_std, X_train, X_test


# ==============================================================================
# TABS
# ==============================================================================

tab_train, tab_predict, tab_ci = st.tabs([
    "🏋️ Train — Watch Model Learn",
    "🔮 Predict — Enter Features",
    "📋 CI/CD Logs",
])


# ==============================================================================
# TAB 1: TRAIN (Streaming)
# ==============================================================================

with tab_train:
    st.header("Real-Time Training — Streaming Loss Curve")

    col_info, col_theory = st.columns([3, 2])

    with col_info:
        st.markdown("""
        **What you'll see:**
        - Loss curve drops in real-time as the model learns
        - Higher loss = model predictions are far from true prices
        - Lower loss = model is getting better

        **How to experiment:**
        1. Try `lr = 0.5` → watch the loss EXPLODE (gradient overshoots)
        2. Try `lr = 0.001` → watch it converge VERY slowly
        3. Try `lr = 0.01` → sweet spot: smooth fast convergence
        """)

    with col_theory:
        st.info("""
        **Why does loss drop?**

        Each epoch: model makes a prediction, measures how wrong it is (loss),
        computes which direction reduces loss (gradient), takes a small step.

        It's like finding the bottom of a bowl while blindfolded:
        feel downhill → step → repeat.
        """)

    n_features_choice = st.radio(
        "Dataset type",
        ["Single feature (house size → price)", "Multi-feature (size + rooms + age → price)"],
        horizontal=True,
    )
    n_features = 1 if "Single" in n_features_choice else 3

    st.markdown("---")

    if st.button("▶️  Start Training", type="primary", use_container_width=True):

        # ── Prepare data ──────────────────────────────────────────────────────
        (X_train_n, X_test_n, y_train, y_test,
         true_w, true_b, feat_mean, feat_std,
         X_train_raw, X_test_raw) = make_dataset(N_SAMPLES, NOISE_STD, SEED, n_features)

        # ── UI Placeholders (will be updated by callback) ─────────────────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Loss During Training")
            chart_placeholder   = st.empty()   # LOCAL: will hold the live line chart

        with col_right:
            st.subheader("Live Metrics")
            epoch_placeholder   = st.empty()   # LOCAL: current epoch display
            loss_placeholder    = st.empty()   # LOCAL: current loss display
            r2_placeholder      = st.empty()   # LOCAL: test R² (updated at end)
            status_placeholder  = st.empty()   # LOCAL: training status

        status_placeholder.info("Training in progress...")

        # ── State for callback ────────────────────────────────────────────────
        loss_log  = []    # LOCAL: accumulates loss at each callback fire
        epoch_log = []    # LOCAL: accumulates epoch indices

        def update_ui(epoch, loss, w, b):
            """
            CALLBACK — called by LinearRegression.fit() every ~1% of training.

            LOCAL variables:
                epoch (int)  — current epoch number
                loss  (float)— current MSE loss
                w     (array)— current weight vector copy
                b     (float)— current bias

            THEORY: This is the "observer" pattern. The model doesn't know
            about Streamlit. It just calls this function. This function
            updates the UI. Separation of concerns.
            """
            loss_log.append(loss)
            epoch_log.append(epoch)

            # Build a DataFrame that Streamlit's line chart can render
            df = pd.DataFrame({       # LOCAL
                "MSE Loss": loss_log,
            }, index=epoch_log)

            chart_placeholder.line_chart(df, color="#e63946", height=300)

            epoch_placeholder.metric("Epoch", f"{epoch} / {EPOCHS}")
            loss_placeholder.metric("MSE Loss", f"{loss:,.2f}")

        # ── Train (with streaming callback) ───────────────────────────────────
        model = LinearRegression(lr=LR, epochs=EPOCHS, verbose=False)
        model.fit(X_train_n, y_train, callback=update_ui)

        # ── Final metrics ──────────────────────────────────────────────────────
        train_r2   = model.score(X_train_n, y_train)   # LOCAL
        test_r2    = model.score(X_test_n,  y_test)    # LOCAL
        test_rmse  = rmse(y_test, model.predict(X_test_n))   # LOCAL

        status_placeholder.success("✅ Training complete!")
        r2_placeholder.metric("Test R²", f"{test_r2:.4f}", delta=f"Train R²: {train_r2:.4f}")

        # ── Results table ──────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Results")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Train R²",  f"{train_r2:.4f}", help="R²=1.0 is perfect. 0.0 = no better than mean.")
        col_b.metric("Test R²",   f"{test_r2:.4f}",  help="Key metric — model never saw test data during training.")
        col_c.metric("Test RMSE", f"${test_rmse:,.0f}", help="Average dollar prediction error.")

        # ── Learned vs True weights ────────────────────────────────────────────
        st.subheader("Learned Weights vs True Weights")
        feat_names = ["size (m²)", "rooms", "age (yrs)"][:n_features]

        rows = []    # LOCAL: list of dicts for DataFrame
        for i, name in enumerate(feat_names):
            # Denormalize: weights in normalized space are scaled by feat_std
            learned_w_real = model.w[i] / feat_std[i]
            rows.append({
                "Feature":         name,
                "True Weight":     f"{true_w[i]:,.0f}",
                "Learned Weight":  f"{learned_w_real:,.0f}",
                "Match?":          "✅" if abs(learned_w_real - true_w[i]) / abs(true_w[i]) < 0.10 else "⚠️ off",
            })
        rows.append({
            "Feature":        "bias (b)",
            "True Weight":    f"{true_b:,.0f}",
            "Learned Weight": f"{model.b:,.0f}",
            "Match?":         "✅" if abs(model.b - true_b) / abs(true_b) < 0.15 else "⚠️ off",
        })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption("""
        **Why do learned ≠ true exactly?**
        Noise in the data + gradient descent approximation.
        With more epochs and less noise they converge closer.
        This is normal — real ML datasets have noise too.
        """)

        # ── Store model in session state for Tab 2 ────────────────────────────
        st.session_state["trained_model"]    = model
        st.session_state["feat_mean"]        = feat_mean
        st.session_state["feat_std"]         = feat_std
        st.session_state["n_features"]       = n_features
        st.session_state["model_test_r2"]    = test_r2


# ==============================================================================
# TAB 2: PREDICT
# ==============================================================================

with tab_predict:
    st.header("Make a Prediction — Enter House Features")

    if "trained_model" not in st.session_state:
        st.warning("No trained model yet. Go to the **Train** tab and click **Start Training** first.")
        st.stop()

    model_pred   = st.session_state["trained_model"]    # LOCAL: model from tab 1
    feat_mean_p  = st.session_state["feat_mean"]        # LOCAL: normalization stats
    feat_std_p   = st.session_state["feat_std"]         # LOCAL
    n_features_p = st.session_state["n_features"]       # LOCAL
    test_r2_p    = st.session_state["model_test_r2"]    # LOCAL

    st.success(f"Using trained model  |  Test R² = {test_r2_p:.4f}")
    st.markdown("---")

    st.subheader("Enter house features:")

    col1, col2, col3 = st.columns(3)

    with col1:
        size_input = st.number_input(
            "House Size (m²)",
            min_value=10.0, max_value=300.0, value=80.0, step=5.0,
            help="Larger house → higher price",
        )

    if n_features_p >= 3:
        with col2:
            rooms_input = st.number_input(
                "Number of Rooms",
                min_value=1, max_value=10, value=3,
                help="More rooms → higher price",
            )
        with col3:
            age_input = st.number_input(
                "House Age (years)",
                min_value=0, max_value=100, value=10,
                help="Older house → lower price (negative weight)",
            )
    else:
        rooms_input = 3
        age_input   = 10

    # ── Real-time prediction (updates on every input change) ──────────────────
    # THEORY: This is "streaming" prediction — no button needed.
    # Streamlit re-runs the entire script on every widget change.
    # So every time the user changes a slider → new prediction appears instantly.
    # This is the same concept as autocomplete: compute while user types.

    if n_features_p == 1:
        raw_features = np.array([[size_input]])              # LOCAL: shape (1,1)
    else:
        raw_features = np.array([[size_input, float(rooms_input), float(age_input)]])

    norm_features  = (raw_features - feat_mean_p) / feat_std_p  # LOCAL: normalize
    predicted_price = model_pred.predict(norm_features)[0]       # LOCAL: scalar

    st.markdown("---")
    st.subheader("Predicted Price")

    pred_col1, pred_col2 = st.columns([2, 3])
    with pred_col1:
        st.metric(
            label="Estimated Price",
            value=f"${predicted_price:,.0f}",
            help="This is the model's prediction: X @ w + b",
        )

    with pred_col2:
        st.markdown(f"""
        **Calculation breakdown:**
        ```
        Input (raw):       size={size_input:.0f} m²
        Input (normalized): {norm_features.flatten()}

        Formula:  y_pred = X @ w + b
        w (learned): {model_pred.w}
        b (learned): {model_pred.b:,.0f}

        Result: ${predicted_price:,.0f}
        ```
        """)

    st.info("""
    **Why normalize before predicting?**
    The model was TRAINED on normalized data (mean=0, std=1).
    We MUST normalize the input the same way before predicting.
    Using the SAME mean and std from training.
    If we forget this step → garbage predictions.
    """)


# ==============================================================================
# TAB 3: CI/CD LOGS
# ==============================================================================

with tab_ci:
    st.header("CI/CD Build Logs — GitHub Actions History")
    st.caption("Run `python scripts/utils/ci_log_collector.py` to fetch latest logs from GitHub.")

    CI_LOG_DIR = Path("outputs/ci_logs")   # LOCAL: where logs are stored

    # ── Find stored log files ──────────────────────────────────────────────────
    log_files = sorted(CI_LOG_DIR.glob("run_*.json"), reverse=True)   # LOCAL

    if not log_files:
        st.warning("""
        No CI logs found yet.

        To fetch logs from GitHub Actions:
        ```bash
        python scripts/utils/ci_log_collector.py
        ```
        Then refresh this page.
        """)
    else:
        st.success(f"Found {len(log_files)} stored CI run(s).")

        # ── Summary table ──────────────────────────────────────────────────────
        rows = []    # LOCAL
        for f in log_files:
            with open(f) as fh:
                data = json.load(fh)
            rows.append({
                "Run ID":        data.get("run_id", "?"),
                "Date":          data.get("created_at", "?")[:10],
                "Branch":        data.get("branch", "?"),
                "Status":        "✅ Pass" if data.get("conclusion") == "success" else "❌ Fail",
                "Python":        data.get("python_version", "?"),
                "Tests Passed":  data.get("tests_passed", "?"),
                "Tests Failed":  data.get("tests_failed", "?"),
                "Duration (s)":  data.get("duration_seconds", "?"),
            })

        df_logs = pd.DataFrame(rows)   # LOCAL
        st.dataframe(df_logs, use_container_width=True, hide_index=True)

        # ── Drill-down: select one run ─────────────────────────────────────────
        st.markdown("---")
        selected_idx = st.selectbox(
            "Select a run to view full log",
            range(len(log_files)),
            format_func=lambda i: f"Run {rows[i]['Run ID']} — {rows[i]['Date']} — {rows[i]['Status']}",
        )

        with open(log_files[selected_idx]) as fh:
            selected = json.load(fh)

        col_meta, col_raw = st.columns([1, 2])

        with col_meta:
            st.subheader("Run Metadata")
            st.json({k: v for k, v in selected.items() if k != "raw_log"})

        with col_raw:
            st.subheader("Raw Log Output")
            raw_log = selected.get("raw_log", "No log captured.")
            st.code(raw_log, language="text")

        # ── Trend chart (if multiple runs) ────────────────────────────────────
        if len(rows) >= 2:
            st.markdown("---")
            st.subheader("Build Health Over Time")
            trend_df = pd.DataFrame(rows[::-1])   # LOCAL: chronological order
            trend_df["pass_rate"] = (trend_df["Status"] == "✅ Pass").astype(int)
            st.bar_chart(trend_df.set_index("Date")["pass_rate"], height=200)
