# QUICKSTART — How to Run mlzero

> Complete step-by-step guide. Follow top to bottom for first-time setup.
> After setup, jump directly to any section you need.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Prerequisites](#2-prerequisites)
3. [Setup — Local (Python venv)](#3-setup--local-python-venv)
4. [Setup — Docker](#4-setup--docker)
5. [Run: Streamlit Dashboard (Live UI)](#5-run-streamlit-dashboard-live-ui)
6. [Run: Demo Script (Terminal)](#6-run-demo-script-terminal)
7. [Run: Tests](#7-run-tests)
8. [Run: CI/CD Log Collector](#8-run-cicd-log-collector)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        mlzero System                            │
│                                                                 │
│  ┌───────────────┐    trains    ┌──────────────────────────┐   │
│  │ Streamlit UI  │ ──────────► │ LinearRegression          │   │
│  │ (browser)     │ ◄────────── │ src/mlzero/supervised/    │   │
│  │ localhost:8501│  streams    │ regression/linear.py      │   │
│  └───────────────┘  loss+preds └──────────────────────────┘   │
│         │                               │                       │
│         ▼                               ▼                       │
│  ┌──────────────┐             ┌─────────────────┐              │
│  │ outputs/     │             │ core/            │              │
│  │ ci_logs/     │             │ losses, metrics  │              │
│  │ plots/       │             │ optimizers       │              │
│  └──────────────┘             └─────────────────┘              │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐     │
│  │ GitHub Actions CI/CD → runs pytest on every git push  │     │
│  │ ci_log_collector.py → saves results to outputs/ci_logs│     │
│  └───────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

**Two ways to run everything:**

| Method | When to use | Start command |
|---|---|---|
| **Local (venv)** | Daily learning & development | `streamlit run app/streamlit_app.py` |
| **Docker** | Clean environment, share with others | `docker-compose up --build` |

---

## 2. Prerequisites

Install these **once**. Skip anything already installed.

### Check what you have
```bash
python3 --version      # need 3.11 or higher (3.12, 3.13, 3.14 all work)
git --version          # need 2.x
docker --version       # only needed for Docker option
gh --version           # only needed for CI log collector
```

### Install missing tools
```bash
# Python (if below 3.11):
brew install python@3.11

# Docker Desktop — download the app:
# https://www.docker.com/products/docker-desktop/
# After installing: open Docker Desktop at least once to start the daemon.

# GitHub CLI (for CI log collector):
brew install gh
gh auth login
# → choose: GitHub.com → SSH → browser login
```

> **Note:** `python` (without the 3) only works **inside the venv**.
> Outside the venv, always use `python3`. This is normal macOS behavior.

---

## 3. Setup — Local (Python venv)

**Do this once.** After setup, only step 3b is needed each session.

### 3a — Get the code

**If you already have it locally (current situation):**
```bash
cd /path/to/Machine_Learning    # your local project folder
```

**If cloning fresh from GitHub:**
```bash
git clone git@github.com:pctcr39/mlzero.git
cd mlzero
```

> Note: The GitHub repo is named `mlzero`. Your local folder may be named
> `Machine_Learning` — both are the same project. The contents are identical.

### 3b — Create and activate venv

```bash
# Create (one time only):
python3 -m venv .venv

# Activate (every new terminal session):
source .venv/bin/activate
```

You'll see `(.venv)` appear in your prompt — this means you're inside the environment.
**You must activate the venv every time you open a new terminal.**

```bash
# To exit the venv when done:
deactivate
```

### 3c — Install all dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

What each command does:
- `pip install -r requirements.txt` — installs numpy, matplotlib, streamlit, sklearn, tqdm, pytest, etc.
- `pip install -e .` — installs mlzero as an editable package so `from mlzero...` imports work

### 3d — Verify setup

```bash
# Check mlzero imports correctly:
python -c "from mlzero.supervised.regression.linear import LinearRegression; print('mlzero OK')"
# Expected: mlzero OK

# Check streamlit:
python -c "import streamlit; print('streamlit', streamlit.__version__)"
# Expected: streamlit 1.x

# Check tests pass:
python -m pytest tests/ -q
# Expected: 24 passed
```

If all three pass, setup is complete.

---

## 4. Setup — Docker

**No Python installation needed.** Docker handles everything inside the container.

### Requirements
- Docker Desktop installed and **running** (look for the whale icon in menu bar)

### Build and start
```bash
# In the project root (where docker-compose.yml is):
docker-compose up --build
```

First run downloads the base image and installs packages — takes 2-4 minutes.
Subsequent runs start in seconds (Docker caches the layers).

### Open the dashboard
```
http://localhost:8501
```

### Stop
```bash
# Ctrl+C in terminal, then:
docker-compose down
```

### Useful Docker commands

```bash
# Start in background (no terminal output):
docker-compose up -d --build

# Watch logs when running in background:
docker-compose logs -f app

# Rebuild after changing requirements.txt or Dockerfile:
docker-compose up --build

# Full reset (remove container + image):
docker-compose down --rmi all

# Check if container is running:
docker ps
```

> **Volumes:** `outputs/` is mounted from your Mac into the container.
> Files saved inside Docker (plots, CI logs) appear in your local `outputs/` folder.

---

## 5. Run: Streamlit Dashboard (Live UI)

### Start it
```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Open: **http://localhost:8501**

### Tab 1 — Train (Streaming Loss Curve)

Watch the model learn in real-time.

**Steps:**
1. Set hyperparameters in the **left sidebar**
2. Choose dataset type: single feature or multi-feature
3. Click **▶ Start Training**
4. Watch the loss curve drop live — the model is learning right now

**Experiment ideas to build intuition:**

| Sidebar setting | What to observe |
|---|---|
| `lr = 0.5` | Loss explodes — overshooting the minimum |
| `lr = 0.0001` | Loss drops very slowly — too cautious |
| `lr = 0.01` | Smooth convergence — the sweet spot |
| Noise = 0 | Near-perfect fit, R² ≈ 1.0 |
| Noise = 20000 | High noise, model struggles |
| Epochs = 200 vs 2000 | See how long convergence takes |

**After training:**
- Table shows learned weights vs true weights
- Big gap = model needs more epochs or less noise
- Small gap = model successfully discovered the pattern

### Tab 2 — Predict (Real-Time Inference)

1. First train a model in Tab 1
2. Switch to Tab 2
3. Enter house features (size in m², rooms, age)
4. Price prediction updates **instantly** as you change inputs — no button needed

This is what real ML inference looks like: model learned from data, now applies
the learned formula `y = X @ w + b` to any new input instantly.

### Tab 3 — CI/CD Logs

Shows stored GitHub Actions build history.
Requires running the CI log collector first (see section 8).

---

## 6. Run: Demo Script (Terminal)

A 4-part educational demo that runs entirely in the terminal and saves plots.

```bash
source .venv/bin/activate
python scripts/supervised/linear_regression_demo.py
```

**Parts:**
- **Part 1** — Single feature: house size → price. Basic training loop.
- **Part 2** — Multi-feature: size + rooms + age → price. Shows normalization.
- **Part 3** — Learning rate comparison: 4 different `lr` values side by side.
- **Part 4** — sklearn comparison: verifies our from-scratch results match sklearn's.

**Output:**
- Terminal: loss values, R² scores, weight comparison table
- `outputs/plots/` — PNG figures saved automatically

**Change hyperparameters without editing code:**
```bash
# Edit the YAML config file:
open configs/supervised/linear_regression.yaml

# Then re-run:
python scripts/supervised/linear_regression_demo.py
```

Config options in `linear_regression.yaml`:
```yaml
lr: 0.0001        # learning rate
epochs: 1000      # training iterations
n_samples: 100    # dataset size
noise_std: 10.0   # data noise level
true_w: 3000.0    # true price per m² (what model should learn)
true_b: 50000.0   # true base price
train_ratio: 0.8  # 80% train, 20% test
random_seed: 42   # reproducibility
```

---

## 7. Run: Tests

Tests verify every function works correctly. Run after any code change.

```bash
source .venv/bin/activate

# Run all tests:
python -m pytest tests/ -v

# Run one file only:
python -m pytest tests/test_supervised/test_linear.py -v
python -m pytest tests/test_core/test_losses.py -v

# Run with coverage (shows which lines aren't tested):
python -m pytest tests/ -v --cov=src/mlzero --cov-report=term-missing

# Run quietly (just pass/fail summary):
python -m pytest tests/ -q
```

**Expected:** `24 passed`

**What each test class checks:**

| Test class | File | Verifies |
|---|---|---|
| `TestInit` | `test_linear.py` | Default values, correct types |
| `TestFit` | `test_linear.py` | Loss decreases, weights update |
| `TestPredict` | `test_linear.py` | Numerically correct predictions |
| `TestScore` | `test_linear.py` | R²=1 on clean data, R²≈0 on noise |
| `TestMultiFeature` | `test_linear.py` | Works with multiple columns |
| `TestMSE` / `TestMAE` | `test_losses.py` | Correct loss values |

> Always use `python -m pytest` not just `pytest`.
> On this Mac, `pytest` without `python -m` sometimes uses the wrong Python.

---

## 8. Run: CI/CD Log Collector

After every `git push`, GitHub Actions runs your tests automatically on Python 3.10, 3.11, and 3.12.
This tool downloads those results and saves them locally.

### Fetch logs
```bash
source .venv/bin/activate

# Fetch latest 10 CI runs:
python scripts/utils/ci_log_collector.py

# Fetch more:
python scripts/utils/ci_log_collector.py --limit 25

# Skip downloading full log text (faster):
python scripts/utils/ci_log_collector.py --no-log
```

### Analyze stored logs
```bash
# Print summary to terminal:
python scripts/utils/ci_log_collector.py --analyze

# Regenerate markdown report:
python scripts/utils/ci_log_collector.py --report
# → outputs/ci_logs/REPORT.md
```

### Where logs are saved
```
outputs/ci_logs/
├── run_123456.json     ← one file per CI run
├── run_123457.json
└── REPORT.md           ← generated trend report
```

Each JSON contains:
```json
{
  "run_id": 123456,
  "branch": "main",
  "conclusion": "success",
  "created_at": "2026-03-13T10:00:00Z",
  "tests_passed": 24,
  "tests_failed": 0,
  "duration_seconds": 45.3
}
```

### View CI results in browser
```
https://github.com/pctcr39/mlzero/actions
```

### Workflow: push → CI runs automatically
```
git push origin main
   → GitHub Actions triggers
   → Tests run on Python 3.10, 3.11, 3.12 simultaneously
   → Results appear in GitHub Actions tab
   → Run ci_log_collector.py to save locally
   → View in Streamlit Tab 3
```

---

## 9. Troubleshooting

### `command not found: python`
```bash
# Outside the venv, use python3:
python3 --version

# Or activate the venv first (then 'python' works):
source .venv/bin/activate
python --version    # now works
```

### `ModuleNotFoundError: No module named 'mlzero'`
```bash
source .venv/bin/activate
pip install -e .
# Then retry your command.
```

### `ModuleNotFoundError: No module named 'streamlit'`
```bash
source .venv/bin/activate
pip install streamlit
```

### `ModuleNotFoundError: No module named 'matplotlib'` or `yaml`
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Port 8501 already in use
```bash
# Find what's using it:
lsof -i :8501

# Kill it (replace PID with the number shown):
kill -9 <PID>

# Or run on a different port:
streamlit run app/streamlit_app.py --server.port 8502
```

### Docker: `Cannot connect to Docker daemon`
Open **Docker Desktop** from your Applications folder. The whale icon must appear in the menu bar.

### Docker: port already in use
```bash
# Stop all running containers:
docker-compose down
docker ps    # should show nothing
```

### `gh: command not found`
```bash
brew install gh
gh auth login
```

### `gh: not logged in` or auth error
```bash
gh auth status    # check current auth
gh auth login     # re-authenticate
```

### Tests fail with `ImportError`
```bash
source .venv/bin/activate
pip install -e .
python -m pytest tests/ -v
```

### Loss is not decreasing / exploding
Most likely: **learning rate too high**.
- Rule of thumb: if loss explodes, divide `lr` by 10 and retry
- `lr = 0.01` is usually a safe starting point for normalized data
- In Streamlit: use the sidebar slider to experiment live

### Streamlit page is blank or shows error
```bash
# Check the terminal output for the actual error message.
# Common fix: the venv isn't activated or mlzero isn't installed.
source .venv/bin/activate
pip install -e .
streamlit run app/streamlit_app.py
```

---

## Quick Reference Card

```bash
# ── Setup (first time only) ────────────────────────────────────────
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# ── Every new terminal ─────────────────────────────────────────────
source .venv/bin/activate

# ── Run the UI ─────────────────────────────────────────────────────
streamlit run app/streamlit_app.py        # → http://localhost:8501

# ── Run via Docker (alternative) ──────────────────────────────────
docker-compose up --build                 # → http://localhost:8501

# ── Demo + tests ───────────────────────────────────────────────────
python scripts/supervised/linear_regression_demo.py
python -m pytest tests/ -v

# ── Fetch CI logs ──────────────────────────────────────────────────
python scripts/utils/ci_log_collector.py
python scripts/utils/ci_log_collector.py --analyze

# ── Push to GitHub (triggers CI automatically) ─────────────────────
git add . && git commit -m "message" && git push origin main
```

---

*Built from zero. Understood deeply. One algorithm at a time.*
