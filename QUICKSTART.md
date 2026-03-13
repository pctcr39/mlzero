# QUICKSTART — How to Run mlzero

> Step-by-step instructions for every way to run this project.
> No assumptions — follow in order top to bottom.

---

## Table of Contents

1. [Prerequisites — Install Once](#1-prerequisites--install-once)
2. [Option A — Run Locally (Python venv)](#2-option-a--run-locally-python-venv)
3. [Option B — Run with Docker (no Python needed)](#3-option-b--run-with-docker-no-python-needed)
4. [Streamlit Dashboard — Live Training UI](#4-streamlit-dashboard--live-training-ui)
5. [Demo Script — Linear Regression](#5-demo-script--linear-regression)
6. [Run Tests](#6-run-tests)
7. [CI/CD Log Collector — Fetch & Analyze Build Logs](#7-cicd-log-collector--fetch--analyze-build-logs)
8. [Common Problems & Fixes](#8-common-problems--fixes)

---

## 1. Prerequisites — Install Once

Install these tools **once on your Mac**. Skip anything already installed.

### Python 3.11+
```bash
# Check if already installed:
python3 --version
# Should show: Python 3.11.x or higher

# If not installed:
brew install python@3.11
```

### Git
```bash
git --version
# Should show: git version 2.x
# If not: brew install git
```

### Docker Desktop *(only needed for Option B)*
Download and install from: https://www.docker.com/products/docker-desktop/

After installing, open Docker Desktop at least once to start the daemon.

```bash
# Verify Docker is running:
docker --version        # Should show: Docker version 24.x or higher
docker-compose version  # Should show: Docker Compose version 2.x
```

### GitHub CLI *(only needed for CI log collector)*
```bash
brew install gh
gh auth login
# Choose: GitHub.com → SSH → browser login
```

---

## 2. Option A — Run Locally (Python venv)

Use this for day-to-day development and learning. Fastest to start.

### Step 1 — Clone the project
```bash
git clone git@github.com:pctcr39/mlzero.git
cd mlzero
```

### Step 2 — Create virtual environment
```bash
# Create a local Python environment inside .venv/
python3 -m venv .venv
```

**What is a venv?**
A virtual environment is an isolated Python installation just for this project.
It prevents conflicts between projects that need different package versions.
Think of it like a sandbox — packages installed here don't affect the rest of your Mac.

### Step 3 — Activate the environment
```bash
source .venv/bin/activate
```

Your terminal prompt will change to show `(.venv)` at the start — this means
you're now "inside" the environment. **You must do this every time you open a new terminal.**

```bash
# To deactivate when done:
deactivate
```

### Step 4 — Install dependencies
```bash
# Install all required packages listed in requirements.txt
pip install -r requirements.txt

# Install mlzero as an editable package
# "editable" (-e) means code changes take effect immediately without reinstalling
pip install -e .
```

### Step 5 — Verify installation
```bash
python -c "from mlzero.supervised.regression.linear import LinearRegression; print('OK')"
# Should print: OK
```

---

## 3. Option B — Run with Docker (no Python needed)

Use this to share the project with others or run in a clean environment.
Docker bundles Python + all dependencies + the app into one container.

### Step 1 — Clone the project
```bash
git clone git@github.com:pctcr39/mlzero.git
cd mlzero
```

### Step 2 — Build and start
```bash
docker-compose up --build
```

**What this does:**
1. Reads `Dockerfile` — builds a Python 3.11 container with all packages installed
2. Starts the Streamlit dashboard on port 8501
3. Mounts `outputs/` so files saved inside the container appear on your Mac

**First run is slow** (~2-3 min) because it downloads the base image and installs packages.
Subsequent runs are fast because Docker caches the layers.

### Step 3 — Open the dashboard
```
http://localhost:8501
```

### Step 4 — Stop
```bash
# Press Ctrl+C in the terminal, then:
docker-compose down
```

### Useful Docker commands
```bash
# Start in background (detached):
docker-compose up -d --build

# View logs when running in background:
docker-compose logs -f app

# Rebuild after changing requirements.txt:
docker-compose up --build

# Remove container and image (clean slate):
docker-compose down --rmi all
```

---

## 4. Streamlit Dashboard — Live Training UI

The dashboard has 3 tabs. Run it locally or via Docker (both work).

### Run locally
```bash
source .venv/bin/activate   # activate venv if not already active
streamlit run app/streamlit_app.py
```

Open: **http://localhost:8501**

### What each tab does

#### Tab 1 — Train (Streaming)
Watch the model learn in real-time.

1. Adjust hyperparameters in the **left sidebar**:
   - **Learning Rate** — controls step size. Try different values to see the effect.
   - **Epochs** — how many training iterations.
   - **Dataset Size** — how many data points.
   - **Noise Level** — how noisy the training data is.

2. Choose single feature (house size → price) or multi-feature (size + rooms + age).

3. Click **▶ Start Training**.

4. Watch:
   - Loss curve drops in real-time as the model learns
   - Live metrics: current epoch, loss value, Test R²
   - After training: comparison table of learned weights vs true weights

**Experiment ideas:**
| Try this | What you'll see |
|---|---|
| `lr = 0.5` | Loss explodes (overshooting) |
| `lr = 0.0001` | Loss drops very slowly |
| `lr = 0.01` | Smooth, fast convergence |
| Noise = 20000 | Model has trouble fitting |
| Noise = 0 | Near-perfect fit |

#### Tab 2 — Predict
Make instant predictions using the trained model.

1. First train a model in Tab 1.
2. Switch to Tab 2.
3. Enter house features (size, rooms, age).
4. Predicted price updates **instantly** as you type — no button needed.
5. Calculation breakdown shows the exact math: `y = X @ w + b`.

#### Tab 3 — CI/CD Logs
View stored GitHub Actions build logs.

1. First run the CI log collector (see section 7 below).
2. Refresh the page.
3. See a table of all stored runs: pass/fail, test counts, duration.
4. Click any run to see full log output.

---

## 5. Demo Script — Linear Regression

A 4-part educational demo that runs in the terminal and saves plots.

```bash
source .venv/bin/activate
python scripts/supervised/linear_regression_demo.py
```

**What it runs:**
- Part 1: Single feature (house size → price)
- Part 2: Multi-feature with normalization
- Part 3: Learning rate comparison (4 different lr values side by side)
- Part 4: Comparison against sklearn (verifies our from-scratch implementation)

**Output:**
- Terminal: loss values, R² scores, weight comparison
- `outputs/plots/` — saved PNG figures

**Change hyperparameters** without editing code — edit the YAML config:
```bash
# Edit learning rate, epochs, etc.:
open configs/supervised/linear_regression.yaml
# Then re-run the script
```

---

## 6. Run Tests

Tests verify that every function works correctly.

```bash
source .venv/bin/activate

# Run all tests:
python -m pytest tests/ -v

# Run only linear regression tests:
python -m pytest tests/test_supervised/test_linear.py -v

# Run only core (loss function) tests:
python -m pytest tests/test_core/test_losses.py -v

# Run with coverage report:
python -m pytest tests/ -v --cov=src/mlzero --cov-report=term-missing
```

**Expected output:** `24 passed in 0.60s`

**What the tests check:**

| Test Class | What It Verifies |
|---|---|
| `TestInit` | Model starts with correct default values |
| `TestFit` | Model trains without crashing; loss decreases |
| `TestPredict` | Predictions are numerically correct |
| `TestScore` | R² is 1.0 on noise-free data; 0.0 on garbage |
| `TestMultiFeature` | Works with multiple input columns |

---

## 7. CI/CD Log Collector — Fetch & Analyze Build Logs

After every `git push`, GitHub Actions runs tests automatically (CI = Continuous Integration).
This tool downloads those results and saves them locally for long-term analysis.

### Prerequisites
```bash
brew install gh      # GitHub CLI
gh auth login        # authenticate once
```

### Usage

```bash
source .venv/bin/activate

# Fetch latest 10 CI runs and save to outputs/ci_logs/:
python scripts/utils/ci_log_collector.py

# Fetch more runs:
python scripts/utils/ci_log_collector.py --limit 25

# Just print analysis (no new fetches):
python scripts/utils/ci_log_collector.py --analyze

# Regenerate the markdown report:
python scripts/utils/ci_log_collector.py --report

# Fast mode (skip downloading full log text):
python scripts/utils/ci_log_collector.py --no-log
```

### What gets saved
Each CI run is saved as: `outputs/ci_logs/run_<id>.json`

```json
{
  "run_id": 123456789,
  "branch": "main",
  "conclusion": "success",
  "created_at": "2026-03-13T10:00:00Z",
  "tests_passed": 24,
  "tests_failed": 0,
  "duration_seconds": 45.3,
  "raw_log": "..."
}
```

A markdown report is generated at: `outputs/ci_logs/REPORT.md`

### View logs in the UI
After fetching logs, open the Streamlit dashboard Tab 3 — all stored runs appear there
in a filterable table with drill-down into full log output.

### CI Pipeline (what runs automatically on every push)

The `.github/workflows/ci.yml` runs:
```
push to any branch → GitHub Actions triggers:
  ├── Python 3.10 → pip install -e . → pytest tests/ -v
  ├── Python 3.11 → pip install -e . → pytest tests/ -v
  └── Python 3.12 → pip install -e . → pytest tests/ -v
```

View results at: **https://github.com/pctcr39/mlzero/actions**

---

## 8. Common Problems & Fixes

### `command not found: python`
```bash
# Use python3 instead:
python3 --version

# Or activate the venv first:
source .venv/bin/activate
python --version  # now works
```

### `ModuleNotFoundError: No module named 'mlzero'`
```bash
# The package isn't installed. Run:
pip install -e .
# Then retry.
```

### `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install streamlit
```

### Port 8501 already in use
```bash
# Find what's using it:
lsof -i :8501

# Kill it:
kill -9 <PID>

# Or run Streamlit on a different port:
streamlit run app/streamlit_app.py --server.port 8502
```

### Docker: `Cannot connect to Docker daemon`
Open **Docker Desktop** app on your Mac first. The daemon must be running.

### `gh: command not found`
```bash
brew install gh
gh auth login
```

### Tests fail: `ImportError`
```bash
# Reinstall the package:
pip install -e .
python -m pytest tests/ -v
```

### Loss is not decreasing during training
Most likely cause: **learning rate too high**.
- Try reducing `lr` by 10×: if `0.1` fails, try `0.01` or `0.001`
- In Streamlit: use the sidebar slider to experiment
- Theory: high lr = steps overshoot the minimum → loss bounces or explodes

---

## Project File Map (Quick Reference)

```
mlzero/
│
├── app/
│   └── streamlit_app.py          ← Streamlit dashboard (run this for the UI)
│
├── scripts/
│   ├── supervised/
│   │   └── linear_regression_demo.py   ← 4-part terminal demo
│   └── utils/
│       └── ci_log_collector.py         ← CI/CD log fetcher + analyzer
│
├── src/mlzero/                    ← all algorithm source code
│   ├── core/                      ← base, losses, optimizers, metrics
│   └── supervised/regression/
│       └── linear.py              ← LinearRegression class
│
├── tests/                         ← pytest unit tests (run: python -m pytest tests/ -v)
├── configs/supervised/            ← linear_regression.yaml (hyperparameters)
├── outputs/
│   ├── plots/                     ← saved figures from demo script
│   └── ci_logs/                   ← stored CI build logs (JSON + REPORT.md)
│
├── docs/
│   ├── guides/LEARNING_GUIDE.md   ← study roadmap + progress checkboxes
│   └── theory/supervised/
│       └── LINEAR_REGRESSION.md   ← full theory: math, intuition, formulas
│
├── Dockerfile                     ← container definition
├── docker-compose.yml             ← `docker-compose up --build` to start
├── CLAUDE.md                      ← project brain (Claude reads this each session)
└── QUICKSTART.md                  ← this file
```

---

*Built from zero. Understood deeply. One algorithm at a time.*
