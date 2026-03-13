# mlzero — Machine Learning from Zero

> Build every AI algorithm from scratch. Understand every concept deeply.
> Goal: Think, design, and build like a top AI Systems Engineer.

---

## Quick Start

```bash
# 1. Clone / navigate to project
cd Machine_Learning

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install package (editable — changes take effect immediately)
pip install -e .

# 4. Run your first algorithm
python scripts/supervised/linear_regression_demo.py

# 5. Run tests
pytest tests/ -v
```

---

## Project Structure

```
Machine_Learning/
│
├── src/mlzero/               ← importable Python package (all source code)
│   ├── core/                 ← engine: BaseModel, losses, optimizers, metrics
│   ├── supervised/           ← regression + classification algorithms
│   │   ├── regression/       ← LinearRegression, Polynomial, Ridge, Lasso
│   │   └── classification/   ← LogisticRegression, DecisionTree, SVM, KNN
│   ├── unsupervised/         ← KMeans, DBSCAN, PCA, t-SNE
│   ├── semi_supervised/      ← SelfTraining, LabelPropagation
│   ├── reinforcement/        ← QLearning, PolicyGradient, DQN
│   ├── deep_learning/        ← MLP, CNN, RNN, Transformer
│   └── modern_ai/            ← LLM finetuning, RAG, Agents
│
├── docs/                     ← all documentation
│   ├── guides/               ← LEARNING_GUIDE.md — study roadmap + progress
│   ├── theory/               ← algorithm theory guides per phase
│   │   ├── core/             ← CORE_GUIDE.md
│   │   └── supervised/       ← SUPERVISED_GUIDE.md
│   ├── architecture/         ← system design, pipeline diagrams
│   └── reference/            ← external guides and resources
│
├── scripts/                  ← runnable training/demo scripts
│   └── supervised/           ← linear_regression_demo.py, ...
│
├── configs/                  ← hyperparameter configs (YAML) — separated from code
│   └── supervised/           ← linear_regression.yaml, ...
│
├── tests/                    ← unit tests
│   ├── test_core/            ← test_losses.py, test_metrics.py
│   └── test_supervised/      ← test_linear.py, ...
│
├── notebooks/                ← Jupyter exploration
│   └── experiments/          ← experiment notebooks
│
├── data/                     ← datasets (not committed to git)
│   ├── raw/                  ← original downloaded data
│   ├── processed/            ← cleaned / feature-engineered data
│   └── synthetic/            ← generated datasets for learning
│
├── outputs/                  ← training artifacts (not committed to git)
│   ├── plots/                ← saved figures
│   ├── logs/                 ← training logs
│   └── checkpoints/          ← saved model weights
│
├── .claude/                  ← AI agent system (ml-tutor, reviewer, experimenter...)
├── CLAUDE.md                 ← project brain — read by Claude every session
├── setup.py                  ← package installation
├── requirements.txt          ← Python dependencies
└── .gitignore
```

---

## Learning Phases

| Phase | Module | Status |
|---|---|---|
| 1 | NumPy Foundations | ✅ Done |
| 2 | Supervised — Regression | 🔄 In Progress |
| 3 | Supervised — Classification | ⏳ Next |
| 4 | Unsupervised | ⏳ Locked |
| 5 | Semi-Supervised | ⏳ Locked |
| 6 | Reinforcement Learning | ⏳ Locked |
| 7 | Neural Networks | ⏳ Locked |
| 8 | Deep Learning | ⏳ Locked |
| 9 | Modern AI | ⏳ Locked |

See [docs/guides/LEARNING_GUIDE.md](docs/guides/LEARNING_GUIDE.md) for detailed progress tracker.

---

## How to Study Each Algorithm

```
1. Read theory:     docs/theory/<phase>/<ALGORITHM_GUIDE.md>
2. Read source:     src/mlzero/<phase>/<algorithm>.py
3. Run demo:        python scripts/<phase>/<algorithm>_demo.py
4. Run tests:       pytest tests/test_<phase>/test_<algorithm>.py -v
5. Experiment:      edit configs/<phase>/<algorithm>.yaml → rerun
6. Track progress:  check off items in docs/guides/LEARNING_GUIDE.md
```

---

## AI Agent System (Claude Code)

This project uses specialized AI agents to support learning:

| Agent | Command | Purpose |
|---|---|---|
| ml-tutor | `/explain-concept X` | Deep theory explanations |
| mathematician | `/explain-concept X` | Math derivations |
| code-reviewer | `/review-code path` | Code review + feedback |
| experimenter | `/experiment X` | Structured experiments |
| debugger | auto-activated | Debug ML bugs |

Full workflow: `/learn-algorithm <AlgorithmName>` walks through the complete cycle.

---

## Import Examples

```python
# After pip install -e . you can import from anywhere:
from mlzero.core.losses import mse, binary_cross_entropy
from mlzero.core.metrics import r2_score, accuracy
from mlzero.core.optimizers import Adam
from mlzero.supervised.regression.linear import LinearRegression

# Train a model
model = LinearRegression(lr=0.0001, epochs=1000)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

---

*Built from zero. Understood deeply. One algorithm at a time.*
