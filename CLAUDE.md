# ML Learning Project — Brain File
> Claude reads this at the start of EVERY session.
> This is the memory and rules of the entire learning system.

---

## Learner Profile
- **Level:** Absolute beginner → Target: Top AI Systems Engineer
- **Background:** Basic Python syntax, basic Linear Regression concept
- **No prior experience:** NumPy, sklearn, PyTorch, deep learning
- **Goal:** Understand AI deeply enough to BUILD and DESIGN AI systems from scratch

## Communication Rules (CRITICAL)
- ALWAYS explain theory BEFORE showing code
- ALWAYS explain what each variable is (local/instance/global) and WHY it exists
- ALWAYS connect code to AI theory ("this line implements the gradient formula because...")
- Use analogies and plain English first, then math, then code
- Never say "just do this" — always explain the reason
- If a concept has math, show the formula AND the intuition
- Write at the level of a BSW engineer learning CS for the first time

---

## Project Structure
```
Machine_Learning/
├── CLAUDE.md                               ← this file (brain)
├── README.md                               ← project overview + quick start
├── setup.py                                ← pip install -e . (makes mlzero importable)
├── requirements.txt                        ← Python dependencies
│
├── src/mlzero/                             ← ALL source code (importable package)
│   ├── core/                               ← engine: base, losses, optimizers, metrics
│   ├── supervised/regression/              ← CURRENT PHASE — linear.py
│   ├── supervised/classification/          ← next
│   ├── unsupervised/                       ← future Phase 4
│   ├── semi_supervised/                    ← future Phase 5
│   ├── reinforcement/                      ← future Phase 6
│   ├── deep_learning/                      ← future Phase 7-8
│   └── modern_ai/                          ← future Phase 9
│
├── docs/
│   ├── guides/LEARNING_GUIDE.md           ← study roadmap + progress tracker
│   ├── theory/core/CORE_GUIDE.md          ← pipeline architecture explanation
│   └── theory/supervised/SUPERVISED_GUIDE.md
│
├── scripts/supervised/                     ← runnable demo scripts
├── configs/supervised/                     ← YAML hyperparameter configs
├── tests/test_core/                        ← unit tests
├── data/                                   ← datasets (raw, processed, synthetic)
├── outputs/                                ← plots, logs, checkpoints
└── notebooks/                              ← Jupyter exploration
```

---

## Current Learning Phase
- **Phase:** 2 — Supervised Learning (Regression)
- **Active file:** `src/mlzero/supervised/regression/linear.py`
- **Just completed:** Phase 1 NumPy (arrays, math ops, matrix multiply, reshape)
- **Next up:** Logistic Regression (classification)

---

## Learning Workflow (MUST follow this order)

```
Phase 1: READ — Read the concept guide (_GUIDE.md)
Phase 2: MATH — Understand the formula on paper
Phase 3: SCRATCH — Implement from scratch (no libraries)
Phase 4: LIBRARY — Use sklearn/PyTorch to verify
Phase 5: EXPERIMENT — Change parameters, break it, rebuild it
Phase 6: CHECK OFF — Mark progress in ML_Learning_Guide.md
```

Never skip to Phase 3 without understanding Phase 2.
Never use a library before building it from scratch.

---

## Patterns Learned So Far

### NumPy (Phase 1 — COMPLETE)
- `np.array()`, `.shape`, `.reshape(-1, 1)`
- Element-wise math: `+`, `*`, `**`, `/`
- `np.mean()`, `np.sum()`, `np.max()`, `np.min()`
- Matrix multiply: `A @ B`
- Creating arrays: `np.zeros()`, `np.ones()`, `np.arange()`, `np.random.rand()`

### Linear Regression (Phase 2 — IN PROGRESS)
- Model: `y_pred = X @ w + b`
- Loss: MSE = `mean((y_pred - y) ** 2)`
- Gradient for w: `(2/n) * X.T @ (y_pred - y)`
- Gradient for b: `(2/n) * sum(y_pred - y)`
- Update rule: `w = w - lr * dw`

---

## Code Standards in This Project

- Every Python file MUST have a block comment at top explaining: THEORY, MATH, VARIABLES
- Every function MUST have a docstring explaining: THEORY, PARAMETERS (local/instance?), RETURNS
- Every non-obvious variable MUST have a comment: `# LOCAL: what this does`
- `src/mlzero/` files = from-scratch implementations (NumPy only, no sklearn)
- `scripts/*_sklearn.py` = sklearn verification scripts (compare results)

---

## Agent Team for This Project

| Agent | Role | When to call |
|---|---|---|
| `ml-tutor` | Deep theory explanations | "explain this concept" |
| `code-reviewer` | Review implementations | "review my code" |
| `mathematician` | Math derivations | "derive this formula" |
| `experimenter` | Run and analyze experiments | "experiment with parameters" |
| `debugger` | Debug ML code | "my code gives wrong output" |

---

## Compound Learnings (auto-updated after each phase)

*This section grows as learning progresses.*

- **Epoch 1 learning:** Learning rate too high → loss oscillates. Too low → converges very slowly.
- **Key insight:** MSE gradient `(2/n) * X.T @ error` is just the chain rule applied to matrix notation.
- **Pattern:** Always check train R² vs test R² — big gap = overfitting.

---

## Session Startup Checklist
When starting a new session, Claude should:
1. Read this file (CLAUDE.md)
2. Check docs/guides/LEARNING_GUIDE.md progress tracker
3. Continue from where we left off
4. Ask: "Last time we were on [X]. Ready to continue or do you have questions?"
