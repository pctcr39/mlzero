---
globs: "core/**/*.py"
---

When editing core/ Python files:

- These are the ENGINE — no algorithm logic here, only infrastructure
- base.py: only abstract interfaces (raise NotImplementedError)
- losses.py: pure functions only, no class state
- optimizers.py: stateless functions OR stateful classes (Adam needs state)
- metrics.py: pure functions only

- Every function MUST explain:
  1. The theory/formula being implemented
  2. When to use this function vs alternatives
  3. Numerical stability tricks (clipping, epsilon) with explanation WHY
  4. Local vs global variable annotation on every variable

- Loss functions: must include worked example with numbers in docstring
- Metrics: must include interpretation of output (what does 0.94 mean?)
