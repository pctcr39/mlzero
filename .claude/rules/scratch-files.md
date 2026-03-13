---
globs: "**/*_scratch.py"
---

When editing or creating *_scratch.py files:

- NO ML libraries (no sklearn, no PyTorch, no TensorFlow)
- NumPy and Matplotlib ONLY
- Every class MUST inherit from core.base.BaseModel
- Every function MUST have a docstring with: THEORY, PARAMETERS, RETURNS
- Every non-obvious variable MUST have a comment:
  `# LOCAL: what this variable stores and why`
  `# INSTANCE: persists across method calls`
  `# GLOBAL: available to entire file`
- At the bottom, ALWAYS have `if __name__ == "__main__":` with:
  - Demo that shows the algorithm working
  - At least 3 exercises for the learner
  - Comparison to expected output (e.g., "Expected: w≈3.0")
- Loss MUST be printed every 100 epochs during training
- After training, ALWAYS plot: data + fitted line/boundary + loss curve
