# ML Theory Knowledge Base

This skill is loaded when explaining theory-heavy concepts.

## The Universal ML Formula

Every ML algorithm minimizes a loss function:
```
θ* = argmin_θ L(θ; X, y)
```
Where:
- θ (theta) = model parameters (weights, biases)
- L = loss function
- X = input features
- y = target labels
- argmin = "find θ that makes L smallest"

## Gradient Descent Family

```
Batch GD:      use all n samples  → stable, slow
Stochastic GD: use 1 sample       → noisy, fast
Mini-batch GD: use k samples      → best of both (default in practice)
```

## Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias    = underfitting (model too simple)
High Variance = overfitting (model too complex)
Goal: find the sweet spot
```

## Regularization

```
L1 (Lasso): Loss + λ × Σ|w|   → drives some weights to exactly 0 (feature selection)
L2 (Ridge): Loss + λ × Σw²    → shrinks all weights toward 0 (smoother model)
```

## When to Use Which Algorithm

```
Task                 → Start with
─────────────────────────────────────────────────
Regression           → Linear Regression
Binary classification → Logistic Regression
Multi-class          → Logistic Regression (softmax)
Tabular data, fast   → Random Forest or XGBoost
Images               → CNN
Text/sequences       → Transformer
Reinforcement        → Q-Learning or PPO
```
