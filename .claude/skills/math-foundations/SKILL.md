# Math Foundations Skill

Loaded when deriving formulas or explaining math.

## Chain Rule (backbone of backpropagation)

```
If y = f(g(x))
Then dy/dx = f'(g(x)) × g'(x)
= "outer derivative × inner derivative"

Example:
  Loss = (w*x - y)²
  Let z = w*x - y
  Loss = z²

  dLoss/dw = dLoss/dz × dz/dw
           = 2z      × x
           = 2(w*x - y) × x
```

## Partial Derivatives

```
f(w, b) = w*x + b
∂f/∂w = x     (treat b as constant, differentiate with respect to w)
∂f/∂b = 1     (treat w as constant, differentiate with respect to b)
```

## Matrix Calculus Reference

```
If y = Xw, then dy/dw = X^T
If L = ||y - Xw||², then dL/dw = -2X^T(y - Xw) = 2X^T(Xw - y)
```

## Common Derivatives

```
d/dx [x²]     = 2x
d/dx [x^n]    = n*x^(n-1)
d/dx [e^x]    = e^x
d/dx [log(x)] = 1/x
d/dx [sigmoid(x)] = sigmoid(x) × (1 - sigmoid(x))
d/dx [max(0,x)]   = 0 if x<0, 1 if x>0  (ReLU)
```

## Matrix Shape Rules

```
(a,b) @ (b,c) = (a,c)    ← inner dims must match
(a,b).T = (b,a)          ← transpose swaps dims

In NumPy:
  X.shape   = (n, m)     ← n samples, m features
  w.shape   = (m,)       ← one weight per feature
  X @ w     = (n,)       ← n predictions
  X.T @ e   = (m,)       ← m gradients (one per weight)
               where e = error vector, shape (n,)
```
