# Core Engine — How Everything Connects
> Read this before touching any algorithm file.
> This explains the big picture pipeline and why the project is built this way.

---

## The Big Picture

Every AI system — no matter how complex — does the same 3 things:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   DATA  →  MODEL  →  LOSS  →  OPTIMIZER  →  repeat │
│                                                     │
│   "input"   "guess"  "how    "get better"           │
│                       wrong?"                       │
└─────────────────────────────────────────────────────┘
```

This is true for:
- Linear Regression (Phase 2)
- Neural Networks (Phase 7)
- ChatGPT (Phase 9)

The only difference is **how complex each piece is.**

---

## Files in `core/`

| File | What it does | Analogy |
|---|---|---|
| `base.py` | Template all models follow | Job description |
| `losses.py` | Measures how wrong the model is | Test score |
| `optimizers.py` | Updates the model to reduce loss | Studying after failing |
| `metrics.py` | Evaluates the final model | Final grade |

---

## `base.py` — The Template

### Why does this exist?

Without a template, every algorithm would be written differently:
```python
# BAD: every algorithm has different names
linear_model.train(X, y)
decision_tree.learn(data, labels)
neural_net.fit(inputs, targets)
```

With a template, everything is consistent:
```python
# GOOD: every algorithm works the same way
linear_model.fit(X, y)
decision_tree.fit(X, y)
neural_net.fit(X, y)
```

This is called an **interface** in software engineering.

### The 3 core methods

```python
model.fit(X, y)        # Step 1: Learn from training data
model.predict(X)       # Step 2: Make predictions on new data
model.score(X, y)      # Step 3: Measure how good predictions are
```

### How inheritance works

```python
# base.py defines the template
class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError   # "you MUST implement this"

# linear_scratch.py uses the template
class LinearRegression(BaseModel):  # ← "inherits from BaseModel"
    def fit(self, X, y):
        # actual implementation goes here
        ...
```

Think of `BaseModel` as a contract:
> "Every model in this project WILL have fit(), predict(), and score()."

---

## `losses.py` — How Wrong Are We?

### The training loop (this is ALL of ML)

```python
for each training step:
    y_pred = model.predict(X)          # 1. guess
    loss   = loss_function(y, y_pred)  # 2. measure how wrong
    update model to reduce loss        # 3. improve
```

### Which loss function to use?

```
Problem Type              → Loss Function
─────────────────────────────────────────
Predict a number          → MSE or MAE
Predict yes/no            → Binary Cross-Entropy
Predict one of N classes  → Cross-Entropy
```

### Visual understanding of MSE

```
Actual:    [3,   5,   2]
Predicted: [4,   4,   3]
Error:     [+1,  -1,  +1]   ← can be positive or negative
Squared:   [1,   1,   1]    ← always positive
MSE = mean([1, 1, 1]) = 1.0
```

### Visual understanding of Cross-Entropy

```
Question: Is this email spam?
True label:      1 (yes, it's spam)
Model predicted: 0.9 (90% confident it's spam)

Loss = -log(0.9) = 0.105  ← small loss, model was right

Model predicted: 0.1 (only 10% confident it's spam)
Loss = -log(0.1) = 2.303  ← large loss, model was wrong and confident
```

Key insight: **the more confident and wrong you are, the bigger the punishment.**

---

## How the Pipeline Flows (Supervised Learning)

```
Step 1: Collect Data
        X = [[size, rooms], [size, rooms], ...]   ← features (input)
        y = [price, price, ...]                    ← labels (output)
           ↓
Step 2: Split Data
        X_train, y_train  ← model learns from this
        X_test,  y_test   ← model is evaluated on this (never seen during training)
           ↓
Step 3: Create Model
        model = LinearRegression(lr=0.001, epochs=1000)
           ↓
Step 4: Train (fit)
        model.fit(X_train, y_train)
        → inside fit(): runs gradient descent loop
        → adjusts w and b to reduce MSE loss
           ↓
Step 5: Predict
        y_pred = model.predict(X_test)
           ↓
Step 6: Evaluate
        score = model.score(X_test, y_test)
        → how well does it generalize to data it never saw?
```

---

## How Future Learning Types Integrate

### Unsupervised (Phase 4)
```python
# No labels — model finds patterns on its own
model = KMeans(k=3)
model.fit(X)          # ← no y! just X
clusters = model.predict(X)
```

### Semi-Supervised (Phase 5)
```python
# Some data has labels, most doesn't
# Common in real world — labeling data is expensive
model = SelfTrainingClassifier()
model.fit(X_labeled, y_labeled, X_unlabeled)
```

### Reinforcement Learning (Phase 6)
```python
# No data at all — agent learns by taking actions and getting rewards
agent = QLearningAgent()
agent.fit(environment)   # ← learns by playing, not from a dataset
```

### Neural Networks (Phase 7+)
```python
# Same interface, but model is much more complex internally
model = NeuralNetwork(layers=[64, 32, 1])
model.fit(X_train, y_train)   # ← same fit() interface!
```

**Key insight:** The interface (`fit`, `predict`, `score`) stays the same.
Only the internal complexity grows.

---

## Local vs Global Variables (Python)

This is important for reading the code files.

```python
import numpy as np   # ← GLOBAL to this file (available in all functions below)

LEARNING_RATE = 0.001  # ← GLOBAL constant (ALL CAPS = constant by convention)

def mse(y_true, y_pred):   # ← parameters are LOCAL to this function
    errors = y_pred - y_true   # ← LOCAL: only exists inside this function
    squared = errors ** 2      # ← LOCAL: disappears when function returns
    return np.mean(squared)    # ← uses np which is GLOBAL to the file
```

Rule of thumb:
- **Global** = defined at the top of the file, available everywhere
- **Local** = defined inside a function, disappears when function ends
- **Parameter** = a local variable passed in when the function is called

---

## Reading Code Like an Engineer

When you open any file in this project, ask:

1. **What problem does this solve?** (read the docstring at the top)
2. **What goes in?** (look at function parameters)
3. **What comes out?** (look at return statement)
4. **What formula is being implemented?** (look at the math comment)
5. **Where does this fit in the pipeline?** (fit? predict? loss? optimizer?)

---

## Files to Read in Order

```
1. core/CORE_GUIDE.md              ← this file (big picture)
2. core/base.py                    ← the template
3. core/losses.py                  ← how we measure error
4. 01_supervised/SUPERVISED_GUIDE.md  ← next chapter
5. 01_supervised/regression/linear_scratch.py  ← first algorithm
```

---

*The core engine never changes. Only the algorithms on top of it change.*
