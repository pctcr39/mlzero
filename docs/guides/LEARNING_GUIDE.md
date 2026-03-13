# Machine Learning — Learning Guide
> Beginner friendly | Built from scratch | Python + NumPy + sklearn

---

## Your Profile
- Python: basic syntax ✅
- Linear Regression: basic concept ✅
- NumPy: no experience → start here
- sklearn: no experience → after NumPy

---

## Learning Strategy
```
Understand concept → Implement from scratch → Use library → Build project
```
**Rule: Never just read. Always code.**

---

## Roadmap Overview

```
Phase 1: NumPy Basics          ← YOU ARE HERE
Phase 2: Linear Regression     (math + scratch + sklearn)
Phase 3: Classification        (Logistic Regression)
Phase 4: Core ML Algorithms
Phase 5: Deep Learning
Phase 6: Modern AI (Transformers, LLMs)
```

---

# Phase 1 — NumPy Basics

> NumPy = Python math library. Every ML framework is built on top of it.

### Install
```bash
pip install numpy matplotlib scikit-learn jupyter
```

### Arrays
```python
import numpy as np

# 1D array (vector)
a = np.array([1, 2, 3, 4, 5])
print(a.shape)   # (5,)

# 2D array (matrix)
b = np.array([[1, 2], [3, 4]])
print(b.shape)   # (2, 2) → 2 rows, 2 cols
```

### Math Operations (apply to ALL elements at once)
```python
a + 10      # [11, 12, 13, 14, 15]
a * 2       # [2, 4, 6, 8, 10]
a ** 2      # [1, 4, 9, 16, 25]
a / 5       # [0.2, 0.4, 0.6, 0.8, 1.0]
```

### Useful Functions
```python
np.sum(a)     # 15       — total sum
np.mean(a)    # 3.0      — average
np.max(a)     # 5        — maximum
np.min(a)     # 1        — minimum
np.std(a)     # standard deviation
```

### Create Special Arrays
```python
np.zeros(5)            # [0. 0. 0. 0. 0.]
np.ones((3, 3))        # 3x3 matrix of 1s
np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]   (start, stop, step)
np.random.rand(5)      # 5 random numbers between 0 and 1
np.random.randn(5)     # 5 random numbers from normal distribution
```

### Matrix Multiplication (critical for ML)
```python
X = np.array([[1, 2], [3, 4]])   # shape (2, 2)
W = np.array([[1], [2]])          # shape (2, 1)

result = X @ W    # matrix multiply → [[5], [11]]
# (2,2) @ (2,1) = (2,1)
```

### Reshape (used everywhere)
```python
a = np.array([1, 2, 3, 4, 5, 6])
a.reshape(2, 3)    # [[1,2,3],[4,5,6]]
a.reshape(-1, 1)   # [[1],[2],[3],[4],[5],[6]] — sklearn needs this
```

### ✏️ NumPy Exercises
- [ ] Create an array of numbers 1–20
- [ ] Calculate mean, std, max of that array
- [ ] Multiply two matrices together
- [ ] Reshape a 1D array into a 2D (4,3) array

---

# Phase 2 — Linear Regression

> Given input X (e.g. house size), predict output y (e.g. price).
> Goal: find the best line through the data.

## The Math

**Model:**
```
y_pred = w * X + b
```
- `w` = weight (slope of the line)
- `b` = bias (intercept)

**Loss function — Mean Squared Error (MSE):**
```
MSE = (1/n) × Σ (y_pred - y_actual)²
```
- Measures how wrong predictions are
- Lower is better
- We square the error so negatives don't cancel positives

**Gradient Descent — how we learn:**
```
w = w - learning_rate × dLoss/dw
b = b - learning_rate × dLoss/db
```
- Gradient = direction of steepest increase in loss
- We subtract it → move downhill → reduce loss
- `learning_rate` controls step size (usually 0.001–0.01)

**Gradients for Linear Regression:**
```
dLoss/dw = (2/n) × Σ (y_pred - y) × X
dLoss/db = (2/n) × Σ (y_pred - y)
```

## From Scratch Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Dataset ---
np.random.seed(42)
X = np.random.rand(50) * 100                        # house sizes
y = 3 * X + 50 + np.random.randn(50) * 10          # prices (true: w=3, b=50)

# --- Initialize ---
w = 0.0
b = 0.0
lr = 0.0001    # learning rate
epochs = 1000  # training iterations
n = len(X)

# --- Training loop ---
for epoch in range(epochs):
    y_pred = w * X + b                              # predict
    loss = np.mean((y_pred - y) ** 2)              # MSE loss

    dw = (2/n) * np.sum((y_pred - y) * X)         # gradient for w
    db = (2/n) * np.sum(y_pred - y)               # gradient for b

    w = w - lr * dw                                # update w
    b = b - lr * db                                # update b

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.2f} | w: {w:.2f} | b: {b:.2f}")

# --- Plot ---
plt.scatter(X, y, label="Data")
plt.plot(X, w * X + b, color="red", label="Fitted line")
plt.legend()
plt.show()
```

**Expected output:** w ≈ 3.0, b ≈ 50.0 (matching the true values)

## With sklearn (5 lines)

```python
from sklearn.linear_model import LinearRegression

X_2d = X.reshape(-1, 1)       # sklearn needs 2D input
model = LinearRegression()
model.fit(X_2d, y)

print(f"w={model.coef_[0]:.2f}, b={model.intercept_:.2f}")
```

## Key Concepts to Understand

| Concept | Question to ask |
|---|---|
| Learning rate too high | What happens if lr = 10? |
| Learning rate too low | What happens if lr = 0.0000001? |
| Too few epochs | What if epochs = 10? |
| Overfitting | What if the model memorizes data? |

### ✏️ Exercises
- [ ] Run scratch implementation, watch loss decrease
- [ ] Change `lr` to `0.1` — observe what happens
- [ ] Change `lr` to `0.000001` — observe what happens
- [ ] Change `epochs` to `50` vs `5000` — compare results
- [ ] Verify scratch vs sklearn give same w and b

---

# Phase 3 — Classification (Logistic Regression)

> Instead of predicting a number, predict a category: yes/no, 0/1, spam/not spam

**Key difference from Linear Regression:**
- Output must be between 0 and 1 (a probability)
- Use **sigmoid** function to squash output

**Sigmoid:**
```
sigmoid(z) = 1 / (1 + e^(-z))
```
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Loss function — Binary Cross-Entropy:**
```
Loss = -(1/n) × Σ [y × log(y_pred) + (1-y) × log(1-y_pred)]
```

*Coming after Phase 2 is complete.*

---

# Phase 4 — Core ML Algorithms

> After you understand gradient descent deeply, these become easy.

| Algorithm | Type | When to use |
|---|---|---|
| Linear Regression | Regression | Predict a number |
| Logistic Regression | Classification | Predict yes/no |
| Decision Tree | Both | Interpretable rules |
| Random Forest | Both | More accurate trees |
| SVM | Classification | Small datasets |
| KNN | Both | Simple, no training |
| K-Means | Clustering | Group unlabeled data |
| PCA | Dimensionality Reduction | Reduce features |

---

# Phase 5 — Neural Networks & Deep Learning

> A neural network is just many linear regressions stacked together with non-linear activations.

```
Input → [Linear → Activation] → [Linear → Activation] → Output
```

**Activation Functions:**
| Function | Formula | Used for |
|---|---|---|
| ReLU | max(0, x) | Hidden layers (default) |
| Sigmoid | 1/(1+e^-x) | Binary output |
| Softmax | e^x / Σe^x | Multi-class output |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | RNNs |

**Key architectures:**
- **MLP** — Fully connected, general purpose
- **CNN** — Images (spatial patterns)
- **RNN/LSTM** — Sequences, time series
- **Transformer** — Everything modern

*Unlocks after Phase 4.*

---

# Phase 6 — Modern AI

> Built on Transformers. All modern AI (GPT, Claude, DALL-E) lives here.

**Self-Attention (the core idea):**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**What to learn:**
- [ ] How Transformers work (Attention is All You Need paper)
- [ ] BERT — understands text
- [ ] GPT — generates text
- [ ] Fine-tuning pre-trained models
- [ ] RAG (Retrieval-Augmented Generation)
- [ ] Agents & Tool Use

---

# Tools & Libraries

| Tool | Purpose | Learn when |
|---|---|---|
| **NumPy** | Math arrays | Phase 1 |
| **Matplotlib** | Plotting | Phase 1 |
| **Pandas** | Data tables | Phase 2 |
| **Scikit-learn** | Classical ML | Phase 2 |
| **PyTorch** | Deep learning | Phase 5 |
| **HuggingFace** | Pre-trained models | Phase 6 |

---

# Quick Reference

### Loss Functions
| Name | Formula | Used for |
|---|---|---|
| MSE | mean((y_pred - y)²) | Regression |
| MAE | mean(\|y_pred - y\|) | Regression (robust) |
| Binary Cross-Entropy | -mean(y*log(p) + (1-y)*log(1-p)) | Binary classification |
| Cross-Entropy | -mean(Σ y*log(p)) | Multi-class |

### Evaluation Metrics
| Metric | Formula | When |
|---|---|---|
| Accuracy | correct / total | Classification |
| Precision | TP / (TP+FP) | When FP is costly |
| Recall | TP / (TP+FN) | When FN is costly |
| F1 | 2 × P×R / (P+R) | Balanced |
| R² | 1 - SS_res/SS_tot | Regression |

### Gradient Descent Variants
| Type | Description |
|---|---|
| Batch GD | Use all data per update (slow, stable) |
| Stochastic GD | Use 1 sample per update (fast, noisy) |
| Mini-batch GD | Use small batch per update (best of both) |

---

# Resources

### Free & Excellent
- [3Blue1Brown — Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) — ML intuition
- [fast.ai](https://www.fast.ai) — Practical deep learning
- [Andrej Karpathy — Neural Networks from scratch](https://www.youtube.com/@AndrejKarpathy)

### Books
- *Hands-On Machine Learning* — Aurélien Géron (practical, sklearn + TF)
- *Deep Learning* — Goodfellow, Bengio (theory)

---

# Progress Tracker

### Phase 1: NumPy
- [x] Arrays and shapes
- [x] Math operations
- [x] Matrix multiplication
- [x] Reshape

### Phase 2: Linear Regression
- [ ] Read docs/theory/supervised/LINEAR_REGRESSION.md
- [ ] Understand MSE loss (compute by hand with 3 samples)
- [ ] Understand gradient descent (bowl analogy + update rule)
- [ ] Understand learning rate effect
- [ ] Run Part 1: `python scripts/supervised/linear_regression_demo.py`
- [ ] Run Part 2: multi-feature + normalization
- [ ] Run Part 3: learning rate experiments
- [ ] Run Part 4: verify vs sklearn
- [ ] All tests pass: `pytest tests/test_supervised/ -v`
- [ ] Complete at least 2 exercises from the demo script

### Phase 3: Logistic Regression
- [ ] Understand sigmoid
- [ ] Understand cross-entropy loss
- [ ] Implement from scratch
- [ ] Use sklearn

### Phase 4: Core Algorithms
- [ ] Decision Tree
- [ ] Random Forest
- [ ] KNN
- [ ] K-Means clustering
- [ ] PCA

### Phase 5: Deep Learning
- [ ] Build MLP from scratch
- [ ] PyTorch basics
- [ ] Train on MNIST
- [ ] CNN for image classification

### Phase 6: Modern AI
- [ ] Transformers & Attention
- [ ] HuggingFace basics
- [ ] Fine-tune a model
- [ ] Build a RAG app

---

*Last updated: Phase 1–2 in progress*
