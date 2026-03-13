# /explain-concept

Get a deep explanation of any ML/AI concept from first principles.

## Input: $ARGUMENTS (concept name, e.g. "gradient descent", "overfitting", "backpropagation")

## Steps

1. Call `ml-tutor` agent to explain "$ARGUMENTS":
   - One-sentence summary
   - Real-world analogy
   - How it fits in the ML pipeline
   - Math formula with every symbol defined
   - Walkthrough with real numbers
   - Python code (minimal, heavily commented)
   - Common mistakes / misconceptions
   - How it connects to future topics

2. If math derivation is needed, call `mathematician` agent to:
   - Derive from scratch using chain rule
   - Show every step
   - Map to NumPy code

3. Suggest: which experiment to run to build intuition
4. Suggest: which file in the project demonstrates this concept

## Examples to Try
- /explain-concept "gradient descent"
- /explain-concept "overfitting vs underfitting"
- /explain-concept "cross-entropy loss"
- /explain-concept "sigmoid function"
- /explain-concept "matrix multiplication in ML"
- /explain-concept "bias-variance tradeoff"
- /explain-concept "backpropagation"
- /explain-concept "attention mechanism"
