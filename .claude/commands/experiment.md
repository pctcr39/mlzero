# /experiment

Run structured experiments to build intuition about an algorithm.

## Input: $ARGUMENTS (e.g. "learning rate on linear regression")

## Steps

1. Read CLAUDE.md to know current algorithm
2. Call `experimenter` agent to:
   - State hypothesis before running
   - Define what to change and what to observe
   - Run experiment (modify parameters in scratch file)
   - Document results in a table
   - Explain WHY results happened
   - Extract a one-sentence rule to remember
3. Output a markdown experiment report saved to:
   `notebooks/experiments/[algorithm]-[parameter]-experiment.md`
4. Add key findings to CLAUDE.md "Compound Learnings"

## Standard Experiments to Run for Each Algorithm

### For Regression:
- /experiment "learning rate sweep on linear regression"
- /experiment "epoch count impact on linear regression"
- /experiment "overfitting test with many features, few samples"

### For Classification:
- /experiment "decision boundary visualization on logistic regression"
- /experiment "class imbalance effect on accuracy vs F1"
- /experiment "threshold tuning on logistic regression"
