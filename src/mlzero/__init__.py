"""
mlzero — Machine Learning from Zero
====================================
A complete ML education library built from scratch.
Every algorithm is implemented with deep theory explanations.

Usage:
    from mlzero.supervised.regression.linear import LinearRegression
    from mlzero.core.losses import mse
    from mlzero.core.metrics import r2_score

Package structure mirrors the learning phases:
    mlzero.core              — engine: BaseModel, losses, optimizers, metrics
    mlzero.supervised        — Phase 2-3: regression + classification
    mlzero.unsupervised      — Phase 4: clustering, dimensionality reduction
    mlzero.semi_supervised   — Phase 5: self-training, label propagation
    mlzero.reinforcement     — Phase 6: Q-learning, policy gradient
    mlzero.deep_learning     — Phase 7-8: MLP, CNN, RNN, Transformer
    mlzero.modern_ai         — Phase 9: LLMs, RAG, Agents
"""

__version__ = "0.1.0"
__author__ = "Learning from Zero"
