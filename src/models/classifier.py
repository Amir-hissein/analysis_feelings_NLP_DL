"""Baseline classifier: Logistic Regression.

Each term gets a weight; the model sums (weight x TF-IDF) and applies a sigmoid
to produce a probability. Interpretable and the strong baseline to beat.
"""
from sklearn.linear_model import LogisticRegression

from src.config import config


def build_model() -> LogisticRegression:
    """Return an unfitted Logistic Regression (C=1.0 regularization)."""
    return LogisticRegression(max_iter=1000, C=1.0, random_state=config.RANDOM_STATE)
