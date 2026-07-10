"""Production inference: raw review text -> sentiment + word contributions.

The deployed model is the negation-aware TF-IDF + Logistic Regression. A new
review must go through the exact same preprocessing as training, so we reuse
`preprocess_neg`. Artifacts are loaded lazily on first call.
"""
import joblib

from src.config import config
from src.preprocessing.nlp_preprocessing import preprocess_neg

_vectorizer = None
_model = None
_names = None   # feature (term) names, cached
_coefs = None   # logistic-regression weights, one per term


def _load():
    global _vectorizer, _model, _names, _coefs
    if _vectorizer is None:
        _vectorizer = joblib.load(config.VECTORIZER_NEG_PATH)
        _model = joblib.load(config.LOGREG_NEG_PATH)
        _names = _vectorizer.get_feature_names_out()
        _coefs = _model.coef_[0]


def _contributions(vector, k: int = 8) -> list[dict]:
    """Per-term contribution = TF-IDF value x learned weight (top k by magnitude)."""
    vector = vector.tocoo()  # iterate over non-zero entries only
    contribs = [
        {"mot": str(_names[j]), "contribution": float(v * _coefs[j])}
        for j, v in zip(vector.col, vector.data)
    ]
    contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contribs[:k]


def predict_sentiment(texte: str) -> dict:
    """Predict sentiment and return label, probability, confidence and contributions."""
    _load()
    vector = _vectorizer.transform([preprocess_neg(texte)])
    proba_pos = float(_model.predict_proba(vector)[0, 1])
    label = "positif" if proba_pos >= 0.5 else "négatif"
    confidence = proba_pos if proba_pos >= 0.5 else 1 - proba_pos
    return {
        "label": label,
        "proba_positif": proba_pos,
        "confiance": confidence,
        "contributions": _contributions(vector),
    }
