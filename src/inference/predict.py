"""
MODULE D'INFÉRENCE (Phase 10)
==============================

Encapsule le MEILLEUR modèle (baseline : TF-IDF + Régression Logistique, 88,6 %)
dans UNE fonction simple : donner un avis brut -> obtenir sentiment + probabilité.

⚠️  RÈGLE D'OR (cohérence entraînement ↔ prédiction) :
    un nouvel avis doit traverser EXACTEMENT le même pipeline qu'à l'entraînement.
    Le TF-IDF a été entraîné sur `review_nlp` = preprocess(clean_text(avis)).
    On rejoue donc ces deux étapes AVANT de vectoriser.

    avis brut → clean_text() → preprocess() spaCy → TF-IDF → modèle → proba

Les artefacts (vectoriseur + modèle) sont chargés UNE fois, au premier appel.
"""
import joblib

from src.config import config
from src.preprocessing.cleaning import clean_text
from src.preprocessing.nlp_preprocessing import preprocess

# ── Chargement paresseux des artefacts (une seule fois) ──────────────
# On les met dans des variables globales remplies au 1er appel : évite de
# recharger le modèle du disque à chaque prédiction.
_vectorizer = None
_model = None


def _charger():
    global _vectorizer, _model
    if _vectorizer is None:
        _vectorizer = joblib.load(config.VECTORIZER_PATH)
        _model = joblib.load(config.LOGREG_MODEL_PATH)


def predict_sentiment(texte: str) -> dict:
    """
    Prédit le sentiment d'un avis brut.

    Renvoie un dict :
      {"label": "positif"/"négatif", "proba_positif": 0.94, "confiance": 0.94}

    - proba_positif : probabilité que l'avis soit positif (entre 0 et 1)
    - confiance     : à quel point le modèle est sûr de SA décision (entre 0.5 et 1)
    """
    _charger()
    # 1. MÊME pipeline qu'à l'entraînement (clean_text puis preprocess spaCy).
    texte_nlp = preprocess(clean_text(texte))
    # 2. Vectorisation avec le TF-IDF ENTRAÎNÉ (transform, jamais fit !).
    vecteur = _vectorizer.transform([texte_nlp])
    # 3. Probabilité de la classe "positif" (colonne 1).
    proba_pos = float(_model.predict_proba(vecteur)[0, 1])
    label = "positif" if proba_pos >= 0.5 else "négatif"
    confiance = proba_pos if proba_pos >= 0.5 else 1 - proba_pos
    return {"label": label, "proba_positif": proba_pos, "confiance": confiance}
