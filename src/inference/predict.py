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
_noms = None    # noms des 5000 termes du vocabulaire (mis en cache)
_coefs = None   # poids appris par la régression logistique (un par terme)


def _charger():
    global _vectorizer, _model, _noms, _coefs
    if _vectorizer is None:
        _vectorizer = joblib.load(config.VECTORIZER_PATH)
        _model = joblib.load(config.LOGREG_MODEL_PATH)
        _noms = _vectorizer.get_feature_names_out()  # index terme -> mot
        _coefs = _model.coef_[0]                      # index terme -> poids


def _contributions(vecteur, k: int = 8) -> list[dict]:
    """
    Explique la décision : pour chaque terme PRÉSENT dans l'avis,
    contribution = score TF-IDF × poids appris.
      - contribution > 0 -> le terme a poussé vers POSITIF
      - contribution < 0 -> vers NÉGATIF
    Renvoie les k termes les plus décisifs (en valeur absolue), triés.
    Note : les mots sont sous leur forme lemmatisée (loved -> love).
    """
    vecteur = vecteur.tocoo()  # parcourir uniquement les cases non nulles
    contribs = [
        {"mot": str(_noms[j]), "contribution": float(v * _coefs[j])}
        for j, v in zip(vecteur.col, vecteur.data)
    ]
    contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contribs[:k]


def predict_sentiment(texte: str) -> dict:
    """
    Prédit le sentiment d'un avis brut.

    Renvoie un dict :
      {"label": ..., "proba_positif": 0.94, "confiance": 0.94, "contributions": [...]}

    - proba_positif : probabilité que l'avis soit positif (entre 0 et 1)
    - confiance     : à quel point le modèle est sûr de SA décision (entre 0.5 et 1)
    - contributions : les mots les plus décisifs (mot + contribution +/-)
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
    return {
        "label": label,
        "proba_positif": proba_pos,
        "confiance": confiance,
        "contributions": _contributions(vecteur),
    }
