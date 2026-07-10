"""
MODULE DE VECTORISATION TF-IDF (Phase 5)
=========================================

Transforme du texte en vecteurs de nombres que le ML classique sait utiliser.

Le TF-IDF donne un score à chaque mot de chaque avis :
    score = TF (fréquent dans CET avis) × IDF (rare dans TOUS les avis)
=> les mots rares et discriminants (masterpiece, awful) ressortent ;
   les mots omniprésents (movie, film) sont écrasés.

⚠️  RÈGLE D'OR (anti-fuite de données) :
    on fait `.fit()` UNIQUEMENT sur le train, puis `.transform()` sur train ET test.
    Le vocabulaire et les IDF ne doivent JAMAIS voir les données de test.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import config


def build_vectorizer() -> TfidfVectorizer:
    """
    Crée un vectoriseur TF-IDF NON entraîné, avec nos hyperparamètres.

    Paramètres choisis :
      - max_features : ne garder que les N mots les plus fréquents (taille des vecteurs maîtrisée)
      - ngram_range=(1, 2) : compter les mots seuls ET les paires de mots consécutifs
        ("not good" capté comme un tout -> important pour le sentiment !)
      - min_df=5 : ignorer les mots présents dans moins de 5 avis (fautes de frappe, mots ultra-rares)
      - sublinear_tf=True : atténuer l'effet des répétitions (log) -> 10 fois "good" ≠ 10× le poids
    """
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True,
    )
