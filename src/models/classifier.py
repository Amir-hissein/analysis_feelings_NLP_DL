"""
MODULE DU CLASSIFIEUR BASELINE (Phase 6)
=========================================

Notre premier modèle : une Régression Logistique.

Intuition : chaque mot du vocabulaire reçoit un POIDS.
  - poids positif  -> le mot pousse vers l'avis POSITIF (brilliant, love...)
  - poids négatif  -> le mot pousse vers l'avis NÉGATIF (awful, boring...)
Le modèle fait la somme (poids × score TF-IDF) puis une sigmoïde -> une probabilité.

C'est notre BASELINE : le score simple à battre par le Deep Learning (Phases 8-9).
"""
from sklearn.linear_model import LogisticRegression

from src.config import config


def build_model() -> LogisticRegression:
    """
    Crée une Régression Logistique NON entraînée.

    Paramètres :
      - max_iter=1000 : nombre d'itérations d'optimisation (assez pour converger sur TF-IDF)
      - C=1.0         : force de régularisation (petit C = modèle plus simple/robuste).
                        1.0 est un bon point de départ ; on pourra l'ajuster plus tard.
      - random_state  : reproductibilité.
    """
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=config.RANDOM_STATE,
    )
