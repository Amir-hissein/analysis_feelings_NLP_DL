"""
MODULE DE NETTOYAGE DE TEXTE (Phase 3)
=======================================

Fonctions RÉUTILISABLES pour nettoyer un avis. On les met dans src/ (et non
dans un notebook) car on les réutilisera plus tard : entraînement ML ET API.
Un texte entrant dans l'API devra être nettoyé EXACTEMENT comme à l'entraînement.

Idée clé : on nettoie de façon "agressive" pour le ML classique (TF-IDF).
Pour BERT (Phase 9), on utilisera un nettoyage BEAUCOUP plus léger.
"""
import re  # 're' = expressions régulières : chercher/remplacer des motifs dans du texte


# Chaque motif est compilé une fois (plus rapide si appelé des milliers de fois).
_HTML_TAG = re.compile(r"<[^>]+>")          # tout ce qui ressemble à une balise : <br />, <b>, ...
_URL = re.compile(r"http\S+|www\.\S+")      # http... ou www... jusqu'au prochain espace
_NON_LETTRE = re.compile(r"[^a-z\s]")       # tout ce qui n'est PAS une lettre a-z ou un espace
_ESPACES = re.compile(r"\s+")               # une ou plusieurs espaces/tabs/retours ligne à la suite


def clean_text(text: str) -> str:
    """
    Nettoyage agressif pour le ML classique.
    Transforme un avis brut en texte propre en minuscules, mots séparés par 1 espace.

    Étapes (dans l'ordre, l'ordre compte !) :
      1. minuscules
      2. retirer les balises HTML (<br />)
      3. retirer les URLs
      4. retirer tout ce qui n'est pas une lettre (chiffres, ponctuation, emojis)
      5. réduire les espaces multiples à un seul, et enlever les espaces aux bords
    """
    text = text.lower()                     # 1. "Good" -> "good"
    text = _HTML_TAG.sub(" ", text)         # 2. "<br />" -> " "
    text = _URL.sub(" ", text)              # 3. "http://x" -> " "
    text = _NON_LETTRE.sub(" ", text)       # 4. "love!!! 😍 5/5" -> "love      "
    text = _ESPACES.sub(" ", text).strip()  # 5. espaces multiples -> un seul, + strip aux bords
    return text
