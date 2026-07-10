"""
MODULE DE PRÉTRAITEMENT NLP (Phase 4)
======================================

Suite logique de `cleaning.py`. Après le nettoyage (Phase 3) le texte est propre
mais reste une grande chaîne de caractères. Ici on le transforme en une liste de
mots "utiles", ramenés à leur forme de base. C'est ce que le ML classique
(TF-IDF, Phase 5-6) sait exploiter.

Trois opérations, mais spaCy les fait TOUTES en une passe :
  1. TOKENISATION   : découper le texte en tokens (mots)
  2. STOPWORDS      : retirer les mots vides (the, is, a, of...) -> bruit inutile
  3. LEMMATISATION  : ramener chaque mot à sa forme de base (loved -> love)

⚠️  RAPPEL IMPORTANT : ce prétraitement agressif est POUR LE ML CLASSIQUE.
    Pour BERT (Phase 9) on N'UTILISERA PAS ce module : les Transformers ont besoin
    des mots entiers, des stopwords et de l'ordre de la phrase.
"""
import spacy

# ── Chargement du modèle spaCy (UNE seule fois) ──────────────────────────────
# 'en_core_web_sm' = petit modèle anglais (tokenizer + tagger + lemmatizer...).
# On DÉSACTIVE 'parser' et 'ner' : on n'a pas besoin de l'analyse syntaxique ni
# de la reconnaissance d'entités -> le traitement est bien plus RAPIDE.
# Le lemmatizer, lui, a besoin du 'tagger' (nature du mot) : on le garde.
_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess(text: str) -> str:
    """
    Transforme un texte nettoyé en une chaîne de LEMMES utiles, séparés par 1 espace.

    Exemple :
        "i really loved this amazing movie it was better than the others"
        -> "love amazing movie well"

    Règle de tri appliquée à chaque token :
      - token.is_alpha       -> on garde uniquement les vrais mots (pas chiffres/ponctuation)
      - not token.is_stop    -> on retire les stopwords (the, is, a...)
      - token.lemma_         -> on prend la forme de base du mot
    """
    doc = _NLP(text)
    lemmes = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(lemmes)


def preprocess_batch(textes, batch_size: int = 200):
    """
    Version RAPIDE pour traiter des milliers de textes d'un coup.

    Pourquoi ne pas juste faire une boucle `for` sur preprocess() ?
    -> `nlp.pipe()` traite les textes par LOTS (batches) et de façon optimisée.
       Sur 49 000 avis, c'est plusieurs fois plus rapide qu'un appel par avis.

    Renvoie un générateur : on l'utilise avec une boucle `for` ou `list(...)`.
    """
    for doc in _NLP.pipe(textes, batch_size=batch_size):
        lemmes = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        yield " ".join(lemmes)
