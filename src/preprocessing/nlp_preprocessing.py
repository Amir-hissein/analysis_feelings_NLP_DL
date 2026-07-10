"""NLP preprocessing with spaCy: tokenization, stop-word removal, lemmatization.

`preprocess`      — standard preprocessing for the classical ML pipeline.
`preprocess_neg`  — negation-aware variant used by the deployed model.
"""
import re

import spacy

# Load spaCy once. Disable parser/NER (unused) for speed; keep the tagger,
# which the lemmatizer depends on.
_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess(text: str) -> str:
    """Return useful lemmas (alphabetic, non stop-word), space-separated.

    "I really loved this amazing movie" -> "love amazing movie"
    """
    doc = _NLP(text)
    lemmas = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
    return " ".join(lemmas)


def preprocess_batch(textes, batch_size: int = 200):
    """Fast batched version of `preprocess` (uses nlp.pipe). Yields strings."""
    for doc in _NLP.pipe(textes, batch_size=batch_size):
        lemmas = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
        yield " ".join(lemmas)


# ── Negation-aware preprocessing ──────────────────────────────────────────────
# Standard preprocessing drops "not" (a stop word), so "not good" -> "good".
# Here we prefix words following a negation with "neg_" until the next punctuation
# or contrastive conjunction, turning "not good" into the distinct token "neg_good".

_HTML = re.compile(r"<[^>]+>")
_NEG_TRIGGERS = {"not", "no", "never", "without", "nor", "cannot", "neither"}  # "n't" lemmatizes to "not"
_NEG_STOPS = {"but", "however", "although", "though", "yet"}


def _negation_tokens(doc) -> list[str]:
    tokens = []
    negate = False
    for token in doc:
        if token.is_punct or token.lemma_ in _NEG_STOPS:
            negate = False
            if token.is_punct:
                continue
        if token.lemma_ in _NEG_TRIGGERS:
            negate = True
            continue
        if not token.is_alpha or token.is_stop:
            continue
        tokens.append("neg_" + token.lemma_ if negate else token.lemma_)
    return tokens


def preprocess_neg(text: str) -> str:
    """Like `preprocess`, but marks negation ("not good" -> "neg_good")."""
    text = _HTML.sub(" ", text.lower())  # keep punctuation (negation boundaries)
    return " ".join(_negation_tokens(_NLP(text)))


def preprocess_neg_batch(textes, batch_size: int = 200):
    """Fast batched version of `preprocess_neg`."""
    textes = (_HTML.sub(" ", str(t).lower()) for t in textes)
    for doc in _NLP.pipe(textes, batch_size=batch_size):
        yield " ".join(_negation_tokens(doc))
