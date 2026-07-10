"""TF-IDF vectorizer builder.

Score(term, doc) = TF (frequent in this doc) x IDF (rare across docs), so rare
discriminative words stand out while ubiquitous ones are down-weighted.
Fit on the training split only; transform both splits (no data leakage).
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import config


def build_vectorizer() -> TfidfVectorizer:
    """Return an unfitted TF-IDF vectorizer.

    - ngram_range=(1, 2): unigrams and bigrams (captures "not good" as one token)
    - min_df=5:           ignore terms appearing in fewer than 5 documents
    - sublinear_tf=True:  log-scale term frequencies
    """
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True,
    )
