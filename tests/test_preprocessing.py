from src.preprocessing.cleaning import clean_text
from src.preprocessing.nlp_preprocessing import preprocess_neg


def test_clean_text_removes_html_and_punctuation():
    assert clean_text("<br />Great MOVIE!!!") == "great movie"


def test_clean_text_removes_urls():
    assert "http" not in clean_text("check http://example.com now")


def test_negation_is_marked():
    assert "neg_good" in preprocess_neg("The movie was not good.").split()


def test_negation_scope_stops_at_contrast():
    tokens = preprocess_neg("not boring but truly great").split()
    assert "neg_boring" in tokens
    assert "great" in tokens  # after "but", not negated
