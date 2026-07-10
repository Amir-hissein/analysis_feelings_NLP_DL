from src.inference.predict import predict_sentiment


def test_predict_returns_expected_shape():
    r = predict_sentiment("This movie was fantastic.")
    assert {"label", "proba_positif", "confiance", "contributions"} <= set(r)
    assert r["label"] in {"positif", "négatif"}
    assert 0.0 <= r["proba_positif"] <= 1.0


def test_clear_positive_and_negative():
    assert predict_sentiment("An absolute masterpiece, brilliant and moving.")["label"] == "positif"
    assert predict_sentiment("A terrible, boring waste of time.")["label"] == "négatif"


def test_negation_is_handled():
    assert predict_sentiment("The movie was not good at all.")["label"] == "négatif"
