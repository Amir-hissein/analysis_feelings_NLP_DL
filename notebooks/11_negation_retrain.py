"""
PHASE 14 — RÉENTRAÎNEMENT AVEC MARQUAGE DE NÉGATION
===================================================

On refait le pipeline baseline, mais en préparant le texte avec preprocess_neg
(qui marque la négation : "not good" -> "neg_good"). Puis on COMPARE à la baseline
pour voir si gérer la négation améliore le modèle et corrige le bug "not good at all".

Le prétraitement spaCy sur 49k avis prend ~10-12 min (comme en Phase 4).

Lancement :
  ./venv/bin/python notebooks/11_negation_retrain.py
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config
from src.features.vectorizer import build_vectorizer
from src.models.classifier import build_model
from src.preprocessing.nlp_preprocessing import preprocess_neg_batch, preprocess_neg


def titre(txt):
    print("\n" + "=" * 70); print(txt); print("=" * 70)


# ── Données : on part du texte BRUT (review) car on a besoin de la ponctuation
#    (bornes de négation), que review_clean avait supprimée.
df = pd.read_csv(config.PROCESSED_DIR / "imdb_clean.csv")
print(f"Chargé : {len(df):,} avis")

titre("1. PRÉTRAITEMENT AVEC NÉGATION (patiente ~10-12 min)")
t0 = time.time()
df["review_neg"] = list(preprocess_neg_batch(df[config.TEXT_COL].astype(str)))
print(f"✅ Terminé en {(time.time() - t0) / 60:.1f} min")

# ── Même découpage que la baseline (comparaison juste) ──────────────
titre("2. DÉCOUPAGE + VECTORISATION + ENTRAÎNEMENT")
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    df["review_neg"], df["label"],
    test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE,
)
vectorizer = build_vectorizer()
X_train = vectorizer.fit_transform(X_train_txt)
X_test = vectorizer.transform(X_test_txt)
model = build_model()
model.fit(X_train, y_train)
print(f"Vocabulaire : {len(vectorizer.vocabulary_):,} termes "
      f"(dont les 'neg_...' appris comme des mots à part)")

# ── Comparaison des performances ────────────────────────────────────
titre("3. RÉSULTAT — négation vs baseline")
acc = model.score(X_test, y_test)
print(f"Accuracy TEST (avec négation) : {acc:.4f}  ({acc * 100:.1f} %)")
print(f"Accuracy TEST (baseline)      : 0.8858  (88.6 %)")
print(f"Différence : {(acc - 0.8858) * 100:+.2f} points")

# ── Le test qui compte : le bug "not good at all" est-il corrigé ? ──
titre("4. LE BUG EST-IL CORRIGÉ ?")
for phrase in ["The movie was not good at all.",
               "This is not a bad film.",
               "It was absolutely wonderful."]:
    vec = vectorizer.transform([preprocess_neg(phrase)])
    proba = model.predict_proba(vec)[0, 1]
    label = "positif" if proba >= 0.5 else "négatif"
    print(f"  {label:8} (p={proba:.2f})  \"{phrase}\"")

# ── Sauvegarde ──────────────────────────────────────────────────────
titre("5. SAUVEGARDE")
joblib.dump(vectorizer, config.VECTORIZER_NEG_PATH)
joblib.dump(model, config.LOGREG_NEG_PATH)
print(f"✅ {config.VECTORIZER_NEG_PATH.name} + {config.LOGREG_NEG_PATH.name}")
