"""
PHASE 5 — FEATURE ENGINEERING (TF-IDF) + DÉCOUPAGE TRAIN/TEST
=============================================================

Transforme le texte prétraité (colonne `review_nlp`) en matrices de nombres
prêtes pour le ML classique (Phase 6).

Étapes :
  1. Découper le dataset en train (80 %) / test (20 %)  <-- AVANT toute vectorisation
  2. Entraîner le TF-IDF sur le TRAIN seulement (règle d'or anti-fuite de données)
  3. Transformer train ET test avec ce même vectoriseur
  4. Sauvegarder les matrices + les labels + le vectoriseur entraîné

Lancement :
  ./venv/bin/python notebooks/05_feature_engineering.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

from src.config import config
from src.features.vectorizer import build_vectorizer


def titre(txt: str) -> None:
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ── Charger le dataset prétraité (Phase 4) ───────────────────────────
in_path = config.PROCESSED_DIR / "imdb_preprocessed.csv"
df = pd.read_csv(in_path)
# Sécurité : de rares avis peuvent être vides -> on les remplace par "" (pas NaN).
df["review_nlp"] = df["review_nlp"].fillna("")
print(f"Chargé : {len(df):,} avis depuis {in_path.name}")

# ─────────────────────────────────────────────────────────────────────
# 1. DÉCOUPAGE TRAIN / TEST  (AVANT de vectoriser !)
# ─────────────────────────────────────────────────────────────────────
# X = le texte (ce qu'on donne au modèle) ; y = le label (ce qu'il doit prédire).
# stratify=y  -> garde le même équilibre 50/50 dans le train ET le test.
# random_state -> découpage reproductible (mêmes lignes à chaque exécution).
titre("1. DÉCOUPAGE TRAIN / TEST")
X = df["review_nlp"]
y = df["label"]
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    X, y,
    test_size=config.TEST_SIZE,
    stratify=y,
    random_state=config.RANDOM_STATE,
)
print(f"Train : {len(X_train_txt):,} avis  |  Test : {len(X_test_txt):,} avis")
print(f"Équilibre train (moyenne des labels) : {y_train.mean():.3f}  (0.5 = parfait)")
print(f"Équilibre test                       : {y_test.mean():.3f}")

# ─────────────────────────────────────────────────────────────────────
# 2. TF-IDF : fit sur le TRAIN seulement, transform sur les DEUX
# ─────────────────────────────────────────────────────────────────────
titre("2. VECTORISATION TF-IDF")
vectorizer = build_vectorizer()
# fit_transform = apprendre le vocabulaire + IDF SUR LE TRAIN, puis vectoriser le train.
X_train = vectorizer.fit_transform(X_train_txt)
# transform SEULEMENT : on réutilise le vocabulaire du train pour le test. On ne re-fit PAS.
X_test = vectorizer.transform(X_test_txt)

print(f"Vocabulaire appris : {len(vectorizer.vocabulary_):,} termes")
print(f"Matrice train : {X_train.shape}  (avis × termes)")
print(f"Matrice test  : {X_test.shape}")
# Les matrices sont "creuses" (sparse) : la plupart des cases sont 0 (un avis n'a
# qu'une poignée des 5000 mots). scipy ne stocke que les cases non nulles -> économe.
densite = X_train.nnz / (X_train.shape[0] * X_train.shape[1]) * 100
print(f"Densité (cases non nulles) : {densite:.2f} %  -> matrice très creuse, normal ✅")

# ─────────────────────────────────────────────────────────────────────
# 3. REGARDER CE QUE LE TF-IDF A COMPRIS (pédagogie)
# ─────────────────────────────────────────────────────────────────────
titre("3. INTERPRÉTATION : les mots les plus 'importants' d'un avis")
noms = vectorizer.get_feature_names_out()   # la liste des 5000 termes
i = 0                                        # on inspecte le 1er avis du train
ligne = X_train[i].toarray().ravel()         # ses 5000 scores TF-IDF
top = ligne.argsort()[::-1][:10]             # indices des 10 scores les plus élevés
print("Top 10 termes (score TF-IDF) du 1er avis d'entraînement :")
for idx in top:
    if ligne[idx] > 0:
        print(f"  {noms[idx]:20} {ligne[idx]:.3f}")

# ─────────────────────────────────────────────────────────────────────
# 4. SAUVEGARDER matrices, labels et vectoriseur
# ─────────────────────────────────────────────────────────────────────
titre("4. SAUVEGARDE")
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Matrices creuses -> format .npz de scipy. Labels -> .npy de numpy.
sparse.save_npz(config.PROCESSED_DIR / "X_train.npz", X_train)
sparse.save_npz(config.PROCESSED_DIR / "X_test.npz", X_test)
np.save(config.PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
np.save(config.PROCESSED_DIR / "y_test.npy", y_test.to_numpy())
# Vectoriseur ENTRAÎNÉ -> l'API l'utilisera pour vectoriser un nouvel avis à l'identique.
joblib.dump(vectorizer, config.VECTORIZER_PATH)

print(f"✅ X_train.npz / X_test.npz / y_train.npy / y_test.npy -> {config.PROCESSED_DIR}")
print(f"✅ Vectoriseur TF-IDF entraîné            -> {config.VECTORIZER_PATH}")
print("\nProchaine étape : Phase 6 — entraîner notre premier classifieur ! 🎯")
