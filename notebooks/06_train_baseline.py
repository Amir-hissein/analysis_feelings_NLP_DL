"""
PHASE 6 — ENTRAÎNEMENT DU MODÈLE BASELINE (Régression Logistique)
==================================================================

Charge les matrices TF-IDF (Phase 5), entraîne une Régression Logistique,
mesure sa performance sur le test (données jamais vues), inspecte les mots
qu'il a appris, et sauvegarde le modèle.

Lancement :
  ./venv/bin/python notebooks/06_train_baseline.py
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from scipy import sparse

from src.config import config
from src.models.classifier import build_model


def titre(txt: str) -> None:
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ── Charger les matrices et labels (Phase 5) ─────────────────────────
X_train = sparse.load_npz(config.PROCESSED_DIR / "X_train.npz")
X_test = sparse.load_npz(config.PROCESSED_DIR / "X_test.npz")
y_train = np.load(config.PROCESSED_DIR / "y_train.npy")
y_test = np.load(config.PROCESSED_DIR / "y_test.npy")
print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────
# 1. ENTRAÎNEMENT  (le fameux .fit())
# ─────────────────────────────────────────────────────────────────────
titre("1. ENTRAÎNEMENT DE LA RÉGRESSION LOGISTIQUE")
model = build_model()
t0 = time.time()
model.fit(X_train, y_train)   # <-- trouve les 5000 poids qui séparent + de -
print(f"✅ Entraîné en {time.time() - t0:.1f} s")

# ─────────────────────────────────────────────────────────────────────
# 2. ÉVALUATION SUR LE TRAIN ET LE TEST
# ─────────────────────────────────────────────────────────────────────
# On compare les deux : si train >> test, le modèle SURAPPREND (overfitting).
titre("2. PERFORMANCE (accuracy = % d'avis bien classés)")
acc_train = model.score(X_train, y_train)
acc_test = model.score(X_test, y_test)
print(f"Accuracy TRAIN : {acc_train:.4f}  ({acc_train * 100:.1f} %)")
print(f"Accuracy TEST  : {acc_test:.4f}  ({acc_test * 100:.1f} %)  <-- la VRAIE mesure")
print(f"Écart train-test : {(acc_train - acc_test) * 100:.1f} pts "
      f"(petit = pas de surapprentissage ✅)")

# ─────────────────────────────────────────────────────────────────────
# 3. INTERPRÉTATION : quels mots le modèle associe au + et au - ?
# ─────────────────────────────────────────────────────────────────────
# Les "poids" appris sont dans model.coef_. On charge les noms des mots
# depuis le vectoriseur sauvegardé pour associer chaque poids à son mot.
titre("3. CE QUE LE MODÈLE A APPRIS (mots les plus décisifs)")
vectorizer = joblib.load(config.VECTORIZER_PATH)
noms = vectorizer.get_feature_names_out()
poids = model.coef_.ravel()

ordre = poids.argsort()          # du plus négatif au plus positif
print("Top 12 mots POSITIFS (poids les plus élevés) :")
for idx in ordre[::-1][:12]:
    print(f"  {noms[idx]:20} {poids[idx]:+.2f}")
print("\nTop 12 mots NÉGATIFS (poids les plus bas) :")
for idx in ordre[:12]:
    print(f"  {noms[idx]:20} {poids[idx]:+.2f}")

# ─────────────────────────────────────────────────────────────────────
# 4. SAUVEGARDER LE MODÈLE
# ─────────────────────────────────────────────────────────────────────
titre("4. SAUVEGARDE")
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, config.LOGREG_MODEL_PATH)
print(f"✅ Modèle sauvegardé : {config.LOGREG_MODEL_PATH}")
print("\nProchaine étape : Phase 7 — évaluation détaillée (matrice de confusion, etc.)")
