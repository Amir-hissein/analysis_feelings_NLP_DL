"""
PHASE 7 — ÉVALUATION DÉTAILLÉE DU MODÈLE BASELINE
==================================================

L'accuracy (88,6 %) est une moyenne : elle cache la structure des erreurs.
Ici on regarde en détail COMMENT le modèle se trompe.

Produit :
  - Un rapport texte (precision / recall / F1 par classe)
  - reports/03_confusion_matrix.png
  - reports/04_roc_curve.png
  - Des exemples concrets d'avis MAL classés (pour comprendre les erreurs)

Lancement :
  ./venv/bin/python notebooks/07_evaluation.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.config import config

# Couleurs (teinte unique séquentielle pour la heatmap, accent pour la ROC).
BLEU = "#2563eb"
GRIS = "#9ca3af"
ENCRE = "#1f2937"


def titre(txt: str) -> None:
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ── Charger modèle, matrices et labels ───────────────────────────────
model = joblib.load(config.LOGREG_MODEL_PATH)
X_test = sparse.load_npz(config.PROCESSED_DIR / "X_test.npz")
y_test = np.load(config.PROCESSED_DIR / "y_test.npy")

# Prédictions : la CLASSE (0/1) et la PROBABILITÉ d'être positif (pour la ROC).
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]   # colonne 1 = proba de la classe "positif"

# ─────────────────────────────────────────────────────────────────────
# 1. RAPPORT DE CLASSIFICATION (precision / recall / F1)
# ─────────────────────────────────────────────────────────────────────
titre("1. RAPPORT DE CLASSIFICATION")
print(classification_report(y_test, y_pred, target_names=["négatif (0)", "positif (1)"]))
auc = roc_auc_score(y_test, y_proba)
print(f"AUC (aire sous la courbe ROC) : {auc:.4f}  (1.0 = parfait, 0.5 = hasard)")

# ─────────────────────────────────────────────────────────────────────
# 2. MATRICE DE CONFUSION (figure)
# ─────────────────────────────────────────────────────────────────────
titre("2. MATRICE DE CONFUSION")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Vrais Négatifs : {tn:>5}   |   Faux Positifs : {fp:>5}")
print(f"Faux Négatifs  : {fn:>5}   |   Vrais Positifs: {tp:>5}")

fig, ax = plt.subplots(figsize=(5.5, 4.8))
im = ax.imshow(cm, cmap="Blues")   # séquentielle : une teinte, clair -> foncé
labels = ["négatif", "positif"]
ax.set_xticks([0, 1], labels=labels)
ax.set_yticks([0, 1], labels=labels)
ax.set_xlabel("Prédit par le modèle", color=ENCRE)
ax.set_ylabel("Vraie classe", color=ENCRE)
ax.set_title("Matrice de confusion — Régression Logistique", color=ENCRE, pad=12)
# Écrire le compte dans chaque case (label direct), couleur lisible selon le fond.
seuil = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > seuil else ENCRE, fontsize=14, fontweight="bold")
fig.tight_layout()
cm_path = config.REPORTS_DIR / "03_confusion_matrix.png"
fig.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Figure sauvegardée : {cm_path}")

# ─────────────────────────────────────────────────────────────────────
# 3. COURBE ROC (figure)
# ─────────────────────────────────────────────────────────────────────
titre("3. COURBE ROC")
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(5.5, 4.8))
ax.plot(fpr, tpr, color=BLEU, linewidth=2, label=f"Modèle (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=GRIS, linewidth=1.5, linestyle="--", label="Hasard (AUC = 0.5)")
ax.set_xlabel("Taux de faux positifs", color=ENCRE)
ax.set_ylabel("Taux de vrais positifs", color=ENCRE)
ax.set_title("Courbe ROC — Régression Logistique", color=ENCRE, pad=12)
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
fig.tight_layout()
roc_path = config.REPORTS_DIR / "04_roc_curve.png"
fig.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✅ Figure sauvegardée : {roc_path}")

# ─────────────────────────────────────────────────────────────────────
# 4. EXEMPLES MAL CLASSÉS (le plus instructif !)
# ─────────────────────────────────────────────────────────────────────
# On refait le MÊME découpage (même random_state) pour récupérer les TEXTES
# du test, alignés avec y_test. Ça permet de LIRE les avis mal classés.
titre("4. EXEMPLES MAL CLASSÉS (pourquoi le modèle se trompe)")
df = pd.read_csv(config.PROCESSED_DIR / "imdb_preprocessed.csv")
df["review_nlp"] = df["review_nlp"].fillna("")
_, X_test_txt, _, _ = train_test_split(
    df[config.TEXT_COL], df["label"],
    test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE,
)
X_test_txt = X_test_txt.reset_index(drop=True)

erreurs = np.where(y_pred != y_test)[0]
print(f"Total d'erreurs sur le test : {len(erreurs)} / {len(y_test)}\n")

# 2 faux positifs (vrai=négatif, prédit=positif) et 2 faux négatifs.
faux_positifs = [i for i in erreurs if y_test[i] == 0][:2]
faux_negatifs = [i for i in erreurs if y_test[i] == 1][:2]

for i in faux_positifs:
    print(f"[FAUX POSITIF] vrai=négatif, prédit=positif (proba {y_proba[i]:.2f})")
    print("  " + X_test_txt.iloc[i][:220].replace("\n", " ") + "...\n")
for i in faux_negatifs:
    print(f"[FAUX NÉGATIF] vrai=positif, prédit=négatif (proba {y_proba[i]:.2f})")
    print("  " + X_test_txt.iloc[i][:220].replace("\n", " ") + "...\n")

print("Prochaine étape : Phase 8 — Deep Learning avec PyTorch ! 🔥")
