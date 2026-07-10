"""
PHASE 3 — NETTOYAGE DU DATASET
===============================

Applique les décisions prises en Phase 2 :
  1. Supprimer les doublons.
  2. Encoder les labels : positive -> 1, negative -> 0.
  3. Créer une colonne `review_clean` (texte nettoyé pour le ML classique),
     tout en GARDANT la colonne `review` originale (utile pour BERT).
  4. Sauvegarder le résultat dans data/processed/imdb_clean.csv

Lancement :
  ./venv/bin/python notebooks/03_nettoyage.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import config
from src.preprocessing.cleaning import clean_text


def titre(txt: str) -> None:
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ── Charger les données brutes ───────────────────────────────────
df = pd.read_csv(config.RAW_DATASET)
print(f"Chargé : {len(df):,} avis")

# ─────────────────────────────────────────────────────────────────
# 1. SUPPRIMER LES DOUBLONS
# ─────────────────────────────────────────────────────────────────
avant = len(df)
df = df.drop_duplicates().reset_index(drop=True)
# reset_index : après suppression, les numéros de ligne ont des trous -> on renumérote.
titre("1. SUPPRESSION DES DOUBLONS")
print(f"Avant : {avant:,}  |  Après : {len(df):,}  |  Retirés : {avant - len(df):,}")

# ─────────────────────────────────────────────────────────────────
# 2. ENCODER LES LABELS (texte -> nombre)
# ─────────────────────────────────────────────────────────────────
# .map() remplace chaque valeur selon un dictionnaire.
df["label"] = df[config.LABEL_COL].map({"negative": 0, "positive": 1})
titre("2. ENCODAGE DES LABELS")
print(df[[config.LABEL_COL, "label"]].value_counts())

# ─────────────────────────────────────────────────────────────────
# 3. NETTOYER LE TEXTE
# ─────────────────────────────────────────────────────────────────
# .apply(fonction) exécute la fonction sur CHAQUE avis de la colonne.
titre("3. NETTOYAGE DU TEXTE (aperçu avant/après)")
df["review_clean"] = df[config.TEXT_COL].apply(clean_text)

# Montrer un exemple concret avant/après.
exemple = df[config.TEXT_COL].iloc[1]
print("AVANT :")
print("  " + exemple[:200] + "...")
print("\nAPRÈS :")
print("  " + df["review_clean"].iloc[1][:200] + "...")

# ─────────────────────────────────────────────────────────────────
# 4. VÉRIFICATIONS DE SÉCURITÉ
# ─────────────────────────────────────────────────────────────────
titre("4. VÉRIFICATIONS")
# Des avis devenus vides après nettoyage ? (ex. un avis qui n'était que ponctuation)
vides = (df["review_clean"].str.len() == 0).sum()
print(f"Avis vides après nettoyage : {vides}")
# Des labels non encodés (NaN) ? (arriverait si une valeur inattendue existait)
print(f"Labels non encodés (NaN)    : {df['label'].isnull().sum()}")

# ─────────────────────────────────────────────────────────────────
# 5. SAUVEGARDER
# ─────────────────────────────────────────────────────────────────
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.PROCESSED_DIR / "imdb_clean.csv"
# On garde : texte original (pour BERT), texte nettoyé (pour ML), label numérique.
df[[config.TEXT_COL, "review_clean", config.LABEL_COL, "label"]].to_csv(out_path, index=False)
titre("5. SAUVEGARDE")
print(f"✅ Dataset nettoyé sauvegardé : {out_path}")
print(f"   {len(df):,} lignes, colonnes : {list(df[[config.TEXT_COL, 'review_clean', config.LABEL_COL, 'label']].columns)}")
