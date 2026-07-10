"""
PHASE 2 — EDA (1/2) : DIAGNOSTIC DU DATASET
============================================

But : répondre AVEC DES CHIFFRES aux questions fondamentales avant de coder
un seul modèle. On ne visualise pas encore (c'est le script 02), on DIAGNOSTIQUE.

Questions auxquelles ce script répond :
  1. Combien de lignes / colonnes ?
  2. À quoi ressemblent les données (un aperçu) ?
  3. Y a-t-il des valeurs manquantes (NaN) ?
  4. Y a-t-il des doublons ?
  5. Les classes positive/negative sont-elles équilibrées ?
  6. Les textes sont-ils longs ? (statistiques de longueur)

Lancement :
  ./venv/bin/python notebooks/01_eda_diagnostic.py
"""

# ── Rendre le package `src` importable ───────────────────────────
# Ce script est dans notebooks/, mais on veut importer src/config/config.py.
# On ajoute donc la racine du projet au chemin de recherche de Python.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import config

# Options d'affichage : ne pas tronquer les colonnes, afficher plus large.
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 120)


def titre(txt: str) -> None:
    """Petit utilitaire pour afficher des titres lisibles dans le terminal."""
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────
# 1. CHARGER LES DONNÉES
# ─────────────────────────────────────────────────────────────────
# pd.read_csv lit le fichier CSV et le met dans un "DataFrame" :
# un tableau 2D (lignes x colonnes), la structure de base de Pandas.
df = pd.read_csv(config.RAW_DATASET)

titre("1. TAILLE DU DATASET")
# df.shape = (nombre_de_lignes, nombre_de_colonnes)
print(f"Lignes   : {df.shape[0]:,}")
print(f"Colonnes : {df.shape[1]}  ->  {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────
# 2. APERÇU DES DONNÉES
# ─────────────────────────────────────────────────────────────────
titre("2. APERÇU (5 premières lignes)")
print(df.head())

titre("2b. TYPES ET INFOS TECHNIQUES")
# df.info() donne le type de chaque colonne et l'usage mémoire.
df.info()

# ─────────────────────────────────────────────────────────────────
# 3. VALEURS MANQUANTES
# ─────────────────────────────────────────────────────────────────
# df.isnull() -> tableau de True/False. .sum() compte les True par colonne.
titre("3. VALEURS MANQUANTES (NaN) par colonne")
print(df.isnull().sum())

# ─────────────────────────────────────────────────────────────────
# 4. DOUBLONS
# ─────────────────────────────────────────────────────────────────
# .duplicated() marque True les lignes déjà vues. .sum() les compte.
nb_doublons = df.duplicated().sum()
titre("4. DOUBLONS")
print(f"Lignes strictement identiques : {nb_doublons:,}")
print(f"Soit {nb_doublons / len(df) * 100:.2f}% du dataset")

# ─────────────────────────────────────────────────────────────────
# 5. ÉQUILIBRE DES CLASSES
# ─────────────────────────────────────────────────────────────────
# .value_counts() compte combien de fois chaque valeur apparaît.
titre("5. ÉQUILIBRE DES CLASSES (colonne 'sentiment')")
counts = df[config.LABEL_COL].value_counts()
percents = df[config.LABEL_COL].value_counts(normalize=True) * 100
for label in counts.index:
    print(f"  {label:<10} : {counts[label]:>7,}  ({percents[label]:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# 6. LONGUEUR DES TEXTES
# ─────────────────────────────────────────────────────────────────
# On crée une colonne temporaire = nombre de MOTS par avis.
# .str.split() découpe sur les espaces, .str.len() compte les morceaux.
titre("6. LONGUEUR DES TEXTES (en nombre de mots)")
longueurs = df[config.TEXT_COL].str.split().str.len()
print(longueurs.describe().round(1))  # count, mean, std, min, quartiles, max

print("\n" + "-" * 70)
print("✅ Diagnostic terminé. Analyse les chiffres ci-dessus,")
print("   puis on passe au script 02 (visualisation).")
print("-" * 70)
