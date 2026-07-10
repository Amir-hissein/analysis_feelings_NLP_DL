"""
PHASE 2 — EDA (2/2) : VISUALISATION
====================================

But : TRANSFORMER les chiffres du script 01 en graphiques, puis générer
un petit rapport EDA (reports/EDA_report.md).

On produit 2 figures :
  1. reports/01_equilibre_classes.png   -> barres positive vs negative
  2. reports/02_distribution_longueur.png -> histogramme des longueurs

Lancement :
  ./venv/bin/python notebooks/02_eda_visualisation.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend "sans écran" : on sauvegarde en PNG, on n'ouvre pas de fenêtre
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import config

# Style global un peu plus joli que le défaut de matplotlib.
sns.set_theme(style="whitegrid")

# S'assurer que le dossier reports/ existe (il existe déjà, mais c'est une bonne habitude).
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Charger les données + recalculer la longueur (comme dans le script 01) ──
df = pd.read_csv(config.RAW_DATASET)
df["n_mots"] = df[config.TEXT_COL].str.split().str.len()

# ─────────────────────────────────────────────────────────────────
# FIGURE 1 : ÉQUILIBRE DES CLASSES
# ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
# countplot compte automatiquement les occurrences de chaque catégorie.
ax = sns.countplot(data=df, x=config.LABEL_COL, hue=config.LABEL_COL,
                   palette="Set2", legend=False)
ax.set_title("Équilibre des classes")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Nombre d'avis")
# Écrire la valeur exacte au-dessus de chaque barre.
for container in ax.containers:
    ax.bar_label(container, fmt="{:,.0f}")
plt.tight_layout()
fig1_path = config.REPORTS_DIR / "01_equilibre_classes.png"
plt.savefig(fig1_path, dpi=120)
plt.close()
print(f"✅ Figure 1 sauvegardée : {fig1_path}")

# ─────────────────────────────────────────────────────────────────
# FIGURE 2 : DISTRIBUTION DES LONGUEURS
# ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
# On coupe l'axe X à 1000 mots : au-delà, il y a trop peu d'avis, ça écrase le graphe.
# (les avis >1000 mots existent, mais on veut voir la MASSE des données)
ax = sns.histplot(df["n_mots"].clip(upper=1000), bins=60, color="#4C72B0")
ax.set_title("Distribution de la longueur des avis (en mots, tronqué à 1000)")
ax.set_xlabel("Nombre de mots par avis")
ax.set_ylabel("Nombre d'avis")

# Lignes verticales : médiane et moyenne, pour VOIR l'asymétrie.
mediane = df["n_mots"].median()
moyenne = df["n_mots"].mean()
ax.axvline(mediane, color="green", linestyle="--", label=f"Médiane = {mediane:.0f}")
ax.axvline(moyenne, color="red", linestyle="--", label=f"Moyenne = {moyenne:.0f}")
ax.legend()
plt.tight_layout()
fig2_path = config.REPORTS_DIR / "02_distribution_longueur.png"
plt.savefig(fig2_path, dpi=120)
plt.close()
print(f"✅ Figure 2 sauvegardée : {fig2_path}")

# ─────────────────────────────────────────────────────────────────
# RAPPORT MARKDOWN
# ─────────────────────────────────────────────────────────────────
# On assemble un petit rapport texte avec les conclusions de la Phase 2.
nb_doublons = df.duplicated(subset=[config.TEXT_COL, config.LABEL_COL]).sum()
rapport = f"""# Rapport EDA — Dataset IMDB (Phase 2)

## Vue d'ensemble
- **Observations** : {len(df):,} avis
- **Colonnes** : `{config.TEXT_COL}` (texte), `{config.LABEL_COL}` (label)
- **Valeurs manquantes** : {int(df.isnull().sum().sum())}
- **Doublons** : {nb_doublons:,} ({nb_doublons/len(df)*100:.2f}%) → à supprimer en Phase 3

## Équilibre des classes
{df[config.LABEL_COL].value_counts().to_string()}

→ Dataset **équilibré (50/50)** : l'accuracy sera une métrique fiable.

![Équilibre des classes](01_equilibre_classes.png)

## Longueur des textes (en mots)
{df['n_mots'].describe().round(1).to_string()}

→ Moyenne ({moyenne:.0f}) > médiane ({mediane:.0f}) : distribution **asymétrique à droite**.
→ Impact : les avis très longs devront être **tronqués** pour BERT (max 512 tokens).

![Distribution des longueurs](02_distribution_longueur.png)

## Décisions pour la suite
1. Supprimer les {nb_doublons} doublons (Phase 3).
2. Encoder les labels : positive → 1, negative → 0 (Phase 3).
3. Nettoyer le texte : HTML (`<br />`), URLs, ponctuation, casse (Phase 3).
"""
report_path = config.REPORTS_DIR / "EDA_report.md"
report_path.write_text(rapport, encoding="utf-8")
print(f"✅ Rapport sauvegardé : {report_path}")
print("\n✅ Phase 2 terminée. Ouvre les 2 PNG et le rapport dans reports/.")
