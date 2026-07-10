"""
PHASE 4 — PRÉTRAITEMENT NLP (tokenisation + stopwords + lemmatisation)
======================================================================

Part de data/processed/imdb_clean.csv (sortie de la Phase 3) et ajoute une
colonne `review_nlp` : le texte réduit à ses LEMMES utiles (sans stopwords).

  review_clean : "i really loved this amazing movie it was better than others"
  review_nlp   : "love amazing movie well"   <-- ce que le ML classique adore

On garde toujours les colonnes précédentes. On n'écrase JAMAIS l'original.

Lancement :
  ./venv/bin/python notebooks/04_pretraitement_nlp.py
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import config
from src.preprocessing.nlp_preprocessing import preprocess, preprocess_batch


def titre(txt: str) -> None:
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ── Charger le dataset nettoyé (Phase 3) ─────────────────────────────
in_path = config.PROCESSED_DIR / "imdb_clean.csv"
df = pd.read_csv(in_path)
print(f"Chargé : {len(df):,} avis depuis {in_path.name}")

# ─────────────────────────────────────────────────────────────────────
# 1. DÉMO SUR UN SEUL EXEMPLE (pour bien VOIR ce que fait le prétraitement)
# ─────────────────────────────────────────────────────────────────────
titre("1. DÉMONSTRATION SUR UN EXEMPLE")
exemple = df["review_clean"].iloc[1]
print("AVANT (review_clean) :")
print("  " + exemple[:200] + "...")
print("\nAPRÈS (review_nlp) :")
print("  " + preprocess(exemple)[:200] + "...")

# ─────────────────────────────────────────────────────────────────────
# 2. APPLIQUER À TOUT LE DATASET
# ─────────────────────────────────────────────────────────────────────
# On utilise preprocess_batch (nlp.pipe) : bien plus rapide qu'un .apply()
# classique sur 49 000 avis. On chronomètre pour se rendre compte du coût réel.
titre("2. PRÉTRAITEMENT DE TOUT LE DATASET (patiente ~1-3 min)")
t0 = time.time()
df["review_nlp"] = list(preprocess_batch(df["review_clean"].astype(str)))
duree = time.time() - t0
print(f"✅ Terminé en {duree:.0f} s ({len(df) / duree:.0f} avis/seconde)")

# ─────────────────────────────────────────────────────────────────────
# 3. VÉRIFICATIONS + IMPACT DU PRÉTRAITEMENT
# ─────────────────────────────────────────────────────────────────────
titre("3. VÉRIFICATIONS")
vides = (df["review_nlp"].str.len() == 0).sum()
print(f"Avis devenus vides après prétraitement : {vides}")

# Combien de mots a-t-on retiré en moyenne ? (mesure l'effet stopwords + filtrage)
mots_avant = df["review_clean"].str.split().str.len().mean()
mots_apres = df["review_nlp"].str.split().str.len().mean()
print(f"Mots par avis  AVANT : {mots_avant:.0f}  |  APRÈS : {mots_apres:.0f} "
      f"(-{(1 - mots_apres / mots_avant) * 100:.0f} %)")

# ─────────────────────────────────────────────────────────────────────
# 4. SAUVEGARDER
# ─────────────────────────────────────────────────────────────────────
titre("4. SAUVEGARDE")
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
out_path = config.PROCESSED_DIR / "imdb_preprocessed.csv"
cols = [config.TEXT_COL, "review_clean", "review_nlp", config.LABEL_COL, "label"]
df[cols].to_csv(out_path, index=False)
print(f"✅ Dataset prétraité sauvegardé : {out_path}")
print(f"   {len(df):,} lignes, colonnes : {cols}")
