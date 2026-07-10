"""
PHASE 10 — DÉMO D'INFÉRENCE
============================

Teste la fonction predict_sentiment() sur des phrases écrites à la main.
C'est le moment où on "parle" enfin au modèle avec nos propres mots.

Lancement :
  ./venv/bin/python notebooks/10_inference_demo.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference.predict import predict_sentiment

# Des phrases variées : évidentes, nuancées, avec négation (le point faible du TF-IDF !).
EXEMPLES = [
    "This movie was absolutely fantastic, one of the best I have ever seen!",
    "A complete waste of time. Boring, poorly acted and predictable.",
    "It was okay, nothing special but not terrible either.",
    "The movie was not good at all.",                       # négation : cas difficile
    "I expected to hate it but it turned out to be brilliant.",  # retournement : cas difficile
    "Great cinematography, terrible story.",                 # sentiment mixte
]

print("=" * 70)
print("DÉMO — predict_sentiment()")
print("=" * 70)
for phrase in EXEMPLES:
    r = predict_sentiment(phrase)
    emoji = "✅" if r["label"] == "positif" else "❌"
    print(f"\n{emoji} {r['label'].upper():8} (confiance {r['confiance']:.0%})")
    print(f'   "{phrase}"')
