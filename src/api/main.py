"""
API D'ANALYSE DE SENTIMENTS (Phase 11)
=======================================

Expose le modèle via le web. N'importe quelle application (notre futur frontend
React) pourra envoyer un avis et recevoir un sentiment.

La logique métier vit dans src/inference/predict.py : l'API ne fait que
"emballer" cette fonction dans des routes HTTP. Séparation des responsabilités.

Lancement du serveur :
  ./venv/bin/uvicorn src.api.main:app --reload

Puis ouvrir la doc interactive : http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.predict import predict_sentiment

# ── L'application ────────────────────────────────────────────────────
app = FastAPI(
    title="API Analyse de Sentiments",
    description="Prédit si un avis de film est positif ou négatif (modèle TF-IDF + Régression Logistique).",
    version="1.0.0",
)

# ── CORS : autoriser le frontend React à appeler l'API ───────────────
# Sans ça, le navigateur BLOQUE les appels de localhost:5173 (React) vers
# localhost:8000 (API) pour raisons de sécurité. On autorise explicitement
# l'origine du frontend de développement (Vite).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # frontend en dev (Vite)
        "http://localhost:8080", "http://127.0.0.1:8080",  # frontend en Docker (nginx)
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas d'entrée/sortie (Pydantic = validation automatique) ──────
class ReviewIn(BaseModel):
    """Ce que le client ENVOIE. Field(...) impose des contraintes validées."""
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Le texte de l'avis à analyser",
                      examples=["This movie was absolutely fantastic!"])


class Contribution(BaseModel):
    """Un mot et son influence sur la décision (+ = positif, - = négatif)."""
    mot: str
    contribution: float


class PredictionOut(BaseModel):
    """Ce que l'API RENVOIE. FastAPI documente ce format tout seul."""
    label: str = Field(description="'positif' ou 'négatif'")
    proba_positif: float = Field(description="Probabilité que l'avis soit positif (0 à 1)")
    confiance: float = Field(description="Confiance du modèle dans sa décision (0.5 à 1)")
    contributions: list[Contribution] = Field(
        default=[], description="Mots les plus décisifs et leur contribution +/-")


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/")
def racine():
    """Page d'accueil : vérifie que l'API tourne + oriente vers la doc."""
    return {"message": "API Analyse de Sentiments en ligne", "docs": "/docs"}


@app.get("/health")
def health():
    """Endpoint de 'santé' : utilisé par les outils de monitoring/déploiement."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(review: ReviewIn):
    """
    Analyse un avis et renvoie son sentiment.

    Pydantic a DÉJÀ validé que `review.text` est une chaîne non vide.
    On appelle simplement notre fonction métier et on renvoie le résultat.
    """
    return predict_sentiment(review.text)
