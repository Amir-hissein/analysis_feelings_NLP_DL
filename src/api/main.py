"""FastAPI application exposing the sentiment model.

Business logic lives in src/inference/predict.py; this module only maps it to
HTTP routes. Run with:  uvicorn src.api.main:app --reload  (docs at /docs).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import config
from src.inference.predict import predict_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts whether a movie review is positive or negative.",
    version="1.0.0",
)

# Allow the React frontend (dev and Docker origins) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",  # Vite dev server
        "http://localhost:8080", "http://127.0.0.1:8080",  # nginx (Docker)
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      examples=["This movie was absolutely fantastic!"])


class Contribution(BaseModel):
    mot: str
    contribution: float


class PredictionOut(BaseModel):
    label: str
    proba_positif: float
    confiance: float
    contributions: list[Contribution] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(review: ReviewIn):
    return predict_sentiment(review.text)


# In production (single-container deploy) the built frontend is copied to
# PROJECT_ROOT/static and served here. In dev this directory is absent, so we
# expose a small JSON landing page instead (the frontend runs on Vite).
_static = config.PROJECT_ROOT / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
else:
    @app.get("/")
    def root():
        return {"message": "Sentiment Analysis API is running", "docs": "/docs"}
