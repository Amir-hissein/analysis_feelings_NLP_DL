# ─────────────────────────────────────────────────────────────
# SINGLE-CONTAINER IMAGE for cloud deployment (Render, HF Spaces…)
# FastAPI serves both the API and the compiled React frontend.
# For local multi-container dev, use docker-compose.yml instead.
# ─────────────────────────────────────────────────────────────

# Stage 1 — build the React frontend
FROM node:22-alpine AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # -> /fe/dist

# Stage 2 — API that also serves the built frontend
FROM python:3.13-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && python -m spacy download en_core_web_sm
COPY src/ ./src/
COPY models/tfidf_vectorizer_neg.joblib models/logreg_model_neg.joblib ./models/
COPY --from=frontend /fe/dist ./static

# Hosting platforms provide the port via $PORT; default to 8000 locally.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
