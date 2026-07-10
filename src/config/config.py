"""Central configuration: paths, column names and hyper-parameters.

Everything is defined here so the rest of the code has a single source of truth.
"""
from pathlib import Path

# Project root: config.py is at src/config/, so parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # raw data, never modified
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned data, ready for modeling
EXTERNAL_DIR = DATA_DIR / "external"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATASET = RAW_DIR / "IMDB Dataset.csv"

# Column names
TEXT_COL = "review"
LABEL_COL = "sentiment"

# Train/test split
TEST_SIZE = 0.2

# TF-IDF: keep the 5000 most frequent terms.
TFIDF_MAX_FEATURES = 5000

# Saved artifacts
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LOGREG_MODEL_PATH = MODELS_DIR / "logreg_model.joblib"
# Negation-aware variant (the deployed model)
VECTORIZER_NEG_PATH = MODELS_DIR / "tfidf_vectorizer_neg.joblib"
LOGREG_NEG_PATH = MODELS_DIR / "logreg_model_neg.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.pt"
LSTM_VOCAB_PATH = MODELS_DIR / "lstm_vocab.json"

# LSTM hyper-parameters
VOCAB_SIZE = 20000      # vocabulary size (most frequent words)
MAX_LEN = 250           # max review length in tokens (truncate/pad)
EMBED_DIM = 100         # embedding dimension
HIDDEN_DIM = 128        # LSTM hidden state size
BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 1e-3

# DistilBERT fine-tuning
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_DIR = MODELS_DIR / "distilbert_sentiment"
BERT_MAX_LEN = 256
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 2
BERT_LR = 2e-5
BERT_TRAIN_SUBSET = 8000  # training subset (speed/quality trade-off on CPU/MPS)

RANDOM_STATE = 42  # reproducibility
