"""
Configuration centrale du projet.

Toute la config (chemins, noms de colonnes, paramètres) vit ICI.
Le reste du code importe depuis ce fichier => une seule source de vérité.
Si un jour tu déplaces le dataset ou renommes une colonne, tu ne changes QU'ICI.
"""
from pathlib import Path

# ── Racine du projet ─────────────────────────────────────────────
# __file__ = ce fichier (src/config/config.py)
# .parents[2] remonte de 2 niveaux : config/ -> src/ -> racine du projet
# Avantage : ça marche peu importe le dossier depuis lequel on lance le code.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Dossiers de données ──────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # données brutes, JAMAIS modifiées
PROCESSED_DIR = DATA_DIR / "processed"  # données nettoyées, prêtes à modéliser
EXTERNAL_DIR = DATA_DIR / "external"

# ── Dossiers de sortie ───────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Fichiers ─────────────────────────────────────────────────────
RAW_DATASET = RAW_DIR / "IMDB Dataset.csv"

# ── Noms de colonnes ─────────────────────────────────────────────
TEXT_COL = "review"      # colonne contenant le texte
LABEL_COL = "sentiment"  # colonne contenant le label (positive/negative)

# ── Découpage train/test (Phase 5) ───────────────────────────────
TEST_SIZE = 0.2  # 20 % des données réservées au test (jamais vues à l'entraînement)

# ── TF-IDF (Phase 5) ─────────────────────────────────────────────
# max_features : on garde les 5000 mots les plus fréquents du vocabulaire.
# Limite la taille des vecteurs (rapidité + évite le surapprentissage sur des mots rares).
TFIDF_MAX_FEATURES = 5000

# ── Artefacts sauvegardés (models/) ──────────────────────────────
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"  # le TF-IDF entraîné (réutilisé par l'API)
LOGREG_MODEL_PATH = MODELS_DIR / "logreg_model.joblib"    # le classifieur baseline (Phase 6)
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.pt"            # le réseau LSTM (Phase 8)
LSTM_VOCAB_PATH = MODELS_DIR / "lstm_vocab.json"          # le vocabulaire mot -> id du LSTM

# ── Hyperparamètres du LSTM (Phase 8) ────────────────────────────
VOCAB_SIZE = 20000      # nb de mots gardés dans le vocabulaire (les plus fréquents)
MAX_LEN = 250           # longueur max d'un avis en tokens (on tronque/complète à cette taille)
EMBED_DIM = 100         # dimension des embeddings (chaque mot -> vecteur de 100 nombres)
HIDDEN_DIM = 128        # taille de la mémoire interne du LSTM
BATCH_SIZE = 64         # nb d'avis traités ensemble à chaque pas
EPOCHS = 4              # nb de passages complets sur les données d'entraînement
LEARNING_RATE = 1e-3    # "vitesse" d'apprentissage de l'optimiseur

# ── DistilBERT / fine-tuning (Phase 9) ───────────────────────────
BERT_MODEL_NAME = "distilbert-base-uncased"   # le modèle pré-entraîné à ajuster
BERT_DIR = MODELS_DIR / "distilbert_sentiment"  # où on sauvegarde le modèle ajusté
BERT_MAX_LEN = 256      # longueur max en tokens : couvre la majeure partie des avis (vs 128 qui tronquait trop)
BERT_BATCH_SIZE = 16    # plus petit que le LSTM : BERT est bien plus gourmand en mémoire
BERT_EPOCHS = 2         # 2 passages suffisent : le modèle connaît déjà la langue
BERT_LR = 2e-5          # learning rate TRÈS petit : on ajuste finement, on ne casse pas le pré-entraînement
BERT_TRAIN_SUBSET = 8000  # sous-échantillon d'entraînement (compromis qualité/temps sur Mac)

# ── Divers ───────────────────────────────────────────────────────
RANDOM_STATE = 42  # graine aléatoire => résultats reproductibles
