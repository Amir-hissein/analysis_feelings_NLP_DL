"""
PHASE 8 — ENTRAÎNEMENT D'UN RÉSEAU LSTM (Deep Learning, PyTorch)
================================================================

Premier vrai réseau de neurones. Contrairement au TF-IDF, il lit l'avis
MOT PAR MOT, DANS L'ORDRE, et apprend ses propres représentations (embeddings).

Points clés :
  - On utilise `review_clean` (ordre + négation préservés), PAS `review_nlp`.
  - Même découpage train/test que la baseline (random_state=42) -> comparaison juste.
  - GPU Apple Silicon (MPS) utilisé automatiquement si disponible.

Lancement :
  ./venv/bin/python notebooks/08_train_lstm.py
"""

import json
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import config
from src.data.text_dataset import IMDBDataset, build_vocab, PAD_IDX
from src.models.lstm import SentimentLSTM


def titre(txt: str) -> None:
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


# ── Choix du matériel : GPU Apple (MPS) sinon CPU ────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(config.RANDOM_STATE)
print(f"Matériel utilisé : {device}")

# ── Charger les données (colonne review_clean !) ─────────────────────
df = pd.read_csv(config.PROCESSED_DIR / "imdb_preprocessed.csv")
df["review_clean"] = df["review_clean"].fillna("")

# ─────────────────────────────────────────────────────────────────────
# 1. DÉCOUPAGE train/test (identique baseline) puis train/validation
# ─────────────────────────────────────────────────────────────────────
titre("1. DÉCOUPAGE DES DONNÉES")
# Même split que la baseline (mêmes y, random_state, stratify) -> mêmes avis en test.
X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    df["review_clean"], df["label"],
    test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE,
)
# Sous-découpe : une validation (10 % du train) pour surveiller l'apprentissage.
X_train, X_val, y_train, y_val = train_test_split(
    X_tr_full, y_tr_full,
    test_size=0.1, stratify=y_tr_full, random_state=config.RANDOM_STATE,
)
print(f"Train : {len(X_train):,}  |  Validation : {len(X_val):,}  |  Test : {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────
# 2. VOCABULAIRE (sur le TRAIN uniquement) + Datasets + DataLoaders
# ─────────────────────────────────────────────────────────────────────
titre("2. VOCABULAIRE ET DATALOADERS")
vocab = build_vocab(X_train, config.VOCAB_SIZE)
print(f"Taille du vocabulaire : {len(vocab):,} mots")

train_ds = IMDBDataset(X_train, y_train, vocab, config.MAX_LEN)
val_ds = IMDBDataset(X_val, y_val, vocab, config.MAX_LEN)
test_ds = IMDBDataset(X_test, y_test, vocab, config.MAX_LEN)

train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=config.BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE)

# ─────────────────────────────────────────────────────────────────────
# 3. MODÈLE, FONCTION DE PERTE, OPTIMISEUR
# ─────────────────────────────────────────────────────────────────────
titre("3. CONSTRUCTION DU RÉSEAU")
model = SentimentLSTM(len(vocab), config.EMBED_DIM, config.HIDDEN_DIM, pad_idx=PAD_IDX).to(device)
# BCEWithLogitsLoss = perte pour classification binaire, prend des logits bruts (stable).
criterion = nn.BCEWithLogitsLoss()
# Adam = optimiseur qui ajuste les poids ; lr = vitesse d'apprentissage.
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
nb_params = sum(p.numel() for p in model.parameters())
print(f"Nombre de paramètres à apprendre : {nb_params:,}")


def evaluer(dataloader) -> float:
    """Calcule l'accuracy sur un dataloader (mode évaluation, sans apprendre)."""
    model.eval()                                  # désactive le dropout
    bons, total = 0, 0
    with torch.no_grad():                         # pas de calcul de gradient -> plus rapide
        for X, lengths, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X, lengths)
            preds = (torch.sigmoid(logits) > 0.5).float()
            bons += (preds == y).sum().item()
            total += y.size(0)
    return bons / total


# ─────────────────────────────────────────────────────────────────────
# 4. BOUCLE D'ENTRAÎNEMENT (le cœur du Deep Learning)
# ─────────────────────────────────────────────────────────────────────
titre("4. ENTRAÎNEMENT")
for epoch in range(1, config.EPOCHS + 1):
    model.train()                                 # active le dropout
    perte_totale = 0.0
    t0 = time.time()
    for X, lengths, y in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()                     # 0. remettre les gradients à zéro
        logits = model(X, lengths)                # 1. FORWARD : prédiction
        loss = criterion(logits, y)               # 2. LOSS : mesure de l'erreur
        loss.backward()                           # 3. BACKWARD : calcul des gradients
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # gradient clipping (stabilise le LSTM)
        optimizer.step()                          # 4. STEP : mise à jour des poids
        perte_totale += loss.item()
    val_acc = evaluer(val_dl)
    print(f"Epoch {epoch}/{config.EPOCHS}  |  perte moy: {perte_totale / len(train_dl):.4f}  "
          f"|  val accuracy: {val_acc:.4f}  |  {time.time() - t0:.0f} s")

# ─────────────────────────────────────────────────────────────────────
# 5. ÉVALUATION FINALE SUR LE TEST + COMPARAISON BASELINE
# ─────────────────────────────────────────────────────────────────────
titre("5. RÉSULTAT FINAL")
test_acc = evaluer(test_dl)
print(f"Accuracy TEST (LSTM)     : {test_acc:.4f}  ({test_acc * 100:.1f} %)")
print(f"Accuracy TEST (baseline) : 0.8858  (88.6 %)")
diff = (test_acc - 0.8858) * 100
print(f"Différence : {diff:+.1f} points  vs la Régression Logistique")

# ─────────────────────────────────────────────────────────────────────
# 6. SAUVEGARDE (poids du modèle + vocabulaire)
# ─────────────────────────────────────────────────────────────────────
titre("6. SAUVEGARDE")
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), config.LSTM_MODEL_PATH)
with open(config.LSTM_VOCAB_PATH, "w") as f:
    json.dump(vocab, f)
print(f"✅ Poids du modèle : {config.LSTM_MODEL_PATH}")
print(f"✅ Vocabulaire     : {config.LSTM_VOCAB_PATH}")
print("\nProchaine étape : Phase 9 — Transformers (DistilBERT) ! 🤖")
