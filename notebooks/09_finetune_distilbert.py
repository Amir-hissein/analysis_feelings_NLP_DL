"""
PHASE 9 — FINE-TUNING DE DISTILBERT (Transformer pré-entraîné)
==============================================================

On prend DistilBERT (déjà entraîné sur des milliards de mots) et on l'AJUSTE
sur notre tâche (avis -> positif/négatif). C'est le transfer learning.

Différences clés avec le LSTM (Phase 8) :
  - On utilise le TEXTE BRUT `review` (le tokenizer de BERT gère tout).
  - Pas de vocabulaire à construire : DistilBERT a le sien (WordPiece).
  - Le modèle part de poids qui "connaissent" déjà l'anglais.
  - MAIS c'est lourd : on fine-tune sur un sous-échantillon (BERT_TRAIN_SUBSET)
    pour tenir en temps raisonnable sur Mac ; on ÉVALUE sur tout le test (comparaison juste).

Lancement :
  ./venv/bin/python notebooks/09_finetune_distilbert.py
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import config


def titre(txt: str) -> None:
    print("\n" + "=" * 70)
    print(txt)
    print("=" * 70)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(config.RANDOM_STATE)
print(f"Matériel utilisé : {device}")

# ── Données : on utilise la colonne `review` BRUTE ───────────────────
df = pd.read_csv(config.PROCESSED_DIR / "imdb_preprocessed.csv")
df[config.TEXT_COL] = df[config.TEXT_COL].fillna("")

# ─────────────────────────────────────────────────────────────────────
# 1. DÉCOUPAGE : même test que la baseline ; train sous-échantillonné
# ─────────────────────────────────────────────────────────────────────
titre("1. PRÉPARATION DES DONNÉES")
X_tr_full, X_test, y_tr_full, y_test = train_test_split(
    df[config.TEXT_COL], df["label"],
    test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE,
)
# Sous-échantillon d'entraînement (stratifié) : BERT apprend vite, pas besoin de tout.
X_train, _, y_train, _ = train_test_split(
    X_tr_full, y_tr_full, train_size=config.BERT_TRAIN_SUBSET,
    stratify=y_tr_full, random_state=config.RANDOM_STATE,
)
print(f"Train (sous-échantillon) : {len(X_train):,}  |  Test (complet) : {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────
# 2. TOKENIZER + MODÈLE PRÉ-ENTRAÎNÉS
# ─────────────────────────────────────────────────────────────────────
titre("2. CHARGEMENT DE DISTILBERT PRÉ-ENTRAÎNÉ")
tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
# num_labels=2 -> on ajoute une petite tête de classification (2 classes) au-dessus de BERT.
model = AutoModelForSequenceClassification.from_pretrained(
    config.BERT_MODEL_NAME, num_labels=2
).to(device)
print(f"Modèle : {config.BERT_MODEL_NAME}  ({sum(p.numel() for p in model.parameters()):,} paramètres)")


def make_loader(textes, labels, batch_size, shuffle):
    """Tokenise les textes (WordPiece) et fabrique un DataLoader."""
    enc = tokenizer(
        list(textes), truncation=True, max_length=config.BERT_MAX_LEN,
        padding="max_length", return_tensors="pt",
    )
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(list(labels), dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


train_dl = make_loader(X_train, y_train, config.BERT_BATCH_SIZE, shuffle=True)
test_dl = make_loader(X_test, y_test, config.BERT_BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────────────────────────
# 3. FINE-TUNING (même boucle qu'en Phase 8 : forward/loss/backward/step)
# ─────────────────────────────────────────────────────────────────────
titre("3. FINE-TUNING")
# AdamW = variante d'Adam recommandée pour les Transformers. LR minuscule (2e-5).
optimizer = torch.optim.AdamW(model.parameters(), lr=config.BERT_LR)


def evaluer(dataloader) -> float:
    model.eval()
    bons, total = 0, 0
    with torch.no_grad():
        for input_ids, mask, y in dataloader:
            input_ids, mask, y = input_ids.to(device), mask.to(device), y.to(device)
            logits = model(input_ids=input_ids, attention_mask=mask).logits
            preds = logits.argmax(dim=1)          # classe prédite = logit le plus élevé
            bons += (preds == y).sum().item()
            total += y.size(0)
    return bons / total


for epoch in range(1, config.BERT_EPOCHS + 1):
    model.train()
    perte_totale = 0.0
    t0 = time.time()
    for i, (input_ids, mask, y) in enumerate(train_dl, start=1):
        input_ids, mask, y = input_ids.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        # Si on passe labels=, le modèle calcule LUI-MÊME la perte (CrossEntropy).
        out = model(input_ids=input_ids, attention_mask=mask, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        perte_totale += out.loss.item()
        # Progression visible toutes les 50 étapes (pour ne pas rester aveugle).
        if i % 50 == 0:
            print(f"  epoch {epoch} - étape {i}/{len(train_dl)}  "
                  f"perte {perte_totale / i:.4f}  ({(time.time() - t0) / 60:.1f} min)")
    print(f"Epoch {epoch}/{config.BERT_EPOCHS}  |  perte moy: {perte_totale / len(train_dl):.4f}  "
          f"|  {(time.time() - t0) / 60:.1f} min")

# ─────────────────────────────────────────────────────────────────────
# 4. RÉSULTAT FINAL + COMPARAISON DES 3 APPROCHES
# ─────────────────────────────────────────────────────────────────────
titre("4. RÉSULTAT FINAL — LE GRAND DUEL")
test_acc = evaluer(test_dl)
print(f"  TF-IDF + Régression Logistique : 88.6 %")
print(f"  LSTM (from scratch)            : 84.6 %")
print(f"  DistilBERT (pré-entraîné)      : {test_acc * 100:.1f} %  <-- fine-tuné sur {len(X_train):,} avis")

# ─────────────────────────────────────────────────────────────────────
# 5. SAUVEGARDE (modèle + tokenizer, format Hugging Face)
# ─────────────────────────────────────────────────────────────────────
titre("5. SAUVEGARDE")
config.BERT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(config.BERT_DIR)
tokenizer.save_pretrained(config.BERT_DIR)
print(f"✅ Modèle + tokenizer sauvegardés : {config.BERT_DIR}")
print("\nProchaine étape : Phase 10 — sauvegarde/chargement propre du meilleur modèle.")
