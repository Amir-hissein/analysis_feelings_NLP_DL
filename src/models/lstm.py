"""
MODÈLE LSTM POUR L'ANALYSE DE SENTIMENTS (Phase 8)
===================================================

Un réseau de neurones qui lit un avis mot par mot, dans l'ordre.

Architecture (le chemin d'un avis dans le réseau) :

  ids des mots        [12, 87, 5, 231, ...]        (un avis = une liste d'entiers)
        │
        ▼
  [ Embedding ]   chaque id -> vecteur de EMBED_DIM nombres (appris)
        │
        ▼
  [ LSTM ]        lit la séquence dans l'ordre, garde une MÉMOIRE
        │         -> produit un vecteur "résumé" de tout l'avis
        ▼
  [ Linear ]      le résumé -> 1 seul nombre (le "score" brut, appelé logit)
        │
        ▼
     logit        > 0 -> positif, < 0 -> négatif (la sigmoïde le convertit en proba)
"""
import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        # Embedding : table (vocab_size × embed_dim). padding_idx=0 -> le token <pad>
        # a un vecteur nul figé (il ne doit rien apporter au sens).
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # LSTM : lit la séquence d'embeddings. batch_first -> tenseurs (batch, longueur, dim).
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # Dropout : éteint aléatoirement 30 % des neurones à l'entraînement -> évite le surapprentissage.
        self.dropout = nn.Dropout(0.3)
        # Couche finale : du résumé (hidden_dim) vers 1 seul nombre (le logit).
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x : (batch, longueur) = ids de mots. lengths : (batch,) = vraie longueur de chaque avis.
        emb = self.embedding(x)                 # -> (batch, longueur, embed_dim)
        # On lit TOUTE la séquence. Le LSTM est causal (lit de gauche à droite) et le
        # padding est À DROITE : la sortie à chaque vrai mot n'est donc PAS polluée par
        # le padding qui suit. (Cette approche reste rapide sur le GPU Apple/MPS.)
        outputs, _ = self.lstm(emb)             # -> (batch, longueur, hidden_dim)
        # On récupère, pour chaque avis, la sortie au DERNIER VRAI mot (index longueur-1).
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(2))
        resume = outputs.gather(1, idx.to(outputs.device)).squeeze(1)  # -> (batch, hidden_dim)
        resume = self.dropout(resume)
        logit = self.fc(resume)                 # -> (batch, 1)
        return logit.squeeze(1)                 # -> (batch,)
