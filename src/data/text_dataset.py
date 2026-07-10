"""
OUTILS DE DONNÉES POUR LE LSTM (Phase 8)
=========================================

Un réseau ne lit pas des mots mais des ENTIERS. Ce module fait le pont :

  "this movie is great"
        │  tokenisation (découpe simple par espaces)
        ▼
  ["this", "movie", "is", "great"]
        │  vocabulaire (mot -> id)
        ▼
  [45, 12, 7, 231]
        │  padding/troncature à MAX_LEN
        ▼
  [45, 12, 7, 231, 0, 0, ... 0]   (0 = <pad>, pour que tous les avis aient la même longueur)

Deux tokens spéciaux :
  <pad> = 0  -> remplissage (avis plus courts que MAX_LEN)
  <unk> = 1  -> mot inconnu (absent du vocabulaire, ex. mot rare ou jamais vu)
"""
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD_TOKEN, PAD_IDX = "<pad>", 0
UNK_TOKEN, UNK_IDX = "<unk>", 1


def tokenize(texte: str) -> list[str]:
    """Tokenisation simple : découpe sur les espaces (le texte est déjà nettoyé)."""
    return texte.split()


def build_vocab(textes, vocab_size: int) -> dict[str, int]:
    """
    Construit le dictionnaire mot -> id à partir des textes d'ENTRAÎNEMENT uniquement.
    On garde les (vocab_size - 2) mots les plus fréquents (+ <pad> et <unk>).
    """
    compteur = Counter()
    for texte in textes:
        compteur.update(tokenize(texte))
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for mot, _ in compteur.most_common(vocab_size - 2):
        vocab[mot] = len(vocab)
    return vocab


def encode(texte: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """Transforme un texte en liste d'ids de longueur EXACTEMENT max_len (tronqué ou complété)."""
    ids = [vocab.get(mot, UNK_IDX) for mot in tokenize(texte)][:max_len]
    ids += [PAD_IDX] * (max_len - len(ids))   # padding à droite
    return ids


def vraie_longueur(texte: str, max_len: int) -> int:
    """Nombre de vrais tokens (hors padding), borné entre 1 et max_len."""
    n = len(tokenize(texte))
    return max(1, min(n, max_len))   # au moins 1 : un avis vide donne quand même 1 pas


class IMDBDataset(Dataset):
    """
    Objet Dataset PyTorch : donne accès aux (avis encodé, longueur réelle, label).
    La LONGUEUR sert à faire ignorer le padding par le LSTM (pack_padded_sequence).
    """
    def __init__(self, textes, labels, vocab: dict[str, int], max_len: int):
        self.X = [encode(t, vocab, max_len) for t in textes]
        self.lengths = [vraie_longueur(t, max_len) for t in textes]
        self.y = list(labels)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.lengths[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float32),
        )
