"""Dataset utilities for the LSTM: vocabulary, encoding and a PyTorch Dataset.

Pipeline: text -> tokens -> word ids -> pad/truncate to max_len.
Special tokens: <pad>=0 (padding), <unk>=1 (out-of-vocabulary).
"""
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD_TOKEN, PAD_IDX = "<pad>", 0
UNK_TOKEN, UNK_IDX = "<unk>", 1


def tokenize(text: str) -> list[str]:
    return text.split()


def build_vocab(textes, vocab_size: int) -> dict[str, int]:
    """Build a word->id map from the training texts (most frequent terms)."""
    counter = Counter()
    for text in textes:
        counter.update(tokenize(text))
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """Encode text into exactly max_len ids (truncated or right-padded)."""
    ids = [vocab.get(w, UNK_IDX) for w in tokenize(text)][:max_len]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


def true_length(text: str, max_len: int) -> int:
    """Number of real (non-pad) tokens, clamped to [1, max_len]."""
    return max(1, min(len(tokenize(text)), max_len))


class IMDBDataset(Dataset):
    """Yields (encoded_review, true_length, label); the length lets the LSTM ignore padding."""

    def __init__(self, textes, labels, vocab: dict[str, int], max_len: int):
        self.X = [encode(t, vocab, max_len) for t in textes]
        self.lengths = [true_length(t, max_len) for t in textes]
        self.y = list(labels)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.lengths[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float32),
        )
