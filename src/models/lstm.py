"""LSTM sentiment classifier.

word ids -> Embedding -> LSTM -> Linear -> logit (>0 positive, <0 negative).
"""
import torch.nn as nn


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x: (batch, seq_len) word ids; lengths: (batch,) true length per review.
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)  # right-padding is causal-safe: pads don't affect earlier steps
        # Take the output at each review's last real token (index length - 1).
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(2))
        summary = outputs.gather(1, idx.to(outputs.device)).squeeze(1)
        return self.fc(self.dropout(summary)).squeeze(1)
