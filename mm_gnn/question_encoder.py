"""Question encoder: GloVe/FastText-style embeddings + LSTM + self-attention (paper Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, question_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Return global question vector [B, H].

        ``question_ids``: [B, T] token ids (0 = pad).
        """
        emb = self.embedding(question_ids)
        if lengths is None:
            lengths = (question_ids != 0).long().sum(dim=-1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # Self-attention pooling over time (Yu et al. MFH-style single-head summary)
        scores = self.attn(out).squeeze(-1)  # [B, T]
        mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (out * w).sum(dim=1)
