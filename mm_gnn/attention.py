"""Top-down attention (Anderson et al., CVPR 2018), used in answer prediction (Eq. 8)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TopDownAttention(nn.Module):
    def __init__(self, dim_feat: int, dim_q: int, hidden: int) -> None:
        super().__init__()
        self.lin_f = nn.Linear(dim_feat, hidden, bias=False)
        self.lin_q = nn.Linear(dim_q, hidden, bias=True)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, feats: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """``feats`` [B, N, D], ``q`` [B, dim_q] -> attended [B, D]."""
        b, n, _ = feats.shape
        h = torch.tanh(self.lin_f(feats) + self.lin_q(q).unsqueeze(1))
        a = self.w(h).squeeze(-1)
        w = torch.softmax(a, dim=-1).unsqueeze(-1)
        return (feats * w).sum(dim=1)
