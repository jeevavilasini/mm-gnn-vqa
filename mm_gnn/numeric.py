"""Numeric token encoding (Sec. 3.1 and supplementary Sec. 6)."""

from __future__ import annotations

import math
import re
from enum import IntEnum

import torch
import torch.nn as nn


class NumericCategory(IntEnum):
    NONE = 0
    MONOTONE = 1
    PERIODIC_TIME = 2


_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def classify_numeric_token(text: str) -> NumericCategory:
    t = text.strip().lower()
    if not t:
        return NumericCategory.NONE
    if _TIME_RE.match(t):
        return NumericCategory.PERIODIC_TIME
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return NumericCategory.MONOTONE
    if t in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}:
        return NumericCategory.PERIODIC_TIME
    return NumericCategory.NONE


def numeric_raw_features(text: str, device: torch.device | None = None) -> torch.Tensor:
    """3-D raw features fed through a linear layer to form x^(0) for numeric nodes."""
    cat = classify_numeric_token(text)
    if cat == NumericCategory.NONE:
        return torch.zeros(3, device=device)
    if cat == NumericCategory.PERIODIC_TIME and _TIME_RE.match(text.strip()):
        parts = text.strip().split(":")
        h = float(parts[0])
        m = float(parts[1])
        frac = (h + m / 60.0) / 24.0
        angle = 2.0 * math.pi * frac
        return torch.tensor([math.cos(angle), math.sin(angle), frac], device=device)
    if cat == NumericCategory.PERIODIC_TIME:
        return torch.tensor([1.0, 0.0, 0.5], device=device)
    try:
        v = float(text.strip())
    except ValueError:
        return torch.tensor([0.5, 0.0, 0.25], device=device)
    u = math.tanh(v / 1000.0)
    sig = 1.0 / (1.0 + math.exp(-(u * 20.0 - 10.0)))
    return torch.tensor([sig, u, u * u], device=device)


class NumericNodeBuilder(nn.Module):
    """Build numeric initial features x^(0) for OCR tokens (Sec. 3.1)."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(3, out_dim)

    def forward(self, ocr_tokens: list[list[str]], numeric_indices: list[list[int]]) -> torch.Tensor:
        device = self.proj.weight.device
        bsz = len(ocr_tokens)
        k_max = max((len(ix) for ix in numeric_indices), default=0)
        if k_max == 0:
            return torch.zeros(bsz, 0, self.out_dim, device=device)
        out = torch.zeros(bsz, k_max, self.out_dim, device=device)
        for b in range(bsz):
            for k, tok_idx in enumerate(numeric_indices[b]):
                tok = ocr_tokens[b][tok_idx]
                raw = numeric_raw_features(tok, device=device)
                out[b, k] = self.proj(raw)
        return out
