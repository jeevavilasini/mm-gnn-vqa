"""Stack VS → SS → SN and build per-OCR token features ``c`` (Sec. 3.2–3.3)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from mm_gnn.aggregators import SemanticNumericAggregator, SemanticSemanticAggregator, VisualSemanticAggregator
from mm_gnn.geometry import encode_boxes_xyxy
from mm_gnn.numeric import NumericCategory, NumericNodeBuilder, classify_numeric_token


def _numeric_indices_from_tokens(ocr_tokens: Sequence[Sequence[str]]) -> list[list[int]]:
    return [[j for j, t in enumerate(row) if classify_numeric_token(t) != NumericCategory.NONE] for row in ocr_tokens]


class MMGNNCore(nn.Module):
    def __init__(
        self,
        visual_in_dim: int,
        semantic_in_dim: int,
        hidden_dim: int,
        box_dim: int,
        question_dim: int,
        attn_dim: int,
        numeric_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.numeric_dim = numeric_dim
        self.numeric_aug_dim = numeric_dim * 2

        self.proj_v = nn.Linear(visual_in_dim, hidden_dim)
        self.proj_s = nn.Linear(semantic_in_dim, hidden_dim)
        self.vs = VisualSemanticAggregator(hidden_dim, hidden_dim, box_dim, question_dim, attn_dim)
        self.ss = SemanticSemanticAggregator(hidden_dim * 2, box_dim, question_dim, attn_dim, hidden_dim)
        self.sn = SemanticNumericAggregator(hidden_dim * 3, numeric_dim, question_dim, attn_dim)
        self.numeric_builder = NumericNodeBuilder(numeric_dim)

    def forward(
        self,
        visual_feats: torch.Tensor,
        visual_boxes_xyxy: torch.Tensor,
        semantic_feats: torch.Tensor,
        semantic_boxes_xyxy: torch.Tensor,
        question_vec: torch.Tensor,
        ocr_tokens: list[list[str]],
        image_hw: tuple[int, int] | None,
        ocr_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``v1`` [B,N,2H], ``c`` [B,M,Hs+num_aug], ``s2`` [B,M,3H].

        Masks are boolean, True for padded/invalid positions to exclude from softmax.
        """
        _ = ocr_tokens
        v0 = self.proj_v(visual_feats)
        s0 = self.proj_s(semantic_feats)
        bv = encode_boxes_xyxy(visual_boxes_xyxy.reshape(-1, 4), image_hw).view(*visual_feats.shape[:2], -1)
        bs = encode_boxes_xyxy(semantic_boxes_xyxy.reshape(-1, 4), image_hw).view(*semantic_feats.shape[:2], -1)
        v_invalid = visual_mask if visual_mask is not None else None
        s_invalid = ocr_mask if ocr_mask is not None else None
        v_mask = ~v_invalid if v_invalid is not None else None
        s_mask = ~s_invalid if s_invalid is not None else None
        v1, s1 = self.vs(v0, s0, bv, bs, question_vec, v_mask=v_mask, s_mask=s_mask)
        s2 = self.ss(s1, bs, question_vec, s_mask=s_mask)
        return v1, s2, bs


class MMGNNCoreWithNumeric(nn.Module):
    """Runs numeric branch and stitches ``c`` for all OCR tokens (non-numeric gets zeros)."""

    def __init__(self, core: MMGNNCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        visual_feats: torch.Tensor,
        visual_boxes_xyxy: torch.Tensor,
        semantic_feats: torch.Tensor,
        semantic_boxes_xyxy: torch.Tensor,
        question_vec: torch.Tensor,
        ocr_tokens: list[list[str]],
        image_hw: tuple[int, int] | None,
        ocr_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v1, s2, _bs = self.core(
            visual_feats,
            visual_boxes_xyxy,
            semantic_feats,
            semantic_boxes_xyxy,
            question_vec,
            ocr_tokens,
            image_hw,
            ocr_mask=ocr_mask,
            visual_mask=visual_mask,
        )
        numeric_ix = _numeric_indices_from_tokens(ocr_tokens)
        x0 = self.core.numeric_builder(ocr_tokens, numeric_ix)
        if x0.size(1) > 0:
            x3 = self.core.sn(s2, x0, question_vec)
        else:
            x3 = torch.zeros(
                s2.size(0),
                0,
                self.core.numeric_aug_dim,
                device=s2.device,
                dtype=s2.dtype,
            )

        b, m, _ = s2.shape
        c = s2.new_zeros(b, m, s2.size(-1) + self.core.numeric_aug_dim)
        c[:, :, : s2.size(-1)] = s2
        for bi in range(b):
            for k, tok_i in enumerate(numeric_ix[bi]):
                if x3.size(1) == 0:
                    break
                c[bi, tok_i, s2.size(-1) :] = x3[bi, k]
        if ocr_mask is not None:
            c = c.masked_fill(ocr_mask.unsqueeze(-1), 0.0)
        return v1, c
