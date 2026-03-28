"""Full MM-GNN VQA model with extended answer space (vocabulary + OCR copy, Sec. 3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mm_gnn.attention import TopDownAttention
from mm_gnn.mm_gnn_core import MMGNNCore, MMGNNCoreWithNumeric
from mm_gnn.question_encoder import QuestionEncoder


class MMGNNVQA(nn.Module):
    """Multi-Modal Graph Neural Network for scene-text VQA (Gao et al., arXiv:2003.13962)."""

    def __init__(
        self,
        answer_vocab_size: int,
        question_vocab_size: int,
        ocr_vocab_size: int,
        visual_feature_dim: int = 2048,
        semantic_feature_dim: int = 300,
        question_embed_dim: int = 300,
        question_hidden_dim: int = 1024,
        hidden_dim: int = 512,
        box_dim: int = 512,
        attn_dim: int = 512,
        numeric_dim: int = 128,
        max_ocr_tokens: int = 50,
        question_layers: int = 1,
        topdown_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.answer_vocab_size = answer_vocab_size
        self.max_ocr_tokens = max_ocr_tokens
        self.num_classes = answer_vocab_size + max_ocr_tokens

        self.question_encoder = QuestionEncoder(
            question_vocab_size,
            question_embed_dim,
            question_hidden_dim,
            num_layers=question_layers,
        )
        self.ocr_embed = nn.Embedding(ocr_vocab_size, semantic_feature_dim, padding_idx=0)

        core = MMGNNCore(
            visual_feature_dim,
            semantic_feature_dim,
            hidden_dim,
            box_dim,
            question_hidden_dim,
            attn_dim,
            numeric_dim,
        )
        self.graph = MMGNNCoreWithNumeric(core)

        ocr_feat_dim = hidden_dim * 3 + core.numeric_aug_dim
        visual_att_dim = hidden_dim * 2
        self.att_v = TopDownAttention(visual_att_dim, question_hidden_dim, topdown_hidden)
        self.att_c = TopDownAttention(ocr_feat_dim, question_hidden_dim, topdown_hidden)

        fused = visual_att_dim + ocr_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused, fused),
            nn.ReLU(inplace=True),
            nn.Linear(fused, self.num_classes),
        )

    def forward(
        self,
        visual_feats: torch.Tensor,
        visual_boxes_xyxy: torch.Tensor,
        ocr_ids: torch.Tensor,
        ocr_boxes_xyxy: torch.Tensor,
        question_ids: torch.Tensor,
        question_lengths: torch.Tensor | None,
        ocr_tokens: list[list[str]],
        image_hw: tuple[int, int] | None = None,
        ocr_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits ``[B, vocab + max_ocr]`` (OCR slots align with first columns after vocab)."""
        q = self.question_encoder(question_ids, question_lengths)
        sem = self.ocr_embed(ocr_ids)
        v1, c = self.graph(
            visual_feats,
            visual_boxes_xyxy,
            sem,
            ocr_boxes_xyxy,
            q,
            ocr_tokens,
            image_hw,
            ocr_mask=ocr_mask,
            visual_mask=visual_mask,
        )
        v_tilde = self.att_v(v1, q)
        c_tilde = self.att_c(c, q)
        return self.classifier(torch.cat([v_tilde, c_tilde], dim=-1))
