"""Multi-modal graph aggregators: VS, SS, SN (Sec. 3.2)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _bilinear_scores(a: torch.Tensor, b: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``a`` [B,M,H], ``b`` [B,N,H], ``q`` [B,H] -> scores [B,M,N], score = sum_h a[m]*(b[n]*q)."""
    scaled = b * q.unsqueeze(1)  # [B,N,H]
    return torch.einsum("bmh,bnh->bmn", a, scaled)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualSemanticAggregator(nn.Module):
    """Eqs. (1)-(4): refine semantic nodes with visual context and vice versa."""

    def __init__(self, dim_v: int, dim_s: int, dim_box: int, dim_q: int, attn_dim: int) -> None:
        super().__init__()
        self.fs = MLP(dim_s + dim_box, attn_dim, attn_dim)
        self.fv = MLP(dim_v + dim_box, attn_dim, attn_dim)
        self.fq = MLP(dim_q, attn_dim, attn_dim)
        self.fb = MLP(10, dim_box, dim_box)
        self.fv_prime = MLP(dim_v, attn_dim, dim_s)
        self.fs_prime = MLP(dim_s, attn_dim, dim_v)

    def forward(
        self,
        v0: torch.Tensor,
        s0: torch.Tensor,
        boxes_v: torch.Tensor,
        boxes_s: torch.Tensor,
        q: torch.Tensor,
        v_mask: torch.Tensor | None = None,
        s_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = v0.shape
        _, m, _ = s0.shape
        bv = self.fb(boxes_v)
        bs = self.fb(boxes_s)
        fs_i = self.fs(torch.cat([s0, bs], dim=-1))
        fv_j = self.fv(torch.cat([v0, bv], dim=-1))
        fq = self.fq(q)
        logits = _bilinear_scores(fs_i, fv_j, fq)
        if v_mask is not None:
            logits = logits.masked_fill(~v_mask.unsqueeze(1), float("-inf"))
        alpha = torch.softmax(logits, dim=-1)  # semantic i over visual j
        agg_s = torch.einsum("bmn,bnf->bmf", alpha, self.fv_prime(v0))
        s1 = torch.cat([s0, agg_s], dim=-1)
        logits_t = logits.transpose(1, 2)  # [B, N, M]
        if s_mask is not None:
            logits_t = logits_t.masked_fill(~s_mask.unsqueeze(1), float("-inf"))
        beta = torch.softmax(logits_t, dim=-1)
        agg_v = torch.einsum("bnm,bms->bns", beta, self.fs_prime(s0))
        v1 = torch.cat([v0, agg_v], dim=-1)
        return v1, s1


class SemanticSemanticAggregator(nn.Module):
    """Eqs. (5)-(6): semantic context between OCR tokens."""

    def __init__(
        self, dim_s1: int, dim_box: int, dim_q: int, attn_dim: int, base_semantic_dim: int
    ) -> None:
        super().__init__()
        d_in = dim_s1 + dim_box
        self.gs1 = MLP(d_in, attn_dim, attn_dim)
        self.gs2 = MLP(d_in, attn_dim, attn_dim)
        self.gq = MLP(dim_q, attn_dim, attn_dim)
        self.gb = MLP(10, dim_box, dim_box)
        self.gs3 = MLP(dim_s1, attn_dim, base_semantic_dim)

    def forward(
        self,
        s1: torch.Tensor,
        boxes_s: torch.Tensor,
        q: torch.Tensor,
        s_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs = self.gb(boxes_s)
        lhs = self.gs1(torch.cat([s1, bs], dim=-1))
        rhs = self.gs2(torch.cat([s1, bs], dim=-1))
        gq = self.gq(q)
        logits = torch.einsum("bmh,bnh->bmn", lhs, rhs * gq.unsqueeze(1))
        eye = torch.eye(logits.size(-1), device=logits.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(eye, float("-inf"))
        if s_mask is not None:
            logits = logits.masked_fill(~s_mask.unsqueeze(1), float("-inf"))
            logits = logits.masked_fill(~s_mask.unsqueeze(2), float("-inf"))
        att = torch.softmax(logits, dim=-1)
        agg = torch.einsum("bmn,bnf->bmf", att, self.gs3(s1))
        return torch.cat([s1, agg], dim=-1)


class SemanticNumericAggregator(nn.Module):
    """Eq. (7): refine numeric nodes from semantic context."""

    def __init__(self, dim_s: int, dim_x: int, dim_q: int, attn_dim: int) -> None:
        super().__init__()
        self.h = MLP(dim_s, attn_dim, dim_x)
        self.score = MLP(dim_s + dim_x + dim_q, attn_dim, 1)

    def forward(self, s2: torch.Tensor, x0: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if x0.size(1) == 0:
            return x0
        b, m, ds = s2.shape
        _, k, dx = x0.shape
        s_exp = s2.unsqueeze(2).expand(b, m, k, ds)
        x_exp = x0.unsqueeze(1).expand(b, m, k, dx)
        q_exp = q.view(b, 1, 1, -1).expand(b, m, k, q.size(-1))
        logits = self.score(torch.cat([s_exp, x_exp, q_exp], dim=-1)).squeeze(-1)
        att = torch.softmax(logits, dim=1)
        h_s = self.h(s2)
        agg = torch.einsum("bmk,bmf->bkf", att, h_s)
        return torch.cat([x0, agg], dim=-1)
