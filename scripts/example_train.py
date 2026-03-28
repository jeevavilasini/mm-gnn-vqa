"""Minimal training loop on synthetic data to verify loss.backward() works."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mm_gnn import MMGNNVQA


def main() -> None:
    torch.manual_seed(0)
    B, N, M = 4, 10, 16
    vocab = 50
    device = torch.device("cpu")
    model = MMGNNVQA(
        answer_vocab_size=vocab,
        question_vocab_size=300,
        ocr_vocab_size=400,
        max_ocr_tokens=M,
        hidden_dim=128,
        box_dim=128,
        attn_dim=128,
        numeric_dim=32,
        question_embed_dim=128,
        question_hidden_dim=256,
        topdown_hidden=128,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(3):
        vf = torch.randn(B, N, 2048, device=device)
        vb = torch.rand(B, N, 4, device=device) * 640
        oi = torch.randint(1, 400, (B, M), device=device)
        ob = torch.rand(B, M, 4, device=device) * 640
        qi = torch.randint(1, 300, (B, 14), device=device)
        ql = torch.full((B,), 14, dtype=torch.long, device=device)
        ocr_tokens = [["42", "road"] + [f"w{k}" for k in range(M - 2)] for _ in range(B)]
        logits = model(vf, vb, oi, ob, qi, ql, ocr_tokens, image_hw=(640, 640))
        target = torch.zeros_like(logits)
        target[torch.arange(B), torch.randint(0, logits.size(1), (B,))] = 1.0
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
