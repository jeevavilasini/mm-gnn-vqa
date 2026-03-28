import torch

from mm_gnn import MMGNNVQA


def test_mmgnn_vqa_forward_shapes():
    B, N, M = 2, 6, 10
    vocab = 200
    q_vocab = 500
    ocr_vocab = 800
    model = MMGNNVQA(
        answer_vocab_size=vocab,
        question_vocab_size=q_vocab,
        ocr_vocab_size=ocr_vocab,
        max_ocr_tokens=M,
        hidden_dim=128,
        box_dim=128,
        attn_dim=128,
        numeric_dim=32,
        question_embed_dim=128,
        question_hidden_dim=256,
        topdown_hidden=128,
    )
    vf = torch.randn(B, N, 2048)
    vb = torch.rand(B, N, 4) * 448
    oi = torch.randint(0, ocr_vocab, (B, M))
    ob = torch.rand(B, M, 4) * 448
    qi = torch.randint(1, q_vocab, (B, 12))
    qi[1, 10:] = 0
    ql = torch.tensor([12, 10], dtype=torch.long)
    ocr_tokens = [["12", "stop", "3:00"] + [f"t{k}" for k in range(M - 3)] for _ in range(B)]

    logits = model(vf, vb, oi, ob, qi, ql, ocr_tokens, image_hw=(448, 448))
    assert logits.shape == (B, vocab + M)
