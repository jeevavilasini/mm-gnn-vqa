# MM-GNN for Scene-Text VQA

PyTorch implementation of **Multi-Modal Graph Neural Network for Joint Reasoning on Vision and Scene Text** (Gao et al., [arXiv:2003.13962](https://arxiv.org/abs/2003.13962)).

The model builds three fully connected sub-graphs (visual, semantic, numeric), runs **Visual–Semantic → Semantic–Semantic → Semantic–Numeric** attention aggregators (Sec. 3.2), then predicts answers over **fixed vocabulary + OCR copy indices** with top-down attention (Sec. 3.3).

## Install

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Optional tests:

```bash
pip install pytest
pytest -q
```

Smoke training on random tensors:

```bash
python scripts/example_train.py
```

## Usage

```python
import torch
from mm_gnn import MMGNNVQA

model = MMGNNVQA(
    answer_vocab_size=3997,       # e.g. TextVQA answers appearing ≥2 times
    question_vocab_size=10000,
    ocr_vocab_size=20000,
    max_ocr_tokens=50,
)
logits = model(
    visual_feats,          # [B, N, 2048] region (e.g. Faster R-CNN) features
    visual_boxes_xyxy,     # [B, N, 4] pixel coords (pass image_hw to normalize)
    ocr_ids,               # [B, M] token ids (embedding table mimics FastText)
    ocr_boxes_xyxy,        # [B, M, 4]
    question_ids,          # [B, T] (GloVe-style table in the paper)
    question_lengths,      # [B] or None
    ocr_tokens,            # list[list[str]] for numeric graph + copy decode
    image_hw=(H, W),      # optional; if None, boxes must be in [0,1]
)
# logits: [B, answer_vocab_size + max_ocr_tokens]
# Indices answer_vocab_size + j correspond to copying the j-th OCR slot.
```

## TextVQA / ST-VQA

This repository provides the **model architecture** and tensor interfaces. Full training matches the paper’s setup: pre-extracted Faster R-CNN regions, Rosetta OCR tokens, GloVe + LSTM (+ self-attention) questions, FastText OCR embeddings, AdaMax optimizer, etc. (Sec. 4.1). Plug in your dataset loaders and vocabularies under the same tensor Shapes as above.

## Layout

| Module | Role |
|--------|------|
| `mm_gnn/geometry.py` | 10-D bounding-box encoding |
| `mm_gnn/numeric.py` | Numeric / periodic encodings (Sec. 3.1, supp. Sec. 6) |
| `mm_gnn/question_encoder.py` | Embeddings + LSTM + attention pooling |
| `mm_gnn/aggregators.py` | VS, SS, SN aggregators |
| `mm_gnn/mm_gnn_core.py` | Graph stack + OCR token features `c` |
| `mm_gnn/model.py` | `MMGNNVQA` with answer head |

## Citation

```bibtex
@article{gao2020mmgnn,
  title={Multi-Modal Graph Neural Network for Joint Reasoning on Vision and Scene Text},
  author={Gao, Difei and Li, Ke and Wang, Ruiping and Shan, Shiguang and Chen, Xilin},
  journal={arXiv preprint arXiv:2003.13962},
  year={2020}
}
```

## License

Implementations here are provided for research and education. Refer to the original paper and dataset licenses (TextVQA, ST-VQA) when training or redistributing weights.
