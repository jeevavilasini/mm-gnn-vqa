"""Bounding-box features (Sec. 3.1 / implementation details): 10-D layout encoding."""

from __future__ import annotations

import torch


def encode_boxes_xyxy(boxes: torch.Tensor, image_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Encode boxes as 10-D features: cx, cy, x1, y1, x2, y2, w, h, area, aspect.

    If ``image_hw`` is provided, coordinates are normalized by (H, W); otherwise
    ``boxes`` are assumed already in [0, 1].
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 10))

    x1, y1, x2, y2 = boxes.unbind(-1)
    if image_hw is not None:
        h, w = float(image_hw[0]), float(image_hw[1])
        x1, x2 = x1 / w, x2 / w
        y1, y2 = y1 / h, y2 / h
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    area = w * h
    aspect = w / h
    return torch.stack([cx, cy, x1, y1, x2, y2, w, h, area, aspect], dim=-1)
