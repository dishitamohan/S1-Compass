"""Shared scalar option scorer."""
from __future__ import annotations
from typing import Any
from s1_compass.model.backbone import mx, nn, _need_mlx


if nn is not None:
    class Head(nn.Module):
        def __init__(self, hidden_size: int = 768):
            super().__init__()
            self.projection = nn.Linear(hidden_size, 1, bias=False)
            self.projection.weight = mx.random.normal(shape=(1, hidden_size), loc=0.0, scale=0.02)

        def __call__(self, values):
            return self.projection(values)
else:
    Head = None


def initialize(seed: int = 17, hidden_size: int = 768) -> Head:
    _need_mlx()
    mx.random.seed(seed)
    return Head(hidden_size)


def score(head: Head, pooled: Any, batch: Any):
    _need_mlx()
    logits = head(pooled.astype(mx.float32)).squeeze(-1).astype(mx.float32)
    mask = mx.array([[j < len(ids) for j in range(logits.shape[1])] for ids in batch.option_ids])
    return mx.where(mask, logits, mx.array(float("-inf"), dtype=mx.float32))
