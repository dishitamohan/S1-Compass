"""Mean pooling over tokenizer-recorded option description spans."""
from __future__ import annotations

from typing import Any
from s1_compass.model.backbone import mx, _need_mlx


def pool(hidden: Any, batch: Any):
    """Return FP32 [N,K,H] means; padded option slots are all zero."""
    _need_mlx()
    if getattr(batch, "architecture", "encoder") == "causal":
        # Causal backbones return [N,K,L,H], one independent sequence per option.
        n, k, _, width = hidden.shape
        rows=[]
        for i in range(n):
            vectors=[]
            for j, start in enumerate(batch.option_prefix_lengths[i]):
                if start < 0:
                    vectors.append(mx.zeros((width,),dtype=mx.float32))
                else:
                    length=len(batch.option_sequences[i][j])
                    vectors.append(mx.mean(hidden[i,j,start:length].astype(mx.float32),axis=0))
            rows.append(mx.stack(vectors,axis=0))
        return mx.stack(rows,axis=0).astype(mx.float32)
    n, _, width = hidden.shape
    k = max(len(row) for row in batch.option_ids)
    rows = []
    for i in range(n):
        if not len(batch.option_ids[i]) <= len(batch.option_spans[i]) <= k:
            raise ValueError("option span row is malformed")
        vectors = []
        for j, span in enumerate(batch.option_spans[i]):
            start, end = span
            if start == -1 and end == -1:
                if j < len(batch.option_ids[i]):
                    raise ValueError("real option is missing its span")
                vectors.append(mx.zeros((width,), dtype=mx.float32))
            else:
                if j >= len(batch.option_ids[i]) or start < 0 or end <= start or end > sum(batch.attention_mask[i]):
                    raise ValueError("option span is outside valid input tokens")
                vectors.append(mx.mean(hidden[i, start:end].astype(mx.float32), axis=0))
        while len(vectors) < k:
            vectors.append(mx.zeros((width,), dtype=mx.float32))
        rows.append(mx.stack(vectors, axis=0))
    return mx.stack(rows, axis=0).astype(mx.float32)
