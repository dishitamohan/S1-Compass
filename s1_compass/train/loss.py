"""Masked cross entropy for ragged option batches."""
from s1_compass.model.backbone import mx, _need_mlx


def cross_entropy(logits, batch):
    _need_mlx()
    labels = batch.label_idx
    if any(label < 0 for label in labels):
        raise ValueError("Training loss requires answerable labels")
    losses = []
    for i, label in enumerate(labels):
        row = logits[i, :len(batch.option_ids[i])].astype(mx.float32)
        m = mx.max(row)
        logsumexp = m + mx.log(mx.sum(mx.exp(row - m)))
        losses.append(logsumexp - row[label])
    return mx.mean(mx.stack(losses)).astype(mx.float32)
