"""End-to-end decision scoring and calibrated probabilities."""
from __future__ import annotations
import math, time
from typing import Any
from s1_compass.model.backbone import mx, forward
from s1_compass.model.pooling import pool
from s1_compass.model.heads import score as score_head


def logits(model: Any, batch: Any, use_bf16: bool = False):
    hidden = forward(model.encoder, batch, use_bf16=use_bf16)
    return score_head(model.head, pool(hidden, batch), batch)


def probabilities(values: Any, temperature: float = 1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    z = values.astype(mx.float32) / temperature
    z = z - mx.max(z, axis=-1, keepdims=True)
    exp = mx.exp(z)
    return exp / mx.sum(exp, axis=-1, keepdims=True)


def make_output(request: dict[str, Any], batch: Any, probs: Any, model_id: str, calibration_version: str, latency_ms: int = 0):
    result = {}
    for i, q in enumerate(request["questions"]):
        ids = [o["id"] for o in q["options"]]
        row = [float(x) for x in probs[i, :len(ids)].tolist()]
        top = max(range(len(ids)), key=lambda j: (row[j], tuple(-ord(c) for c in ids[j])))
        mapping = {oid: row[j] for j, oid in enumerate(ids)}
        if q["type"] == "boolean":
            ptrue = mapping["true"]
            result[q["id"]] = {"type":"boolean", "value":ptrue >= 0.5, "probability":ptrue}
        else:
            key = "choice" if q["type"] == "choice" else "level"
            result[q["id"]] = {"type":q["type"], key:ids[top], "probabilities":mapping, "confidence":row[top]}
    return {"answers":result,"model":model_id,"calibration_version":calibration_version,"latency_ms":latency_ms}


def decide(model: Any, request: dict[str, Any], batch: Any, calibration: dict[str, Any], model_id: str):
    start = time.perf_counter()
    p = probabilities(logits(model, batch), float(calibration["T"]))
    mx.eval(p)
    return make_output(request, batch, p, model_id, calibration["version"], int((time.perf_counter()-start)*1000))
