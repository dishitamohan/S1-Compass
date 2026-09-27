"""Stable identities for base models supported by S1 Compass."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BaseModelSpec:
    """Project-owned metadata for a base model, separate from trained releases."""

    selector: str
    model_id: str
    display_name: str
    architecture: str
    repository: str
    snapshot_path: Path
    hidden_size: int | None
    preprocessing_version: str = "modernbert-v1"


BASE_MODELS: dict[str, BaseModelSpec] = {
    "modernbert-base": BaseModelSpec(
        selector="modernbert-base",
        model_id="answerdotai/ModernBERT-base",
        display_name="ModernBERT base",
        architecture="encoder",
        repository="answerdotai/ModernBERT-base",
        snapshot_path=PROJECT_ROOT / "artifacts/modernbert-base",
        hidden_size=768,
        preprocessing_version="modernbert-v1",
    ),
    "qwen3-0.6b": BaseModelSpec(
        selector="qwen3-0.6b",
        model_id="Qwen/Qwen3-0.6B",
        display_name="Qwen3 0.6B",
        architecture="causal",
        repository="Qwen/Qwen3-0.6B",
        snapshot_path=PROJECT_ROOT / "artifacts/qwen3-0.6b",
        hidden_size=1024,
        preprocessing_version="qwen3-option-causal-v1",
    ),
}


def get_base_model(selector: str) -> BaseModelSpec:
    """Resolve a CLI/UI selector; never infer model choice from artifacts."""
    try:
        return BASE_MODELS[selector]
    except KeyError as exc:
        choices = ", ".join(BASE_MODELS)
        raise ValueError(f"Unknown base model {selector!r}; choose one of: {choices}.") from exc
