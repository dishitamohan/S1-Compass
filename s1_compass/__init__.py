"""S1 Compass Decision Studio engine."""

PRODUCT_NAME = "S1 Compass"
PUBLIC_MODEL_ID = "s1-compass-modernbert-base"

def public_model_id(base_model_selector: str, model_sha: str) -> str:
    """Build an allowlisted, candidate-specific public model identifier."""
    import re
    if base_model_selector not in {"modernbert-base", "qwen3-0.6b"}:
        raise ValueError("Unsupported public base model selector.")
    if not re.fullmatch(r"[a-f0-9]{64}", model_sha):
        raise ValueError("model_sha must be a lowercase SHA-256 digest.")
    return f"s1-compass-{base_model_selector}:{model_sha}"

__version__ = "0.1.0"
