"""MLX implementation and strict checkpoint mapping for ModernBERT-base.

The implementation follows the public ModernBERT encoder structure. Loading a
checkpoint requires a local immutable-snapshot lock and a successful parity
record; this module never downloads mutable model files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from s1_compass.data.tokenize import BACKBONE_REVISION, CONFIG_SHA256, MODEL_ID, TOKENIZER_SHA256, WEIGHTS_SHA256

try:  # Keep contract tooling importable on machines without MLX installed.
    if os.environ.get("S1_COMPASS_DISABLE_MLX") == "1":
        raise ImportError("MLX disabled for contract-only CLI commands")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.nn.utils import checkpoint as _checkpoint
except ImportError:  # pragma: no cover - exercised on non-MLX hosts
    mx = None
    nn = None
    _checkpoint = None

HIDDEN_SIZE = 768
NUM_HEADS = 12
NUM_LAYERS = 22
INTERMEDIATE_SIZE = 1152
LOCAL_ATTENTION = 128
GLOBAL_ATTENTION_EVERY = 3
MODEL_LOCK = "backbone.lock.json"
PARITY_RECORD = "parity.json"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMMIT = re.compile(r"^[a-f0-9]{40}$")


class BackboneError(RuntimeError):
    """Invalid or unsupported backbone configuration or artifact."""


def _need_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError("MLX is required for ModernBERT execution; install the pinned MLX dependency on Apple Silicon.")


def _linear(layer: Any, value: Any, use_bf16: bool = False):
    """Dense layer with FP32 parameters and optional BF16 matrix operands."""
    if not use_bf16:
        return layer(value.astype(mx.float32)).astype(mx.float32)
    x = value.astype(mx.bfloat16)
    weight = layer.weight.astype(mx.bfloat16)
    result = mx.matmul(x, weight.T)
    if getattr(layer, "bias", None) is not None:
        result = result + layer.bias.astype(mx.bfloat16)
    return result.astype(mx.float32)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _weights_digest(paths: Sequence[Path]) -> str:
    """Hash sorted relative names and per-file hashes for single or sharded weights."""
    entries = []
    for path in sorted(paths, key=lambda p: p.name):
        entries.append({"name": path.name, "sha256": _sha256(path)})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _weight_files(snapshot: Path) -> list[Path]:
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(data.get("weight_map"), dict) or not data["weight_map"]:
            raise BackboneError("Malformed safetensors weight index.")
        names = sorted(set(data["weight_map"].values()))
        files = [snapshot / name for name in names]
        if any(not p.is_file() for p in files):
            raise BackboneError("Checkpoint index refers to a missing shard.")
        return files
    single = snapshot / "model.safetensors"
    if single.is_file():
        return [single]
    raise BackboneError("Expected model.safetensors or model.safetensors.index.json in snapshot.")


def _validate_snapshot(snapshot: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = snapshot / MODEL_LOCK
    if not lock_path.is_file():
        raise BackboneError(f"Immutable ModernBERT lock is required: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = {"model_id", "revision", "config_sha256", "tokenizer_sha256", "weights_sha256", "python", "macos", "mlx", "transformers"}
    if set(lock) != expected:
        raise BackboneError(f"{MODEL_LOCK} must contain exactly {sorted(expected)}.")
    if lock["model_id"] != MODEL_ID or not isinstance(lock["revision"], str) or not _COMMIT.fullmatch(lock["revision"]):
        raise BackboneError("Snapshot must identify answerdotai/ModernBERT-base at an immutable 40-character commit.")
    for key in ("config_sha256", "tokenizer_sha256", "weights_sha256"):
        if not isinstance(lock[key], str) or not _SHA256.fullmatch(lock[key]):
            raise BackboneError(f"Invalid {key} in snapshot lock.")
    if (BACKBONE_REVISION is not None and lock["revision"] != BACKBONE_REVISION
            or CONFIG_SHA256 is not None and lock["config_sha256"] != CONFIG_SHA256
            or TOKENIZER_SHA256 is not None and lock["tokenizer_sha256"] != TOKENIZER_SHA256
            or WEIGHTS_SHA256 is not None and lock["weights_sha256"] != WEIGHTS_SHA256):
        raise BackboneError("Snapshot lock does not match this project's pinned ModernBERT revision and hashes.")
    config_path = snapshot / "config.json"
    tokenizer_path = snapshot / "tokenizer.json"
    if not config_path.is_file() or _sha256(config_path) != lock["config_sha256"]:
        raise BackboneError("ModernBERT config hash does not match lock.")
    if not tokenizer_path.is_file() or _sha256(tokenizer_path) != lock["tokenizer_sha256"]:
        raise BackboneError("Tokenizer hash does not match lock.")
    weight_files = _weight_files(snapshot)
    # A single blob's lock uses its direct content hash. Sharded snapshots use
    # a canonical manifest digest over sorted relative filenames and hashes.
    actual_weights_hash = _sha256(weight_files[0]) if len(weight_files) == 1 else _weights_digest(weight_files)
    if actual_weights_hash != lock["weights_sha256"]:
        raise BackboneError("ModernBERT weight hash does not match lock.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    return lock, config


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "model_type": "modernbert", "hidden_size": HIDDEN_SIZE, "num_attention_heads": NUM_HEADS,
        "num_hidden_layers": NUM_LAYERS, "intermediate_size": INTERMEDIATE_SIZE,
        "global_attn_every_n_layers": GLOBAL_ATTENTION_EVERY, "local_attention": LOCAL_ATTENTION,
    }
    for key, wanted in required.items():
        if config.get(key) != wanted:
            raise BackboneError(f"Unsupported ModernBERT config: {key}={config.get(key)!r}, expected {wanted!r}.")
    if config.get("attention_bias", False) or config.get("mlp_bias", False) or config.get("norm_bias", False):
        raise BackboneError("Pinned ModernBERT-base expects bias-free attention/MLP projections and norms.")
    if config.get("hidden_activation") != "gelu":
        raise BackboneError("Pinned ModernBERT-base requires GELU gated MLP activation.")
    if config.get("position_embedding_type", "absolute") != "absolute":
        raise BackboneError("Unexpected position_embedding_type.")


if nn is not None:
    class _LayerNorm(nn.Module):
        def __init__(self, dims: int, eps: float, bias: bool = False):
            super().__init__()
            self.weight = mx.ones((dims,), dtype=mx.float32)
            self.bias = mx.zeros((dims,), dtype=mx.float32) if bias else None
            self.eps = eps

        def __call__(self, x):
            x = x.astype(mx.float32)
            mean = mx.mean(x, axis=-1, keepdims=True)
            variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
            y = (x - mean) * mx.rsqrt(variance + self.eps) * self.weight
            if self.bias is not None:
                y = y + self.bias
            return y


    class _ModernBertLayer(nn.Module):
        def __init__(self, config: Mapping[str, Any], index: int):
            super().__init__()
            hidden = int(config["hidden_size"])
            heads = int(config["num_attention_heads"])
            inner = int(config["intermediate_size"])
            self.index = index
            self.heads = heads
            self.head_dim = hidden // heads
            eps = float(config.get("norm_eps", config.get("layer_norm_eps", 1e-5)))
            self.attn_norm = _LayerNorm(hidden, eps) if index != 0 else None
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=bool(config.get("attention_bias", False)))
            self.attn_out = nn.Linear(hidden, hidden, bias=bool(config.get("attention_bias", False)))
            self.mlp_norm = _LayerNorm(hidden, eps)
            self.mlp_in = nn.Linear(hidden, inner * 2, bias=bool(config.get("mlp_bias", False)))
            self.mlp_out = nn.Linear(inner, hidden, bias=bool(config.get("mlp_bias", False)))
            self.attn_dropout = nn.Dropout(float(config.get("attention_dropout", 0.0)))
            self.mlp_dropout = nn.Dropout(float(config.get("mlp_dropout", 0.0)))
            self.out_dropout = nn.Dropout(float(config.get("attention_dropout", 0.0)))
            self.local = (index % int(config["global_attn_every_n_layers"])) != 0
            self.local_radius = int(config["local_attention"]) // 2
            self.local_theta = float(config.get("local_rope_theta", 10000.0))
            self.global_theta = float(config.get("global_rope_theta", 160000.0))

        def _rope(self, x, theta: float):
            # x is [B,H,L,D]. ModernBERT duplicates freqs in the two halves,
            # then applies rotate_half([x1,x2]) = [-x2,x1].
            length = x.shape[2]
            half = self.head_dim // 2
            inv = 1.0 / mx.power(mx.array(theta, dtype=mx.float32), mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
            positions = mx.arange(length, dtype=mx.float32)
            freqs = positions[:, None] * inv[None, :]
            emb = mx.concatenate([freqs, freqs], axis=-1)
            cos = mx.cos(emb)[None, None, :, :]
            sin = mx.sin(emb)[None, None, :, :]
            return cos, sin

        def __call__(self, hidden, attention_mask, use_bf16: bool = False):
            residual = hidden
            attn_input = self.attn_norm(hidden) if self.attn_norm is not None else hidden
            # We keep FP32 master parameters and normalization/output. Matrix
            # operands can be BF16 during training as specified by the plan.
            qkv_input = attn_input.astype(mx.bfloat16) if use_bf16 else attn_input
            qkv = _linear(self.qkv, qkv_input, use_bf16=use_bf16)
            batch, length, _ = qkv.shape
            qkv = qkv.reshape(batch, length, 3, self.heads, self.head_dim)
            q = qkv[:, :, 0, :, :].transpose(0, 2, 1, 3)
            k = qkv[:, :, 1, :, :].transpose(0, 2, 1, 3)
            v = qkv[:, :, 2, :, :].transpose(0, 2, 1, 3)
            cos, sin = self._rope(q, self.local_theta if self.local else self.global_theta)
            q = q * cos + mx.concatenate([-q[..., self.head_dim // 2:], q[..., :self.head_dim // 2]], axis=-1) * sin
            k = k * cos + mx.concatenate([-k[..., self.head_dim // 2:], k[..., :self.head_dim // 2]], axis=-1) * sin
            scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)
            # Full bidirectional attention excludes right-padding keys. Local
            # layers use an inclusive +/-64 token window (128-token setting).
            allowed = attention_mask[:, None, None, :] == 1
            if self.local:
                pos = mx.arange(length)
                window = mx.abs(pos[:, None] - pos[None, :]) <= self.local_radius
                allowed = allowed & window[None, None, :, :]
            scores = mx.where(allowed, scores, mx.array(-1e30, dtype=mx.float32))
            # Match eager FP32 attention explicitly instead of using the
            # backend's potentially lower-precision fused softmax kernel.
            shifted = scores - mx.max(scores, axis=-1, keepdims=True)
            exp_scores = mx.exp(shifted)
            probs = (exp_scores / mx.sum(exp_scores, axis=-1, keepdims=True)).astype(mx.float32)
            probs = self.attn_dropout(probs)
            context = mx.matmul(probs, v).transpose(0, 2, 1, 3).reshape(batch, length, -1)
            context = _linear(self.attn_out, context, use_bf16=use_bf16)
            hidden = residual + self.out_dropout(context)
            mlp_input = self.mlp_norm(hidden)
            projected = _linear(self.mlp_in, mlp_input, use_bf16=use_bf16)
            value, gate = mx.split(projected, 2, axis=-1)
            gelu = 0.5 * value * (1.0 + mx.erf(value / math.sqrt(2.0)))
            mlp = _linear(self.mlp_out, gelu * gate, use_bf16=use_bf16)
            return hidden + self.mlp_dropout(mlp)


    class ModernBertEncoder(nn.Module):
        """FP32-master MLX ModernBERT encoder returning float32 hidden states."""

        def __init__(self, config: Mapping[str, Any]):
            super().__init__()
            _validate_config(config)
            hidden = int(config["hidden_size"])
            eps = float(config.get("norm_eps", config.get("layer_norm_eps", 1e-5)))
            # MLX treats nested dictionaries as parameter-tree branches. Keep
            # scalar/string config values out of the Module tree so optimizer
            # state only sees array leaves.
            self.embeddings = nn.Embedding(int(config["vocab_size"]), hidden)
            self.embedding_norm = _LayerNorm(hidden, eps)
            self.layers = [_ModernBertLayer(config, i) for i in range(int(config["num_hidden_layers"]))]
            self.final_norm = _LayerNorm(hidden, eps)
            self.embedding_dropout = nn.Dropout(float(config.get("embedding_dropout", 0.0)))

        def __call__(self, input_ids, attention_mask, use_bf16: bool = False, return_hidden_states: bool = False):
            if len(input_ids.shape) != 2 or input_ids.shape != attention_mask.shape:
                raise ValueError("input_ids and attention_mask must both have shape [N,L].")
            if input_ids.shape[0] < 1 or input_ids.shape[1] > 512:
                raise ValueError("Encoder expects 1..512 sequence rows.")
            hidden = self.embeddings(input_ids).astype(mx.float32)
            hidden = self.embedding_dropout(self.embedding_norm(hidden)).astype(mx.float32)
            layer_outputs = [hidden] if return_hidden_states else None
            for layer in self.layers:
                if self.training and use_bf16:
                    hidden = _checkpoint(layer)(hidden, attention_mask, use_bf16=use_bf16)
                else:
                    hidden = layer(hidden, attention_mask, use_bf16=use_bf16)
                if layer_outputs is not None:
                    layer_outputs.append(hidden)
            hidden = self.final_norm(hidden).astype(mx.float32)
            return (hidden, layer_outputs) if return_hidden_states else hidden

else:
    class ModernBertEncoder:  # pragma: no cover - placeholder for non-MLX hosts
        def __init__(self, config: Mapping[str, Any]):
            _need_mlx()


_MLM_DISCARD_PREFIXES = ("head.", "decoder.")
_BASE_PREFIXES = ("model.", "")


def map_checkpoint_weights(weights: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Map HF encoder tensors to MLX names and reject all unknown tensors.

    Only the recognized masked-language-model prediction head and decoder are
    discarded. All embeddings, norms, attention, and MLP tensors are required.
    """
    _validate_config(config)
    mapped: dict[str, Any] = {}
    discarded = []
    for original_name, value in weights.items():
        name = original_name
        if name.startswith("model."):
            name = name[len("model."):]
        if name.startswith(_MLM_DISCARD_PREFIXES):
            discarded.append(name)
            continue
        if name == "embeddings.tok_embeddings.weight":
            target = "embeddings.weight"
        elif name.startswith("embeddings.norm."):
            target = "embedding_norm." + name.removeprefix("embeddings.norm.")
        elif name.startswith("final_norm."):
            target = "final_norm." + name.removeprefix("final_norm.")
        else:
            match = re.fullmatch(r"layers\.(\d+)\.(attn_norm|mlp_norm|attn\.Wqkv|attn\.Wo|mlp\.Wi|mlp\.Wo)\.(weight|bias)", name)
            if not match:
                raise BackboneError(f"Unrecognized ModernBERT checkpoint tensor: {original_name}")
            layer_index, component, parameter = match.groups()
            if int(layer_index) >= int(config["num_hidden_layers"]):
                raise BackboneError(f"Checkpoint tensor references nonexistent layer: {original_name}")
            component_map = {
                "attn_norm": "attn_norm", "mlp_norm": "mlp_norm", "attn.Wqkv": "qkv",
                "attn.Wo": "attn_out", "mlp.Wi": "mlp_in", "mlp.Wo": "mlp_out",
            }
            if component == "attn_norm" and int(layer_index) == 0:
                raise BackboneError("Layer 0 uses identity attention normalization; checkpoint contains unexpected norm weights.")
            target = f"layers.{layer_index}.{component_map[component]}.{parameter}"
        if target in mapped:
            raise BackboneError(f"Duplicate checkpoint mapping to {target}.")
        mapped[target] = value
    if not discarded:
        # Encoder-only checkpoints are valid; no MLM tensors are required.
        discarded = []
    expected = expected_parameter_names(config)
    missing = expected - mapped.keys()
    unexpected = mapped.keys() - expected
    if missing or unexpected:
        raise BackboneError(f"Checkpoint mapping mismatch; missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}.")
    # The tied decoder is sometimes serialized twice; the original model's
    # tied-weight metadata may omit one copy. Its exact known names are ignored.
    return mapped


def expected_parameter_names(config: Mapping[str, Any]) -> set[str]:
    """Return the complete parameter namespace expected from the encoder."""
    names = {"embeddings.weight", "embedding_norm.weight", "final_norm.weight"}
    if bool(config.get("norm_bias", False)):
        names |= {"embedding_norm.bias", "final_norm.bias"}
    for i in range(int(config["num_hidden_layers"])):
        if i != 0:
            names.add(f"layers.{i}.attn_norm.weight")
            if bool(config.get("norm_bias", False)):
                names.add(f"layers.{i}.attn_norm.bias")
        names |= {
            f"layers.{i}.qkv.weight", f"layers.{i}.attn_out.weight",
            f"layers.{i}.mlp_norm.weight", f"layers.{i}.mlp_in.weight", f"layers.{i}.mlp_out.weight",
        }
        if bool(config.get("attention_bias", False)):
            names |= {f"layers.{i}.qkv.bias", f"layers.{i}.attn_out.bias"}
        if bool(config.get("mlp_bias", False)):
            names |= {f"layers.{i}.mlp_in.bias", f"layers.{i}.mlp_out.bias"}
        if bool(config.get("norm_bias", False)):
            names.add(f"layers.{i}.mlp_norm.bias")
    return names


def _parameter_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    h, inner, vocab = int(config["hidden_size"]), int(config["intermediate_size"]), int(config["vocab_size"])
    shapes = {"embeddings.weight": (vocab, h), "embedding_norm.weight": (h,), "final_norm.weight": (h,)}
    for i in range(int(config["num_hidden_layers"])):
        if i != 0:
            shapes[f"layers.{i}.attn_norm.weight"] = (h,)
        shapes.update({
            f"layers.{i}.qkv.weight": (3 * h, h), f"layers.{i}.attn_out.weight": (h, h),
            f"layers.{i}.mlp_norm.weight": (h,), f"layers.{i}.mlp_in.weight": (2 * inner, h),
            f"layers.{i}.mlp_out.weight": (h, inner),
        })
    if bool(config.get("norm_bias", False)):
        for name in list(shapes):
            if name.endswith("norm.weight"):
                shapes[name.removesuffix("weight") + "bias"] = shapes[name]
    if bool(config.get("attention_bias", False)):
        for i in range(int(config["num_hidden_layers"])):
            shapes[f"layers.{i}.qkv.bias"] = (3 * h,)
            shapes[f"layers.{i}.attn_out.bias"] = (h,)
    if bool(config.get("mlp_bias", False)):
        for i in range(int(config["num_hidden_layers"])):
            shapes[f"layers.{i}.mlp_in.bias"] = (2 * inner,)
            shapes[f"layers.{i}.mlp_out.bias"] = (h,)
    return shapes


def _check_tensor_shapes(mapped: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    shapes = _parameter_shapes(config)
    for name, value in mapped.items():
        shape = tuple(value.shape)
        if shape != shapes[name]:
            raise BackboneError(f"Tensor {name} has shape {shape}; expected {shapes[name]}.")


def load(snapshot_path: str | Path, *, require_parity: bool = True) -> ModernBertEncoder:
    """Load a strictly hash-bound local snapshot into MLX FP32 master weights."""
    _need_mlx()
    path = Path(snapshot_path)
    lock, config = _validate_snapshot(path)
    parity_path = path / PARITY_RECORD
    if require_parity:
        if not parity_path.is_file():
            raise BackboneError("A passing parity.json is required before loading a production encoder.")
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if (parity.get("status") != "passed" or parity.get("mlx_device") not in {"cpu", "gpu"}
                or parity.get("max_abs_error") is None or parity["max_abs_error"] >= 0.001):
            raise BackboneError("The recorded FP32 ModernBERT parity check did not pass.")
        if (parity.get("revision") != lock["revision"] or parity.get("weights_sha256") != lock["weights_sha256"]
                or parity.get("config_sha256") != lock["config_sha256"]):
            raise BackboneError("Parity record does not match this pinned snapshot.")
    files = _weight_files(path)
    tensors = {}
    for file in files:
        tensors.update(mx.load(str(file)))
    mapped = map_checkpoint_weights(tensors, config)
    _check_tensor_shapes(mapped, config)
    mapped = {name: value.astype(mx.float32) for name, value in mapped.items()}
    model = ModernBertEncoder(config)
    model.load_weights(list(mapped.items()), strict=True)
    model.eval()
    return model


def load_base_model(snapshot_path: str | Path, selector: str, *, require_parity: bool = True, training_mode: str = "inference"):
    """Load the registered architecture implementation for a pinned snapshot."""
    if selector == "modernbert-base":
        if training_mode not in {"inference", "full"}:
            raise ValueError("ModernBERT supports only full fine-tuning mode.")
        return load(snapshot_path, require_parity=require_parity)
    if selector != "qwen3-0.6b":
        raise ValueError(f"Unknown base model selector: {selector}")
    from s1_compass.models.registry import get_base_model
    from s1_compass.models.snapshot import verify_snapshot
    spec = get_base_model(selector)
    lock = verify_snapshot(spec, Path(snapshot_path))
    if require_parity:
        parity_path = Path(snapshot_path).resolve().parent / (Path(snapshot_path).name + ".parity.json")
        if not parity_path.is_file():
            raise BackboneError(f"A passing Qwen3 parity sidecar is required: {parity_path}. Run python -m s1_compass.models parity qwen3-0.6b.")
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        expected_lock_sha = _sha256(Path(snapshot_path) / "s1-compass-snapshot.lock.json")
        if parity.get("status") != "passed" or parity.get("revision") != lock["revision"] or parity.get("snapshot_lock_sha256") != expected_lock_sha:
            raise BackboneError("Qwen3 parity sidecar is failed or bound to a different snapshot.")
    try:
        import mlx_lm
        from mlx_lm.utils import load_model
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise RuntimeError("Install the pinned MLX-LM release with Qwen3 support.") from exc
    if tuple(map(int, mlx_lm.__version__.split(".")[:2])) < (0, 25):
        raise RuntimeError("Qwen3 requires MLX-LM >=0.25.0.")
    if training_mode not in {"inference", "lora"}:
        raise ValueError("Qwen3 supports only inference or explicit LoRA training mode.")
    model, config = load_model(Path(snapshot_path), lazy=False, strict=True)
    if config.get("model_type") != "qwen3" or not hasattr(model, "model"):
        raise BackboneError("MLX-LM did not load the expected Qwen3 causal model.")
    tokenizer_config=json.loads((Path(snapshot_path)/"tokenizer_config.json").read_text(encoding="utf-8"))
    from tokenizers import Tokenizer as HFTokenizer
    raw_tokenizer=HFTokenizer.from_file(str(Path(snapshot_path)/"tokenizer.json"))
    pad_token=tokenizer_config.get("pad_token")
    if isinstance(pad_token,dict): pad_token=pad_token.get("content")
    config["pad_token_id"]=raw_tokenizer.token_to_id(pad_token)
    return Qwen3MLXBackbone(model, config, lock, training_mode, linear_to_lora_layers)


if nn is not None:
    class Qwen3MLXBackbone(nn.Module):
        """MLX-LM Qwen3 hidden states with independent option sequences."""

        def __init__(self, model: Any, config: Mapping[str, Any], lock: Mapping[str, Any], training_mode: str, lora_factory: Any):
            super().__init__()
            self.model = model
            self.config = dict(config)
            self.lock = dict(lock)
            self.trainable_mode = training_mode
            self.lora_config = None
            if training_mode == "lora":
                self.model.freeze()
                self.lora_config = {"rank": 8, "scale": 16.0, "dropout": 0.0, "layers": 4,
                                    "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
                lora_factory(self.model, self.lora_config["layers"],
                             {key: value for key, value in self.lora_config.items() if key in {"rank", "scale", "dropout", "keys"}})
            self.model.eval()

        def forward_options(self, sequences: Any, starts: Any):
            _need_mlx()
            n = len(sequences)
            k = max(len(row) for row in sequences)
            flat = [seq for row in sequences for seq in row]
            max_len = max(len(seq) for seq in flat if seq)
            pad_id = int(self.config["pad_token_id"])
            ids = [seq + [pad_id] * (max_len-len(seq)) if seq else [pad_id] * max_len for seq in flat]
            valid = [[True] * len(seq) + [False] * (max_len-len(seq)) if seq else [False] * max_len for seq in flat]
            x = mx.array(ids, dtype=mx.int32)
            key_mask = mx.array(valid, dtype=mx.bool_)
            positions = mx.arange(max_len)
            causal = positions[:, None] >= positions[None, :]
            attention = causal[None, None, :, :] & key_mask[:, None, None, :]
            hidden = self.model.model(x, mask=attention)
            return hidden.reshape(n, k, max_len, int(self.config["hidden_size"])).astype(mx.float32)
else:
    class Qwen3MLXBackbone:
        def __init__(self, *args: Any, **kwargs: Any): _need_mlx()


def forward(encoder: ModernBertEncoder, batch: Any, use_bf16: bool = False):
    """Return FP32 ``[N,L,768]`` hidden states for a tokenize.Batch."""
    if getattr(batch, "architecture", "encoder") == "causal":
        return encoder.forward_options(batch.option_sequences, batch.option_prefix_lengths)
    _need_mlx()
    input_ids = mx.array(batch.input_ids, dtype=mx.int32)
    attention_mask = mx.array(batch.attention_mask, dtype=mx.int32)
    hidden = encoder(input_ids, attention_mask, use_bf16=use_bf16)
    if hidden.shape != (len(batch.input_ids), len(batch.input_ids[0]), HIDDEN_SIZE):
        raise BackboneError(f"Unexpected encoder output shape: {hidden.shape}.")
    return hidden.astype(mx.float32)


def hidden_state_parity_error(actual: Any, reference: Any, attention_mask: Any) -> float:
    """Compute max valid-token absolute error, ignoring padded positions."""
    _need_mlx()
    mask = mx.array(attention_mask).astype(mx.bool_)
    diff = mx.abs(mx.array(actual).astype(mx.float32) - mx.array(reference).astype(mx.float32))
    valid = mx.broadcast_to(mask[..., None], diff.shape)
    return float(mx.max(mx.where(valid, diff, 0)).item())


def verify_parity(
    snapshot_path: str | Path,
    cases: Sequence[tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]],
    *,
    mlx_device: str = "gpu",
) -> dict[str, Any]:
    """Compare MLX outputs to a separate-process HF FP32 CPU reference.

    Each case is ``(input_ids, attention_mask)``. The reference process exits
    before the MLX model is loaded to avoid holding both model copies together.
    A local Transformers installation and audited checkpoint are required.
    """
    _need_mlx()
    if mlx_device not in {"gpu", "cpu"}:
        raise ValueError("mlx_device must be gpu or cpu")
    previous_device = mx.default_device()
    mx.set_default_device(mx.Device(mx.DeviceType.gpu if mlx_device == "gpu" else mx.DeviceType.cpu))
    import subprocess
    import sys
    import tempfile
    from array import array

    path = Path(snapshot_path)
    lock, config = _validate_snapshot(path)
    if not cases:
        raise ValueError("At least one parity case is required.")
    # Worker receives JSON inputs and emits raw float32 hidden states per case.
    # Its process fully exits before the MLX copy is instantiated below.
    request = {"snapshot": str(path.resolve()), "cases": [{"input_ids": ids, "attention_mask": mask} for ids, mask in cases]}
    with tempfile.TemporaryDirectory(prefix="s1-compass-modernbert-parity-") as temp:
        temp_path = Path(temp)
        request_path = temp_path / "request.json"
        request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
        worker = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
req=json.loads(Path(sys.argv[1]).read_text())
model=AutoModel.from_pretrained(req["snapshot"], local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager")
model.eval()
with torch.no_grad():
 for i,case in enumerate(req["cases"]):
  ids=torch.tensor(case["input_ids"],dtype=torch.long)
  mask=torch.tensor(case["attention_mask"],dtype=torch.long)
  result=model(input_ids=ids,attention_mask=mask,output_hidden_states=True)
  states=np.stack([x.cpu().contiguous().numpy().astype("<f4") for x in result.hidden_states]+[result.last_hidden_state.cpu().contiguous().numpy().astype("<f4")])
  states.tofile(Path(sys.argv[2])/f"case-{i}.f32")
  Path(sys.argv[2],f"case-{i}.shape").write_text(json.dumps(list(states.shape)))
'''
        proc = subprocess.run([sys.executable, "-c", worker, str(request_path), str(temp_path)], capture_output=True, text=True)
        if proc.returncode:
            raise BackboneError("Separate-process Transformers reference failed: " + proc.stderr[-3000:])
        references = []
        shapes = []
        for i, _ in enumerate(cases):
            shape = tuple(json.loads((temp_path / f"case-{i}.shape").read_text()))
            raw = array("f")
            with (temp_path / f"case-{i}.f32").open("rb") as f:
                raw.fromfile(f, os.path.getsize(temp_path / f"case-{i}.f32") // raw.itemsize)
            references.append(mx.array(raw, dtype=mx.float32).reshape(shape))
            shapes.append(shape)
        model = load(path, require_parity=False)
        errors = []
        layer_errors = []
        for (ids, mask), reference in zip(cases, references):
            actual, actual_states = model(mx.array(ids, dtype=mx.int32), mx.array(mask, dtype=mx.int32), use_bf16=False, return_hidden_states=True)
            all_actual = mx.stack(actual_states + [actual], axis=0)
            layer_case_errors = [hidden_state_parity_error(all_actual[i], reference[i], mask) for i in range(len(actual_states) + 1)]
            layer_errors.append(layer_case_errors)
            errors.append(layer_case_errors[-1])
        maximum = max(errors)
        record = {"status": "passed" if maximum < 0.001 else "failed", "max_abs_error": maximum,
                  "case_errors": errors, "layer_case_errors": layer_errors, "case_shapes": shapes, "revision": lock["revision"],
                  "config_sha256": lock["config_sha256"], "weights_sha256": lock["weights_sha256"],
                  "dtype": "float32", "mlx_device": mlx_device,
                  "reference": "transformers-cpu-eager"}
        record_path = path / (PARITY_RECORD if mlx_device == "cpu" else "parity-gpu.json")
        record_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        mx.set_default_device(previous_device)
        if maximum >= 0.001:
            raise BackboneError(f"ModernBERT FP32 parity failed: maximum valid hidden-state error {maximum:.8g} >= 0.001.")
        return record
