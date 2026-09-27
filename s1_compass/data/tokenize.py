"""Deterministic ModernBERT input assembly and padded Batch construction."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

from s1_compass.data.schema import ValidationError, validate_row

MODEL_ID = "answerdotai/ModernBERT-base"
BACKBONE_REVISION: Optional[str] = "94032bb66234a691cf6248265170006a7ced4970"
TOKENIZER_SHA256: Optional[str] = "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1"
CONFIG_SHA256: Optional[str] = "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f"
WEIGHTS_SHA256: Optional[str] = "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9"
MAX_LENGTH = 512
MAX_INSTRUCTION_TOKENS = 96
MAX_OPTION_TOKENS = 48
STATE_RESERVE = 32
FIXED_TEMPLATE_LIMIT = MAX_LENGTH - STATE_RESERVE


class Tokenizer(Protocol):
    cls_token_id: int
    sep_token_id: int
    pad_token_id: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class _PinnedTokenizer:
    """Adapter around the low-level ``tokenizers`` file format."""

    def __init__(self, raw: Any, config: dict[str, Any], tokenizer_config: dict[str, Any]):
        self._raw = raw
        for attr, key, token_name in (
            ("cls_token_id", "cls_token_id", "cls_token"),
            ("sep_token_id", "sep_token_id", "sep_token"),
            ("pad_token_id", "pad_token_id", "pad_token"),
        ):
            token_id = config.get(key)
            token_spec = tokenizer_config.get(token_name)
            if token_id is None and isinstance(token_spec, dict):
                token_id = raw.token_to_id(token_spec.get("content"))
            elif token_id is None and isinstance(token_spec, str):
                token_id = raw.token_to_id(token_spec)
            if not isinstance(token_id, int) or raw.id_to_token(token_id) is None:
                raise ValueError(f"Pinned tokenizer has no valid {token_name} ID.")
            if isinstance(token_spec, dict) and token_spec.get("content") != raw.id_to_token(token_id):
                raise ValueError(f"Tokenizer special token mismatch for {token_name}.")
            if isinstance(token_spec, str) and token_spec != raw.id_to_token(token_id):
                raise ValueError(f"Tokenizer special token mismatch for {token_name}.")
            setattr(self, attr, token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._raw.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self._raw.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets


class _QwenTokenizer:
    """Hash-locked Qwen tokenizer adapter with no vocabulary mutation."""

    def __init__(self, raw: Any, config: dict[str, Any], tokenizer_config: dict[str, Any]):
        self._raw = raw
        self.pad_token_id = tokenizer_config.get("pad_token_id")
        if self.pad_token_id is None:
            pad_token = tokenizer_config.get("pad_token")
            if isinstance(pad_token, dict): pad_token = pad_token.get("content")
            self.pad_token_id = raw.token_to_id(pad_token) if pad_token else None
        self.eos_token_id = config.get("eos_token_id")
        if self.pad_token_id is None or self.eos_token_id is None:
            raise ValueError("Qwen snapshot must define its actual pad and EOS token IDs.")
        if raw.id_to_token(self.pad_token_id) is None or raw.id_to_token(self.eos_token_id) is None:
            raise ValueError("Qwen pad/EOS token IDs are not present in tokenizer vocabulary.")
        self.cls_token_id = self.eos_token_id
        self.sep_token_id = self.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._raw.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self._raw.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets


@dataclass(frozen=True)
class Batch:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    option_spans: list[list[list[int]]]
    label_idx: list[int]
    question_type: list[str]
    option_ids: list[list[str]]
    architecture: str = "encoder"
    option_sequences: list[list[list[int]]] | None = None
    option_prefix_lengths: list[list[int]] | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.input_ids), len(self.input_ids[0]), max(map(len, self.option_ids)))


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, (list, tuple)) or any(isinstance(x, bool) or not isinstance(x, int) for x in ids):
        raise TypeError("Tokenizer.encode must return a sequence of integer token IDs.")
    return list(ids)


def _field_tokens(tokenizer: Tokenizer, text: str, limit: int, name: str) -> list[int]:
    ids = _encode(tokenizer, text)
    if len(ids) > limit:
        raise ValidationError(f"{name} exceeds its {limit}-token limit.", "/")
    return ids


def _stable_option_order(option_ids: Sequence[str], seed: int, epoch: int, row_id: str) -> list[int]:
    """Stable per-row permutation, independent of input iteration order."""
    material = f"{seed}:{epoch}:{row_id}".encode("utf-8")
    row_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = list(range(len(option_ids)))
    random.Random(row_seed).shuffle(order)
    return order


def _encode_one(row: dict[str, Any], tokenizer: Tokenizer, training: bool, seed: int, epoch: int) -> tuple[list[int], list[list[int]], int]:
    question = row["question"]
    instruction = _field_tokens(tokenizer, "Instructions:\n" + question["instructions"], MAX_INSTRUCTION_TOKENS, "Instruction")
    state = _encode(tokenizer, "\nState:\n" + row["state_text"])
    option_parts = []
    for option in row["options"]:
        delimiter = "\nOption:\n"
        segment = delimiter + option["description"]
        encode_offsets = getattr(tokenizer, "encode_with_offsets", None)
        if callable(encode_offsets):
            segment_ids, offsets = encode_offsets(segment)
            if any(start < len(delimiter) < end for start, end in offsets):
                raise ValidationError("An option token crosses the delimiter/description boundary.", "/options")
            description_positions = [i for i, (start, end) in enumerate(offsets) if start >= len(delimiter) and end > start]
            description = [segment_ids[i] for i in description_positions]
            if len(description) > MAX_OPTION_TOKENS:
                raise ValidationError(f"Option description exceeds its {MAX_OPTION_TOKENS}-token limit.", "/options")
            if description_positions and description_positions != list(range(description_positions[0], len(segment_ids))):
                raise ValidationError("Tokenizer offsets do not form a contiguous description span.", "/options")
            description_start = description_positions[0] if description_positions else len(segment_ids)
        else:
            # Simple deterministic tokenizers used by fixtures can lack offset
            # support; production's pinned tokenizer always uses the branch above.
            delimiter_ids = _encode(tokenizer, delimiter)
            description = _field_tokens(tokenizer, option["description"], MAX_OPTION_TOKENS, "Option description")
            segment_ids = delimiter_ids + description
            description_start = len(delimiter_ids)
        option_parts.append((segment_ids, description_start, len(description)))
    if any(description_length == 0 for _, _, description_length in option_parts):
        raise ValidationError("Every option must encode to a nonempty token span.", "/options")

    # Template: CLS + instruction + state + descriptions + SEP. Each segment
    # already includes its textual delimiter, so only CLS/SEP are special IDs.
    fixed_length = 1 + len(instruction) + sum(len(segment_ids) for segment_ids, _, _ in option_parts) + 1
    if fixed_length > FIXED_TEMPLATE_LIMIT:
        raise ValidationError("Fixed template content exceeds 480 tokens.", "/")
    state_capacity = MAX_LENGTH - fixed_length
    if len(state) > state_capacity:
        first = (state_capacity + 1) // 2
        last = state_capacity // 2
        state = state[:first] + (state[-last:] if last else [])

    order = list(range(len(option_parts)))
    label = row["label"]
    if training:
        order = _stable_option_order([o["id"] for o in row["options"]], seed, epoch, row["id"])
    ids = [tokenizer.cls_token_id] + instruction + state
    spans: list[list[int]] = []
    for index in order:
        segment_ids, description_start, description_length = option_parts[index]
        segment_start = len(ids)
        ids.extend(segment_ids)
        start = segment_start + description_start
        end = start + description_length
        if start >= end:
            raise ValidationError("Every option span must be nonempty.", "/options")
        spans.append([start, end])
    ids.append(tokenizer.sep_token_id)
    if len(ids) > MAX_LENGTH:
        raise AssertionError("Tokenizer assembly exceeded its fixed maximum length.")
    label_idx = -1 if label == "ABSTAIN" else next(i for i, original in enumerate(order) if row["options"][original]["id"] == label)
    return ids, spans, label_idx


def encode_qwen(row: dict[str, Any], tokenizer: Tokenizer) -> tuple[list[list[int]], list[int]]:
    """Build independent causal inputs with a common prefix and final option span."""
    instruction = "Decision instructions:\n" + row["question"]["instructions"].strip()
    fixed_ids = _encode(tokenizer, instruction + "\n\nContext:\n")
    tail_ids = _encode(tokenizer, "\n\nCandidate:\n")
    options = [(option, _encode(tokenizer, option["description"])) for option in row["options"]]
    if any(not tokens for _, tokens in options):
        raise ValidationError("Qwen option description must encode to at least one token.", "/options")
    longest_option_length = max(len(tail_ids) + len(tokens) for _, tokens in options)
    capacity = MAX_LENGTH - len(fixed_ids) - longest_option_length - 8
    if capacity < 16:
        raise ValidationError("Qwen fixed instruction and option text leave less than 16 context tokens.", "/")
    state_ids = _encode(tokenizer, row["state_text"])
    if len(state_ids) > capacity:
        first, last = (capacity + 1) // 2, capacity // 2
        state_ids = state_ids[:first] + (state_ids[-last:] if last else [])
    sequences, spans = [], []
    for _, tokens in options:
        sequence = fixed_ids + state_ids + tail_ids + tokens
        sequences.append(sequence)
        spans.append(len(fixed_ids) + len(state_ids) + len(tail_ids))
    return sequences, spans


def encode(
    rows: Sequence[dict[str, Any]],
    training: bool,
    seed: int,
    *,
    epoch: int = 0,
    tokenizer: Tokenizer,
    architecture: str = "encoder",
) -> Batch:
    """Encode validated rows; training rows are deterministically option-shuffled.

    For inference, pass validated temporary rows with label ``ABSTAIN``. Their
    label index is always -1 and input option order is retained.
    """
    if not 1 <= len(rows) <= 20:
        raise ValueError("Batch size must be between 1 and 20.")
    rows = [validate_row(row) for row in rows]
    if architecture == "causal":
        if training:
            shuffled = []
            for row in rows:
                order = _stable_option_order([o["id"] for o in row["options"]], seed, epoch, row["id"])
                shuffled.append({**row, "options": [row["options"][i] for i in order]})
            rows = shuffled
        built_causal = [encode_qwen(row, tokenizer) for row in rows]
        max_options = max(len(row["options"]) for row in rows)
        sequences = [seqs + [[] for _ in range(max_options - len(seqs))] for seqs, _ in built_causal]
        starts = [pos + [-1 for _ in range(max_options - len(pos))] for _, pos in built_causal]
        labels = [(-1 if row["label"] == "ABSTAIN" else next(i for i, option in enumerate(row["options"]) if option["id"] == row["label"])) for row in rows]
        max_len = max(len(seq) for group in sequences for seq in group if seq)
        return Batch(
            input_ids=[[tokenizer.pad_token_id] * max_len for _ in rows],
            attention_mask=[[0] * max_len for _ in rows],
            option_spans=[[[start, start + 1] if start >= 0 else [-1, -1] for start in group] for group in starts],
            label_idx=labels,
            question_type=[row["question"]["type"] for row in rows],
            option_ids=[[o["id"] for o in row["options"]] for row in rows],
            architecture="causal", option_sequences=sequences, option_prefix_lengths=starts,
        )
    if architecture != "encoder":
        raise ValueError("architecture must be 'encoder' or 'causal'.")
    built = [_encode_one(row, tokenizer, training, seed, epoch) for row in rows]
    max_length = max(len(item[0]) for item in built)
    max_options = max(len(row["options"]) for row in rows)
    pad_id = tokenizer.pad_token_id
    input_ids = [ids + [pad_id] * (max_length - len(ids)) for ids, _, _ in built]
    attention = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids, _, _ in built]
    spans: list[list[list[int]]] = []
    option_ids: list[list[str]] = []
    for row, (_, row_spans, _) in zip(rows, built):
        order = list(range(len(row["options"])))
        if training:
            order = _stable_option_order([o["id"] for o in row["options"]], seed, epoch, row["id"])
        spans.append(row_spans + [[-1, -1] for _ in range(max_options - len(row_spans))])
        option_ids.append([row["options"][i]["id"] for i in order])
    return Batch(
        input_ids=input_ids,
        attention_mask=attention,
        option_spans=spans,
        label_idx=[item[2] for item in built],
        question_type=[row["question"]["type"] for row in rows],
        option_ids=option_ids,
    )


def load_tokenizer(snapshot_path: str | Path) -> Tokenizer:
    """Load a verified local snapshot; mutable/unrecorded artifacts are refused.

    Uses the pinned low-level tokenizer format without importing Transformers.
    Hashes are recorded in a sibling ``tokenizer.lock.json`` manifest.
    """
    path = Path(snapshot_path)
    config_path = path / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")).get("model_type") == "qwen3":
        from s1_compass.models.registry import get_base_model
        from s1_compass.models.snapshot import verify_snapshot
        verify_snapshot(get_base_model("qwen3-0.6b"), path)
        lock = json.loads((path / "s1-compass-snapshot.lock.json").read_text(encoding="utf-8"))
        tokenizer_path = path / "tokenizer.json"
        if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != lock["tokenizer_sha256"]:
            raise ValueError("Qwen tokenizer hash does not match immutable snapshot lock.")
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the pinned tokenizers dependency to load Qwen tokenizer.") from exc
        raw = HFTokenizer.from_file(str(tokenizer_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer_config_path = path / "tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        return _QwenTokenizer(raw, config, tokenizer_config)
    lock_path = path / "tokenizer.lock.json"
    if not lock_path.is_file():
        raise ValueError(f"Pinned tokenizer manifest is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {"model_id", "revision", "tokenizer_sha256", "config_sha256", "weights_sha256"}
    if set(lock) != required or lock["model_id"] != MODEL_ID or not all(
        isinstance(lock[key], str) and len(lock[key]) == 64 and all(c in "0123456789abcdef" for c in lock[key])
        for key in ("tokenizer_sha256", "config_sha256", "weights_sha256")
    ) or not isinstance(lock["revision"], str) or len(lock["revision"]) != 40 or any(c not in "0123456789abcdef" for c in lock["revision"]):
        raise ValueError("Tokenizer lock must contain the exact model ID, immutable 40-character revision, and three SHA-256 hashes.")
    if (BACKBONE_REVISION is not None and lock["revision"] != BACKBONE_REVISION
            or TOKENIZER_SHA256 is not None and lock["tokenizer_sha256"] != TOKENIZER_SHA256
            or CONFIG_SHA256 is not None and lock["config_sha256"] != CONFIG_SHA256
            or WEIGHTS_SHA256 is not None and lock["weights_sha256"] != WEIGHTS_SHA256):
        raise ValueError("Tokenizer lock does not match this project's pinned ModernBERT snapshot.")
    tokenizer_file = path / "tokenizer.json"
    config_file = path / "config.json"
    if not tokenizer_file.is_file() or hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != lock["tokenizer_sha256"]:
        raise ValueError(f"Pinned artifact hash mismatch: {tokenizer_file}")
    if not config_file.is_file() or hashlib.sha256(config_file.read_bytes()).hexdigest() != lock["config_sha256"]:
        raise ValueError(f"Pinned artifact hash mismatch: {config_file}")
    try:
        from tokenizers import Tokenizer as HFTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the pinned tokenizers dependency to load ModernBERT tokenizer.") from exc
    config = json.loads(config_file.read_text(encoding="utf-8"))
    tokenizer_config_path = path / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        raise ValueError(f"Tokenizer configuration is missing: {tokenizer_config_path}")
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    raw = HFTokenizer.from_file(str(tokenizer_file))
    return _PinnedTokenizer(raw, config, tokenizer_config)
