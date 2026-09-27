"""Download and verify immutable Hugging Face snapshots for known base models."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import struct
from pathlib import Path
from typing import Any

from s1_compass.models.registry import BaseModelSpec


COMMIT = re.compile(r"^[a-f0-9]{40}$")
LOCK_NAME = "s1-compass-snapshot.lock.json"
MODERNBERT_REQUIRED_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
QWEN3_REQUIRED_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _snapshot_files(root: Path) -> list[Path]:
    resolved_root = root.resolve()
    return sorted(
        (path for path in root.rglob("*")
         if path.is_file() and ".cache" not in path.relative_to(root).parts
         and path.resolve().is_relative_to(resolved_root)
         and path.name != LOCK_NAME and not path.name.endswith(".lock")),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _safetensors_headers(snapshot: Path, names: set[str]) -> dict[str, tuple[str, tuple[int, ...]]]:
    headers={}
    for name in sorted(value for value in names if value.endswith(".safetensors")):
        path=snapshot/name
        with path.open("rb") as stream:
            raw=stream.read(8)
            if len(raw)!=8: raise ValueError(f"Malformed safetensors header in {name}.")
            size=struct.unpack("<Q",raw)[0]
            if size>128*1024*1024: raise ValueError(f"Unreasonable safetensors header size in {name}.")
            header=json.loads(stream.read(size))
        for key,value in header.items():
            if key=="__metadata__": continue
            if key in headers: raise ValueError(f"Duplicate checkpoint tensor: {key}")
            headers[key]=(value.get("dtype"),tuple(value.get("shape",())))
    return headers


def create_lock(spec: BaseModelSpec, snapshot: Path, revision: str) -> dict[str, Any]:
    files = _snapshot_files(snapshot)
    names = {path.relative_to(snapshot).as_posix() for path in files}
    required_files = QWEN3_REQUIRED_FILES if spec.architecture == "causal" else MODERNBERT_REQUIRED_FILES
    missing = required_files - names
    has_weights = any(name.endswith((".safetensors", ".safetensors.index.json")) for name in names)
    if missing or not has_weights:
        missing_text = ", ".join(sorted(missing)) or "model safetensors"
        raise ValueError(f"Snapshot is incomplete; missing {missing_text}.")
    entries = [
        {"path": path.relative_to(snapshot).as_posix(), "sha256": _sha256(path), "size": path.stat().st_size}
        for path in files
    ]
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != ("qwen3" if spec.architecture == "causal" else "modernbert"):
        raise ValueError(f"Snapshot architecture does not match {spec.model_id}.")
    hidden_size = config.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size != spec.hidden_size:
        raise ValueError(f"Snapshot hidden size {hidden_size!r} does not match registry value {spec.hidden_size!r}.")
    if spec.architecture == "causal":
        if config.get("architectures") != ["Qwen3ForCausalLM"] or config.get("tie_word_embeddings") is not True:
            raise ValueError("Qwen3 snapshot must be the expected tied-embedding causal LM architecture.")
        if config.get("num_hidden_layers") != 28 or config.get("num_attention_heads") != 16 or config.get("num_key_value_heads") != 8:
            raise ValueError("Qwen3 snapshot attention configuration does not match the registered model.")
        tokenizer_config=json.loads((snapshot/"tokenizer_config.json").read_text(encoding="utf-8"))
        from tokenizers import Tokenizer
        raw_tokenizer=Tokenizer.from_file(str(snapshot/"tokenizer.json"))
        pad_token=tokenizer_config.get("pad_token")
        if isinstance(pad_token,dict): pad_token=pad_token.get("content")
        pad_id=raw_tokenizer.token_to_id(pad_token) if isinstance(pad_token,str) else None
        if (config.get("bos_token_id")!=151643 or config.get("eos_token_id")!=151645 or pad_id!=151643
                or raw_tokenizer.id_to_token(151643)!="<|endoftext|>" or raw_tokenizer.id_to_token(151645)!="<|im_end|>"):
            raise ValueError("Qwen3 bos/eos/pad token IDs do not match the pinned tokenizer contract.")
        headers=_safetensors_headers(snapshot,names)
        expected={"model.embed_tokens.weight":("BF16",(151936,1024)),"model.norm.weight":("BF16",(1024,)),"lm_head.weight":("BF16",(151936,1024))}
        for layer in range(28):
            prefix=f"model.layers.{layer}."
            expected.update({
                prefix+"input_layernorm.weight":("BF16",(1024,)),prefix+"post_attention_layernorm.weight":("BF16",(1024,)),
                prefix+"self_attn.q_proj.weight":("BF16",(2048,1024)),prefix+"self_attn.k_proj.weight":("BF16",(1024,1024)),
                prefix+"self_attn.v_proj.weight":("BF16",(1024,1024)),prefix+"self_attn.o_proj.weight":("BF16",(1024,2048)),
                prefix+"self_attn.q_norm.weight":("BF16",(128,)),prefix+"self_attn.k_norm.weight":("BF16",(128,)),
                prefix+"mlp.gate_proj.weight":("BF16",(3072,1024)),prefix+"mlp.up_proj.weight":("BF16",(3072,1024)),
                prefix+"mlp.down_proj.weight":("BF16",(1024,3072)),
            })
        if headers!=expected:
            missing=sorted(expected.keys()-headers.keys())[:5]
            extra=sorted(headers.keys()-expected.keys())[:5]
            mismatched=[key for key in expected.keys()&headers.keys() if expected[key]!=headers[key]][:5]
            raise ValueError(f"Qwen3 checkpoint tensor manifest mismatch; missing={missing}, extra={extra}, shape_or_dtype={mismatched}.")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    weight_files = [entry for entry in entries if entry["path"].endswith((".safetensors", ".safetensors.index.json"))]
    return {
        "format_version": 1,
        "model_id": spec.model_id,
        "selector": spec.selector,
        "repository": spec.repository,
        "revision": revision,
        "architecture": spec.architecture,
        "hidden_size": hidden_size,
        "config_sha256": _sha256(snapshot / "config.json"),
        "files": entries,
        "files_sha256": hashlib.sha256(canonical).hexdigest(),
        "weights_sha256": hashlib.sha256(json.dumps(weight_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "tokenizer_sha256": _sha256(snapshot / "tokenizer.json"),
        "environment": {
            "python": __import__("sys").version.split()[0],
            "huggingface_hub": _version("huggingface-hub"),
            "tokenizers": _version("tokenizers"),
            "mlx": _version("mlx"),
        },
    }


def verify_snapshot(spec: BaseModelSpec, snapshot: Path | None = None) -> dict[str, Any]:
    root = (snapshot or spec.snapshot_path).resolve()
    lock_path = root / LOCK_NAME
    if not lock_path.is_file():
        raise ValueError(f"Pinned S1 Compass snapshot lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("format_version") != 1 or lock.get("model_id") != spec.model_id
            or lock.get("selector") != spec.selector or lock.get("repository") != spec.repository
            or lock.get("architecture") != spec.architecture):
        raise ValueError("Snapshot lock identifies a different base model or architecture.")
    if not isinstance(lock.get("revision"), str) or not COMMIT.fullmatch(lock["revision"]):
        raise ValueError("Snapshot revision must be an immutable 40-character commit.")
    for entry in lock.get("files", []):
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Snapshot lock contains an unsafe path.")
        path = root / rel
        if not path.resolve().is_relative_to(root):
            raise ValueError("Snapshot lock path resolves outside the snapshot directory.")
        if not path.is_file() or _sha256(path) != entry["sha256"] or path.stat().st_size != entry["size"]:
            raise ValueError(f"Snapshot file is missing or changed: {rel}")
    if not lock.get("files"):
        raise ValueError("Snapshot lock has no file entries.")
    current = create_lock(spec, root, lock["revision"])
    checked_keys=["config_sha256","files_sha256","hidden_size","model_id","architecture","repository","revision"]
    if spec.architecture=="causal": checked_keys.extend(["weights_sha256","tokenizer_sha256"])
    if any(current.get(key) != lock.get(key) for key in checked_keys):
        raise ValueError("Snapshot file manifest hash does not match lock.")
    return lock


def lock_existing_snapshot(spec: BaseModelSpec, revision: str, snapshot: Path | None = None) -> Path:
    """Hash an already-downloaded snapshot without replacing any model files."""
    if not COMMIT.fullmatch(revision):
        raise ValueError("Pass an immutable 40-character lowercase Hugging Face commit revision.")
    root = (snapshot or spec.snapshot_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Snapshot directory does not exist: {root}")
    lock_path = root / LOCK_NAME
    if lock_path.exists():
        raise FileExistsError(f"Snapshot is already locked: {lock_path}")
    lock = create_lock(spec, root, revision)
    temporary = lock_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, lock_path)
    return lock_path


def download_snapshot(spec: BaseModelSpec, revision: str, destination: Path | None = None) -> Path:
    """Download one immutable revision into an empty directory and lock its files."""
    if not COMMIT.fullmatch(revision):
        raise ValueError("Pass an immutable 40-character lowercase Hugging Face commit revision.")
    target = (destination or spec.snapshot_path).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to replace nonempty snapshot directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-download-", dir=target.parent))
    try:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install project dependencies before downloading a model snapshot.") from exc
        snapshot_download(
            repo_id=spec.repository,
            revision=revision,
            local_dir=str(staging),
            local_files_only=False,
        )
        lock = create_lock(spec, staging, revision)
        lock_path = staging / LOCK_NAME
        lock_tmp = lock_path.with_suffix(".tmp")
        lock_tmp.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(lock_tmp, lock_path)
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
