"""Deterministic, group-isolated dataset partitioning."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from s1_compass.data.schema import ValidationError, strict_dumps, strict_loads, validate_row
from s1_compass.data.tokenize import (
    MAX_INSTRUCTION_TOKENS,
    MAX_LENGTH,
    MAX_OPTION_TOKENS,
    Tokenizer,
    _stable_option_order,
    encode,
)

COHORTS = {"normal", "ambiguous", "adversarial", "long_context", "ood"}
SPLITS = ("train", "validation", "calibration", "test_normal")
SPLIT_FRACTIONS = (0.50, 0.10, 0.20, 0.20)
DEFAULT_MINIMUMS = {
    "train": 10_000,
    "validation": 2_000,
    "calibration": 5_000,
    "test_normal": 5_000,
    "other_cohort": 500,
    "answerable_adversarial": 200,
    "test_normal_per_primitive": 1_000,
    "train_per_question_option": 50,
    "permutations": 5_000,
}


@dataclass(frozen=True)
class Manifest:
    data: dict[str, Any]
    path: Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_context(text: str) -> str:
    """NFKC normalize and collapse all Unicode whitespace to one ASCII space."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    seen = set()
    with path.open("rb") as f:
        raw_bytes = f.read()
    for line_no, line in enumerate(raw_bytes.splitlines(), 1):
        if not line.strip():
            raise ValidationError("Blank JSONL records are not allowed.", f"line:{line_no}")
        try:
            row = validate_row(strict_loads(line))
        except ValidationError as exc:
            raise ValidationError(exc.message, f"line:{line_no}{exc.path}") from exc
        if row["id"] in seen:
            raise ValidationError("Row IDs must be globally unique.", f"line:{line_no}/id")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValidationError("Input JSONL must contain at least one row.", "/")
    return rows, _sha256_bytes(raw_bytes)


def _read_annotations(path: Path, row_ids: set[str]) -> tuple[dict[str, dict[str, str]], str]:
    raw = path.read_bytes()
    parsed = strict_loads(raw)
    if not isinstance(parsed, dict):
        raise ValidationError("Annotation mapping must be a JSON object keyed by row ID.", "/")
    if set(parsed) != row_ids:
        missing = sorted(row_ids - parsed.keys())
        extra = sorted(parsed.keys() - row_ids)
        if missing:
            raise ValidationError(f"Missing annotations for row {missing[0]}.", "/")
        raise ValidationError(f"Annotation references unknown row {extra[0]}.", "/")
    result = {}
    for row_id, value in parsed.items():
        p = "/" + row_id
        if not isinstance(value, dict) or not {"group_id", "cohort"} <= set(value) or set(value) - {"group_id", "cohort", "evidence"}:
            raise ValidationError("Annotation must contain group_id and cohort, with optional evidence.", p)
        group_id, cohort = value["group_id"], value["cohort"]
        if not isinstance(group_id, str) or not group_id:
            raise ValidationError("group_id must be a nonempty string.", p + "/group_id")
        if cohort not in COHORTS:
            raise ValidationError("Unknown cohort.", p + "/cohort")
        result[row_id] = {"group_id": group_id, "cohort": cohort}
        if "evidence" in value:
            evidence = value["evidence"]
            if not isinstance(evidence, str) or not evidence:
                raise ValidationError("evidence must be a nonempty string.", p + "/evidence")
            result[row_id]["evidence"] = evidence
    return result, _sha256_bytes(raw)


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {v: v for v in values}
        self.rank = {v: 0 for v in values}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _rows_digest(rows: Sequence[dict[str, Any]]) -> str:
    body = "".join(strict_dumps(row) + "\n" for row in rows).encode("utf-8")
    return _sha256_bytes(body)


def _split_groups(groups: Mapping[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """Assign whole normal groups by deterministic hash order and row targets."""
    ordered = sorted(groups, key=lambda group_id: (_sha256_bytes(("17:" + group_id).encode()), group_id))
    total = sum(len(groups[group]) for group in ordered)
    targets = [total * fraction for fraction in SPLIT_FRACTIONS]
    assigned = {split: [] for split in SPLITS}
    counts = [0, 0, 0, 0]
    split_index = 0
    for group_id in ordered:
        # Move forward when the current split target is reached. Keep a group
        # whole, even when it makes a split exceed its target by that group.
        while split_index < 3 and counts[split_index] >= targets[split_index]:
            split_index += 1
        assigned[SPLITS[split_index]].append(group_id)
        counts[split_index] += len(groups[group_id])
    return assigned


def _validate_cohort_counts(
    split_rows: Mapping[str, Sequence[dict[str, Any]]],
    cohort_rows: Mapping[str, Sequence[dict[str, Any]]],
    minimums: Mapping[str, int],
    small_data: bool,
    tokenizer: Optional[Tokenizer],
    evidence_by_row: Mapping[str, str],
    architecture: str = "encoder",
) -> None:
    if any(r["label"] != "ABSTAIN" for c in ("ambiguous", "ood") for r in cohort_rows[c]):
        raise ValidationError("Ambiguous and OOD rows must have ABSTAIN labels.", "/")
    if not small_data:
        for split in SPLITS:
            n = sum(row["label"] != "ABSTAIN" for row in split_rows[split])
            if n < minimums[split]:
                raise ValidationError(f"{split} requires at least {minimums[split]} answerable rows; found {n}.", "/")
        for cohort in ("ambiguous", "adversarial", "long_context", "ood"):
            if len(cohort_rows[cohort]) < minimums["other_cohort"]:
                raise ValidationError(f"{cohort} requires at least {minimums['other_cohort']} rows.", "/")
        adversarial_answerable = sum(r["label"] != "ABSTAIN" for r in cohort_rows["adversarial"])
        if adversarial_answerable < minimums["answerable_adversarial"]:
            raise ValidationError("Insufficient answerable adversarial rows.", "/")
    if not cohort_rows["long_context"]:
        return
    if tokenizer is None:
        if small_data:
            return
        raise ValidationError("Tokenizer required to verify retained long-context evidence.", "/")
    for row in cohort_rows["long_context"]:
        if row["label"] == "ABSTAIN":
            continue
        batch = encode([row], False, 17, tokenizer=tokenizer, architecture=architecture)
        evidence = evidence_by_row.get(row["id"])
        if not isinstance(evidence, str) or not evidence:
            raise ValidationError("Long-context answerable rows need an evidence string.", "/evidence")
        # Evidence is expected to be an exact retained substring. Tokenizing the
        # retained state region separately lets us detect evidence lost at truncation.
        evidence_ids = tokenizer.encode(evidence, add_special_tokens=False)
        if architecture == "causal":
            from s1_compass.data.tokenize import encode_qwen
            sequences, starts = encode_qwen(row, tokenizer)
            fixed_ids=tokenizer.encode("Decision instructions:\n"+row["question"]["instructions"].strip()+"\n\nContext:\n",add_special_tokens=False)
            tail_ids=tokenizer.encode("\n\nCandidate:\n",add_special_tokens=False)
            option_ids=tokenizer.encode(row["options"][0]["description"],add_special_tokens=False)
            state_end=starts[0]-len(tail_ids)-len(option_ids)
            retained_state_ids=sequences[0][len(fixed_ids):state_end]
        else:
            full_state_ids = tokenizer.encode("\nState:\n" + row["state_text"], add_special_tokens=False)
            instruction_ids = tokenizer.encode("Instructions:\n" + row["question"]["instructions"], add_special_tokens=False)
            option_fixed = sum(
                len(tokenizer.encode("\nOption:\n", add_special_tokens=False))
                + len(tokenizer.encode(option["description"], add_special_tokens=False))
                for option in row["options"]
            )
            state_capacity = MAX_LENGTH - (2 + len(instruction_ids) + option_fixed)
            if len(full_state_ids) > state_capacity:
                first, last = (state_capacity + 1) // 2, state_capacity // 2
                retained_state_ids = full_state_ids[:first] + (full_state_ids[-last:] if last else [])
            else:
                retained_state_ids = full_state_ids
        if not evidence_ids or not any(retained_state_ids[i:i + len(evidence_ids)] == evidence_ids for i in range(len(retained_state_ids) - len(evidence_ids) + 1)):
            raise ValidationError("Long-context evidence is not present in the encoded state.", "/evidence")
        valid_length=max(len(sequence) for sequence in batch.option_sequences[0]) if architecture=="causal" else sum(batch.attention_mask[0])
        if valid_length < 450:
            raise ValidationError("Long-context rows must encode to at least 450 tokens.", "/")


def partition(
    rows_path: str | Path,
    annotations_path: str | Path,
    output_dir: str | Path,
    seed: int = 17,
    *,
    tokenizer: Optional[Tokenizer] = None,
    small_data: bool = False,
    minimums: Optional[Mapping[str, int]] = None,
    permutation_count: Optional[int] = None,
    source_identity: Optional[str] = None,
    preprocessing_hashes: Optional[Mapping[str, str]] = None,
    base_model_id: str = "modernbert-base",
    preprocessing_version: str = "modernbert-v1",
    snapshot_lock_sha256: Optional[str] = None,
    architecture: str = "encoder",
) -> Manifest:
    """Partition annotated DecisionRows and atomically write immutable artifacts.

    ``small_data=True`` disables only corpus-size gates, not data integrity,
    split isolation, cohort policy, or determinism checks.
    """
    rows_path, annotations_path, output_dir = Path(rows_path), Path(annotations_path), Path(output_dir)
    rows, row_file_hash = _read_rows(rows_path)
    annotations, annotations_hash = _read_annotations(annotations_path, {r["id"] for r in rows})
    minimums = {**DEFAULT_MINIMUMS, **(minimums or {})}
    n_permutations = minimums["permutations"] if permutation_count is None else permutation_count
    if n_permutations < 0:
        raise ValueError("permutation_count must be nonnegative")

    # First connect explicit related groups, then union normalized identical
    # states across group IDs so near-duplicate context never crosses splits.
    uf = _UnionFind(r["id"] for r in rows)
    first_by_group: dict[str, str] = {}
    first_by_context: dict[str, str] = {}
    for row in rows:
        ann = annotations[row["id"]]
        if ann["group_id"] in first_by_group:
            uf.union(row["id"], first_by_group[ann["group_id"]])
        else:
            first_by_group[ann["group_id"]] = row["id"]
        context_key = _sha256_bytes(normalized_context(row["state_text"]).encode("utf-8"))
        if context_key in first_by_context:
            uf.union(row["id"], first_by_context[context_key])
        else:
            first_by_context[context_key] = row["id"]

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_cohorts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        root = uf.find(row["id"])
        components[root].append(row)
        component_cohorts[root].add(annotations[row["id"]]["cohort"])
    if any(len(cohorts) != 1 for cohorts in component_cohorts.values()):
        raise ValidationError("Related/duplicate context components cannot mix cohorts.", "/annotations")

    # Use a stable component identity derived from member group IDs. This is
    # persisted so split assignments can be audited and reproduced.
    comp_info: dict[str, dict[str, Any]] = {}
    for root, members in components.items():
        group_ids = sorted({annotations[row["id"]]["group_id"] for row in members})
        cohort = next(iter(component_cohorts[root]))
        group_key = "\0".join(group_ids)
        comp_id = _sha256_bytes(group_key.encode("utf-8"))
        comp_info[root] = {"members": members, "cohort": cohort, "component_id": comp_id, "group_ids": group_ids}

    normal_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for info in comp_info.values():
        if info["cohort"] == "normal":
            normal_groups[info["component_id"]].extend(info["members"])
    assignment = _split_groups(normal_groups)
    split_ids: dict[str, list[str]] = {split: [] for split in SPLITS}
    for split, group_ids in assignment.items():
        for group_id in group_ids:
            split_ids[split].extend(row["id"] for row in normal_groups[group_id])
    split_map = {row_id: split for split, ids in split_ids.items() for row_id in ids}

    cohort_rows: dict[str, list[dict[str, Any]]] = {cohort: [] for cohort in COHORTS}
    for info in comp_info.values():
        cohort_rows[info["cohort"]].extend(info["members"])
    _validate_cohort_counts(
        {s: [r for r in rows if split_map.get(r["id"]) == s] for s in SPLITS},
        cohort_rows, minimums, small_data, tokenizer,
        {row_id: ann["evidence"] for row_id, ann in annotations.items() if "evidence" in ann},
        architecture,
    )

    # Non-normal cohorts remain held out with explicit cohort filenames; they
    # never enter train/validation/calibration/test_normal.
    artifact_rows: dict[str, list[dict[str, Any]]] = {
        split: [r for r in rows if split_map.get(r["id"]) == split and r["label"] != "ABSTAIN"]
        for split in SPLITS
    }
    for cohort in ("ambiguous", "adversarial", "long_context", "ood"):
        artifact_rows[f"test_{cohort}"] = sorted(cohort_rows[cohort], key=lambda r: r["id"])
    exclusions = {
        split: [r["id"] for r in rows if split_map.get(r["id"]) == split and r["label"] == "ABSTAIN"]
        for split in SPLITS
    }

    # Evaluate primitive and label coverage over answerable rows.
    normal_test = artifact_rows["test_normal"]
    primitive_counts = Counter(r["question"]["id"] for r in normal_test)
    if not small_data:
        for primitive in {r["question"]["id"] for r in rows if annotations[r["id"]]["cohort"] == "normal"}:
            if primitive_counts[primitive] < minimums["test_normal_per_primitive"]:
                raise ValidationError(f"test_normal primitive {primitive} has insufficient rows.", "/")
    train_pairs = Counter((r["question"]["id"], o["id"]) for r in artifact_rows["train"] for o in r["options"])
    evaluated_pairs = {(r["question"]["id"], o["id"]) for r in normal_test for o in r["options"]}
    if not small_data:
        for pair in evaluated_pairs:
            if train_pairs[pair] < minimums["train_per_question_option"]:
                raise ValidationError(f"Training coverage below minimum for {pair}.", "/")

    # Paired permutations contain row ID first in each JSONL record and a
    # deterministic permutation index, allowing probabilities to align later.
    permutation_rows = []
    permutation_sources = sorted(normal_test, key=lambda r: r["id"])
    for pair_index in range(n_permutations):
        if not permutation_sources:
            break
        row = permutation_sources[pair_index % len(permutation_sources)]
        option_ids = [o["id"] for o in row["options"]]
        order = _stable_option_order(option_ids, seed, pair_index, row["id"])
        # Rotate unchanged permutations deterministically so every pair tests
        # an actual order change while preserving the original row association.
        if len(order) > 1 and order == list(range(len(order))):
            offset = (pair_index % (len(order) - 1)) + 1
            order = order[offset:] + order[:offset]
        digest = _sha256_bytes(f"{seed}:{row['id']}:{pair_index}".encode("utf-8"))
        permutation_rows.append({"row_id": row["id"], "pair_id": digest, "options": [option_ids[i] for i in order], "label": row["label"]})

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, split_data in artifact_rows.items():
        path = output_dir / f"{name}.jsonl"
        body = "".join(strict_dumps(row) + "\n" for row in split_data).encode("utf-8")
        _atomic_write(path, body)
        files[path.name] = {"sha256": _sha256_bytes(body), "rows": len(split_data)}
    permutation_path = output_dir / "test_option_permuted.jsonl"
    perm_body = "".join(strict_dumps(item) + "\n" for item in permutation_rows).encode("utf-8")
    _atomic_write(permutation_path, perm_body)
    files[permutation_path.name] = {"sha256": _sha256_bytes(perm_body), "rows": len(permutation_rows)}

    groups_manifest = []
    for root, info in sorted(comp_info.items(), key=lambda item: item[1]["component_id"]):
        groups_manifest.append({"component_id": info["component_id"], "group_ids": info["group_ids"],
                                "cohort": info["cohort"], "row_ids": sorted(r["id"] for r in info["members"]),
                                "split": split_map.get(info["members"][0]["id"], "test_" + info["cohort"])})
    manifest_data = {
        "format_version": 1, "seed": seed, "source_identity": source_identity or str(rows_path.name),
        "base_model_id": base_model_id, "preprocessing_version": preprocessing_version,
        "snapshot_lock_sha256": snapshot_lock_sha256, "architecture": architecture,
        "source_sha256": row_file_hash, "annotations_sha256": annotations_hash,
        "preprocessing_hashes": dict(sorted((preprocessing_hashes or {}).items())),
        "files": dict(sorted(files.items())), "groups": groups_manifest,
        "exclusions": exclusions, "permutation_count": len(permutation_rows),
        "counts": {name: value["rows"] for name, value in files.items()},
        "minimums": dict(sorted(minimums.items())), "small_data_mode": small_data,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write(manifest_path, (json.dumps(manifest_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    return Manifest(manifest_data, manifest_path)


def batches(path: str | Path, batch_size: int, seed: int = 17, *, tokenizer: Tokenizer, training: bool = False, epoch: int = 0, architecture: str = "encoder") -> Iterator[Any]:
    """Yield deterministic tokenized batches from a JSONL split file."""
    if not 1 <= batch_size <= 20:
        raise ValueError("batch_size must be between 1 and 20")
    rows, _ = _read_rows(Path(path))
    if training:
        rows.sort(key=lambda row: _sha256_bytes(f"{seed}:{epoch}:{row['id']}".encode()))
    else:
        rows.sort(key=lambda row: row["id"])
    for start in range(0, len(rows), batch_size):
        yield encode(rows[start:start + batch_size], training, seed, epoch=epoch, tokenizer=tokenizer, architecture=architecture)
