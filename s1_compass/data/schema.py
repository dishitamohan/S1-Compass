"""Strict JSON and semantic validation for the S1 Compass Decision Studio contracts.

The validators deliberately use only the standard library so malformed input is
rejected consistently even before optional JSON Schema tooling is installed.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Mapping, Optional

SCHEMA_BASE = "https://s1-compass.local/schemas/"
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
INTEGER_ID = re.compile(r"^(0|-?[1-9][0-9]*)$")
PUBLIC_MODEL_PATTERN = re.compile(r"^s1-compass-(?:modernbert-base|qwen3-0\.6b):[a-f0-9]{64}$")


class ValidationError(ValueError):
    """A contract violation, with a JSON Pointer identifying the bad value."""

    def __init__(self, message: str, path: str = "") -> None:
        self.message = message
        self.path = path or "/"
        super().__init__(f"{self.path}: {message}")


def _fail(message: str, path: str) -> None:
    raise ValidationError(message, path)


def _object(value: Any, path: str, required: set[str], allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail("Expected an object.", path)
    missing = required - value.keys()
    if missing:
        _fail(f"Missing required property: {sorted(missing)[0]}.", path)
    extra = value.keys() - allowed
    if extra:
        name = sorted(extra)[0]
        _fail(f"Unknown property: {name}.", f"{path}/{name}")
    return value


def _string(value: Any, path: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        _fail(f"Expected a string of length {minimum}..{maximum}.", path)
    return value


def _finite_number(value: Any, path: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("Expected a finite number.", path)
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        _fail(f"Expected a finite number in [{minimum}, {maximum}].", path)
    return number


def _validate_options(options: Any, path: str, question_type: str) -> list[dict[str, str]]:
    if not isinstance(options, list) or not 2 <= len(options) <= 8:
        _fail("Expected an array containing 2..8 options.", path)
    validated = []
    ids = set()
    for i, option in enumerate(options):
        p = f"{path}/{i}"
        option = _object(option, p, {"id", "description"}, {"id", "description"})
        oid = _string(option["id"], f"{p}/id", 1, 64)
        if oid == "ABSTAIN":
            _fail("ABSTAIN is reserved and cannot be an option ID.", f"{p}/id")
        if oid in ids:
            _fail("Option IDs must be unique.", f"{p}/id")
        ids.add(oid)
        description = _string(option["description"], f"{p}/description", 1, 2048)
        validated.append({"id": oid, "description": description})
    if question_type == "boolean" and ids != {"false", "true"}:
        _fail('Boolean options must have IDs "false" and "true" exactly once.', path)
    if question_type == "ordinal":
        values = []
        for i, option in enumerate(validated):
            oid = option["id"]
            if not INTEGER_ID.fullmatch(oid):
                _fail("Ordinal option IDs must be canonical decimal integers.", f"{path}/{i}/id")
            value = int(oid)
            if not INT32_MIN <= value <= INT32_MAX:
                _fail("Ordinal option ID must fit signed int32.", f"{path}/{i}/id")
            values.append(value)
        if sorted(values) != list(range(min(values), max(values) + 1)):
            _fail("Ordinal option IDs must be contiguous.", path)
    return validated


def validate_row(value: Any) -> dict[str, Any]:
    row = _object(value, "", {"id", "state_text", "question", "options", "label", "annotator_confidence", "source"},
                  {"id", "state_text", "question", "options", "label", "annotator_confidence", "source"})
    row_id = _string(row["id"], "/id", 1, 128)
    state = _string(row["state_text"], "/state_text", 1, 65536)
    question = _object(row["question"], "/question", {"id", "type", "instructions"}, {"id", "type", "instructions"})
    qid = _string(question["id"], "/question/id", 1, 128)
    qtype = question["type"]
    if qtype not in ("choice", "boolean", "ordinal"):
        _fail('Expected one of "choice", "boolean", or "ordinal".', "/question/type")
    instructions = _string(question["instructions"], "/question/instructions", 1, 4096)
    options = _validate_options(row["options"], "/options", qtype)
    label = _string(row["label"], "/label", 1, 64)
    if label != "ABSTAIN" and label not in {o["id"] for o in options}:
        _fail("Label must match an option ID or ABSTAIN.", "/label")
    confidence = _finite_number(row["annotator_confidence"], "/annotator_confidence", 0, 1)
    source = row["source"]
    if source not in ("human_review", "dataset_name", "adjudicated"):
        _fail("Unknown source.", "/source")
    return {"id": row_id, "state_text": state,
            "question": {"id": qid, "type": qtype, "instructions": instructions},
            "options": options, "label": label,
            "annotator_confidence": confidence, "source": source}


def validate_request(value: Any) -> dict[str, Any]:
    request = _object(value, "", {"state_text", "questions"}, {"state_text", "questions"})
    state = _string(request["state_text"], "/state_text", 1, 65536)
    questions = request["questions"]
    if not isinstance(questions, list) or not 1 <= len(questions) <= 20:
        _fail("Expected 1..20 questions.", "/questions")
    ids = set()
    result = []
    for i, item in enumerate(questions):
        p = f"/questions/{i}"
        q = _object(item, p, {"id", "type", "instructions", "options"}, {"id", "type", "instructions", "options"})
        qid = _string(q["id"], f"{p}/id", 1, 128)
        if qid in ids:
            _fail("Question IDs must be unique.", f"{p}/id")
        ids.add(qid)
        qtype = q["type"]
        if qtype not in ("choice", "boolean", "ordinal"):
            _fail('Expected one of "choice", "boolean", or "ordinal".', f"{p}/type")
        instructions = _string(q["instructions"], f"{p}/instructions", 1, 4096)
        options = _validate_options(q["options"], f"{p}/options", qtype)
        result.append({"id": qid, "type": qtype, "instructions": instructions, "options": options})
    return {"state_text": state, "questions": result}


def validate_output(value: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    req = validate_request(request)
    out = _object(value, "", {"answers", "model", "calibration_version", "latency_ms"},
                  {"answers", "model", "calibration_version", "latency_ms"})
    if not isinstance(out["answers"], dict) or set(out["answers"]) != {q["id"] for q in req["questions"]}:
        _fail("Answer IDs must match the request question IDs exactly.", "/answers")
    if not isinstance(out["model"], str) or not PUBLIC_MODEL_PATTERN.fullmatch(out["model"]):
        _fail("Invalid model identifier.", "/model")
    if not isinstance(out["calibration_version"], str) or not re.fullmatch(r"[a-f0-9]{64}", out["calibration_version"]):
        _fail("Invalid calibration version.", "/calibration_version")
    if isinstance(out["latency_ms"], bool) or not isinstance(out["latency_ms"], int) or out["latency_ms"] < 0:
        _fail("Latency must be a nonnegative integer.", "/latency_ms")
    for i, q in enumerate(req["questions"]):
        p = f"/answers/{q['id']}"
        answer = _object(out["answers"][q["id"]], p, {"type"}, {"type", "choice", "probabilities", "confidence", "value", "probability", "level"})
        if answer["type"] != q["type"]:
            _fail("Answer type must match request type.", f"{p}/type")
        ids = [o["id"] for o in q["options"]]
        if q["type"] == "boolean":
            if set(answer) != {"type", "value", "probability"} or not isinstance(answer["value"], bool):
                _fail("Malformed boolean answer.", p)
            probability = _finite_number(answer["probability"], f"{p}/probability", 0, 1)
            expected = answer["value"]
            if expected != (probability >= 0.5):
                _fail("Boolean value must select the most probable option.", f"{p}/value")
        else:
            expected_key = "choice" if q["type"] == "choice" else "level"
            if set(answer) != {"type", expected_key, "probabilities", "confidence"}:
                _fail("Malformed categorical answer.", p)
            if answer[expected_key] not in ids:
                _fail("Selected option must belong to the request.", f"{p}/{expected_key}")
            probs = answer["probabilities"]
            if not isinstance(probs, dict) or set(probs) != set(ids):
                _fail("Probability keys must match request option IDs exactly.", f"{p}/probabilities")
            values = [_finite_number(probs[k], f"{p}/probabilities/{k}", 0, 1) for k in ids]
            if abs(sum(values) - 1.0) > 1e-6:
                _fail("Probabilities must sum to one within 1e-6.", f"{p}/probabilities")
            maximum = max(values)
            selected = min(k for k in ids if probs[k] == maximum)
            if answer[expected_key] != selected:
                _fail("Selected option must be the maximum probability, with lexicographic tie breaking.", f"{p}/{expected_key}")
            confidence = _finite_number(answer["confidence"], f"{p}/confidence", 0, 1)
            if abs(confidence - maximum) > 1e-6:
                _fail("Confidence must equal the maximum probability.", f"{p}/confidence")
    return dict(out)


def strict_loads(raw: str | bytes) -> Any:
    """Decode JSON while rejecting duplicate keys and NaN/Infinity constants."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValidationError(f"Duplicate JSON key: {key}.", "/")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValidationError(f"Nonfinite JSON number {value} is not allowed.", "/")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except ValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Malformed JSON.", "/") from exc


def strict_dumps(value: Any) -> str:
    """Serialize JSON with finite numbers only and stable compact formatting."""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Value cannot be represented as finite JSON.", "/") from exc


DECISION_ROW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_BASE + "DecisionRow.json",
    "title": "DecisionRow", "type": "object", "additionalProperties": False,
    "required": ["id", "state_text", "question", "options", "label", "annotator_confidence", "source"],
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 128},
        "state_text": {"type": "string", "minLength": 1, "maxLength": 65536},
        "question": {"type": "object", "additionalProperties": False, "required": ["id", "type", "instructions"], "properties": {
            "id": {"type": "string", "minLength": 1, "maxLength": 128}, "type": {"enum": ["choice", "boolean", "ordinal"]},
            "instructions": {"type": "string", "minLength": 1, "maxLength": 4096}}},
        "options": {"type": "array", "minItems": 2, "maxItems": 8, "items": {"type": "object", "additionalProperties": False,
            "required": ["id", "description"], "properties": {"id": {"type": "string", "minLength": 1, "maxLength": 64, "not": {"const": "ABSTAIN"}},
                "description": {"type": "string", "minLength": 1, "maxLength": 2048}}}},
        "label": {"type": "string", "minLength": 1, "maxLength": 64}, "annotator_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "source": {"enum": ["human_review", "dataset_name", "adjudicated"]}},
    "$comment": "Semantic invariants (unique IDs, label membership, boolean and contiguous ordinal options) are enforced by s1_compass.data.schema.validate_row.",
}

DECISION_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": SCHEMA_BASE + "DecisionOutput.json",
    "title": "DecisionOutput", "type": "object", "additionalProperties": False,
    "required": ["answers", "model", "calibration_version", "latency_ms"],
    "properties": {"answers": {"type": "object", "minProperties": 1, "maxProperties": 20},
        "model": {"type": "string", "pattern": PUBLIC_MODEL_PATTERN.pattern},
        "calibration_version": {"type": "string", "pattern": "^[a-f0-9]{64}$"}, "latency_ms": {"type": "integer", "minimum": 0}},
}

SCHEMA_REGISTRY = {
    DECISION_ROW_SCHEMA["$id"]: DECISION_ROW_SCHEMA,
    DECISION_OUTPUT_SCHEMA["$id"]: DECISION_OUTPUT_SCHEMA,
}


def resolve_schema(uri: str) -> dict[str, Any]:
    """Resolve only known local HTTPS schema IDs; no network fetches are made."""
    try:
        return SCHEMA_REGISTRY[uri]
    except KeyError as exc:
        raise ValidationError(f"Unknown schema URI: {uri}.", "/$schema") from exc
