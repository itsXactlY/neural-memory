"""JSON schema for `ResultRecord` instances written by runners.

The comparator validates every result file against this schema before
considering it for matrix inclusion. Validation failure is logged but
does not abort the run — the offending row is rendered as ERROR.
"""
from __future__ import annotations

from typing import Any

RESULT_RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BenchPodResultRecord",
    "type": "object",
    "required": [
        "timestamp", "bench_pod_version", "system", "system_version",
        "system_config", "dataset", "metrics", "host",
    ],
    "additionalProperties": True,
    "properties": {
        "timestamp": {"type": "string"},
        "bench_pod_version": {"type": "string"},
        "system": {
            "type": "string",
            "enum": ["mazemaker", "hindsight", "letta", "mem0", "amem", "cognee"],
        },
        "system_version": {"type": "string"},
        "system_config": {"type": "object"},
        "dataset": {
            "type": "object",
            "required": ["name", "size", "hash"],
            "properties": {
                "name": {"type": "string"},
                "size": {"type": "integer", "minimum": 0},
                "hash": {"type": "string"},
            },
        },
        "metrics": {
            "type": "object",
            "required": ["errors"],
            "properties": {
                "r_at_1":            {"type": ["number", "null"]},
                "r_at_5":            {"type": ["number", "null"]},
                "r_at_10":           {"type": ["number", "null"]},
                "mrr":               {"type": ["number", "null"]},
                "p50_recall_ms":     {"type": ["number", "null"]},
                "p95_recall_ms":     {"type": ["number", "null"]},
                "wall_seconds_ingest": {"type": ["number", "null"]},
                "wall_seconds_query":  {"type": ["number", "null"]},
                "llm_tokens_extraction": {"type": ["integer", "null"]},
                "errors": {"type": "integer", "minimum": 0},
                "failed_questions": {"type": "array", "items": {"type": "integer"}},
            },
        },
        "host": {
            "type": "object",
            "properties": {
                "hostname": {"type": "string"},
                "platform": {"type": "string"},
                "python_version": {"type": "string"},
                "cpu_count": {"type": "integer"},
                "cuda_available": {"type": "boolean"},
            },
        },
    },
}


def validate(record: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty if valid).

    Uses `jsonschema` if available; otherwise a minimal hand-rolled
    check so the comparator still works on a stripped venv.
    """
    try:
        import jsonschema
    except ImportError:
        return _fallback_validate(record)
    try:
        jsonschema.validate(record, RESULT_RECORD_SCHEMA)
        return []
    except jsonschema.ValidationError as e:
        return [f"{'/'.join(str(p) for p in e.path)}: {e.message}"]


def _fallback_validate(record: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    for k in RESULT_RECORD_SCHEMA["required"]:
        if k not in record:
            errs.append(f"missing required key: {k}")
    ds = record.get("dataset") or {}
    for k in ("name", "size", "hash"):
        if k not in ds:
            errs.append(f"dataset missing required key: {k}")
    metrics = record.get("metrics") or {}
    if "errors" not in metrics:
        errs.append("metrics missing required key: errors")
    sys_name = record.get("system")
    valid_systems = RESULT_RECORD_SCHEMA["properties"]["system"]["enum"]
    if sys_name not in valid_systems:
        errs.append(f"system '{sys_name}' not in {valid_systems}")
    return errs
