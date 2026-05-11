"""Shared runner machinery.

Every runner emits a `ResultRecord` matching the JSON schema declared
in `compare/schema.py`. This module owns the dataclass, dataset
loading (with hash verification), the latency timer context manager,
and the result writer.

Hardfact rule: a runner that does not have a complete result MUST
write the partial record with explicit `None` / `"ERROR"` markers
rather than fabricated numbers. The comparator surfaces these as-is.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

POD_VERSION = "0.1.0"

VALID_SYSTEMS = ("mazemaker", "hindsight", "letta", "mem0", "amem", "cognee")


# ---------------------------------------------------------------------------
# ResultRecord — mirrors the existing bench_*.json shape so the
# comparator can ingest old runs too.
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    r_at_1: Optional[float] = None
    r_at_5: Optional[float] = None
    r_at_10: Optional[float] = None
    mrr: Optional[float] = None
    p50_recall_ms: Optional[float] = None
    p95_recall_ms: Optional[float] = None
    wall_seconds_ingest: Optional[float] = None
    wall_seconds_query: Optional[float] = None
    llm_tokens_extraction: Optional[int] = None
    errors: int = 0
    failed_questions: list[int] = field(default_factory=list)


@dataclass
class DatasetMeta:
    name: str
    size: int
    hash: str


@dataclass
class HostInfo:
    hostname: str
    platform: str
    python_version: str
    cpu_count: int
    cuda_available: bool


@dataclass
class ResultRecord:
    timestamp: str
    bench_pod_version: str
    system: str
    system_version: str
    system_config: dict[str, Any]
    dataset: DatasetMeta
    metrics: Metrics
    host: HostInfo

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def host_info() -> HostInfo:
    cuda = False
    try:
        import torch  # type: ignore
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return HostInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 0,
        cuda_available=cuda,
    )


# ---------------------------------------------------------------------------
# Dataset loading — hash-verified against datasets/manifest.json
# ---------------------------------------------------------------------------

def _read_manifest(pod_root: Path) -> dict[str, Any]:
    mpath = pod_root / "datasets" / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"manifest not found: {mpath}")
    return json.loads(mpath.read_text())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_dataset_path(work_dir: Path, pod_root: Path, name: str) -> Path:
    """Locate a dataset on disk, honoring the bundled-vs-fetched split.

    `synthetic_v1` lives in-repo under `datasets/bundled/`. Anything
    else is expected under `$WORK/datasets/` after `datasets/fetch.py`.
    """
    if name == "synthetic_v1":
        return pod_root / "datasets" / "bundled" / "synthetic_v1.json"
    return work_dir / "datasets" / f"{name}.json"


def load_dataset(work_dir: Path, pod_root: Path, name: str) -> tuple[list[dict[str, Any]], DatasetMeta]:
    """Load + hash-verify a dataset by name.

    Raises if the on-disk SHA-256 does not match the manifest entry.
    Hardfact: a benchmark run with the wrong dataset hash is worse
    than no run at all — the comparator would silently merge
    apples and oranges.
    """
    manifest = _read_manifest(pod_root)
    entry = next((d for d in manifest["datasets"] if d["name"] == name), None)
    if entry is None:
        raise KeyError(f"dataset '{name}' not declared in manifest.json")

    path = resolve_dataset_path(work_dir, pod_root, name)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset '{name}' not found at {path}. "
            f"Run `python datasets/fetch.py --work {work_dir}` first."
        )

    actual = file_sha256(path)
    expected = entry.get("sha256") or ""
    if expected and expected != "TODO:fill-after-upload" and actual != expected:
        raise ValueError(
            f"dataset '{name}' hash mismatch:\n"
            f"  path:     {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"Refusing to run — bench results would be uncomparable."
        )

    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"dataset '{name}' is not a JSON array (got {type(raw).__name__})")

    return raw, DatasetMeta(name=name, size=len(raw), hash=actual[:16])


# ---------------------------------------------------------------------------
# Latency timer
# ---------------------------------------------------------------------------

@contextmanager
def latency_ms() -> Iterator[list[float]]:
    """Context manager that yields a one-element list; element is set
    to the elapsed milliseconds at __exit__.

    Usage:
        with latency_ms() as t:
            do_thing()
        elapsed = t[0]
    """
    holder: list[float] = [0.0]
    t0 = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = (time.perf_counter() - t0) * 1000.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return s[idx]


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_result(work_dir: Path, system: str, record: ResultRecord) -> Path:
    if system not in VALID_SYSTEMS:
        raise ValueError(f"unknown system '{system}' (valid: {VALID_SYSTEMS})")
    out_dir = work_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{system}.json"
    path.write_text(json.dumps(record.to_json(), indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Common CLI surface — every runner sub-class shares this argparse skeleton
# ---------------------------------------------------------------------------

def base_argparser(system: str, description: str):
    import argparse
    p = argparse.ArgumentParser(prog=f"runners.{system}_runner", description=description)
    p.add_argument("--work", required=True,
                   help="Work directory ($WORK). Datasets live under $WORK/datasets, "
                        "results land under $WORK/results.")
    p.add_argument("--dataset", default="longmemeval_s",
                   help="Dataset name as declared in datasets/manifest.json.")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap on number of questions (0 = all).")
    p.add_argument("--quiet", action="store_true")
    return p


def pod_root() -> Path:
    return Path(__file__).resolve().parents[1]
