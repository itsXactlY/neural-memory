"""Dataset fetcher — downloads (or sources from bundled/) every entry
in manifest.json into $WORK/datasets/, verifying SHA-256 before
declaring success.

Hard rules:
  - if a download finishes but its SHA-256 disagrees with the manifest,
    delete the file and exit nonzero. Never leave a tainted dataset on
    disk where a runner might pick it up.
  - if the manifest entry has a `TODO:` sentinel hash, recompute the
    hash and *warn* (so the operator notices the manifest needs a
    bump) but do not abort — bundled-asset workflows are how a new
    dataset gets added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

POD_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = POD_ROOT / "datasets" / "manifest.json"
BUNDLED_DIR = POD_ROOT / "datasets" / "bundled"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[fetch] {url} -> {dst}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"download failed: {url}: {e}") from e
    tmp.rename(dst)


def _handle_entry(entry: dict[str, Any], work_dir: Path) -> tuple[str, str]:
    """Return (status, message). status in {OK, WARN, FAIL}."""
    name = entry["name"]
    expected = entry.get("sha256") or ""
    url = entry.get("url") or ""

    if url.startswith("bundled://"):
        rel = url[len("bundled://"):]
        src = POD_ROOT / rel
        if not src.exists():
            return ("FAIL", f"{name}: bundled file missing at {src}")
        actual = _sha256(src)
        if expected.startswith("TODO:"):
            return ("WARN", f"{name}: bundled, sha256={actual} (manifest still has TODO sentinel)")
        if expected and expected != actual:
            return ("FAIL",
                    f"{name}: bundled file hash mismatch — expected {expected}, got {actual}")
        return ("OK", f"{name}: bundled OK ({actual[:12]})")

    if not url:
        return ("FAIL", f"{name}: no url")

    dst = work_dir / "datasets" / f"{name}.json"
    if dst.exists():
        actual = _sha256(dst)
        if expected.startswith("TODO:"):
            return ("WARN", f"{name}: cached, sha256={actual} (manifest still has TODO sentinel)")
        if expected and actual == expected:
            return ("OK", f"{name}: cached OK ({actual[:12]})")
        # Hash mismatch on a previously cached file — refetch.
        print(f"[fetch] {name}: cached hash mismatch, refetching", flush=True)
        try:
            dst.unlink()
        except OSError:
            pass

    if expected.startswith("TODO:") or "TODO:upload-tag" in (entry.get("_url_note") or ""):
        return ("WARN",
                f"{name}: skip download — manifest URL is a TODO sentinel ({url}). "
                f"Once the GitHub Release tag exists, this entry will fetch automatically.")

    try:
        _download(url, dst)
    except Exception as e:
        return ("FAIL", f"{name}: {e}")

    actual = _sha256(dst)
    if expected and actual != expected:
        # Tainted: nuke the file so no runner picks it up.
        try:
            dst.unlink()
        except OSError:
            pass
        return ("FAIL",
                f"{name}: hash mismatch after download — expected {expected}, got {actual}; deleted")
    return ("OK", f"{name}: fetched + verified ({actual[:12]})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch + SHA-verify Comparison Pod datasets")
    p.add_argument("--work", required=True, help="$WORK dir, e.g. ~/.bench-pod")
    args = p.parse_args(argv)

    work = Path(args.work).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_PATH.read_text())
    statuses: list[tuple[str, str]] = []
    for entry in manifest.get("datasets", []):
        statuses.append(_handle_entry(entry, work))

    print()
    n_ok = sum(1 for s, _ in statuses if s == "OK")
    n_warn = sum(1 for s, _ in statuses if s == "WARN")
    n_fail = sum(1 for s, _ in statuses if s == "FAIL")
    for status, msg in statuses:
        marker = {"OK": "  OK  ", "WARN": " WARN ", "FAIL": " FAIL "}[status]
        print(f"[{marker}] {msg}")
    print(f"\n[fetch] {n_ok} OK, {n_warn} WARN, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
