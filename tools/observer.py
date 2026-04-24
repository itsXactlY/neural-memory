#!/usr/bin/env python3
"""A6 — always-on neural-memory observer (v1 minimal).

Polls a handful of git repos + Obsidian vault for changes since last run, and
writes a compact note per change into neural-memory via the `remember` CLI.

v1 scope (no LLM required):
  - git commits in known project dirs since last observed SHA
  - new/significantly-edited markdown notes in known vault paths

v2 (not implemented here):
  - Haiku filter to drop noise
  - Opus extraction for richer summaries

State tracked in ~/.neural_memory/observer-state.json so each run is idempotent.

Run every 15 minutes via launchd (see com.ae.neural-observer.plist).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STATE_PATH = Path.home() / ".neural_memory" / "observer-state.json"
LOG_DIR = Path.home() / ".neural_memory" / "logs"
REMEMBER_BIN = os.environ.get("REMEMBER_BIN", str(Path.home() / ".local" / "bin" / "remember"))

# Git repos to watch
DEFAULT_REPOS = [
    "/Users/tito/lWORKSPACEl/research/neural-memory",
    "/Users/tito/lWORKSPACEl/research/pulse-hermes",
    "/Users/tito/lWORKSPACEl/Projects/AngelsElectric/LangGraph",
    str(Path.home() / ".hermes" / "hermes-agent"),
]

# Obsidian vault paths to watch (md-only, exclude .obsidian/)
DEFAULT_VAULTS = [
    "/Users/tito/.obsidian/obsidian-vaults/neural-memory-vault",
    "/Users/tito/.obsidian/obsidian-vaults/ae-ai-vault-hub",
]

# Cutoff: don't re-ingest files modified > MAX_AGE_MIN ago (they should have
# been caught by an earlier run; running at launch after long downtime could
# flood memory if we don't cap the lookback)
MAX_AGE_MIN = 60


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"git": {}, "vault_files": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _git(repo: str, *args: str) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return res.stdout.strip()
    except Exception:
        return ""


def poll_git(repos: list[str], state: dict, events: list[dict]) -> None:
    git_state = state.setdefault("git", {})
    for repo in repos:
        if not Path(repo).is_dir():
            continue
        if not (Path(repo) / ".git").exists() and not (Path(repo) / ".git").is_file():
            # worktree or non-git dir
            continue

        last_sha = git_state.get(repo, {}).get("last_sha", "")
        current_sha = _git(repo, "rev-parse", "HEAD")
        if not current_sha:
            continue
        if current_sha == last_sha:
            continue  # nothing new

        # Fetch commits between last seen and HEAD
        if last_sha:
            fmt = f"{last_sha}..HEAD"
        else:
            fmt = "-5"  # first time: last 5 commits
        log_out = _git(repo, "log", fmt, "--pretty=format:%h|%an|%ci|%s", "--no-merges")
        for line in log_out.splitlines():
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            sha, author, ci, subject = parts
            repo_name = Path(repo).name
            events.append({
                "kind": "git",
                "repo": repo_name,
                "sha": sha,
                "author": author,
                "date": ci,
                "claim": f"[{repo_name}] {subject} ({sha} by {author})",
                "source": f"observer:git:{repo_name}",
                "label": f"git:{repo_name}:{sha}",
            })

        git_state[repo] = {"last_sha": current_sha, "last_poll": time.strftime("%Y-%m-%dT%H:%M:%S")}


def poll_vault(vaults: list[str], state: dict, events: list[dict]) -> None:
    vault_state = state.setdefault("vault_files", {})
    now = time.time()
    cutoff = now - MAX_AGE_MIN * 60
    for vault in vaults:
        vp = Path(vault)
        if not vp.is_dir():
            continue
        for md in vp.rglob("*.md"):
            if any(p.name.startswith(".") for p in md.parents):
                continue  # skip .obsidian/ etc.
            try:
                mtime = md.stat().st_mtime
            except Exception:
                continue
            if mtime < cutoff:
                continue
            rel = str(md.relative_to(vp))
            key = str(md)
            last_mtime = vault_state.get(key, 0)
            if mtime <= last_mtime:
                continue

            # Read first line as summary
            try:
                first_line = md.read_text(errors="ignore").splitlines()[0:2]
                summary = " ".join(first_line)[:200]
            except Exception:
                summary = ""
            events.append({
                "kind": "vault",
                "vault": vp.name,
                "file": rel,
                "claim": f"[vault:{vp.name}] edited {rel}: {summary}",
                "source": f"observer:vault:{vp.name}",
                "label": f"vault:{vp.name}:{rel[:60]}",
            })
            vault_state[key] = mtime


def write_events(events: list[dict]) -> int:
    if not events:
        return 0
    if not Path(REMEMBER_BIN).exists():
        print(f"ERROR: remember CLI not at {REMEMBER_BIN}", file=sys.stderr)
        return 0

    written = 0
    for e in events:
        try:
            subprocess.run(
                [REMEMBER_BIN, e["claim"], "--label", e["label"], "--source", e["source"]],
                check=False, timeout=20, capture_output=True,
            )
            written += 1
        except Exception as exc:
            print(f"WARN remember failed for {e.get('label')}: {exc}", file=sys.stderr)
    return written


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print events without calling remember")
    ap.add_argument("--repos", default=None, help="Comma-separated repo override")
    ap.add_argument("--vaults", default=None, help="Comma-separated vault override")
    ap.add_argument("--max-events", type=int, default=50, help="Cap events per run to prevent flooding")
    args = ap.parse_args()

    repos = args.repos.split(",") if args.repos else DEFAULT_REPOS
    vaults = args.vaults.split(",") if args.vaults else DEFAULT_VAULTS

    state = load_state()
    events: list[dict] = []

    poll_git(repos, state, events)
    poll_vault(vaults, state, events)

    # Cap to prevent flooding
    if len(events) > args.max_events:
        events = events[: args.max_events]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "events_found": len(events),
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        for e in events:
            print(json.dumps(e))
        summary["action"] = "dry-run"
    else:
        written = write_events(events)
        save_state(state)
        summary["written"] = written
        summary["action"] = "wrote"

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
