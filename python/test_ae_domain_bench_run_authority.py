"""Authority tests for tools/ae_domain_bench_run.sh eligibility filter.

Sonnet S1 packet 2026-05-03 deliverable. Per synth contract (LIVE_FEED
2026-05-03T10:41:47Z): F9 shell must filter previous artifacts to eligible
current-format scored artifacts, pass explicit canonical --db, rerun at
current HEAD, and promote only if artifact fields are complete.

The shell exposes a `--select-eligible-prev <dir> [--current-head <sha>]`
dry-run mode that exercises the eligibility filter without touching the
substrate or running the bench. These tests build a temp bench-history
directory with synthetic artifacts covering the rejection criteria and
assert the selector prints the right artifact (or empty) on stdout.

Rejection criteria covered (per packet spec):
  - stale HEAD vs --current-head
  - missing per_query rows
  - db_path == "(default)"  (provenance present but DB not pinned)
  - null memory_count / connections_active
  - copy/ablation filename markers (bge-small, copy, ablation, mpnet, clean)
  - missing provenance block (legacy schema)
  - mode != "scored"
  - missing model name / env block / query_file_md5 / category_regression_gate

Plus: fallback when NO eligible artifact exists (empty stdout, no crash).
And: positive case — a fully-formed eligible artifact IS selected.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SHELL = _REPO / "tools" / "ae_domain_bench_run.sh"
_CANONICAL_DB = str(Path.home() / ".neural_memory" / "memory.db")


def _full_eligible_artifact(git_head: str = "1b82848e110983d962751d62561b71e33c548140") -> dict:
    """Build a synthetic JSON dict that passes EVERY eligibility check."""
    return {
        "mode": "scored",
        "queries_evaluated": 1,
        "per_query": [
            {
                "id": "q1",
                "category": "electrical_contracting",
                "ground_truth_ids": [42],
                "retrieved_ids": [42],
                "hit_at_5": 1,
                "hit_at_10": 1,
                "mrr": 1.0,
                "first_hit_rank": 1,
                "latency_ms": 12.3,
            }
        ],
        "per_category": {
            "electrical_contracting": {
                "n": 1, "r@5": 1.0, "r@10": 1.0,
                "mrr": 1.0, "threshold": 0.7, "passed": True,
            }
        },
        "global_r@5": 1.0,
        "global_r@5_target": 0.760,
        "global_r@5_passed": True,
        "categories_failed": [],
        "provenance": {
            "git_head": git_head,
            "db_path": _CANONICAL_DB,
            "substrate_counts": {"memories": 12810, "connections_active": 45000},
            "models": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "reranker_en": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "reranker_es": None,
            },
            "env": {"NM_EMBED_MODEL": None, "NM_SPANISH_TRANSLATE": "1"},
            "query_file_md5": "e54f49e1abcdef0123456789abcdef01",
            "args": {
                "mode": "scored", "category": None, "k": 10,
                "embedding_backend": "auto", "rerank": True,
                "mmr_lambda": 0.0, "percentile_floor": 0.0,
            },
            "queries_loaded": 38,
            "ts_iso": "2026-05-03T10:00:00Z",
            "ts_epoch": 1746268800,
        },
        "category_regression_gate": {
            "enabled": False, "regressions": [],
        },
    }


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2))
    return path


def _run_selector(bench_dir: Path, current_head: str = "") -> tuple[str, str, int]:
    """Invoke the shell's --select-eligible-prev mode. Returns (stdout, stderr, rc)."""
    cmd = ["bash", str(_SHELL), "--select-eligible-prev", str(bench_dir)]
    if current_head:
        cmd += ["--current-head", current_head]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


class TestEligibilityFilter(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aedombench-test-"))
        self.head = "1b82848e110983d962751d62561b71e33c548140"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- positive case ----------------------------------------------------

    def test_eligible_artifact_is_selected(self):
        good = self.tmp / "ae-domain-2026-05-03-100000.json"
        _write(good, _full_eligible_artifact(git_head=self.head))
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0, msg=f"selector failed: {err}")
        self.assertEqual(out, str(good), msg=f"wrong selection. stderr:\n{err}")
        self.assertIn("OK ", err, msg="expected OK trace on stderr")

    # ---- rejection cases --------------------------------------------------

    def test_rejects_stale_head_artifact(self):
        d = _full_eligible_artifact(git_head="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "", msg=f"stale-HEAD artifact should be rejected. stderr:\n{err}")
        self.assertIn("stale-head", err)

    def test_rejects_missing_per_query(self):
        d = _full_eligible_artifact(git_head=self.head)
        d["per_query"] = []
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "", msg=f"empty per_query should be rejected. stderr:\n{err}")
        self.assertIn("per_query-empty", err)

    def test_rejects_default_db_path(self):
        # The 2026-05-03-032052 incident: provenance present, but db_path
        # is the literal sentinel "(default)" because --db wasn't passed.
        d = _full_eligible_artifact(git_head=self.head)
        d["provenance"]["db_path"] = "(default)"
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "", msg=f"(default) db_path should be rejected. stderr:\n{err}")
        self.assertIn("db_path", err)

    def test_rejects_null_memory_count(self):
        d = _full_eligible_artifact(git_head=self.head)
        d["provenance"]["substrate_counts"]["memories"] = None
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("substrate_memories=None", err)

    def test_rejects_null_active_connection_count(self):
        d = _full_eligible_artifact(git_head=self.head)
        d["provenance"]["substrate_counts"]["connections_active"] = None
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("substrate_conn", err)

    def test_rejects_legacy_no_provenance(self):
        d = _full_eligible_artifact(git_head=self.head)
        d.pop("provenance")
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("no-provenance", err)

    def test_rejects_mode_not_scored(self):
        d = _full_eligible_artifact(git_head=self.head)
        d["mode"] = "diagnostic"
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("mode=", err)

    def test_rejects_missing_embedding_model(self):
        d = _full_eligible_artifact(git_head=self.head)
        d["provenance"]["models"]["embedding_model"] = None
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("embedding_model", err)

    def test_rejects_missing_query_file_md5(self):
        d = _full_eligible_artifact(git_head=self.head)
        d["provenance"]["query_file_md5"] = ""
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("query_file_md5", err)

    def test_rejects_missing_category_regression_gate(self):
        d = _full_eligible_artifact(git_head=self.head)
        d.pop("category_regression_gate")
        _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertIn("category_regression_gate-missing", err)

    def test_rejects_copy_ablation_filename(self):
        # Even a fully-formed artifact in the file gets rejected on filename
        # tag — by-design defense against accidentally promoting an
        # experimental copy.
        d = _full_eligible_artifact(git_head=self.head)
        for tag in ("ae-domain-bge-small-100000.json",
                    "ae-domain-copy-100000.json",
                    "ae-domain-ablation-100000.json",
                    "ae-domain-mpnet-100000.json",
                    "ae-domain-clean-100000.json"):
            p = self.tmp / tag
            _write(p, d)
            out, err, rc = _run_selector(self.tmp, current_head=self.head)
            self.assertEqual(rc, 0)
            self.assertEqual(out, "", msg=f"{tag} should be rejected on filename. stderr:\n{err}")
            self.assertIn("tag:", err)
            p.unlink()

    # ---- fallback case ----------------------------------------------------

    def test_fallback_when_no_eligible_artifact(self):
        """Empty bench-history dir → empty stdout, no crash, rc=0."""
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0, msg=f"selector failed on empty dir: {err}")
        self.assertEqual(out, "", msg="empty dir should yield empty selection")

    def test_fallback_when_only_ineligible_artifacts(self):
        """Mix of priors all rejected → empty stdout, no crash."""
        # Stale HEAD
        _write(self.tmp / "ae-domain-2026-05-02-100000.json",
               _full_eligible_artifact(git_head="cafebabecafebabecafebabecafebabecafebabe"))
        # Missing provenance
        d2 = _full_eligible_artifact(git_head=self.head); d2.pop("provenance")
        _write(self.tmp / "ae-domain-2026-05-02-110000.json", d2)
        # Tag-blacklisted
        _write(self.tmp / "ae-domain-bge-small-110000.json",
               _full_eligible_artifact(git_head=self.head))
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "", msg=f"all-rejected should yield empty. stderr:\n{err}")

    # ---- ordering ---------------------------------------------------------

    def test_picks_newest_eligible_by_mtime(self):
        """Two eligible artifacts → the newer (by mtime) wins."""
        older = self.tmp / "ae-domain-2026-05-03-090000.json"
        newer = self.tmp / "ae-domain-2026-05-03-100000.json"
        _write(older, _full_eligible_artifact(git_head=self.head))
        _write(newer, _full_eligible_artifact(git_head=self.head))
        # Force older < newer mtime
        os.utime(older, (1746260000, 1746260000))
        os.utime(newer, (1746268800, 1746268800))
        out, err, rc = _run_selector(self.tmp, current_head=self.head)
        self.assertEqual(rc, 0)
        self.assertEqual(out, str(newer))

    def test_no_head_pin_skips_stale_head_check(self):
        """When current-head is empty, stale-HEAD is NOT rejected (bench may
        legitimately be run by a wrapper that doesn't pre-fetch HEAD).
        Other eligibility checks still apply.
        """
        d = _full_eligible_artifact(git_head="cafebabecafebabecafebabecafebabecafebabe")
        p = _write(self.tmp / "ae-domain-2026-05-03-100000.json", d)
        out, err, rc = _run_selector(self.tmp, current_head="")
        self.assertEqual(rc, 0)
        self.assertEqual(out, str(p), msg=f"no head-pin → should accept: {err}")


class TestShellInvocationContract(unittest.TestCase):
    """Sanity: shell file exists, is executable-flagged or invokable, and
    the python bench script still accepts --db + --prev-results.
    """

    def test_shell_file_exists(self):
        self.assertTrue(_SHELL.is_file(), f"missing {_SHELL}")

    def test_shell_passes_canonical_db_to_python(self):
        text = _SHELL.read_text()
        self.assertIn('--db "$CANONICAL_DB"', text,
                      "shell must pass --db to python invocation")
        self.assertIn('CANONICAL_DB="${HOME}/.neural_memory/memory.db"', text,
                      "shell must hard-code canonical DB path")

    def test_shell_does_not_use_naive_ls_t_for_prev(self):
        text = _SHELL.read_text()
        # The original 'ls -t ... | head -1' selector must be gone for the
        # primary --prev-results selection. The delta-comparison block was
        # also rewritten to reuse _select_eligible_prev.
        self.assertNotIn(
            'PREV_FOR_GATE=$(ls -t "${HIST}"/ae-domain-*.json',
            text,
            "naive ls -t selector for --prev-results must be removed",
        )

    def test_python_bench_accepts_db_and_prev_results(self):
        """Confirms the shell's --db and --prev-results args are accepted by
        the bench script's argparse — protects against CLI drift."""
        bench = _REPO / "benchmarks" / "ae_domain_memory_bench" / "run_ae_domain_bench.py"
        proc = subprocess.run(
            ["/Users/tito/.hermes/hermes-agent/venv/bin/python3", str(bench), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--db", proc.stdout)
        self.assertIn("--prev-results", proc.stdout)
        self.assertIn("--mode", proc.stdout)
        self.assertIn("--rerank", proc.stdout)


if __name__ == "__main__":
    unittest.main()
