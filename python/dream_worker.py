"""Standalone dream daemon — runs the dream engine outside the Hermes
plugin / mcp container, so it's not gated by Hermes idle hooks and
doesn't fight the in-pod writers for the SQLite write lock.

Loop is dumb: cycle → log stats → short sleep → cycle. No
idle threshold, no memory threshold, no `should_dream` check. The
mcp container should be started with MM_DREAM_DISABLED=1 to keep its
own engine inert; this daemon owns consolidation.

Reads/writes the same SQLite memory.db the pod uses, so consolidation
state (dream_sessions, connection_history, dream_insights) shows up
unchanged in `mazemaker_dream_stats`.

Usage:
    python dream_worker.py                          # default loop
    python dream_worker.py --once                   # single cycle
    python dream_worker.py --phase nrem             # one phase only
    python dream_worker.py --no-think               # skip think() — fast NREM, no real consolidation
    python dream_worker.py --db /path/to/memory.db
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger("dream_worker")


def _build_memory(db_path: str, embedding_backend: str, retrieval_mode: str,
                  think_engine: str):
    """Construct a NeuralMemory-compatible `Memory` instance the same way
    the Hermes plugin does. GPU is auto-selected when CUDA is available
    (sentence-transformers / GpuRecallEngine pick it up internally)."""
    from mazemaker import Memory
    # lazy_graph=True is critical for the daemon: _load_from_store loads all
    # ~190k embeddings (~750MB) into a Python dict at construction time,
    # which can take many minutes on a big corpus and consume RAM the GPU
    # PPR path doesn't need — it queries the connections table directly.
    return Memory(
        db_path=db_path,
        embedding_backend=embedding_backend,
        retrieval_mode=retrieval_mode,
        think_engine=think_engine,
        lazy_graph=True,
    )


def _run_phase(engine, phase: str):
    if phase == "all":
        return engine._run_dream_cycle()
    if phase == "nrem":
        return {"nrem": engine._phase_nrem()}
    if phase == "rem":
        return {"rem": engine._phase_rem()}
    if phase == "insight":
        return {"insights": engine._phase_insights()}
    raise ValueError(f"unknown phase: {phase}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--db", default=os.environ.get(
        "MM_DREAM_DB",
        os.path.expanduser("~/.mazemaker/data/memory.db"),
    ), help="SQLite memory.db path (must match the pod's mount)")
    ap.add_argument("--max-memories", type=int, default=2000,
                    help="NREM batch size per cycle")
    ap.add_argument("--max-isolated", type=int, default=800,
                    help="REM batch size per cycle")
    ap.add_argument("--recent-pct", type=float, default=0.5)
    ap.add_argument("--random-pct", type=float, default=0.3)
    ap.add_argument("--low-salience-pct", type=float, default=0.2)
    ap.add_argument("--cycle-interval", type=float, default=5.0,
                    help="seconds to sleep between cycles after one finishes")
    ap.add_argument("--phase", choices=["all", "nrem", "rem", "insight"],
                    default="all")
    ap.add_argument("--once", action="store_true",
                    help="run a single cycle then exit")
    ap.add_argument("--no-think", action="store_true",
                    help="skip self._memory.think() in NREM — much faster but "
                         "skips the spreading-activation step that picks "
                         "which edges to strengthen")
    ap.add_argument("--embedding-backend", default="auto",
                    help="auto | sentence-transformers | fastembed | tfidf | hash")
    ap.add_argument("--retrieval-mode", default="hybrid")
    ap.add_argument("--think-engine", default="ppr",
                    help="bfs | ppr — ppr is the default GPU-friendlier path")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dream_engine import DreamEngine, SQLiteDreamBackend

    db_path = args.db
    if not Path(db_path).exists():
        logger.error("memory.db not found at %s", db_path)
        return 2

    if args.no_think:
        memory = None
        logger.info("running --no-think — sample+strengthen+prune only, "
                    "skipping self._memory.think()")
    else:
        logger.info("loading mazemaker.Memory (embedding=%s, retrieval=%s, think=%s) ...",
                    args.embedding_backend, args.retrieval_mode, args.think_engine)
        memory = _build_memory(
            db_path=db_path,
            embedding_backend=args.embedding_backend,
            retrieval_mode=args.retrieval_mode,
            think_engine=args.think_engine,
        )
        logger.info("Memory loaded — GPU is used automatically when CUDA is present "
                    "(sentence-transformers / GpuRecallEngine)")

    backend = SQLiteDreamBackend(db_path)
    engine = DreamEngine(
        backend,
        neural_memory=memory,
        # idle/memory thresholds aren't used by this script — we don't call
        # engine.start(); we drive cycles directly. Keep them at 0 so the
        # engine doesn't stash any "wait for idle" state if someone later
        # decides to engine.start() this instance.
        idle_threshold=0,
        memory_threshold=0,
        max_memories_per_cycle=args.max_memories,
        max_isolated_per_cycle=args.max_isolated,
        sample_recent_pct=args.recent_pct,
        sample_random_pct=args.random_pct,
        sample_low_salience_pct=args.low_salience_pct,
    )

    stop_flag = {"v": False}

    def _handle(signum, _frame):
        if stop_flag["v"]:
            logger.warning("second signal %d — exiting NOW (mid-cycle)", signum)
            sys.exit(130)
        logger.info("signal %d received — finishing current cycle then stopping", signum)
        stop_flag["v"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    cycle_n = 0
    logger.info("dream daemon up — db=%s, max_memories=%d, max_isolated=%d, "
                "mix=%.2f/%.2f/%.2f, phase=%s",
                db_path, args.max_memories, args.max_isolated,
                args.recent_pct, args.random_pct, args.low_salience_pct, args.phase)

    while not stop_flag["v"]:
        cycle_n += 1
        t0 = time.time()
        try:
            stats = _run_phase(engine, args.phase)
            duration = time.time() - t0
            logger.info("cycle #%d (%s) done in %.1fs: %s",
                        cycle_n, args.phase, duration, stats)
        except Exception:
            logger.exception("cycle #%d failed", cycle_n)

        if args.once:
            break
        if stop_flag["v"]:
            break
        time.sleep(args.cycle_interval)

    logger.info("dream daemon stopped after %d cycles", cycle_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
