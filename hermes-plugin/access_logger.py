#!/usr/bin/env python3
"""
access_logger.py - Records every recall event for LSTM training data.

Provides:
- Circular buffer of recent access events
- JSON Lines persistence for training data generation
- Co-occurrence analysis for graph edge strengthening
- LSTM training pair extraction
"""

import json
import os
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional


class AccessLogger:
    """Logs memory recall events for temporal pattern learning.

    Usage:
        logger = AccessLogger.instance()
        logger.log_recall(
            query_embedding=[0.1, 0.2, ...],
            result_ids=[42, 17, 88],
            result_scores=[0.95, 0.87, 0.72],
            timestamp=time.time()
        )
        sequence = logger.get_sequence(20)
        pair = logger.get_training_pair()
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls, log_dir: str = "~/.neural_memory/access_logs",
                 max_sequence: int = 20) -> "AccessLogger":
        """Singleton accessor. Creates on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(log_dir=log_dir, max_sequence=max_sequence)
        return cls._instance

    def __init__(self, log_dir: str = "~/.neural_memory/access_logs",
                 max_sequence: int = 20):
        self._log_dir = Path(os.path.expanduser(log_dir))
        self._max_sequence = max_sequence
        self._buffer: list[dict] = []
        self._max_buffer = 1000
        self._flush_threshold = 100
        self._ops_since_flush = 0
        self._file_lock = threading.Lock()

        # Auto-create log directory
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "access_log.jsonl"

    def log_recall(self, query_embedding: list[float], result_ids: list[int],
                   result_scores: list[float], timestamp: Optional[float] = None):
        """Record a single recall event.

        Args:
            query_embedding: The query vector (stored sparsely/truncated for log).
            result_ids: Memory IDs returned by the recall.
            result_scores: Similarity scores for each result.
            timestamp: Event time (epoch seconds). Defaults to now.
        """
        if timestamp is None:
            timestamp = time.time()

        # Store truncated embedding (first 64 dims) to keep logs manageable
        emb_sample = query_embedding[:64] if len(query_embedding) > 64 else query_embedding

        event = {
            "ts": timestamp,
            "query_emb": emb_sample,
            "result_ids": result_ids[:20],  # cap at 20 results
            "result_scores": [round(s, 4) for s in result_scores[:20]],
            "n_results": len(result_ids),
        }

        with self._file_lock:
            self._buffer.append(event)
            self._ops_since_flush += 1

            # Circular buffer eviction
            if len(self._buffer) > self._max_buffer:
                self._buffer = self._buffer[-self._max_buffer:]

            # Periodic disk flush
            if self._ops_since_flush >= self._flush_threshold:
                self._flush_buffer()

    def get_sequence(self, n: int = 20) -> list[dict]:
        """Return last N recall events (most recent last).

        Args:
            n: Number of events to return.
        Returns:
            List of event dicts in chronological order.
        """
        with self._file_lock:
            return list(self._buffer[-n:])

    def get_co_occurrence_pairs(self, min_count: int = 3) -> list[tuple[int, int, int]]:
        """Find memory pairs frequently recalled together.

        Scans buffer for result IDs appearing together in the same recall event.
        Useful for strengthening graph edges between co-accessed memories.

        Args:
            min_count: Minimum co-occurrence threshold.
        Returns:
            List of (id_a, id_b, count) tuples sorted by count descending.
            id_a < id_b to avoid duplicates.
        """
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)

        with self._file_lock:
            for event in self._buffer:
                ids = sorted(set(event.get("result_ids", [])))
                # Count all pairs from this event
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pair_counts[(ids[i], ids[j])] += 1

        # Filter and sort
        results = [(a, b, c) for (a, b), c in pair_counts.items() if c >= min_count]
        results.sort(key=lambda x: -x[2])
        return results

    def save(self):
        """Persist current buffer to disk (append to JSONL file)."""
        with self._file_lock:
            self._flush_buffer()

    def load(self, n: int = 1000):
        """Load last N events from disk into buffer.

        Args:
            n: Maximum events to load.
        """
        if not self._log_file.exists():
            return

        events = []
        try:
            with open(self._log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except IOError:
            return

        with self._file_lock:
            self._buffer = events[-n:]

    def get_training_pair(self, max_seq: int = 20) -> Optional[tuple[list[list[float]], list[float]]]:
        """Extract one LSTM training sample from recent access history.

        Returns a (sequence_embeddings, target_embedding) pair where:
        - sequence: last max_seq-1 query embeddings
        - target: the query embedding of the most recent event

        If not enough events, returns None.

        Args:
            max_seq: Minimum events needed.
        Returns:
            (sequence, target) or None if insufficient data.
        """
        with self._file_lock:
            events = list(self._buffer[-max_seq:])

        if len(events) < 2:
            return None

        # Build sequence from all but last, target from last
        sequence = [e["query_emb"] for e in events[:-1]]
        target = events[-1]["query_emb"]

        return (sequence, target)

    def get_training_batch(self, batch_size: int = 32,
                           max_seq: int = 20) -> Optional[list[tuple[list[list[float]], list[float]]]]:
        """Extract multiple LSTM training pairs.

        Slides a window across the buffer to generate training samples.
        Returns None if insufficient data.

        Args:
            batch_size: Number of samples to generate.
            max_seq: Sequence length per sample.
        Returns:
            List of (sequence, target) pairs, or None.
        """
        with self._file_lock:
            events = list(self._buffer)

        if len(events) < max_seq + 1:
            return None

        pairs = []
        # Slide window backwards from most recent
        for i in range(len(events) - 1, max_seq - 1, -1):
            if len(pairs) >= batch_size:
                break
            seq = [events[j]["query_emb"] for j in range(i - max_seq + 1, i)]
            target = events[i]["query_emb"]
            pairs.append((seq, target))

        return pairs if pairs else None

    def flush(self):
        """Force flush buffer to disk."""
        with self._file_lock:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered events to JSONL file (internal, assumes lock held)."""
        if not self._buffer:
            return

        try:
            with open(self._log_file, "a") as f:
                # Write events since last flush
                start = max(0, len(self._buffer) - self._ops_since_flush)
                for event in self._buffer[start:]:
                    f.write(json.dumps(event, separators=(",", ":")) + "\n")
            self._ops_since_flush = 0
        except IOError as e:
            print(f"[AccessLogger] Flush error: {e}")

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return f"AccessLogger(events={len(self._buffer)}, log_dir={self._log_dir})"


if __name__ == "__main__":
    # Quick smoke test
    import random

    logger = AccessLogger(log_dir="/tmp/test_access_logs")
    print(f"Created: {logger}")

    # Simulate some recalls
    for i in range(50):
        emb = [random.uniform(-1, 1) for _ in range(1024)]
        ids = random.sample(range(1000), k=random.randint(3, 10))
        scores = [random.uniform(0.5, 1.0) for _ in ids]
        logger.log_recall(emb, ids, scores)

    print(f"After 50 recalls: {logger}")
    seq = logger.get_sequence(5)
    print(f"Last 5 events: {len(seq)} events")
    print(f"First event keys: {list(seq[0].keys())}")

    pairs = logger.get_co_occurrence_pairs(min_count=2)
    print(f"Co-occurrence pairs (min=2): {len(pairs)}")
    if pairs:
        print(f"  Top pair: {pairs[0]}")

    pair = logger.get_training_pair()
    if pair:
        seq_emb, target = pair
        print(f"Training pair: sequence len={len(seq_emb)}, target len={len(target)}")

    # Test singleton
    logger2 = AccessLogger.instance(log_dir="/tmp/test_access_logs")
    print(f"Singleton same? {logger is logger2}")

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/test_access_logs", ignore_errors=True)
    print("AccessLogger: OK")
