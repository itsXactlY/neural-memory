"""Comparison Pod comparator: reads per-system ResultRecord JSON,
validates them against a strict schema, and emits matrix.md /
matrix.json / verdict.md.

Hardfact: a missing system renders as PENDING, not as zero. A
dataset hash that disagrees across results is a hard abort — same
dataset or no comparison.
"""
