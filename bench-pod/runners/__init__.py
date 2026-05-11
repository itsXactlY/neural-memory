"""Comparison Pod runner package.

Each `<system>_runner` module implements a thin shim that drives one
memory system through the canonical bench and emits a `ResultRecord`
JSON under `$WORK/results/<system>.json`.

v0.1: only `mazemaker_runner` is a full implementation. The other five
modules are stubs that raise `NotImplementedError`; the comparator
renders them as PENDING in the matrix.
"""
