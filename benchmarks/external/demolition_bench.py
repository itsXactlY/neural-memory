"""Compatibility shim — renamed to comparison_bench.py 2026-05-11.

This module proxies imports to the new name so existing pinned-link
documentation and external repro scripts continue to work. New code
and links should reference `comparison_bench` directly.
"""
from comparison_bench import *  # noqa: F401,F403
from comparison_bench import main, SYNTHETIC_RECORDS  # explicit re-export

if __name__ == "__main__":
    main()
