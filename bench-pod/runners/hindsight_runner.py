"""NotImplementedError — see /destruction/hindsight/ for locked methodology, runner pending implementation."""
from __future__ import annotations

from runners.common import base_argparser

SYSTEM = "hindsight"


def build_parser():
    return base_argparser(SYSTEM, __doc__)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        f"{SYSTEM} runner is a v0.1 stub. "
        f"See https://mazemaker.online/destruction/{SYSTEM}/ for the locked methodology."
    )


if __name__ == "__main__":
    raise SystemExit(main())
