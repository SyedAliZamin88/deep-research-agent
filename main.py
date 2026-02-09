# main.py

from __future__ import annotations

import sys
from typing import Iterable

from scripts.run_agent import main as _run_agent_cli


def main(argv: Iterable[str] | None = None) -> int:
    """
    Entry point that delegates to the CLI runner located in scripts/run_agent.py.

    Args:
        argv: Optional iterable of command-line arguments. When None, sys.argv[1:]
            will be used, matching the behavior of the CLI module.

    Returns:
        Exit code produced by the underlying CLI.
    """
    if argv is None:
        argv = sys.argv[1:]
    return _run_agent_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
