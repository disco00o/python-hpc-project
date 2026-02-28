#!/usr/bin/env python
"""Convenience wrapper for Task 5 static scheduling experiment."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_solver.py"


def main() -> None:
    cmd = [sys.executable, str(SCRIPT), *sys.argv[1:], "--schedule", "static"]
    print("Running:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=os.environ.copy()))


if __name__ == "__main__":
    main()
