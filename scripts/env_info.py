#!/usr/bin/env python
"""Print environment and dependency versions for report appendix."""

from __future__ import annotations

import importlib
import platform
import sys


MODS = ["numpy", "pandas", "matplotlib", "numba", "cupy", "line_profiler"]


def version_of(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover
        return f"not available ({type(exc).__name__})"
    return getattr(mod, "__version__", "unknown")


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"processor: {platform.processor()}")
    for mod in MODS:
        print(f"{mod}: {version_of(mod)}")


if __name__ == "__main__":
    main()
