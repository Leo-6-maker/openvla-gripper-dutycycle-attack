#!/usr/bin/env python3
"""Deterministic Z1 runtime entrypoint for the frozen runtime tree."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SRC = RUNTIME_ROOT / "src"
RUNNER = RUNTIME_ROOT / "scripts" / "stage_z" / "run_stage_z_z1_runtime_canary.py"


def configure_runtime_environment() -> tuple[Path, Path]:
    if not RUNTIME_SRC.is_dir():
        raise RuntimeError(f"Z1_RUNTIME_SRC_MISSING:{RUNTIME_SRC}")
    if not RUNNER.is_file():
        raise RuntimeError(f"Z1_RUNTIME_RUNNER_MISSING:{RUNNER}")
    os.chdir(RUNTIME_ROOT)
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(RUNTIME_SRC) + (os.pathsep + existing if existing else "")
    runtime_src = str(RUNTIME_SRC)
    sys.path[:] = [entry for entry in sys.path if entry != runtime_src]
    sys.path.insert(0, runtime_src)
    return RUNTIME_ROOT, RUNTIME_SRC


def main() -> None:
    configure_runtime_environment()
    runpy.run_path(str(RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
