#!/usr/bin/env python3
"""CPU-only smoke for the frozen OFT import boundary; no model or LIBERO runtime."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from types import ModuleType


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"SMOKE_MODULE_SPEC_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--oft-root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SMOKE_REQUIRES_CUDA_VISIBLE_DEVICES_EMPTY")

    runtime_root = args.runtime_root.resolve()
    oft_root = args.oft_root.resolve()
    launcher = load_module(runtime_root / "scripts/stage_z/launch_stage_z_z1_runtime.py", "stage_z_z1r2_launcher")
    configured_root, configured_src = launcher.configure_runtime_environment()
    if configured_root != runtime_root or configured_src != runtime_root / "src":
        raise RuntimeError("SMOKE_RUNTIME_ROOT_BINDING_MISMATCH")
    if Path.cwd() != runtime_root:
        raise RuntimeError("SMOKE_CWD_BINDING_MISMATCH")

    sys.path.insert(0, str(oft_root))
    runner = load_module(runtime_root / "scripts/stage_z/run_stage_z_z1_runtime_canary.py", "stage_z_z1r2_runner")
    runner._install_optional_import_shims()
    from experiments.robot.libero.run_libero_eval import process_action  # type: ignore
    from experiments.robot.openvla_utils import get_vla_action  # type: ignore

    process_action_path = Path(inspect.getsourcefile(process_action) or "").resolve()
    get_vla_action_path = Path(inspect.getsourcefile(get_vla_action) or "").resolve()
    if oft_root not in process_action_path.parents or oft_root not in get_vla_action_path.parents:
        raise RuntimeError("SMOKE_OFFICIAL_OFT_MODULE_PATH_MISMATCH")
    print(
        json.dumps(
            {
                "schema": "STAGE_Z_Z1R2_M1_ENTRYPOINT_AND_IMPORT_STATIC_SMOKE_V1",
                "status": "STAGE_Z_Z1R2_M1_ENTRYPOINT_AND_IMPORT_STATIC_PASS",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "runtime_root": str(runtime_root),
                "runtime_src": str(runtime_root / "src"),
                "oft_root": str(oft_root),
                "runner_path": str(runtime_root / "scripts/stage_z/run_stage_z_z1_runtime_canary.py"),
                "launcher_path": str(runtime_root / "scripts/stage_z/launch_stage_z_z1_runtime.py"),
                "process_action_module": process_action.__module__,
                "process_action_path": str(process_action_path),
                "get_vla_action_module": get_vla_action.__module__,
                "get_vla_action_path": str(get_vla_action_path),
                "model_construction": False,
                "checkpoint_reads": 0,
                "simulator_creation": False,
                "env_step": 0,
                "scientific_parent_exposure": 0,
                "protected_reads": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
