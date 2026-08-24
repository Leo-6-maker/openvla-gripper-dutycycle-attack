from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
RUNNER_MODULE = "stage_z.run_stage_z_z1_runtime_canary"


def test_optional_shims_precede_oft_dynamic_model_import() -> None:
    source = (ROOT / "scripts/stage_z/run_stage_z_z1_runtime_canary.py").read_text(encoding="utf-8")
    load_start = source.index("def load_openvla(")
    load_source = source[load_start:]
    assert load_source.index("_install_optional_import_shims()") < load_source.index("AutoProcessor.from_pretrained")
    assert load_source.index("_install_optional_import_shims()") < load_source.index("ModelClass.from_pretrained")


def test_broken_optional_imports_are_replaced_by_module_spec_shims(monkeypatch) -> None:
    runner = importlib.import_module(RUNNER_MODULE)
    original = {name: sys.modules.get(name) for name in ("wandb", "json_numpy")}
    sys.modules.pop("wandb", None)
    sys.modules.pop("json_numpy", None)
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name in {"wandb", "json_numpy"}:
            raise ImportError(f"mock broken optional import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    try:
        runner._install_optional_import_shims()
        assert sys.modules["wandb"].__spec__.name == "wandb"
        assert sys.modules["json_numpy"].__spec__.name == "json_numpy"
        assert callable(sys.modules["json_numpy"].patch)
    finally:
        for name, value in original.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_runner_module_import_is_cpu_only(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    runner = importlib.import_module(RUNNER_MODULE)
    assert runner.PHASE == "Z1"
