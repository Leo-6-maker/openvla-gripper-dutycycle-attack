from __future__ import annotations

import builtins
import importlib
import importlib.util
import json
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


def test_runtime_launcher_binds_root_src_and_cwd(monkeypatch) -> None:
    path = ROOT / "scripts/stage_z/launch_stage_z_z1_runtime.py"
    spec = importlib.util.spec_from_file_location("stage_z_z1_launcher_test", path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    changed: list[Path] = []
    monkeypatch.setattr(launcher.os, "chdir", changed.append)
    monkeypatch.setenv("PYTHONPATH", "existing-path")
    monkeypatch.setattr(launcher.sys, "path", list(sys.path))
    runtime_root, runtime_src = launcher.configure_runtime_environment()
    assert runtime_root == ROOT
    assert runtime_src == ROOT / "src"
    assert changed == [ROOT]
    assert launcher.sys.path[0] == str(ROOT / "src")
    assert launcher.os.environ["PYTHONPATH"].split(launcher.os.pathsep)[0] == str(ROOT / "src")


def test_oft_unnorm_key_exact_suite_wins() -> None:
    runner = importlib.import_module(RUNNER_MODULE)
    resolved, mode = runner.resolve_official_unnorm_key({"libero_10": {}, "libero_10_no_noops": {}}, "libero_10")
    assert resolved == "libero_10"
    assert mode == "EXACT_SUITE_KEY"


def test_oft_unnorm_key_uses_official_no_noops_fallback() -> None:
    runner = importlib.import_module(RUNNER_MODULE)
    resolved, mode = runner.resolve_official_unnorm_key({"libero_10_no_noops": {}}, "libero_10")
    assert resolved == "libero_10_no_noops"
    assert mode == "OFFICIAL_NO_NOOPS_FALLBACK"


def test_oft_unnorm_key_missing_fails_closed() -> None:
    runner = importlib.import_module(RUNNER_MODULE)
    try:
        runner.resolve_official_unnorm_key({"libero_goal": {}}, "libero_10")
    except runner.OFTUnnormKeyResolutionError as exc:
        assert str(exc) == "MODEL_UNNORM_KEY_MISSING:libero_10"
    else:
        raise AssertionError("missing OFT norm key did not fail closed")


def test_oft_resolved_key_reaches_get_vla_action_config() -> None:
    source = (ROOT / "scripts/stage_z/run_stage_z_z1_runtime_canary.py").read_text(encoding="utf-8")
    load_source = source[source.index("def load_openvla("):]
    assert "unnorm_key=resolved_unnorm_key" in load_source
    assert "unnorm_key=suite" not in load_source
    assert "get_vla_action(" in load_source


def test_oft_uses_official_dataset_stats_loader_before_key_resolution() -> None:
    source = (ROOT / "scripts/stage_z/run_stage_z_z1_runtime_canary.py").read_text(encoding="utf-8")
    load_source = source[source.index("def load_openvla("):]
    import_line = "from experiments.robot.openvla_utils import _load_dataset_stats"
    assert import_line in load_source
    assert load_source.index("_load_dataset_stats(model, checkpoint)") < load_source.index("resolve_official_unnorm_key")
    assert '"stats_loader": "official_openvla_utils._load_dataset_stats" if oft else' in load_source


def test_oft_key_resolution_does_not_rewrite_checkpoint_json(tmp_path) -> None:
    runner = importlib.import_module(RUNNER_MODULE)
    stats_path = tmp_path / "dataset_statistics.json"
    original = b'{"libero_10_no_noops": {"q01": [0], "q99": [1]}}\n'
    stats_path.write_bytes(original)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    runner.resolve_official_unnorm_key(stats, "libero_10")
    assert stats_path.read_bytes() == original


def test_m0_compatibility_branch_remains_unchanged() -> None:
    source = (ROOT / "scripts/stage_z/run_stage_z_z1_runtime_canary.py").read_text(encoding="utf-8")
    load_source = source[source.index("def load_openvla("):]
    assert "if not oft:" in load_source
    assert "predict_action_compat" in load_source
    assert "M0_PREDICT_ACTION_SHAPE_INVALID" in load_source
