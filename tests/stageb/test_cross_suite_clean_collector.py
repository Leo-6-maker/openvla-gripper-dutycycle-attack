import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.run_sc5_cross_suite_clean import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    build_base_manifest,
    fail_if_output_exists,
    max_steps_for_suite,
    parse_args,
    run_clean_collection,
)


def test_suite_max_steps_are_explicit_and_overridable():
    assert set(DEFAULT_MAX_STEPS) == {"libero_spatial", "libero_goal", "libero_10"}
    assert max_steps_for_suite("libero_spatial") == (400, "suite_default")
    assert max_steps_for_suite("libero_goal", 321) == (321, "cli_override")


def test_clean_collector_dry_run_writes_clean_only_manifest(tmp_path, monkeypatch):
    out = tmp_path / "dry"
    argv = [
        "run_sc5_cross_suite_clean.py",
        "--suite", "libero_spatial",
        "--model_path", "/models/spatial",
        "--unnorm_key", "libero_spatial",
        "--task_idx", "0",
        "--state_id", "0",
        "--eval_seed", "0",
        "--detector_path", "/detector.pt",
        "--source_commit", "abc123",
        "--output_dir", str(out),
        "--render_gpu", "6",
        "--dry_run",
        "--no_gpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    manifest = build_base_manifest(args)
    assert manifest["condition"] == "CLEAN"
    assert manifest["attack_enabled"] is False
    assert manifest["teacher_anchor_required"] is False

    run_clean_collection(args)
    written = json.loads((out / "episode_manifest.json").read_text(encoding="utf-8"))
    assert written["dry_run"] is True
    assert written["suite"] == "libero_spatial"
    assert written["vis_enabled"] is False
    assert written["rand_enabled"] is False


def test_no_gpu_without_dry_run_rejected(tmp_path, monkeypatch):
    argv = [
        "run_sc5_cross_suite_clean.py",
        "--suite", "libero_goal",
        "--model_path", "/models/goal",
        "--unnorm_key", "libero_goal",
        "--task_idx", "1",
        "--state_id", "0",
        "--eval_seed", "0",
        "--detector_path", "/detector.pt",
        "--source_commit", "abc123",
        "--output_dir", str(tmp_path / "bad"),
        "--render_gpu", "4",
        "--no_gpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    try:
        run_clean_collection(args)
    except SystemExit as exc:
        assert "--no_gpu requires" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_output_dir_fail_closed_when_nonempty(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "artifact.txt").write_text("x", encoding="utf-8")
    try:
        fail_if_output_exists(out)
    except SystemExit as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
