import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.run_sc5_cross_suite_clean import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    build_overlay_frames,
    build_base_manifest,
    fail_if_output_exists,
    max_steps_for_suite,
    parse_args,
    run_clean_collection,
    write_sim_state_archive,
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


def test_overlay_frames_mark_emit_and_invalid_steps():
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    telemetry = [
        {"step": 0, "feat_valid": True, "mlp_emit": -1, "mlp_triggered": False},
        {"step": 1, "feat_valid": False, "mlp_emit": -1, "mlp_triggered": False},
        {"step": 2, "feat_valid": True, "mlp_emit": 2, "mlp_triggered": True},
    ]
    overlay = build_overlay_frames(frames, telemetry)
    assert len(overlay) == 3
    assert not np.array_equal(overlay[1], frames[1])
    assert not np.array_equal(overlay[2], frames[2])
    assert tuple(overlay[1][0, 0]) == (170, 45, 210)
    assert tuple(overlay[2][13, 0]) == (255, 220, 0)


def test_sim_state_archive_records_generic_arrays(tmp_path):
    states = [
        {
            "qpos": np.zeros((2,), dtype=np.float32),
            "qvel": np.ones((2,), dtype=np.float32),
            "body_xpos": np.zeros((3, 3), dtype=np.float32),
            "body_xquat": np.zeros((3, 4), dtype=np.float32),
            "site_xpos": np.zeros((4, 3), dtype=np.float32),
            "ctrl": np.zeros((1,), dtype=np.float32),
        },
        {
            "qpos": np.ones((2,), dtype=np.float32),
            "qvel": np.zeros((2,), dtype=np.float32),
            "body_xpos": np.ones((3, 3), dtype=np.float32),
            "body_xquat": np.ones((3, 4), dtype=np.float32),
            "site_xpos": np.ones((4, 3), dtype=np.float32),
            "ctrl": np.ones((1,), dtype=np.float32),
        },
    ]
    manifest = write_sim_state_archive(
        tmp_path / "sim_state_stream.npz",
        states,
        {"body_names": ["a", "b", "c"], "site_names": ["s"], "joint_names": ["j"]},
    )
    assert manifest["steps"] == 2
    assert manifest["arrays"]["qpos"] == [2, 2]
    assert len(manifest["sha256"]) == 64
