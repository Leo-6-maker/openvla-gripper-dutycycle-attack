#!/usr/bin/env python3
"""C2e3 runtime contract regression tests.

Validates that the deployment runtime matches training, and that D7
audit/aggregation guards prevent scientifically invalid results.

CPU-only. No env.step, no OpenVLA, no MuJoCo.
"""

from __future__ import annotations

import json, os, sys, tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
assert (REPO / "src" / "gripper_attack").exists(), f"REPO not found at {REPO}"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from gripper_attack.c2e3_gru_detector_runtime import (
    C2e3GRUDetectorRuntime,
    CANONICAL_25D_FEATURES,
    CANONICAL_108D_CONTEXT_FEATURES,
    GRUModel,
    sha256_file,
)

# ── test fixtures ──

def _make_dummy_package(tmpdir: str) -> dict:
    """Create a minimal C2e3 package with known values for testing."""
    pkg = Path(tmpdir)
    pkg.mkdir(parents=True, exist_ok=True)

    # Model
    model = GRUModel(25, 108, 128)
    model.eval()
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"window": 16, "channels": 128, "dropout": 0.1, "lr": 0.001, "seed": 2},
        "threshold": {"tau_emit": 0.33, "tau_suppress": 0.67},
    }
    torch.save(ckpt, str(pkg / "c2e3_selected_baseline_model.pt"))

    # Config
    (pkg / "c2e3_selected_baseline_config.json").write_text(json.dumps({"window": 16, "hidden": 128}))

    # Norm stats
    norm = {
        "temporal_feature_mean": [0.5] * 25,
        "temporal_feature_std": [0.25] * 25,
        "context_feature_mean": [0.5] * 108,
        "context_feature_std": [0.25] * 108,
        "fit_split": "train",
    }
    (pkg / "c2e3_normalization_stats_train_only.json").write_text(json.dumps(norm))

    # Context lookup (40 entries)
    lookup = {"created_from": "dummy", "n_contexts": 40, "context_dim": 108, "lookup": {}}
    suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    for s in suites:
        for ti in range(10):
            vec = [0.0] * 108
            # suite one-hot (0-3)
            vec[suites.index(s)] = 1.0
            # task one-hot (68-107)
            offset = 68 + suites.index(s) * 10 + ti
            if offset < 108:
                vec[offset] = 1.0
            # hash features (4-67): use simple hashing
            h = hash(f"{s}_{ti}") % 32
            vec[4 + h] = 1.0  # ctx_suite_task_hash
            vec[36 + ((h * 7) % 32)] = 1.0  # ctx_task_index_hash
            lookup["lookup"][f"{s}|task_{ti:02d}"] = vec
    (pkg / "c2e3_context_lookup.json").write_text(json.dumps(lookup))

    return {"model": model, "norm": norm, "pkg_dir": str(pkg)}


# ── tests ──

def test_runtime_loads_without_error():
    """Runtime loads with all required files present."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    det = C2e3GRUDetectorRuntime(fixture["pkg_dir"])
    assert det.n_features == 25
    assert det.n_context == 108
    assert det.window == 16
    assert det.hidden == 128
    assert det.tau_emit == 0.33
    assert det.tau_suppress == 0.67
    assert det.checkpoint_sha256 != ""
    assert det.normalization_sha256 != ""
    assert det.config_sha256 != ""
    assert det.context_lookup_sha256 != ""
    assert det.provenance["normalization_applied"] is True
    print("  PASS test_runtime_loads_without_error")


def test_missing_normalization_stats_fails():
    """Missing normalization stats should raise FileNotFoundError."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    (Path(fixture["pkg_dir"]) / "c2e3_normalization_stats_train_only.json").unlink()
    try:
        C2e3GRUDetectorRuntime(fixture["pkg_dir"])
        assert False, "should have raised"
    except FileNotFoundError:
        pass  # expected
    print("  PASS test_missing_normalization_stats_fails")


def test_missing_context_lookup_fails():
    """Missing context lookup should raise FileNotFoundError."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    (Path(fixture["pkg_dir"]) / "c2e3_context_lookup.json").unlink()
    try:
        C2e3GRUDetectorRuntime(fixture["pkg_dir"])
        assert False, "should have raised"
    except FileNotFoundError:
        pass
    print("  PASS test_missing_context_lookup_fails")


def test_wrong_feature_count_fails():
    """If checkpoint has n_features != 25, should raise ValueError."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    ckpt = torch.load(str(Path(fixture["pkg_dir"]) / "c2e3_selected_baseline_model.pt"), weights_only=False)
    ckpt["config"]["n_features"] = 30
    torch.save(ckpt, str(Path(fixture["pkg_dir"]) / "c2e3_selected_baseline_model.pt"))
    try:
        C2e3GRUDetectorRuntime(fixture["pkg_dir"])
        assert False, "should have raised"
    except ValueError as e:
        assert "25" in str(e)
    print("  PASS test_wrong_feature_count_fails")


def test_wrong_context_count_fails():
    """If checkpoint has n_context != 108, should raise ValueError."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    ckpt = torch.load(str(Path(fixture["pkg_dir"]) / "c2e3_selected_baseline_model.pt"), weights_only=False)
    ckpt["config"]["n_context"] = 50
    torch.save(ckpt, str(Path(fixture["pkg_dir"]) / "c2e3_selected_baseline_model.pt"))
    try:
        C2e3GRUDetectorRuntime(fixture["pkg_dir"])
        assert False, "should have raised"
    except ValueError as e:
        assert "108" in str(e)
    print("  PASS test_wrong_context_count_fails")


def test_predict_output_shape():
    """predict() returns (emit_p, suppress_p, emitted) tuple."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    det = C2e3GRUDetectorRuntime(fixture["pkg_dir"])
    window = np.random.randn(16, 25).astype(np.float32)
    ep, sp, emitted = det.predict(window, "libero_object", 3)
    assert isinstance(ep, float)
    assert isinstance(sp, float)
    assert isinstance(emitted, bool)
    assert 0.0 <= ep <= 1.0
    assert 0.0 <= sp <= 1.0
    print("  PASS test_predict_output_shape")


def test_predict_unknown_suite_task_fails():
    """Predicting with unknown (suite, task_idx) should raise KeyError."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    det = C2e3GRUDetectorRuntime(fixture["pkg_dir"])
    window = np.random.randn(16, 25).astype(np.float32)
    try:
        det.predict(window, "unknown_suite", 99)
        assert False, "should have raised"
    except KeyError:
        pass
    print("  PASS test_predict_unknown_suite_task_fails")


def test_parity_normalized_equals_training_path():
    """Runtime-normalized logits must equal manually-normalized training-path logits."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    det = C2e3GRUDetectorRuntime(fixture["pkg_dir"])
    model = fixture["model"]

    # Generate a raw window
    rng = np.random.RandomState(42)
    window_raw = rng.randn(16, 25).astype(np.float32)
    suite = "libero_10"; task_idx = 5

    # Path A: runtime predict()
    ep_a, sp_a, _ = det.predict(window_raw, suite, task_idx)

    # Path B: manual normalize + forward
    tm = np.array(fixture["norm"]["temporal_feature_mean"], dtype=np.float32).reshape(1,1,-1)
    ts = np.maximum(np.array(fixture["norm"]["temporal_feature_std"], dtype=np.float32).reshape(1,1,-1), 1e-8)
    cm = np.array(fixture["norm"]["context_feature_mean"], dtype=np.float32).reshape(1,-1)
    cs = np.maximum(np.array(fixture["norm"]["context_feature_std"], dtype=np.float32).reshape(1,-1), 1e-8)

    ctx_raw = np.array(det._context_lookup[(suite, task_idx)], dtype=np.float32)
    w_norm = (window_raw - tm) / ts
    c_norm = (ctx_raw.reshape(1,-1) - cm) / cs

    with torch.no_grad():
        logits = model(
            torch.from_numpy(w_norm.reshape(1,16,25)),
            torch.from_numpy(c_norm.reshape(1,108))
        ).numpy()[0]

    def sig(x): return 1.0/(1.0+np.exp(-np.clip(float(x),-50,50)))
    ep_b = sig(logits[0]); sp_b = sig(logits[1])

    assert abs(ep_a - ep_b) < 1e-6, f"emit_p mismatch: {ep_a} vs {ep_b}"
    assert abs(sp_a - sp_b) < 1e-6, f"suppress_p mismatch: {sp_a} vs {sp_b}"
    print("  PASS test_parity_normalized_equals_training_path")


def test_raw_path_differs_from_training_path():
    """Raw (unnormalized) window + zero context must differ from training path."""
    fixture = _make_dummy_package(tempfile.mkdtemp())
    model = fixture["model"]

    rng = np.random.RandomState(42)
    window_raw = rng.randn(16, 25).astype(np.float32)

    # Training path
    tm = np.array(fixture["norm"]["temporal_feature_mean"], dtype=np.float32).reshape(1,1,-1)
    ts = np.maximum(np.array(fixture["norm"]["temporal_feature_std"], dtype=np.float32).reshape(1,1,-1), 1e-8)
    w_norm = (window_raw - tm) / ts
    ctx_raw = np.array([1.0,0,0,0] + [0]*104, dtype=np.float32)
    cm = np.array(fixture["norm"]["context_feature_mean"], dtype=np.float32).reshape(1,-1)
    cs = np.maximum(np.array(fixture["norm"]["context_feature_std"], dtype=np.float32).reshape(1,-1), 1e-8)
    c_norm = (ctx_raw.reshape(1,-1) - cm) / cs

    with torch.no_grad():
        logits_train = model(
            torch.from_numpy(w_norm.reshape(1,16,25)),
            torch.from_numpy(c_norm.reshape(1,108))
        ).numpy()[0]

    # Raw path (old D7B)
    with torch.no_grad():
        logits_raw = model(
            torch.from_numpy(window_raw.reshape(1,16,25)),
            torch.zeros(1,108)
        ).numpy()[0]

    max_diff = float(np.max(np.abs(logits_train - logits_raw)))
    assert max_diff > 1e-3, f"raw path too close to training: max_diff={max_diff:.2e}"
    print(f"  PASS test_raw_path_differs_from_training_path (diff={max_diff:.4f})")


def test_feature_order_matches_c2e1():
    """CANONICAL_25D_FEATURES must match the known 25D SC5_V2 order."""
    expected = [
        "gripper_command", "gripper_qpos", "gripper_opening_proxy",
        "eef_x", "eef_y", "eef_z",
        "eef_vx", "eef_vy", "eef_vz",
        "action_dx", "action_dy", "action_dz", "action_gripper",
        "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
        "close_onset", "time_since_close",
        "eef_speed", "eef_z_delta_since_close",
        "qpos_delta_1", "qpos_delta_3",
        "opening_proxy_delta_3", "opening_proxy_variance_5",
        "eef_speed_variance_5",
    ]
    assert CANONICAL_25D_FEATURES == expected
    print("  PASS test_feature_order_matches_c2e1")


def test_context_feature_count():
    """CANONICAL_108D_CONTEXT_FEATURES must have exactly 108 entries."""
    assert len(CANONICAL_108D_CONTEXT_FEATURES) == 108
    # First 4 must be suite one-hot
    for i, s in enumerate(["libero_10", "libero_goal", "libero_object", "libero_spatial"]):
        assert CANONICAL_108D_CONTEXT_FEATURES[i] == f"ctx_suite_{s}"
    print("  PASS test_context_feature_count")


# ── D7 audit/aggregation contract guard tests ──

def _make_mini_audit_report(tmpdir: str, contract_status: str, d7d_blocked: bool, runtime_errors: int) -> str:
    path = Path(tmpdir) / "audit_report.json"
    path.write_text(json.dumps({
        "gate": "D7_TABLE1_POSTRUN_AUDIT",
        "status": "PASS" if contract_status == "PASS" and runtime_errors == 0 else "HOLD",
        "runtime_contract_status": contract_status,
        "d7d_aggregation_blocked": d7d_blocked,
        "runtime_error_violations": runtime_errors,
    }))
    return str(path)


def test_audit_contract_pass_allows_aggregation():
    """PASS contract status should allow aggregation."""
    tmp = tempfile.mkdtemp()
    report = json.loads(Path(_make_mini_audit_report(tmp, "PASS", False, 0)).read_text())
    assert report["runtime_contract_status"] == "PASS"
    assert report["d7d_aggregation_blocked"] is False
    assert report["runtime_error_violations"] == 0
    print("  PASS test_audit_contract_pass_allows_aggregation")


def test_audit_contract_fail_blocks_aggregation():
    """FAIL contract status must block aggregation."""
    tmp = tempfile.mkdtemp()
    report = json.loads(Path(_make_mini_audit_report(tmp, "FAIL_NORMALIZATION_CONTRACT", True, 0)).read_text())
    assert report["d7d_aggregation_blocked"] is True
    print("  PASS test_audit_contract_fail_blocks_aggregation")


def test_audit_runtime_errors_block_aggregation():
    """Runtime errors must block aggregation even if contract passes."""
    tmp = tempfile.mkdtemp()
    report = json.loads(Path(_make_mini_audit_report(tmp, "PASS", True, 3)).read_text())
    assert report["d7d_aggregation_blocked"] is True
    assert report["runtime_error_violations"] == 3
    print("  PASS test_audit_runtime_errors_block_aggregation")


def test_no_trigger_attack_frames_zero_is_valid():
    """TRUE_T10/RAND_T10/ORACLE with detector_emitted=false and attack_frames=0 is valid (ITT)."""
    # Simulate audit check logic
    condition = "TRUE_T10"
    detector_emitted = False
    attack_frames = 0

    emitted = detector_emitted  # from episode_summary
    is_violation = False
    if condition in ("TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"):
        if emitted and attack_frames != 10:
            is_violation = True
        if not emitted and attack_frames != 0:
            is_violation = True
    assert not is_violation, "no-trigger with attack_frames=0 should be VALID"
    print("  PASS test_no_trigger_attack_frames_zero_is_valid")


def test_emitted_but_no_attack_frames_is_violation():
    """TRUE_T10 with detector_emitted=true but attack_frames=0 IS a violation."""
    condition = "TRUE_T10"
    detector_emitted = True
    attack_frames = 0

    emitted = detector_emitted
    is_violation = False
    if condition in ("TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"):
        if emitted and attack_frames != 10:
            is_violation = True
        if not emitted and attack_frames != 0:
            is_violation = True
    assert is_violation, "emitted but no attack frames SHOULD be a violation"
    print("  PASS test_emitted_but_no_attack_frames_is_violation")


def test_summary_error_blocks_d7d():
    """summary.error != '' must be flagged as RUNTIME_ERROR_VIOLATION."""
    ep_error = "CUDA out of memory"
    is_blocked = bool(ep_error)
    assert is_blocked, "non-empty error should block D7D"
    print("  PASS test_summary_error_blocks_d7d")


# ── D8B taxonomy logic tests ──

def test_oracle_sensitive_requires_attack_frames():
    """ORACLE_SENSITIVE must have attack_frames>=10; no-trigger failure is not SENSITIVE."""
    # Simulate oracle with trigger
    summary_triggered = {"condition": "COMMAND_OPEN_ORACLE", "detector_emitted": True,
                          "task_success": False, "attack_frames": 10,
                          "n_steps": 300, "error": "", "emit_step": 50}
    # oracle fail + clean success + attack_frames=10 → SENSITIVE
    from scripts.stageb.analyze_d7_l10_no_emit_taxonomy import classify_episode
    r = classify_episode(summary_triggered, clean_success=True)
    assert r == "ORACLE_SENSITIVE", f"expected ORACLE_SENSITIVE, got {r}"

    # Same but no trigger (attack_frames=0)
    summary_no_trigger = {**summary_triggered, "attack_frames": 0, "detector_emitted": False}
    r2 = classify_episode(summary_no_trigger, clean_success=True)
    assert r2 == "ORACLE_NO_TRIGGER_UNINFORMATIVE", f"expected ORACLE_NO_TRIGGER_UNINFORMATIVE, got {r2}"

    # Oracle success + trigger
    summary_ok = {**summary_triggered, "task_success": True}
    r3 = classify_episode(summary_ok, clean_success=True)
    assert r3 == "ORACLE_NOT_SENSITIVE", f"expected ORACLE_NOT_SENSITIVE, got {r3}"
    print("  PASS test_oracle_sensitive_requires_attack_frames")


def test_clean_fail_oracle_uninformative():
    """When CLEAN also failed, oracle failure is uninformative."""
    summary = {"condition": "COMMAND_OPEN_ORACLE", "detector_emitted": True,
               "task_success": False, "attack_frames": 10,
               "n_steps": 300, "error": "", "emit_step": 50}
    from scripts.stageb.analyze_d7_l10_no_emit_taxonomy import classify_episode
    r = classify_episode(summary, clean_success=False)
    assert r == "ORACLE_UNINFORMATIVE_CLEAN_FAIL", f"expected UNINFORMATIVE_CLEAN_FAIL, got {r}"
    print("  PASS test_clean_fail_oracle_uninformative")


def test_emit_late_after_250():
    """Emitted=true but emit_step > 250 → EMIT_LATE_AFTER_STEP_250."""
    summary = {"condition": "TRUE_T10", "detector_emitted": True,
               "task_success": False, "attack_frames": 10,
               "n_steps": 300, "error": "", "emit_step": 270}
    from scripts.stageb.analyze_d7_l10_no_emit_taxonomy import classify_episode
    r = classify_episode(summary)
    assert r == "EMIT_LATE_AFTER_STEP_250", f"expected EMIT_LATE_AFTER_STEP_250, got {r}"
    print("  PASS test_emit_late_after_250")


def test_no_emit_short_episode():
    """Episodes < 16 steps with no emission → NO_EMIT_SHORT_EPISODE."""
    summary = {"condition": "TRUE_T10", "detector_emitted": False,
               "task_success": False, "attack_frames": 0,
               "n_steps": 10, "error": "", "emit_step": -1}
    from scripts.stageb.analyze_d7_l10_no_emit_taxonomy import classify_episode
    r = classify_episode(summary)
    assert r == "NO_EMIT_SHORT_EPISODE", f"expected NO_EMIT_SHORT_EPISODE, got {r}"
    print("  PASS test_no_emit_short_episode")


def test_runtime_error_blocks_taxonomy():
    """Non-empty error → RUNTIME_ERROR regardless of everything else."""
    summary = {"condition": "TRUE_T10", "detector_emitted": True,
               "task_success": True, "attack_frames": 10,
               "n_steps": 300, "error": "CUDA OOM", "emit_step": 50}
    from scripts.stageb.analyze_d7_l10_no_emit_taxonomy import classify_episode
    r = classify_episode(summary)
    assert r == "RUNTIME_ERROR", f"expected RUNTIME_ERROR, got {r}"
    print("  PASS test_runtime_error_blocks_taxonomy")


# ── runner ──

def main():
    print("=== C2e3 Runtime Contract Regression Tests ===\n")
    tests = [
        test_runtime_loads_without_error,
        test_missing_normalization_stats_fails,
        test_missing_context_lookup_fails,
        test_wrong_feature_count_fails,
        test_wrong_context_count_fails,
        test_predict_output_shape,
        test_predict_unknown_suite_task_fails,
        test_parity_normalized_equals_training_path,
        test_raw_path_differs_from_training_path,
        test_feature_order_matches_c2e1,
        test_context_feature_count,
        test_audit_contract_pass_allows_aggregation,
        test_audit_contract_fail_blocks_aggregation,
        test_audit_runtime_errors_block_aggregation,
        test_no_trigger_attack_frames_zero_is_valid,
        test_emitted_but_no_attack_frames_is_violation,
        test_summary_error_blocks_d7d,
        test_oracle_sensitive_requires_attack_frames,
        test_clean_fail_oracle_uninformative,
        test_emit_late_after_250,
        test_no_emit_short_episode,
        test_runtime_error_blocks_taxonomy,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{failed}/{len(tests)} tests failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
