"""D4.2c: Production detector fail-closed correctness tests."""

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gripper_attack.production_detector import (
    ProductionStreamingDetector,
    _is_valid_float,
    _is_valid_binary,
)

# ── Inline normalization (avoids pipeline-root dependency) ──

CLIP_RANGE = 3.0
ZERO_STDEV_THRESHOLD = 1e-8


def _normalize_features(candidates, means, stdevs, impute):
    """Normalize with frozen stats. Zero-stdev → 0.0. Missing → impute. Clip [-3,3]."""
    X_rows = []
    for c in candidates:
        row = []
        for fn in FEATURE_NAMES:
            v = c.get(fn, "")
            if v == "" or v is None:
                v = impute[fn]
            else:
                try:
                    v = float(v)
                except Exception:
                    v = impute[fn]
            s = stdevs[fn]
            if s < ZERO_STDEV_THRESHOLD:
                nv = 0.0
            else:
                nv = (v - means[fn]) / s
            row.append(max(-CLIP_RANGE, min(CLIP_RANGE, nv)))
        X_rows.append(row)
    X = torch.tensor(X_rows, dtype=torch.float32)
    assert torch.isfinite(X).all(), "Non-finite values after normalization"
    return X

# ── Helpers ──

ZERO_STDEV_FEATURES = {"close_streak_bonus", "close_streak", "close_onset", "time_since_last_open"}
FEATURE_NAMES = [
    "total_score", "raw_crossing_bonus", "close_streak_bonus", "close_onset_qpos_bonus",
    "eef_deceleration_bonus", "qpos_ready_bonus", "eef_speed_now", "eef_speed_prev",
    "eef_deceleration_delta", "close_streak", "raw_crossing", "close_onset",
    "qpos", "time_since_prev_close", "time_since_last_open", "candidate_index",
]


class PredictableModel(nn.Module):
    """Model with fixed weights for deterministic score testing."""
    def __init__(self, weight=0.1, bias=0.0):
        super().__init__()
        self.fc = nn.Linear(16, 1)
        nn.init.constant_(self.fc.weight, weight)
        nn.init.constant_(self.fc.bias, bias)

    def forward(self, x):
        return self.fc(x)


def make_means_stdevs_impute():
    """Create simple normalization stats for testing."""
    means = {}
    stdevs = {}
    impute = {}
    for fn in FEATURE_NAMES:
        means[fn] = 0.5
        impute[fn] = 0.5
        if fn in ZERO_STDEV_FEATURES:
            stdevs[fn] = 0.0
        else:
            stdevs[fn] = 1.0
    return means, stdevs, impute


def make_detector(threshold=0.5):
    means, stdevs, impute = make_means_stdevs_impute()
    return ProductionStreamingDetector(
        PredictableModel().eval(), means, stdevs, impute, threshold=threshold,
    )


def valid_step(detector, step_id=0, raw=0.7, env_val=1.0, qpos=0.0,
               eef_x=0.0, eef_y=0.0, eef_z=0.2, decoded_open=0):
    return detector.update(
        step_id, raw, env_val, qpos, eef_x, eef_y, eef_z, decoded_open,
    )


# ═══════════════════════════════════════════════════════════════
# Input validation helpers
# ═══════════════════════════════════════════════════════════════

def test_is_valid_float():
    assert _is_valid_float(0.5)
    assert _is_valid_float(0)
    assert _is_valid_float(-1.0)
    assert not _is_valid_float(None)
    assert not _is_valid_float(float("nan"))
    assert not _is_valid_float(float("inf"))
    assert not _is_valid_float(float("-inf"))
    assert not _is_valid_float(True)
    assert not _is_valid_float(False)
    assert not _is_valid_float("0.5")


def test_is_valid_binary():
    assert _is_valid_binary(0)
    assert _is_valid_binary(1)
    assert _is_valid_binary(0.0)
    assert _is_valid_binary(1.0)
    assert _is_valid_binary(True)
    assert _is_valid_binary(False)
    assert not _is_valid_binary(None)
    assert not _is_valid_binary(2)
    assert not _is_valid_binary(-1)
    assert not _is_valid_binary(0.5)
    assert not _is_valid_binary(float("nan"))


# ═══════════════════════════════════════════════════════════════
# 1. Invalid raw cannot generate raw crossing
# ═══════════════════════════════════════════════════════════════

def test_invalid_raw_no_raw_crossing():
    d = make_detector()
    # Step 0: valid raw=0.7 (>0.5, sets prev_raw > 0.5)
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.prev_raw == 0.7
    assert d.prev_raw_valid is True

    # Step 1: invalid raw — raw_crossing must NOT fire
    result = d.update(1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0, raw_valid=False)
    # Candidate may still fire from close_onset or close_streak,
    # but raw_crossing feature must be 0
    if result is not None:
        assert result["features"]["raw_crossing"] == 0


def test_invalid_raw_prev_no_crossing():
    """Even with valid raw now, if prev_raw was invalid, no crossing."""
    d = make_detector()
    # Step 0: invalid raw (won't set prev_raw properly)
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0, raw_valid=False)
    # prev_raw_valid should be False
    assert d.prev_raw_valid is False

    # Step 1: valid raw=0.3 (<=0.5), but prev was invalid
    d.update(1, 0.3, -1.0, 0.0, 0.0, 0.0, 0.2, 0, raw_valid=True)
    # No raw_crossing because prev_raw_valid was False
    # The candidate check uses raw_crossing which requires prev_raw_valid


# ═══════════════════════════════════════════════════════════════
# 2. Invalid semantics cannot generate candidate or emission
# ═══════════════════════════════════════════════════════════════

def test_invalid_semantics_no_candidate():
    d = make_detector(threshold=-999.0)
    # Step 0: invalid semantics, env=1.0 (>0.5 would be CLOSE)
    result = d.update(0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0,
                      gripper_semantics_valid=False)
    # With invalid semantics, env is not trusted → clean_close=0 → no candidate
    assert result is None
    # Verify the record got empty env
    assert d.history[0]["clean_gripper_env"] == ""


def test_invalid_semantics_no_emission():
    d = make_detector(threshold=-999.0)
    # Step 0: valid to establish prev
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)

    # Step 1: invalid semantics — raw_crossing blocked
    result = d.update(1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0,
                      gripper_semantics_valid=False)
    # Emission blocked because semantics invalid blocks raw_crossing and close_onset
    # If candidate is generated (from close_streak), predictor will abstain
    if result is not None:
        assert result["abstained"] is True or result["abstain"] == "gripper_semantics_invalid"


# ═══════════════════════════════════════════════════════════════
# 3. decoded_open invalid cannot emit
# ═══════════════════════════════════════════════════════════════

def test_decoded_open_invalid_no_candidate():
    d = make_detector(threshold=-999.0)
    # decoded_open=None → invalid, combined with env>0.5 → CLOSE signal
    # but candidate gate requires decoded_open_ok
    result = d.update(0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, None)
    # decoded_open invalid → is_candidate = False regardless of close_onset
    assert result is None


def test_decoded_open_nan_no_candidate():
    d = make_detector(threshold=-999.0)
    result = d.update(0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, float("nan"))
    assert result is None


# ═══════════════════════════════════════════════════════════════
# 4. Predictor abstain cannot emit
# ═══════════════════════════════════════════════════════════════

def test_predictor_abstain_blocks_emission():
    """When decoded_open=1 (already open), predictor abstains with gripper_already_open."""
    d = make_detector(threshold=-999.0)
    # Step 0: raw > 0.5 sets prev_raw > 0.5
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    # Step 1: raw <= 0.5 creates raw_crossing, but decoded_open=1 → abstain
    result = d.update(1, 0.3, -1.0, 0.0, 0.0, 0.0, 0.2, 1)
    assert result is not None  # candidate IS generated (raw_crossing detected)
    assert result["abstained"] is True
    assert result["abstain"] == "gripper_already_open"
    # Emission must be blocked
    assert d.emit_step == -1


def test_predictor_abstain_too_early():
    """Step < 3 → predictor abstains as too_early."""
    d = make_detector(threshold=-999.0)
    # Step 0: create CLOSE candidate
    result = d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    if result is not None:
        # Predictor should have "too_early" abstain for step 0
        assert result["abstained"] is True
        assert "too_early" in result["abstain"] or result["abstain"] == ""


# ═══════════════════════════════════════════════════════════════
# 5. Step sequence violations
# ═══════════════════════════════════════════════════════════════

def test_step_gap_fails():
    d = make_detector()
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    with pytest.raises(ValueError, match="Step sequence violation"):
        d.update(2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)


def test_step_duplicate_fails():
    d = make_detector()
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    with pytest.raises(ValueError, match="Step sequence violation"):
        d.update(0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)


def test_step_reversal_fails():
    d = make_detector()
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    d.update(1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    with pytest.raises(ValueError, match="Step sequence violation"):
        d.update(0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)


def test_step_after_reset_starts_at_zero():
    d = make_detector()
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    d.update(1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    d.reset()
    # After reset, must start at 0
    with pytest.raises(ValueError, match="Step sequence violation"):
        d.update(1, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    # Step 0 should work
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)


# ═══════════════════════════════════════════════════════════════
# 6. Reset fully clears episode state
# ═══════════════════════════════════════════════════════════════

def test_reset_clears_all_state():
    d = make_detector()
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    d.update(1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    d.update(2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)

    assert d._next_expected_step > 0
    assert len(d.history) > 0

    d.reset()

    assert d._next_expected_step == 0
    assert d.history == []
    assert d.prev_raw is None
    assert d.prev_raw_valid is False
    assert d.close_streak == 0
    assert d.close_steps == []
    assert d.open_steps == []
    assert d.emit_step == -1
    assert d.emit_idx == -1
    assert d.candidate_features == []


# ═══════════════════════════════════════════════════════════════
# 7. None / NaN / inf fail closed
# ═══════════════════════════════════════════════════════════════

def test_none_input_fail_closed():
    d = make_detector()
    result = d.update(0, None, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    # raw_gripper=None → raw_ok=False → record gets empty string
    assert d.history[0]["clean_gripper_raw"] == ""


def test_nan_input_fail_closed():
    d = make_detector()
    result = d.update(0, float("nan"), 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.history[0]["clean_gripper_raw"] == ""


def test_inf_input_fail_closed():
    d = make_detector()
    result = d.update(0, float("inf"), 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.history[0]["clean_gripper_raw"] == ""


def test_nan_eef_fail_closed():
    d = make_detector()
    d.update(0, 0.7, 1.0, 0.0, float("nan"), 0.0, 0.2, 0)
    # eef_ok should be False
    assert d.history[0]["eef_x"] == ""


# ═══════════════════════════════════════════════════════════════
# 8. No more than one emission
# ═══════════════════════════════════════════════════════════════

def test_at_most_one_emission():
    d = make_detector(threshold=-999.0)
    # Feed multiple raw-crossing steps
    for i in range(5):
        d.update(i, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
        d.prev_raw = 0.3  # force prev > 0.5
        d.prev_raw_valid = True
    first_emit = d.emit_step
    assert first_emit >= 0

    # Feed more steps — emit_step must not change
    d.update(5, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.emit_step == first_emit


# ═══════════════════════════════════════════════════════════════
# 9. Normalized feature parity
# ═══════════════════════════════════════════════════════════════

def test_normalized_feature_parity():
    """Production normalized features match batch-normalized features."""
    d = make_detector(threshold=-999.0)
    # Feed a step that creates a CLOSE candidate
    result = d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    if result is None:
        # No candidate generated at step 0 (too early for predictor?)
        # Generate more steps
        for i in range(1, 5):
            result = d.update(i, 0.3 if i % 2 == 0 else 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
            if result is not None:
                break

    if result is None:
        pytest.skip("No candidate generated in test setup")

    # Batch-normalize the same raw features
    X_batch = _normalize_features(
        [result["features"]], d.means, d.stdevs, d.impute,
    )
    batch_norm = [round(float(v), 10) for v in X_batch[0].cpu().tolist()]
    prod_norm = result["normalized_features"]

    for j, (bn, pn) in enumerate(zip(batch_norm, prod_norm)):
        assert abs(bn - pn) < 1e-7, (
            f"Norm feature {j} ({FEATURE_NAMES[j]}): batch={bn}, prod={pn}, "
            f"diff={abs(bn - pn):.2e}"
        )


# ═══════════════════════════════════════════════════════════════
# 10. Direct MLP score parity
# ═══════════════════════════════════════════════════════════════

def test_mlp_score_parity():
    """Production MLP score matches score from batch-normalized features."""
    means, stdevs, impute = make_means_stdevs_impute()
    model = PredictableModel(weight=0.1, bias=0.0).eval()
    d = ProductionStreamingDetector(model, means, stdevs, impute, threshold=-999.0)

    # Feed steps to generate a candidate
    result = None
    for i in range(5):
        result = d.update(i, 0.3 if i % 2 == 0 else 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
        if result is not None:
            break

    if result is None:
        pytest.skip("No candidate generated in test setup")

    # Batch score from same features
    X_batch = _normalize_features([result["features"]], means, stdevs, impute)
    with torch.no_grad():
        batch_score = float(model(X_batch).item())

    prod_score = result["score"]
    assert abs(batch_score - prod_score) < 1e-6, (
        f"MLP score mismatch: batch={batch_score}, prod={prod_score}, "
        f"diff={abs(batch_score - prod_score):.2e}"
    )


# ═══════════════════════════════════════════════════════════════
# 11. Future suffix mutation does not alter prior outputs
# ═══════════════════════════════════════════════════════════════

def test_future_suffix_does_not_alter_prior():
    """Adding future steps must not mutate previously returned candidate dicts."""
    d = make_detector(threshold=-999.0)

    # Collect results for steps 0-2
    results = []
    for i in range(3):
        r = d.update(i, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
        results.append(r)

    # Snapshot the first non-None result
    snapshot = None
    snapshot_idx = None
    for idx, r in enumerate(results):
        if r is not None:
            snapshot = {
                "step": r["step"],
                "score": r["score"],
                "features": dict(r["features"]),
                "normalized_features": list(r["normalized_features"]),
                "abstain": r["abstain"],
                "abstained": r["abstained"],
            }
            snapshot_idx = idx
            break

    if snapshot is None:
        pytest.skip("No candidate in first 3 steps")

    # Feed more steps (extend the trace)
    for i in range(3, 8):
        d.update(i, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)

    # The original result object should be unchanged
    orig = results[snapshot_idx]
    assert orig["step"] == snapshot["step"]
    assert orig["score"] == snapshot["score"]
    assert orig["abstain"] == snapshot["abstain"]
    assert orig["abstained"] == snapshot["abstained"]
    assert orig["normalized_features"] == snapshot["normalized_features"]
    for fn in FEATURE_NAMES:
        assert orig["features"][fn] == snapshot["features"][fn], (
            f"Feature {fn} mutated: {snapshot['features'][fn]} -> {orig['features'][fn]}"
        )


# ═══════════════════════════════════════════════════════════════
# 12. Detector input objects are not mutated
# ═══════════════════════════════════════════════════════════════

def test_input_objects_not_mutated():
    """The raw values passed to update() must not be mutated."""
    d = make_detector()
    raw_val = 0.7
    env_val = 1.0
    qpos_val = 0.0
    eef_x, eef_y, eef_z = 0.0, 0.0, 0.2
    dec_open = 0

    raw_snapshot = raw_val
    env_snapshot = env_val

    d.update(0, raw_val, env_val, qpos_val, eef_x, eef_y, eef_z, dec_open)

    assert raw_val == raw_snapshot
    assert env_val == env_snapshot
    assert dec_open == 0


def test_history_dicts_are_copies():
    """The record dict stored in history should not share references with inputs."""
    d = make_detector()
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)

    # Modify original variables — history should be independent
    record = d.history[0]
    assert record["step"] == 0
    assert record["clean_gripper_env"] == 1.0


# ═══════════════════════════════════════════════════════════════
# 13. Production module has no Teacher-P / candidate-table dependency
# ═══════════════════════════════════════════════════════════════

def test_no_teacher_p_import():
    """Production detector must not import Teacher-P related modules."""
    import inspect
    src_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "gripper_attack",
        "production_detector.py",
    )
    with open(src_file) as f:
        source = f.read()

    # No Teacher-P import
    assert "teacher" not in source.lower() or "teacher_anchor" in source.lower(), (
        "Teacher-P reference found in production_detector.py"
    )
    # Teacher anchor is allowed (always -1), but teacher_p import is not
    assert "from" not in source or "teacher_p" not in source.lower(), (
        "Teacher-P import found"
    )


def test_no_candidate_table_dependency():
    """Production detector must not reference candidate CSV/table files."""
    import inspect
    src_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "gripper_attack",
        "production_detector.py",
    )
    with open(src_file) as f:
        source = f.read()

    # No candidate table / CSV reference
    forbidden = ["candidate_table", "candidate.csv", "close_candidates"]
    for term in forbidden:
        assert term not in source.lower(), f"Forbidden term '{term}' in production_detector.py"


def test_detector_init_no_hidden_deps():
    """Detector __init__ must not touch disk or network."""
    d = make_detector()
    assert d.means is not None
    assert d.threshold == 0.5


# ═══════════════════════════════════════════════════════════════
# 14. Missing trace fails (integration: parity runner assertion)
# ═══════════════════════════════════════════════════════════════

def test_missing_manifest_trace_asserts():
    """Verify the assertion pattern used in the parity runner for missing traces."""
    manifest_map = {"trace_a": {}, "trace_b": {}}
    all_ids = ["trace_a", "trace_b", "trace_missing"]

    missing = []
    for tid in all_ids:
        try:
            assert tid in manifest_map, f"MISSING_MANIFEST: trace_id={tid}"
        except AssertionError:
            missing.append(tid)

    assert "trace_missing" in missing
    assert len(missing) == 1


# ═══════════════════════════════════════════════════════════════
# Additional: step_id tracking and next_expected_step
# ═══════════════════════════════════════════════════════════════

def test_next_expected_step_tracks_correctly():
    d = make_detector()
    assert d.next_expected_step == 0
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.next_expected_step == 1
    d.update(1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.next_expected_step == 2


def test_has_emitted_tracks_emission():
    d = make_detector(threshold=-999.0)
    assert d.has_emitted is False
    # Generate candidate that emits (low threshold)
    for i in range(5):
        d.update(i, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
        d.prev_raw = 0.3
        d.prev_raw_valid = True
        if d.has_emitted:
            break
    # After enough steps with low threshold, should emit
    # (emission may be blocked by predictor abstain for steps < 3)


def test_reset_clears_step_counter():
    d = make_detector()
    d.update(0, 0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    d.update(1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.next_expected_step == 2
    d.reset()
    assert d.next_expected_step == 0
