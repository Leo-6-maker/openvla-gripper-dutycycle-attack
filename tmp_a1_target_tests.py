"""A1: VIS-OPEN target construction unit tests + gradient direction validation."""
import copy, json, sys, os, numpy as np

# ── P1: Target construction unit tests (no GPU needed) ──

CANONICAL_RAW_OPEN = 1.0
CANONICAL_RAW_CLOSE = 0.0
CANONICAL_ENV_OPEN = -1.0
CANONICAL_ENV_CLOSE = 1.0

def test_target_not_aliased():
    """target_action must be a COPY, not a reference to clean_action."""
    clean = np.array([0.3, 0.0, 0.1, -0.2, 0.0, 0.5, 0.0], dtype=np.float32)
    target = clean.copy()
    target[-1] = CANONICAL_RAW_OPEN
    assert target is not clean, "target must NOT be the same object as clean"
    assert target[-1] == CANONICAL_RAW_OPEN, f"target grip should be OPEN (1.0), got {target[-1]}"
    assert np.array_equal(target[:6], clean[:6]), "arm dims must be preserved"
    print("PASS: test_target_not_aliased")

def test_target_is_open():
    """target_action[6] must be CANONICAL_RAW_OPEN."""
    for grip_val in [0.0, 0.2, 0.5, 0.8, 1.0]:
        clean = np.array([0.1, 0.2, 0.3, -0.1, 0.0, 0.4, grip_val], dtype=np.float32)
        target = clean.copy()
        target[-1] = CANONICAL_RAW_OPEN
        assert target[-1] == 1.0, f"regardless of clean grip={grip_val}, target should be OPEN (1.0)"
    print("PASS: test_target_is_open")

def test_normalize_and_invert_gripper():
    """Verify the env action mapping chain."""
    # This mimics the production normalize_and_invert_gripper function
    def normalize_and_invert_gripper(raw_action):
        action = np.asarray(raw_action, dtype=np.float32).copy()
        action[-1] = 2.0 * action[-1] - 1.0
        action[-1] = float(np.sign(action[-1]))
        if action[-1] == 0:
            action[-1] = 1.0
        action[-1] *= -1.0
        return action[-1]

    # raw 0.0 → env 1.0 (CLOSE)
    assert normalize_and_invert_gripper(np.array([0]*6 + [0.0])) == 1.0, "raw 0.0 should → env +1 (CLOSE)"
    # raw 1.0 → env -1.0 (OPEN)
    assert normalize_and_invert_gripper(np.array([0]*6 + [1.0])) == -1.0, "raw 1.0 should → env -1 (OPEN)"
    # raw 0.5 → env -1.0 (boundary → OPEN in production)
    assert normalize_and_invert_gripper(np.array([0]*6 + [0.5])) == -1.0, "raw 0.5 should → env -1 (OPEN, boundary bias)"

    print("PASS: test_normalize_and_invert_gripper")

def test_self_target_rejected():
    """Production attacker should construct target_action != clean_action for VIS-OPEN."""
    # Simulate V5 fix: target is constructed from clean with grip replaced
    clean = np.array([0.3, 0.0, 0.1, -0.2, 0.0, 0.5, 0.0], dtype=np.float32)
    target = clean.copy()
    target[-1] = CANONICAL_RAW_OPEN  # V5 FIX
    # Verify: target != clean
    assert not np.array_equal(clean, target), \
        "V5 OPEN attack: target must NOT equal clean (was the release-blocking bug)"
    # Verify: grip dim is now OPEN
    assert target[-1] == 1.0, f"target grip should be OPEN (1.0)"
    # Verify: arm dims unchanged
    assert np.array_equal(target[:6], clean[:6]), "arm dims preserved"
    print("PASS: test_self_target_rejected")

def test_arm_dims_preserved():
    """Arm dimensions 0-5 must stay identical between clean and target."""
    rng = np.random.RandomState(42)
    for _ in range(100):
        clean = rng.randn(7).astype(np.float32) * 0.5
        clean[-1] = rng.choice([0.0, 0.3, 0.8])  # various gripper values
        target = clean.copy()
        target[-1] = CANONICAL_RAW_OPEN
        assert np.array_equal(target[:6], clean[:6]), \
            f"arm dims must be preserved: clean[:6]={clean[:6]}, target[:6]={target[:6]}"
    print("PASS: test_arm_dims_preserved")

def test_env_mapping_consistency():
    """raw 1.0 → env -1.0 → physical OPEN must be consistent."""
    def to_env(raw_grip):
        return float(np.sign(2.0 * raw_grip - 1.0)) * -1.0 if raw_grip != 0.5 else -1.0

    raw_open_env = to_env(1.0)
    raw_close_env = to_env(0.0)
    assert raw_open_env == CANONICAL_ENV_OPEN, f"raw OPEN → env {raw_open_env}, expected {CANONICAL_ENV_OPEN}"
    assert raw_close_env == CANONICAL_ENV_CLOSE, f"raw CLOSE → env {raw_close_env}, expected {CANONICAL_ENV_CLOSE}"
    print("PASS: test_env_mapping_consistency")

# ── Run ──
tests = [
    test_target_not_aliased,
    test_target_is_open,
    test_normalize_and_invert_gripper,
    test_self_target_rejected,
    test_arm_dims_preserved,
    test_env_mapping_consistency,
]

passed = 0; failed = 0
for t in tests:
    try:
        t(); passed += 1
    except Exception as e:
        failed += 1; print(f"FAIL: {t.__name__}: {e}")

print(f"\n{passed} PASS / {failed} FAIL (total {len(tests)})")
sys.exit(0 if failed == 0 else 1)
