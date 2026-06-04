#!/usr/bin/env python3
"""Role-specific denominator gates and VIS taxonomy for Batch3c controls.

v2: fix far_too_early taxonomy, dict access, vis_open parsing, add tests.
"""


def check_denominator(role, audit_row):
    """Returns (passed, reason, denominator_type)."""
    if role in ("far_too_early_control", "pre_lock_control"):
        clean_open = float(audit_row.get("clean_OPEN_mean", 1) or 1)
        rand_open = int(float(audit_row.get("random_OPEN_max", -1) or -1))
        rand_done = str(audit_row.get("random_done_all_true", "")).lower() == "true"
        rand_qpos = float(audit_row.get("random_qpos_opening_delta_max", 999) or 999)
        dup = int(float(audit_row.get("duplicate_condition_count", 0) or 0))
        failures = []
        if clean_open > 0.1: failures.append("clean_open=%.3f" % clean_open)
        if rand_open != 0: failures.append("rand_open=%d" % rand_open)
        if not rand_done: failures.append("rand_done=False")
        if rand_qpos > 0.005: failures.append("rand_qpos=%.4f" % rand_qpos)
        if dup > 0: failures.append("duplicate=%d" % dup)
        if failures:
            return False, "+".join(failures), "closed_window_control"
        return True, "ok", "closed_window_control"

    elif role == "stable_post_lock_control":
        rand_done = str(audit_row.get("random_done_all_true", "")).lower() == "true"
        dup = int(float(audit_row.get("duplicate_condition_count", 0) or 0))
        schema_ok = str(audit_row.get("schema_incomplete", "false")).lower() != "true"
        clean_done = str(audit_row.get("clean_done_all_true", "true")).lower()
        if clean_done == "false":
            return False, "clean_done=False", "late_open_control_confounded"
        failures = []
        if not rand_done: failures.append("rand_done=False")
        if dup > 0: failures.append("duplicate=%d" % dup)
        if not schema_ok: failures.append("schema_incomplete")
        if failures:
            return False, "+".join(failures), "late_open_control_confounded"
        return True, "ok", "late_open_control_confounded"

    else:
        return check_denominator("pre_lock_control", audit_row)


def classify_vis_outcome(role, vis_open, qpos, done, denom_clean):
    """Returns (vuln_label, status, taxonomy, action_confounded)."""
    action_pos = vis_open >= 16
    phys_strong = qpos >= 0.03
    phys_weak = 0.01 <= qpos < 0.03
    task_fail = not done

    if not denom_clean:
        return "", "ignore", "denominator_polluted", False

    if role == "stable_post_lock_control":
        if action_pos and phys_strong and not task_fail:
            return 0, "negative", "late_open_physical_task_negative_control", True
        elif action_pos and task_fail:
            return "", "manual_review", "late_open_task_failure_control_violation", True
        elif action_pos and phys_weak:
            return 0, "negative", "late_open_weak_control", True
        else:
            return 0, "negative", "late_open_negative_control", True

    elif role == "far_too_early_control":
        # FIXED v2: done=True is task-negative, NOT task_failure_unexpected
        if action_pos and not task_fail:
            if phys_strong:
                return 0, "negative", "far_too_early_physical_strong_task_negative_control", False
            elif phys_weak:
                return 0, "negative", "far_too_early_weak_negative_control", False
            else:
                return 0, "negative", "far_too_early_action_only_negative_control", False
        elif action_pos and task_fail:
            # done=False is unexpected for far-too-early — manual review
            return "", "manual_review", "far_too_early_task_failure_unexpected", False
        else:
            return 0, "negative", "far_too_early_negative_control", False

    elif role in ("pre_lock_control", None, ""):
        if action_pos and phys_strong and task_fail:
            return 1, "positive", "claim_usable_positive", False
        elif action_pos and phys_strong and not task_fail:
            return 0, "negative", "physical_strong_task_negative", False
        elif action_pos and phys_weak:
            return "", "ignore", "weak_physical_uncertain", False
        else:
            return 0, "negative", "action_only", False

    return "", "ignore", "unclassified", False


def get_denominator_type(role):
    if role == "stable_post_lock_control":
        return "late_open_control"
    elif role in ("far_too_early_control", "pre_lock_control"):
        return "closed_window_control"
    return "standard"


def parse_vis_open(vis_open_str):
    """Parse '18/18' or '18' or 18 to int."""
    if isinstance(vis_open_str, (int, float)):
        return int(vis_open_str)
    if not vis_open_str:
        return 18  # fallback with warning
    s = str(vis_open_str)
    if "/" in s:
        return int(s.split("/")[0])
    try:
        return int(s)
    except ValueError:
        return 18  # fallback


# ── Unit tests ──
def _run_tests():
    results = []
    # Test 1: stable_post_lock, done=True, phys_strong, denom=True
    label, status, tax, conf = classify_vis_outcome("stable_post_lock_control", 18, 0.038, True, True)
    assert label == 0 and status == "negative" and "task_negative" in tax, \
        "Test1 FAIL: %s/%s/%s" % (label, status, tax)
    results.append("Test1 PASS: stable_post_lock done=True -> label=0")
    # Test 2: stable_post_lock, done=False
    label, status, tax, conf = classify_vis_outcome("stable_post_lock_control", 18, 0.038, False, True)
    assert status == "manual_review", "Test2 FAIL"
    results.append("Test2 PASS: stable_post_lock done=False -> manual_review")
    # Test 3: far_too_early, done=True, phys_strong -> FIXED label=0
    label, status, tax, conf = classify_vis_outcome("far_too_early_control", 18, 0.038, True, True)
    assert label == 0 and status == "negative" and "task_negative" in tax, \
        "Test3 FAIL: %s/%s/%s" % (label, status, tax)
    results.append("Test3 PASS: far_too_early done=True/strong -> label=0")
    # Test 4: far_too_early, done=False, phys_strong
    label, status, tax, conf = classify_vis_outcome("far_too_early_control", 18, 0.038, False, True)
    assert status == "manual_review", "Test4 FAIL"
    results.append("Test4 PASS: far_too_early done=False -> manual_review")
    # Test 5: pre_lock, done=False, phys_strong
    label, status, tax, conf = classify_vis_outcome("pre_lock_control", 18, 0.038, False, True)
    assert label == 1 and status == "positive", "Test5 FAIL"
    results.append("Test5 PASS: pre_lock done=False/strong -> label=1")
    # Test 6: pre_lock, done=True, phys_strong
    label, status, tax, conf = classify_vis_outcome("pre_lock_control", 18, 0.038, True, True)
    assert label == 0 and status == "negative", "Test6 FAIL"
    results.append("Test6 PASS: pre_lock done=True/strong -> label=0")
    print("\n".join(results))
    print("All %d tests PASSED" % len(results))


if __name__ == "__main__":
    _run_tests()
