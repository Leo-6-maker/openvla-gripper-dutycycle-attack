#!/usr/bin/env python3
"""Role-specific denominator gates and VIS taxonomy for Batch3c controls.

stable_post_lock: late/open controls, high clean_open_ratio expected.
far_too_early: approach-phase CLOSED controls.
pre_lock: standard closed-window taxonomy.
"""

# ── Role-specific denominator checks ──
def check_denominator(role, audit_row):
    """Returns (passed, reason, denominator_type)."""
    denom = {}

    if role in ("far_too_early_control", "pre_lock_control"):
        # Standard closed-window gate
        clean_open = float(audit_row.get("clean_OPEN_mean", 1) or 1)
        rand_open = int(float(audit_row.get("random_OPEN_max", -1) or -1))
        rand_done = str(audit_row.get("random_done_all_true", "")).lower() == "true"
        rand_qpos = float(audit_row.get("random_qpos_opening_delta_max", 999) or 999)
        dup = int(float(audit_row.get("duplicate_condition_count", 0) or 0))

        failures = []
        if clean_open > 0.1: failures.append(f"clean_open={clean_open}")
        if rand_open != 0: failures.append(f"rand_open={rand_open}")
        if not rand_done: failures.append("rand_done=False")
        if rand_qpos > 0.005: failures.append(f"rand_qpos={rand_qpos}")
        if dup > 0: failures.append(f"duplicate={dup}")

        if failures:
            return False, "+".join(failures), "closed_window_control"
        return True, "ok", "closed_window_control"

    elif role == "stable_post_lock_control":
        # Late/open control: do NOT require clean_OPEN <= 0.1
        rand_done = str(audit_row.get("random_done_all_true", "")).lower() == "true"
        dup = int(float(audit_row.get("duplicate_condition_count", 0) or 0))
        schema_ok = str(audit_row.get("schema_incomplete", "True")).lower() != "true"

        failures = []
        if not rand_done: failures.append("rand_done=False")
        if dup > 0: failures.append(f"duplicate={dup}")
        if not schema_ok: failures.append("schema_incomplete")

        if failures:
            return False, "+".join(failures), "late_open_control_confounded"
        return True, "ok", "late_open_control_confounded"

    else:
        # Unknown role — use standard gate
        return check_denominator("pre_lock_control", audit_row)


# ── Role-specific VIS taxonomy ──
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
            return 0, "negative", "late_open_action_only_control", True

    elif role == "far_too_early_control":
        if action_pos and not task_fail:
            if phys_strong:
                return "", "manual_review", "far_too_early_strong_task_failure_unexpected", False
            elif phys_weak:
                return 0, "negative", "far_too_early_weak_negative_control", False
            else:
                return 0, "negative", "far_too_early_action_only_negative_control", False
        elif action_pos and task_fail and phys_strong:
            return "", "manual_review", "far_too_early_task_failure_unexpected", False
        else:
            return 0, "negative", "far_too_early_negative_control", False

    elif role in ("pre_lock_control", "approach_far_closed_proxy", "approach_near_closed_proxy",
                  "pre_lock_closed_proxy", "grasp_formation_pre_lock_proxy", None, ""):
        # Standard closed-window taxonomy
        if action_pos and phys_strong and task_fail:
            return 1, "positive", "claim_usable_positive", False
        elif action_pos and phys_strong and not task_fail:
            return 0, "negative", "physical_strong_task_negative", False
        elif action_pos and phys_weak:
            return "", "ignore", "weak_physical_uncertain", False
        else:
            return 0, "negative", "action_only", False

    return "", "ignore", "unclassified_role", False


# ── Get denominator type for label metadata ──
def get_denominator_type(role):
    if role == "stable_post_lock_control":
        return "late_open_control"
    elif role in ("far_too_early_control", "pre_lock_control"):
        return "closed_window_control"
    return "standard_closed"
