#!/usr/bin/env python3
"""Remap V4 trace CSV to L12-expected field format.

RC1a frozen gripper semantics (official OpenVLA):
  raw > 0.5  →  env = -1.0  →  physical OPEN
  raw < 0.5  →  env = +1.0  →  physical CLOSE

Consequently:
  env < -0.5  →  physical OPEN   →  raw proxy = 1.0
  env > +0.5  →  physical CLOSE  →  raw proxy = 0.0

The V4 runner does not save raw gripper values. This remapper outputs
clean_gripper_raw_proxy (≠ the true raw) and clean_gripper_raw_is_proxy=True.
It does NOT fabricate a false-precision raw value.

Field validity: missing/empty/NaN coords are left as NA, not defaulted to 0.
"""

import csv
import math
import sys


def _is_numeric(val) -> bool:
    """Check if a value is a valid finite number (not empty, not NaN)."""
    if val is None:
        return False
    s = str(val).strip()
    if s == "" or s.lower() == "na" or s.lower() == "nan":
        return False
    try:
        f = float(s)
        return not math.isnan(f) and not math.isinf(f)
    except (ValueError, TypeError):
        return False


def _safe_float(val):
    """Convert to float; returns None if invalid."""
    if not _is_numeric(val):
        return None
    return float(val)


def _check_env_decoded_invariant(env: float, decoded_open: int, step: int) -> list[str]:
    """Verify env action and decoded_open agree per frozen RC1a semantics.

    env < -0.5 → physical OPEN → decoded_open_bool must be 1
    env > +0.5 → physical CLOSE → decoded_open_bool must be 0
    abs(env) <= 0.5 → neutral (not forced to either side)
    """
    issues = []
    if env < -0.5:   # physical OPEN
        if decoded_open != 1:
            issues.append(f"step {step}: env={env} OPEN but decoded_open={decoded_open}")
    elif env > 0.5:  # physical CLOSE
        if decoded_open != 0:
            issues.append(f"step {step}: env={env} CLOSE but decoded_open={decoded_open}")
    # neutral: no invariant to check
    return issues


def remap_v4_to_l12(input_path: str, output_path: str,
                     raise_on_invariant: bool = True) -> tuple:
    """Read V4-format trace, write L12-format CSV.

    Returns (rows, invariant_issues, field_issues).
    """
    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        v4_rows = list(reader)

    prev_clean_close = 0
    close_streak = 0
    invariant_issues = []
    field_issues = []

    l12_rows = []
    for t, r in enumerate(v4_rows):
        # ── Extract V4 fields with validity checking ──
        clean_gripper_env_raw = r.get("clean_gripper_env", "")
        decoded_open_bool_raw = r.get("decoded_open_bool", "")
        qpos_raw = r.get("gripper_qpos_before", "")
        eef_x_raw = r.get("eef_x", "")
        eef_y_raw = r.get("eef_y", "")
        eef_z_raw = r.get("eef_z", "")
        obj_x_raw = r.get("obj_x", "")
        obj_y_raw = r.get("obj_y", "")
        obj_z_raw = r.get("obj_z", "")

        gripper_semantics_valid = True

        # Gripper env — required
        if not _is_numeric(clean_gripper_env_raw):
            field_issues.append(f"step {t}: clean_gripper_env invalid/missing")
            gripper_semantics_valid = False
            clean_gripper_env = None
        else:
            clean_gripper_env = float(clean_gripper_env_raw)

        # Decoded open — required
        if not _is_numeric(decoded_open_bool_raw):
            field_issues.append(f"step {t}: decoded_open_bool invalid/missing")
            gripper_semantics_valid = False
            decoded_open_bool = None
        else:
            decoded_open_bool = int(float(decoded_open_bool_raw))

        # ── RC1a env↔decoded invariant check ──
        if gripper_semantics_valid and clean_gripper_env is not None:
            issues = _check_env_decoded_invariant(
                clean_gripper_env, decoded_open_bool, t)
            if issues:
                invariant_issues.extend(issues)
                if raise_on_invariant:
                    raise ValueError(
                        f"RC1a env-decoded invariant violated in {input_path}:\n" +
                        "\n".join(issues[:10]))

        # ── Reconstruct raw proxy (RC1a-corrected) ──
        if gripper_semantics_valid and clean_gripper_env is not None:
            if clean_gripper_env < -0.5:
                # physical OPEN → raw > 0.5
                clean_gripper_raw_proxy = 1.0
                clean_gripper_raw_is_proxy = True
            elif clean_gripper_env > 0.5:
                # physical CLOSE → raw < 0.5
                clean_gripper_raw_proxy = 0.0
                clean_gripper_raw_is_proxy = True
            else:
                # neutral env — cannot classify
                clean_gripper_raw_proxy = None
                clean_gripper_raw_is_proxy = True
                gripper_semantics_valid = False
        else:
            clean_gripper_raw_proxy = None
            clean_gripper_raw_is_proxy = True

        # ── Derived semantics ──
        if clean_gripper_raw_proxy is not None:
            clean_close = int(clean_gripper_raw_proxy <= 0.5)
            close_onset = int(clean_close and not prev_clean_close)
            if clean_close:
                close_streak += 1
            else:
                close_streak = 0
        else:
            clean_close = None
            close_onset = None
            close_streak = None

        # ── Qpos ──
        gripper_qpos_before = _safe_float(qpos_raw)
        qpos_abs_before = abs(gripper_qpos_before) if gripper_qpos_before is not None else None

        # ── EEF pose validity ──
        eef_x = _safe_float(eef_x_raw)
        eef_y = _safe_float(eef_y_raw)
        eef_z = _safe_float(eef_z_raw)
        eef_pose_valid = all(v is not None for v in [eef_x, eef_y, eef_z])

        # ── Object pose validity ──
        obj_x = _safe_float(obj_x_raw)
        obj_y = _safe_float(obj_y_raw)
        obj_z = _safe_float(obj_z_raw)
        object_pose_valid = all(v is not None for v in [obj_x, obj_y, obj_z])

        # ── Distance: only compute when both poses valid ──
        if eef_pose_valid and object_pose_valid:
            eef_to_obj_distance = (
                (eef_x - obj_x)**2 + (eef_y - obj_y)**2 + (eef_z - obj_z)**2
            ) ** 0.5
        else:
            eef_to_obj_distance = None

        # ── Target coords — NOT in V4 trace ──
        # Marked NA; placement privilege is unavailable per-trace

        row = {
            "step": t,
            "clean_gripper_env": clean_gripper_env if clean_gripper_env is not None else "",
            "clean_gripper_raw_proxy": clean_gripper_raw_proxy if clean_gripper_raw_proxy is not None else "",
            "clean_gripper_raw_is_proxy": int(clean_gripper_raw_is_proxy),
            "clean_gripper_raw_source": "reconstructed_from_env_rc1a",
            "gripper_qpos_before": gripper_qpos_before if gripper_qpos_before is not None else "",
            "gripper_qpos_after": _safe_float(r.get("gripper_qpos_after", "")) or "",
            "qpos_abs_before": qpos_abs_before if qpos_abs_before is not None else "",
            "qpos_abs_after": abs(_safe_float(r.get("gripper_qpos_after", "")) or 0.0) if _is_numeric(r.get("gripper_qpos_after", "")) else "",
            "eef_x": eef_x if eef_x is not None else "",
            "eef_y": eef_y if eef_y is not None else "",
            "eef_z": eef_z if eef_z is not None else "",
            "eef_pose_valid": int(eef_pose_valid),
            "eef_to_obj_distance": eef_to_obj_distance if eef_to_obj_distance is not None else "",
            "clean_close": clean_close if clean_close is not None else "",
            "close_onset": close_onset if close_onset is not None else "",
            "close_streak": close_streak if close_streak is not None else "",
            "decoded_open_bool": decoded_open_bool if decoded_open_bool is not None else "",
            "gripper_semantics_valid": int(gripper_semantics_valid),
            "obj_x": obj_x if obj_x is not None else "",
            "obj_y": obj_y if obj_y is not None else "",
            "obj_z": obj_z if obj_z is not None else "",
            "object_pose_valid": int(object_pose_valid),
            "target_obj_x": "",
            "target_obj_y": "",
            "target_obj_z": "",
            "obj_to_target_distance": "",
            "placement_privilege_valid": False,  # target coords absent in V4
            "success": int(r.get("success_done", 0)),
            "done": int(r.get("success_done", 0)),
            "source_trace": input_path,
            "remapper_version": "rc1a_corrected_v1",
        }
        l12_rows.append(row)
        if clean_close is not None:
            prev_clean_close = clean_close

    fieldnames = list(l12_rows[0].keys()) if l12_rows else []
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(l12_rows)

    return l12_rows, invariant_issues, field_issues


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: remap_v4_trace_for_l12.py <input_v4.csv> <output_l12.csv>")
        sys.exit(1)
    rows, inv_issues, field_issues = remap_v4_to_l12(sys.argv[1], sys.argv[2])
    print(f"Remapped {len(rows)} rows to {sys.argv[2]}")
    if inv_issues:
        print(f"WARNING: {len(inv_issues)} invariant violations")
        for i in inv_issues[:5]:
            print(f"  {i}")
    if field_issues:
        print(f"WARNING: {len(field_issues)} field validity issues")
        for i in field_issues[:5]:
            print(f"  {i}")
