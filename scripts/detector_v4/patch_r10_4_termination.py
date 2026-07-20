#!/usr/bin/env python3
"""R10.4E Gate E-R1: Termination classification patch for r10_4d_passive.py.

Applies the 4-type termination classifier to run_passive_episode().
Run on the server to patch the worktree copy in-place."""

import re
from pathlib import Path


def patch_r10_4d_passive(filepath: str) -> bool:
    content = Path(filepath).read_text()

    # ── Patch 1: Add classify_termination function ──
    classifier_code = '''
# ── Termination classification (Gate E-R1) ─────────────────────────────────

def _classify_termination(
    step_records: list[dict],
    detector_records: list[dict],
    configured_horizon: int,
    env: Any,
    violations: list[str],
) -> dict[str, Any]:
    """Classify episode termination into one of four types.

    SUCCESS_TERMINATION        — done=True AND check_success()=True
    HORIZON_TERMINATION        — done=True at configured horizon, not success
    FULL_LOOP_TASK_FAILURE     — ran full horizon without done, not success
    EARLY_DONE_WITHOUT_SUCCESS — done=True before horizon, not success (HARD FAIL)
    """
    observed_horizon = len(step_records)
    last_done = bool(step_records[-1]["done"]) if step_records else False

    # Simulator timestep (best-effort, MuJoCo)
    simulator_timestep = -1.0
    try:
        sim = getattr(env, "sim", None)
        if sim is not None and hasattr(sim, "data"):
            simulator_timestep = float(sim.data.time)
    except Exception:
        pass

    # env.check_success()
    env_success = None
    if not violations and hasattr(env, "check_success"):
        try:
            env_success = bool(env.check_success())
        except Exception:
            env_success = None

    # Classification
    if not step_records:
        termination_reason = "NO_STEPS"
        task_success = False
        is_hard_failure = True
    elif last_done and env_success is True:
        termination_reason = "SUCCESS_TERMINATION"
        task_success = True
        is_hard_failure = False
    elif last_done and observed_horizon >= configured_horizon and env_success is not True:
        termination_reason = "HORIZON_TERMINATION"
        task_success = False
        is_hard_failure = False
    elif not last_done and observed_horizon >= configured_horizon:
        termination_reason = "FULL_LOOP_TASK_FAILURE"
        task_success = False
        is_hard_failure = False
    elif last_done and observed_horizon < configured_horizon and env_success is not True:
        termination_reason = "EARLY_DONE_WITHOUT_SUCCESS"
        task_success = False
        is_hard_failure = True
    else:
        termination_reason = "UNCLASSIFIED"
        task_success = False
        is_hard_failure = True

    if is_hard_failure and termination_reason not in {"NO_STEPS"}:
        violations.append(f"HARD_FAILURE:{termination_reason}")

    return {
        "termination_reason": termination_reason,
        "task_success": task_success,
        "is_hard_failure": is_hard_failure,
        "configured_horizon": configured_horizon,
        "observed_horizon": observed_horizon,
        "done": last_done,
        "env_check_success": env_success,
        "simulator_timestep": simulator_timestep,
    }
'''

    # Insert classifier before def run_passive_episode
    marker = "\ndef run_passive_episode("
    if marker not in content:
        print("ERROR: marker 'def run_passive_episode(' not found")
        return False
    if "_classify_termination" in content:
        print("WARNING: _classify_termination already present, skipping insert")
    else:
        content = content.replace(marker, classifier_code + marker, 1)
        print("Patch 1 applied: _classify_termination inserted")

    # ── Patch 2: Save info in step_records ──
    # Old: next_observation, reward, done, info = env.step(...)
    #       step_records.append({...})  # no info field
    # New: add "info" serialization to step_records

    old_info_pattern = r'(next_observation, reward, done, info = env\.step\(executed_action\.tolist\(\)\))'
    if re.search(old_info_pattern, content):
        # After env.step, the step_records.append needs "info" added
        # Find the step_records.append call
        old_append = """"done": bool(done),
                "reward": float(reward),
            }"""
        new_append = """"done": bool(done),
                "reward": float(reward),
                "info": _safe_info(info),
            }"""
        if old_append in content and "_safe_info" not in content:
            content = content.replace(old_append, new_append, 1)
            print("Patch 2a applied: info field added to step_records")
        elif "_safe_info" in content:
            print("Patch 2a skipped: _safe_info already present")

    # ── Patch 3: Add _safe_info helper ──
    if "_safe_info" not in content:
        safe_info_code = '''
def _safe_info(info: Any) -> dict[str, Any]:
    """Serialize env.step info dict safely (some LIBERO info values are non-serializable)."""
    if info is None:
        return {}
    try:
        return dict(info)
    except Exception:
        pass
    result = {}
    for k, v in (info.items() if isinstance(info, dict) else []):
        try:
            json.dumps({k: v})
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(type(v).__name__)
    return result
'''
        # Insert before def run_passive_episode
        content = content.replace("\ndef run_passive_episode(", safe_info_code + "\ndef run_passive_episode(", 1)
        print("Patch 3 applied: _safe_info helper inserted")

    # ── Patch 4: Replace the broken task_success + status logic ──
    # Find and replace the old termination block
    old_term_block = """    emit_count = sum(1 for row in detector_records if row.get("emit") is True)
    if emit_count > FROZEN["max_episode_emits"]:
        violations.append(f"DUPLICATE_EMIT:{emit_count}")
    if any(float(row["action_max_abs_error"]) != 0.0 for row in step_records):
        violations.append("ACTION_PARITY")
    if any(int(row["generation_passes_per_step"]) != 1 for row in step_records):
        violations.append("GENERATION_COUNT")
    task_success = bool(step_records and step_records[-1]["done"])
    if not task_success and hasattr(env, "check_success"):
        task_success = bool(env.check_success())

    status = "FAIL_RUNTIME" if violations else (
        "PASS_RUNTIME_EMIT_OBSERVED" if emit_count else "PASS_RUNTIME_NO_EMIT"
    )
    return {
        "schema": "R10_4D_SINGLE_EPISODE_PASSIVE_RESULT_V1",
        "identity": identity,
        "status": status,
        "n_steps": len(step_records),
        "emit_count": emit_count,
        "task_success": task_success,
        "violations": violations,
        "step_records": step_records,
        "detector_records": detector_records,
        "privileged_records": privileged_records,
        "privileged_runtime_input": False,
        "action_mutation": False,
    }"""

    new_term_block = """    emit_count = sum(1 for row in detector_records if row.get("emit") is True)
    if emit_count > FROZEN["max_episode_emits"]:
        violations.append(f"DUPLICATE_EMIT:{emit_count}")
    if any(float(row["action_max_abs_error"]) != 0.0 for row in step_records):
        violations.append("ACTION_PARITY")
    if any(int(row["generation_passes_per_step"]) != 1 for row in step_records):
        violations.append("GENERATION_COUNT")

    # ── Gate E-R1: proper termination classification ──
    termination = _classify_termination(
        step_records=step_records,
        detector_records=detector_records,
        configured_horizon=max_steps,
        env=env,
        violations=violations,
    )

    if termination["is_hard_failure"]:
        status = "FAIL_TERMINATION"
    elif violations:
        status = "FAIL_RUNTIME"
    elif emit_count > 0:
        status = "PASS_RUNTIME_EMIT_OBSERVED"
    else:
        status = "PASS_RUNTIME_NO_EMIT"

    return {
        "schema": "R10_4E_SINGLE_EPISODE_PASSIVE_RESULT_V1",
        "identity": identity,
        "status": status,
        "n_steps": len(step_records),
        "emit_count": emit_count,
        "task_success": termination["task_success"],
        "termination_reason": termination["termination_reason"],
        "configured_horizon": termination["configured_horizon"],
        "observed_horizon": termination["observed_horizon"],
        "done": termination["done"],
        "env_check_success": termination["env_check_success"],
        "simulator_timestep": termination["simulator_timestep"],
        "violations": violations,
        "step_records": step_records,
        "detector_records": detector_records,
        "privileged_records": privileged_records,
        "privileged_runtime_input": False,
        "action_mutation": False,
    }"""

    if old_term_block in content:
        content = content.replace(old_term_block, new_term_block, 1)
        print("Patch 4 applied: termination classification + extended summary fields")
    elif "termination_reason" in content and "configured_horizon" in content:
        print("Patch 4 skipped: termination fields already present")
    else:
        print("ERROR: could not find termination block to patch")
        # Try to find what's there
        idx = content.find("emit_count = sum(1 for row")
        if idx >= 0:
            print(f"  Context: ...{content[idx:idx+100]}...")
        return False

    # ── Write back ──
    Path(filepath).write_text(content)
    print(f"\nPatched: {filepath}")
    return True


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "/mnt/sdc/dty_user/worktrees/r10_4d_smoke_3d741847/src/gripper_attack/r10_4d_passive.py"
    ok = patch_r10_4d_passive(target)
    sys.exit(0 if ok else 1)
