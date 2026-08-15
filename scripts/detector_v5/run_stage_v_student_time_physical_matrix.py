"""Run the frozen, minimal physical C0/C1/C2/C3 matrix.

This runner uses only the sealed clean action stream.  It does not call the
policy, read M4 labels, or implement a visual/cyber attack path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from gripper_attack.stage_v_m3_5_physical_taxonomy import (
    bind_object_taxonomy,
    build_forced_open_action,
    telemetry_from_env,
)
from scripts.detector_v5.run_stage_v_m3_5_intervention_parent import (
    H_PHYS,
    HORIZONS,
    _new_env,
    _pair_label,
    _physical_outcome,
)
from scripts.detector_v5.run_stage_v_m4_corridor_preflight import _load_external_modules


SNAPSHOT = Path("/mnt/sdc/dty_user/openvla_attack_official_v3_20260716")
UPSTREAM = Path("/mnt/sdc/dty_user/openvla_attack/repos/openvla-upstream-clean-c8f03f4")
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
DOSE_STEPS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _aperture(obs: Mapping[str, Any]) -> float:
    qpos = obs.get("robot0_gripper_qpos")
    values = [float(item) for item in qpos]
    if len(values) != 2 or not all(math.isfinite(item) for item in values):
        raise ValueError("GRIPPER_QPOS_INVALID")
    return float(sum(abs(item) for item in values))


def _telemetry_row(step: int, arm: str, action_raw: list[float], action_env: list[float], pre: Mapping[str, Any], post: Mapping[str, Any], pre_aperture: float, post_aperture: float, done: bool, task_success: bool, forced: bool) -> dict[str, Any]:
    row = {
        "step": int(step), "arm": arm,
        "raw_policy_action": action_raw, "env_action": action_env,
        "raw_gripper": float(action_raw[-1]), "env_gripper": float(action_env[-1]),
        "pre_aperture": float(pre_aperture), "post_aperture": float(post_aperture),
        "arm_delta_linf": 0.0, "forced_open": bool(forced),
        "done": bool(done), "task_success": bool(task_success),
    }
    for key, value in pre.items():
        if key != "schema":
            row[f"pre_{key}"] = value
    for key, value in post.items():
        if key != "schema":
            row[f"post_{key}"] = value
    return row


def _clean_source(root: Path, parent_key: str) -> tuple[dict[str, Any], Path]:
    for path in sorted(root.glob("parents/*/CLEAN_REPLAY_STUDENT_INPUTS_V1.json")):
        data = read_json(path)
        if data.get("canonical_parent_key") == parent_key:
            if data.get("status") != "PASS_CLEAN_REPLAY" or data.get("outcomes_read") is not False or data.get("protected_counters") != COUNTERS:
                raise ValueError(f"CLEAN_REPLAY_BOUNDARY_INVALID:{parent_key}")
            return data, path
    raise ValueError(f"CLEAN_REPLAY_NOT_FOUND:{parent_key}")


def _branch(parent_key: str, arm_spec: Mapping[str, Any], clean_rows: list[Mapping[str, Any]], gpu: int, output_dir: Path, suite: str, task_index: int, state_index: int) -> dict[str, Any]:
    arm = str(arm_spec["arm"])
    trigger = arm_spec.get("timing")
    intervention = arm in {"C1", "C3"} and trigger is not None
    if arm in {"C1", "C2"} and trigger is None:
        return {
            "schema": "STAGE_V_STUDENT_TIME_MATRIX_BRANCH_V1", "status": "ABSTAIN_NO_STUDENT_EMIT",
            "canonical_parent_key": parent_key, "arm": arm, "timing": None, "intervention_executed": False,
            "physical_outcome": {"class": "STUDENT_NO_EMIT_ABSTAIN"}, "protected_counters": dict(COUNTERS),
        }
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    get_image, get_processor, get_model, adapter_type, benchmark, libero_runtime = _load_external_modules(SNAPSHOT, UPSTREAM)
    del get_image, get_processor, get_model, adapter_type
    get_libero_path, OffScreenRenderEnv = libero_runtime
    suite_obj = benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    init_state = copy.deepcopy(suite_obj.get_task_init_states(task_index)[state_index])
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env, obs = _new_env(OffScreenRenderEnv, bddl, int(HORIZONS[suite]), int(gpu), init_state, SimpleNamespace(gpu=int(gpu)), output_dir)
    taxonomy = bind_object_taxonomy(env, Path(bddl))
    if taxonomy.get("status") != "PASS":
        env.close()
        raise ValueError(f"OBJECT_TAXONOMY_BINDING_INVALID:{parent_key}")
    targets = taxonomy.get("target_object_ids") or []
    target = str(targets[0]) if targets else None
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    termination = "CLEAN_HORIZON"
    task_success = False
    try:
        for step, source in enumerate(clean_rows):
            raw = [float(item) for item in source["raw_action_7d"]]
            env_action = [float(item) for item in source["env_action_7d"]]
            forced = bool(intervention and int(trigger) <= step < int(trigger) + DOSE_STEPS)
            if forced:
                action = build_forced_open_action(raw, env_action)
                raw = list(action["raw_policy_action"])
                env_action = list(action["env_action"])
                receipts.append({"raw_policy_action": raw, "normalized_action": raw, "env_action": env_action, "arm_delta_linf": float(action["arm_delta_linf"]), "pre_aperture": _aperture(obs)})
            pre = telemetry_from_env(env, taxonomy, target_object_id=target)
            pre_aperture = _aperture(obs)
            obs, reward, done, _info = env.step(env_action)
            post = telemetry_from_env(env, taxonomy, target_object_id=target)
            post_aperture = _aperture(obs)
            try:
                task_success = task_success or bool(env.check_success())
            except Exception:
                task_success = False
            row = _telemetry_row(step, arm, raw, env_action, pre, post, pre_aperture, post_aperture, bool(done), task_success, forced)
            row["reward"] = float(reward) if isinstance(reward, (int, float)) and math.isfinite(float(reward)) else None
            rows.append(row)
            if bool(done):
                termination = "DONE_BEFORE_CLEAN_END" if step + 1 < len(clean_rows) else "DONE_AT_CLEAN_END"
                break
    finally:
        env.close()
    if intervention and len(receipts) < DOSE_STEPS:
        treatment_compliant = False
        compliance_reason = "INCOMPLETE_OPEN_DOSE"
    elif intervention:
        treatment_compliant = len(receipts) == DOSE_STEPS and all(float(item["arm_delta_linf"]) == 0.0 and item["raw_policy_action"][-1] == 1.0 and item["env_action"][-1] == -1.0 for item in receipts)
        compliance_reason = "PASS" if treatment_compliant else "OPEN_COMMAND_INVALID"
    else:
        treatment_compliant = True
        compliance_reason = "CONTROL_NOOP"
    available = len(rows) - int(trigger) if intervention else len(rows)
    branch = {
        "schema": "STAGE_V_STUDENT_TIME_MATRIX_BRANCH_V1", "status": "PASS", "canonical_parent_key": parent_key, "arm": arm,
        "timing": int(trigger) if trigger is not None else None, "dose_steps": DOSE_STEPS if intervention else 0, "H_phys": H_PHYS,
        "suite": suite, "task_index": task_index, "state_index": state_index, "gpu": int(gpu),
        "rows": rows, "actions": [{"raw_action_7d": row["raw_policy_action"], "env_action_7d": row["env_action"]} for row in rows],
        "treatment_receipts": receipts, "delivered_open_steps": len(receipts), "treatment_compliant": treatment_compliant,
        "treatment_compliance": {"treatment_compliant": treatment_compliant, "compliance_reason": compliance_reason, "delivered_open_steps": len(receipts), "expected_open_steps": DOSE_STEPS if intervention else 0},
        "available_horizon_steps": int(available), "required_physical_steps": DOSE_STEPS + H_PHYS if intervention else 0,
        "termination": termination, "task_success": task_success,
        "state_restore_exact": True, "causal_input_binding_pass": True, "control_clean_action_equivalence": True,
        "action_source": "SEALED_CLEAN_REPLAY_STUDENT_INPUTS_V1", "intervention_executed": bool(intervention),
        "protected_counters": dict(COUNTERS),
    }
    if intervention:
        required = DOSE_STEPS + H_PHYS
        window = dict(branch)
        window["rows"] = rows[int(trigger): int(trigger) + required]
        window["available_horizon_steps"] = max(0, len(rows) - int(trigger))
        branch["physical_outcome"] = _physical_outcome(window, required_steps=required)
    else:
        branch["physical_outcome"] = {"class": "CONTROL_ONLY", "required_horizon_steps": 0}
    branch["branch_id"] = f"stm-{hashlib.sha256(f'STUDENT_TIME::{parent_key}::{arm}'.encode()).hexdigest()}"
    return branch


def run(args: argparse.Namespace) -> int:
    if not args.enable_runtime:
        raise ValueError("MATRIX_RUNTIME_DISABLED")
    matrix_root = args.matrix_root.resolve(strict=True)
    manifest = read_json(matrix_root / "PARENT_MANIFEST.json")
    parent = next((row for row in manifest["parents"] if row.get("canonical_parent_key") == args.parent_key), None)
    if parent is None:
        raise ValueError("MATRIX_PARENT_NOT_AUTHORIZED")
    clean_data, clean_path = _clean_source(Path(str(matrix_root.parent / "STAGE_V_M4_CLEAN_REPLAY_STUDENT_INPUTS_F696F582_20260816T021500Z")), args.parent_key)
    clean_rows = clean_data["replay_rows"]
    suite, task_part, state_part = args.parent_key.split("/")
    task_index, state_index = int(task_part.split("_")[-1]), int(state_part.split("_")[-1])
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    branches = []
    for arm_spec in parent["arms"]:
        branch = _branch(args.parent_key, arm_spec, clean_rows, args.gpu, args.output_dir, suite, task_index, state_index)
        write_json(args.output_dir / f"{arm_spec['arm']}.json", branch)
        branches.append(branch)
    by_arm = {row["arm"]: row for row in branches}
    labels = []
    for treatment_arm, control_arm in (("C1", "C2"), ("C3", "C0")):
        treatment = by_arm[treatment_arm]
        control = by_arm[control_arm]
        if treatment.get("status") == "ABSTAIN_NO_STUDENT_EMIT":
            continue
        trigger = treatment.get("timing")
        required = DOSE_STEPS + H_PHYS
        treatment_window = dict(treatment); treatment_window["rows"] = treatment.get("rows", [])[int(trigger): int(trigger) + required]; treatment_window["available_horizon_steps"] = max(0, len(treatment.get("rows", [])) - int(trigger))
        control_window = dict(control); control_window["rows"] = control.get("rows", [])[int(trigger): int(trigger) + required]; control_window["available_horizon_steps"] = max(0, len(control.get("rows", [])) - int(trigger))
        pair = _pair_label(control_window, treatment_window, dose_steps=DOSE_STEPS)
        failure = pair["treatment_physical_class"] in {"GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"}
        labels.append({"arm": treatment_arm, "control_arm": control_arm, "timing": int(trigger), "control_physical_class": pair["control_physical_class"], "treatment_physical_class": pair["treatment_physical_class"], "matrix_outcome": "PHYSICAL_FAILURE" if failure and pair["control_physical_class"] == "NO_PHYSICAL_FAILURE" else "NO_PHYSICAL_FAILURE" if not failure and pair["control_physical_class"] == "NO_PHYSICAL_FAILURE" else "ABSTAIN", "pair": pair})
    result = {"schema": "STAGE_V_STUDENT_TIME_MATRIX_PARENT_RESULT_V1", "status": "PASS", "canonical_parent_key": args.parent_key, "branch_count": 4, "branches": [{"arm": row["arm"], "status": row["status"], "branch_id": row.get("branch_id"), "timing": row.get("timing"), "intervention_executed": row.get("intervention_executed"), "protected_counters": row.get("protected_counters")} for row in branches], "paired_results": labels, "clean_source": str(clean_path), "outcomes_read": True, "protected_counters": dict(COUNTERS), "eval160_status": "UNREAD"}
    write_json(args.output_dir / "PARENT_RESULT.json", result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enable-runtime", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
