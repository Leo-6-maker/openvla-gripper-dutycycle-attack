#!/usr/bin/env python3
"""Run one formal M4 parent with a matched clean-action physical window."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gripper_attack.stage_v_causal_observation_snapshot import (  # noqa: E402
    CausalSnapshotError,
    capture_rng_state,
    capture_runtime_state,
    capture_simulator_state,
    load_snapshot,
    write_snapshot,
)
from gripper_attack.stage_v_canonical_execution_core import canonical_sha256  # noqa: E402
from gripper_attack.stage_v_m3_5_phase_classifier import classify_trajectory  # noqa: E402
from scripts.detector_v5.build_stage_v_m3_5_probe_plan import (  # noqa: E402
    H_PHYS,
    PROBE_COUNT,
    select_probe_steps,
)
from scripts.detector_v5.run_stage_v_canonical_clean import (  # noqa: E402
    _load_external_modules,
    _load_policy,
    _write_runtime_binding_receipt,
)
from scripts.detector_v5.run_stage_v_m3_5_intervention_parent import (  # noqa: E402
    HORIZONS,
    M35RunnerError,
    _new_env,
)
from scripts.detector_v5.run_stage_v_m3_5_v1_4_gate_a import (  # noqa: E402
    _clean_row,
    _policy_capture,
    _replay_canary,
    _snapshot_payload,
)
from scripts.detector_v5.run_stage_v_m3_5_v1_4_gate_b import (  # noqa: E402
    DOSES,
    _pair_label,
    _run_branch,
)
from scripts.detector_v5.stage_v_m4_governance import (  # noqa: E402
    M4GovernanceError,
    validate_formal_m4_corridor_gate,
)


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
SOURCE_COMMIT = "17fb76971da28dc9a61aaead52cebea62b653a46"
SOURCE_TREE = "6efec29765121710456ae93ffe292965490022bf"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise M35RunnerError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _seal(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": _sha(path)})
    (root / "SHA256SUMS").write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{_sha(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def _validate(protocol: Mapping[str, Any], authorization: Mapping[str, Any], *, protocol_path: Path, authorization_path: Path, split_path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if protocol.get("schema") != "STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V1" or protocol.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise M35RunnerError("M4_PROTOCOL_NOT_FROZEN_OR_AUTHORIZED")
    source = protocol.get("source_binding", {})
    if source.get("runtime_commit") != args.source_commit or source.get("runtime_tree") != args.source_tree:
        raise M35RunnerError("M4_SOURCE_BINDING_MISMATCH")
    matrix = protocol.get("matrix", {})
    if matrix.get("parents") != 40 or matrix.get("probes_per_parent") != 24 or matrix.get("repetitions") != 1 or matrix.get("conditions") != ["CONTROL", "T3", "T5", "T10"]:
        raise M35RunnerError("M4_MATRIX_INVALID")
    if protocol.get("protected_counters") != COUNTERS or protocol.get("operation", {}).get("fresh_render_primary_consumption") is not False or protocol.get("operation", {}).get("native_policy_calls_in_primary_window") != 0:
        raise M35RunnerError("M4_PRIMARY_CONTRACT_INVALID")
    if not split_path.is_file() or _sha(split_path) != protocol.get("inputs", {}).get("formal_parent_split_sha256"):
        raise M35RunnerError("M4_SPLIT_BINDING_INVALID")
    if authorization.get("status") != "PASS" or authorization.get("protocol_sha256") != _sha(protocol_path) or authorization.get("formal_parent_split_sha256") != _sha(split_path) or authorization.get("source_commit") != args.source_commit or authorization.get("source_tree") != args.source_tree:
        raise M35RunnerError("M4_AUTHORIZATION_BINDING_INVALID")
    try:
        corridor_gate = validate_formal_m4_corridor_gate(
            protocol,
            protocol_path=protocol_path,
            split_path=split_path,
            source_commit=args.source_commit,
            source_tree=args.source_tree,
            authorization=authorization,
        )
    except M4GovernanceError as exc:
        raise M35RunnerError(str(exc)) from exc
    return _load(split_path), corridor_gate


def _parent_row(split: Mapping[str, Any], parent_key: str) -> Mapping[str, Any]:
    rows = split.get("parents", [])
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("canonical_parent_key")) == parent_key:
            if row.get("split") not in {"TRAIN", "VAL", "TEST"}:
                raise M35RunnerError("M4_PARENT_SPLIT_INVALID")
            return row
    raise M35RunnerError("M4_PARENT_NOT_IN_FORMAL_SPLIT")


def _branch_record(parent_key: str, probe: Mapping[str, Any], arm: str, branch: Mapping[str, Any], pair: Mapping[str, Any] | None = None, control: Mapping[str, Any] | None = None) -> dict[str, Any]:
    probe_id = str(probe["probe_id"])
    branch_id = f"m4-v1-{hashlib.sha256(f'M4_V1::{parent_key}::{probe_id}::R0::{arm}'.encode()).hexdigest()}"
    return {
        "schema": "STAGE_V_M4_PHYSICAL_EXECUTION_V1",
        "canonical_parent_key": parent_key,
        "probe_id": probe_id,
        "probe_step": int(probe["step"]),
        "repetition": 0,
        "arm": arm,
        "branch_id": branch_id,
        "branch_result_sha256": canonical_sha256(branch),
        "shared_control_branch_id": control.get("branch_id") if control else None,
        "shared_control_result_sha256": control.get("branch_result_sha256") if control else None,
        "branch": dict(branch),
        "pair": dict(pair) if pair is not None else None,
        "protected_counters": dict(COUNTERS),
    }


def _label(parent_key: str, probe: Mapping[str, Any], dose: str, treatment: Mapping[str, Any], control: Mapping[str, Any], pair: Mapping[str, Any]) -> dict[str, Any]:
    identity = {"parent": parent_key, "probe": probe["probe_id"], "dose": dose}
    label_class = str(pair["label_class"])
    return {
        "schema": "STAGE_V_M4_V_PHYS_LABEL_V1",
        "label_id": f"m4-label-{hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}",
        "canonical_parent_key": parent_key,
        "probe_id": probe["probe_id"],
        "probe_step": int(probe["step"]),
        "dose": dose,
        "primary_estimand": "V_phys@T5",
        "label_class": label_class,
        "binary_label_consumable": label_class in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"},
        "repeatability_status": "NOT_APPLICABLE_SINGLE_EXECUTION",
        "control_valid": pair["control_valid"],
        "treatment_valid": pair["treatment_valid"],
        "f_control": pair["f_control"],
        "f_open": pair["f_open"],
        "control_physical_class": pair["control_physical_class"],
        "treatment_physical_class": pair["treatment_physical_class"],
        "control_branch_id": control["branch_id"],
        "treatment_branch_id": treatment["branch_id"],
        "control_result_sha256": control["branch_result_sha256"],
        "treatment_result_sha256": treatment["branch_result_sha256"],
        "treatment_compliant": treatment["branch"].get("treatment_compliant") is True,
        "matched_action_window": "T+H_phys",
        "protected_counters": dict(COUNTERS),
    }


def run(args: argparse.Namespace) -> int:
    if not args.enable_runtime:
        raise M35RunnerError("M4_RUNTIME_DISABLED")
    protocol_path = args.protocol.resolve()
    authorization_path = args.authorization_receipt.resolve()
    protocol = _load(protocol_path)
    authorization = _load(authorization_path)
    split_path = Path(str(protocol["inputs"]["formal_parent_split_path"])).resolve()
    split, corridor_gate = _validate(protocol, authorization, protocol_path=protocol_path, authorization_path=authorization_path, split_path=split_path, args=args)
    parent = _parent_row(split, args.parent_key)
    output = args.output_dir.resolve()
    if output.exists():
        allowed_worker_files = {"JOB.json", "RESOURCE_PRE.json", "SCIENCE_RUNNER.log"}
        unexpected = [path.name for path in output.iterdir() if path.name not in allowed_worker_files]
        if unexpected:
            raise M35RunnerError(f"M4_OUTPUT_NOT_NEW:{output}:{sorted(unexpected)}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, output / "M4_PROTOCOL.json")
    shutil.copy2(authorization_path, output / "M4_AUTHORIZATION.json")
    shutil.copy2(corridor_gate["receipt_path"], output / "M4_CORRIDOR_PASS_RECEIPT.json")
    suite, task_part, state_part = args.parent_key.split("/")
    args.suite = suite
    task_index = int(task_part.removeprefix("task_"))
    state_index = int(state_part.removeprefix("state_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    get_image, get_processor, get_model, adapter_type, benchmark, libero_runtime = _load_external_modules(args.official_snapshot_root, args.upstream_root)
    get_libero_path, OffScreenRenderEnv = libero_runtime
    suite_obj = benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    init_state = copy.deepcopy(suite_obj.get_task_init_states(task_index)[state_index])
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    adapter, model, _processor, _unnorm_key = _load_policy(args, get_processor, get_model, adapter_type)
    horizon = int(HORIZONS[suite])
    episode_rng = capture_rng_state()
    first_env, first_obs = _new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, init_state, args, output)
    _write_runtime_binding_receipt(args, first_env, output)
    captures: dict[int, dict[str, Any]] = {}
    clean_rows: list[dict[str, Any]] = []
    clean_actions: list[list[float]] = []
    task_success = False
    terminal_seen = False
    baseline_z: float | None = None
    try:
        from gripper_attack.stage_v_m3_5_physical_taxonomy import bind_object_taxonomy, aperture_metric, telemetry_from_env
        taxonomy = bind_object_taxonomy(first_env, Path(bddl))
        if taxonomy.get("status") != "PASS":
            raise M35RunnerError("M4_OBJECT_TAXONOMY_BINDING_FAIL")
        for step in range(horizon):
            simulator = capture_simulator_state(first_env)
            runtime = capture_runtime_state(first_env, model=model, adapter=adapter)
            capture = _policy_capture(adapter, get_image, first_obs, str(task.language))
            captures[step] = {**capture, "simulator": simulator, "runtime": runtime}
            aperture = None
            if isinstance(first_obs, Mapping):
                for key in ("robot0_gripper_qpos", "gripper_qpos"):
                    if key in first_obs:
                        aperture = aperture_metric(first_obs[key])
                        if aperture is not None:
                            break
            telemetry = telemetry_from_env(first_env, taxonomy)
            row, baseline_z = _clean_row(step, horizon, capture, telemetry, aperture, baseline_z)
            row["state_sha256"] = hashlib.sha256(np.asarray(simulator["registered_flat_state"]).tobytes(order="C")).hexdigest()
            row["clean_terminal"] = terminal_seen
            clean_rows.append(row)
            clean_actions.append(list(capture["env_action"]))
            first_obs, _reward, done, _info = first_env.step(capture["env_action"])
            try:
                current_success = bool(first_env.check_success())
            except Exception:
                current_success = False
            task_success = task_success or current_success
            if current_success or bool(done):
                terminal_seen = True
                row["clean_terminal"] = True
            if bool(done):
                break
    finally:
        first_env.close()
    if not task_success:
        raise M35RunnerError("M4_CLEAN_PARENT_NOT_SUCCESSFUL")
    for index, row in enumerate(clean_rows):
        row["remaining_horizon"] = len(clean_rows) - index
    for row, label in zip(clean_rows, classify_trajectory(clean_rows)):
        row.update(label)
    _write(output / "CLEAN_TRAJECTORY_V1_4.json", {"schema": "STAGE_V_M4_CLEAN_TRAJECTORY_V1", "outcomes_read": False, "rows": clean_rows, "task_success": True, "protected_counters": dict(COUNTERS)})
    plan = select_probe_steps(clean_rows, args.parent_key)
    probes = plan.get("probe_steps")
    if not isinstance(probes, list) or len(probes) != PROBE_COUNT:
        raise M35RunnerError("M4_PROBE_PLAN_INVALID")
    _write(output / "PROBE_PLAN_V1_4.json", {**plan, "outcomes_read": False, "protected_counters": dict(COUNTERS)})
    snapshot_rows = []
    for probe in probes:
        step = int(probe["step"])
        capture = captures[step]
        if str(probe.get("policy_input_sha256")) != str(capture["policy_input_sha256"]) or str(probe.get("policy_rgb_224_sha256")) != str(capture["policy_rgb_224_sha256"]):
            raise M35RunnerError(f"M4_PROBE_INPUT_BINDING_MISMATCH:{probe['probe_id']}")
        if str(probe.get("state_sha256")) != str(capture["clean_state_sha256"] if "clean_state_sha256" in capture else clean_rows[step]["state_sha256"]):
            raise M35RunnerError(f"M4_PROBE_STATE_BINDING_MISMATCH:{probe['probe_id']}")
        snapshot_root = output / "CAUSAL_SNAPSHOTS" / str(probe["probe_id"])
        manifest = write_snapshot(snapshot_root, _snapshot_payload(capture, probe=probe, simulator=capture["simulator"], runtime=capture["runtime"], episode_rng=episode_rng, action_rows=clean_rows), binding={"parent_key": args.parent_key, "probe_id": probe["probe_id"], "step": step, "object_identity": probe.get("object_identity"), "source_commit": args.source_commit, "source_tree": args.source_tree})
        snapshot_rows.append({"probe_id": probe["probe_id"], "step": step, "path": snapshot_root.relative_to(output).as_posix(), "manifest_sha256": manifest["manifest_sha256"]})
    canaries = []
    for row in snapshot_rows:
        canaries.append(_replay_canary(output / row["path"], OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output, clean_actions=clean_actions, model=model, adapter=adapter))
    if len(canaries) != PROBE_COUNT or any(item.get("status") != "PASS" for item in canaries):
        raise M35RunnerError("M4_CAUSAL_SNAPSHOT_CANARY_FAIL")
    _write(output / "M4_CAUSAL_SNAPSHOT_CANARY.json", {"schema": "STAGE_V_M4_CAUSAL_SNAPSHOT_CANARY_V1", "status": "PASS", "snapshots": snapshot_rows, "canaries": canaries, "fresh_render_equality_gate_used": False, "protected_counters": dict(COUNTERS)})
    branches: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for probe in probes:
        snapshot_root = output / "CAUSAL_SNAPSHOTS" / str(probe["probe_id"])
        control = _run_branch(snapshot_root=snapshot_root, gate_a_root=output, OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output, clean_actions=clean_actions, model=model, adapter=adapter, arm="CONTROL", dose=0, probe_id=str(probe["probe_id"]), repetition=0)
        control_record = _branch_record(args.parent_key, probe, "CONTROL", control)
        branches.append(control_record)
        for dose_name, dose_steps in DOSES.items():
            treatment = _run_branch(snapshot_root=snapshot_root, gate_a_root=output, OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output, clean_actions=clean_actions, model=model, adapter=adapter, arm=dose_name, dose=dose_steps, probe_id=str(probe["probe_id"]), repetition=0)
            pair = _pair_label(control, treatment, dose_steps)
            treatment_record = _branch_record(args.parent_key, probe, dose_name, treatment, pair=pair, control=control_record)
            branches.append(treatment_record)
            labels.append(_label(args.parent_key, probe, dose_name, treatment_record, control_record, pair))
            observations.append({"schema": "STAGE_V_M4_TREATMENT_OBSERVATION_V1", "canonical_parent_key": args.parent_key, "probe_id": probe["probe_id"], "dose": dose_name, **{key: pair[key] for key in ("label_class", "control_valid", "treatment_valid", "f_control", "f_open", "control_physical_class", "treatment_physical_class")}, "treatment_compliant": treatment.get("treatment_compliant") is True, "treatment_branch_id": treatment_record["branch_id"], "control_branch_id": control_record["branch_id"], "protected_counters": dict(COUNTERS)})
    _write_jsonl(output / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl", branches)
    _write_jsonl(output / "M4_TREATMENT_OBSERVATIONS_V1.jsonl", observations)
    _write_jsonl(output / "M4_V_PHYS_LABELS_V1.jsonl", labels)
    result = {"schema": "STAGE_V_M4_PARENT_RESULT_V1", "status": "PASS", "parent_atomic": True, "canonical_parent_key": args.parent_key, "suite": suite, "task_index": task_index, "state_index": state_index, "split": parent["split"], "source_commit": args.source_commit, "source_tree": args.source_tree, "runner_sha256": _sha(Path(__file__)), "protocol_sha256": _sha(protocol_path), "authorization_receipt_sha256": _sha(authorization_path), "clean_success": True, "probe_count": len(probes), "branch_count": len(branches), "treatment_label_count": len(labels), "expected_physical_executions": 96, "expected_treatment_labels": 72, "primary_estimand": "V_phys@T5", "primary_window": "MATCHED_CANONICAL_ACTION_T_PLUS_H_PHYS", "native_policy_calls_in_primary_window": 0, "fresh_render_equality_gate_used": False, "fresh_render_primary_consumption": False, "selection_outcomes_read": False, "causal_snapshot_canary_status": "PASS", "label_status": "VALID", "independent_audit_status": "PENDING", "protected_counters": dict(COUNTERS)}
    _write(output / "PARENT_RESULT.json", result)
    _seal(output)
    audit_path = output / "M4_INDEPENDENT_AUDIT.json"
    auditor = REPO_ROOT / "scripts/detector_v5/audit_stage_v_m4_matched_parent.py"
    process = subprocess.run([sys.executable, str(auditor), "--root", str(output), "--parent-key", args.parent_key, "--source-commit", args.source_commit, "--source-tree", args.source_tree], cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    audit = _load(audit_path) if audit_path.is_file() else {"status": "FAIL", "errors": [process.stderr[-1000:]]}
    result["independent_audit_status"] = audit.get("status")
    result["independent_audit_sha256"] = _sha(audit_path) if audit_path.is_file() else None
    if audit.get("status") != "PASS_M4_PARENT_INDEPENDENT":
        result["status"] = "HOLD_SEALED"
        result["label_status"] = "HOLD"
    _write(output / "PARENT_RESULT.json", result)
    _seal(output)
    return 0 if result["status"] == "PASS" and audit.get("status") == "PASS_M4_PARENT_INDEPENDENT" and process.returncode == 0 else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("protocol", "output_dir", "official_snapshot_root", "upstream_root", "model_path", "authorization_receipt"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--enable-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, KeyError, ValueError, CausalSnapshotError, M35RunnerError) as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
