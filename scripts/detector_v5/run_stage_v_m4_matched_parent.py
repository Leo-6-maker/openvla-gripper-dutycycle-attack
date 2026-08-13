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
    assert_primary_observation_exact,
    load_snapshot,
)
from gripper_attack.stage_v_canonical_execution_core import canonical_sha256, canonical_value  # noqa: E402
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
from scripts.detector_v5.run_stage_v_m3_5_v1_4_gate_b import (  # noqa: E402
    DOSES,
    H_PHYS,
    _pair_label,
    _run_branch,
)
from scripts.detector_v5.stage_v_m4_governance import (  # noqa: E402
    M4GovernanceError,
    validate_formal_m4_v2_authority,
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


def _sha_json(value: Any) -> str:
    return canonical_sha256(canonical_value(value))


def _inside(root: Path, relative: Any) -> Path:
    path = (root / str(relative)).resolve()
    path.relative_to(root.resolve())
    return path


def _load_exact_plan_authority(root: Path, parent_key: str, expected_manifest_sha256: str) -> dict[str, Any]:
    """Load the plan-only authority; formal runtime never selects probes."""
    root = root.resolve()
    manifest_path = root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"
    audit_path = root / "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT.json"
    result_path = root / "PLAN_RESULT.json"
    seal_path = root / "ROOT_SEAL.sha256"
    sums_path = root / "SHA256SUMS"
    if not root.is_dir() or not all(path.is_file() for path in (manifest_path, audit_path, result_path, seal_path, sums_path)):
        raise M35RunnerError("EXACT_PLAN_ROOT_INCOMPLETE")
    if _sha(manifest_path) != expected_manifest_sha256:
        raise M35RunnerError("EXACT_PLAN_MANIFEST_HASH_MISMATCH")
    seal_tokens = seal_path.read_text(encoding="utf-8").split()
    if not seal_tokens or seal_tokens[0] != _sha(sums_path):
        raise M35RunnerError("EXACT_PLAN_ROOT_SEAL_INVALID")
    manifest = _load(manifest_path)
    audit = _load(audit_path)
    result = _load(result_path)
    if manifest.get("schema") != "STAGE_V_M4_EXACT_PROBE_AND_SNAPSHOT_MANIFEST_V1" or manifest.get("status") != "PASS_EXACT_40X24_PLAN_ONLY":
        raise M35RunnerError("EXACT_PLAN_MANIFEST_NOT_PASS")
    if audit.get("status") != "PASS" or result.get("status") != "PASS" or result.get("manifest_status") != "PASS_EXACT_40X24_PLAN_ONLY":
        raise M35RunnerError("EXACT_PLAN_AUDIT_NOT_PASS")
    if result.get("audit_sha256") != _sha(audit_path) or manifest.get("independent_audit_sha256") != _sha(audit_path):
        raise M35RunnerError("EXACT_PLAN_AUDIT_HASH_MISMATCH")
    if manifest.get("parent_count") != 40 or manifest.get("probe_count_per_parent") != 24 or manifest.get("probe_count_total") != 960 or manifest.get("planned_branch_authority_count") != 3840:
        raise M35RunnerError("EXACT_PLAN_MATRIX_INVALID")
    if manifest.get("selection_outcomes_read") is not False or manifest.get("intervention_executed") is not False or manifest.get("v_phys_generated") is not False or manifest.get("teacher_predictions_read") is not False or manifest.get("student_predictions_read") is not False or manifest.get("protected_counters") != COUNTERS:
        raise M35RunnerError("EXACT_PLAN_BOUNDARY_INVALID")
    parent = next((row for row in manifest.get("parents", []) if isinstance(row, Mapping) and str(row.get("canonical_parent_key")) == parent_key), None)
    if not isinstance(parent, Mapping) or parent.get("status") != "PASS" or parent.get("probe_count") != 24:
        raise M35RunnerError("EXACT_PLAN_PARENT_NOT_PASS")
    probes = sorted([row for row in manifest.get("probe_authorities", []) if isinstance(row, Mapping) and str(row.get("canonical_parent_key")) == parent_key], key=lambda row: str(row.get("probe_id")))
    if len(probes) != 24 or len({str(row.get("probe_id")) for row in probes}) != 24:
        raise M35RunnerError("EXACT_PLAN_PROBE_AUTHORITY_INVALID")
    branches = [row for row in manifest.get("branch_authorities", []) if isinstance(row, Mapping) and str(row.get("canonical_parent_key")) == parent_key]
    if len(branches) != 96 or any(row.get("execution_status") != "PLANNED_NOT_EXECUTED" or row.get("outcomes_read") is not False or row.get("protected_counters") != COUNTERS for row in branches):
        raise M35RunnerError("EXACT_PLAN_BRANCH_AUTHORITY_INVALID")
    clean_path = _inside(root, parent.get("clean_trajectory_path"))
    if _sha(clean_path) != parent.get("clean_trajectory_sha256"):
        raise M35RunnerError("EXACT_PLAN_CLEAN_TRAJECTORY_HASH_MISMATCH")
    clean = _load(clean_path)
    if clean.get("outcomes_read") is not False or clean.get("task_success") is not True or not isinstance(clean.get("rows"), list):
        raise M35RunnerError("EXACT_PLAN_CLEAN_TRAJECTORY_INVALID")
    actions = [{"step": int(row["step"]), "raw": row["raw_action"], "env": row["env_action"]} for row in clean["rows"]]
    if _sha_json(actions) != parent.get("clean_reference_action_sequence_sha256"):
        raise M35RunnerError("EXACT_PLAN_CLEAN_ACTION_SEQUENCE_HASH_MISMATCH")
    gate_a_root = _inside(root, parent.get("output_dir"))
    gate_a_receipt = _load(gate_a_root / "M3_5_V1_4_GATE_A_RECEIPT.json")
    canaries = gate_a_receipt.get("canary_receipts")
    if gate_a_receipt.get("status") != "PASS" or not isinstance(canaries, list) or len(canaries) != 24 or not all(isinstance(row, Mapping) and row.get("status") == "PASS" for row in canaries):
        raise M35RunnerError("EXACT_PLAN_PARENT_CANARY_INVALID")
    for probe in probes:
        snapshot_root = _inside(root, probe.get("snapshot_path"))
        snapshot_manifest_path = snapshot_root / "CAUSAL_PROBE_SNAPSHOT_V2.json"
        if not snapshot_manifest_path.is_file() or _sha(snapshot_manifest_path) != probe.get("snapshot_manifest_sha256"):
            raise M35RunnerError(f"EXACT_PLAN_SNAPSHOT_HASH_MISMATCH:{probe.get('probe_id')}")
        loaded = load_snapshot(snapshot_root, materialize_torch=True)
        snapshot_manifest = loaded["manifest"]
        payload = loaded["payload"]
        hashes = assert_primary_observation_exact(payload)
        if snapshot_manifest.get("binding", {}).get("parent_key") != parent_key or snapshot_manifest.get("binding", {}).get("probe_id") != probe.get("probe_id") or snapshot_manifest.get("primary_input_authority") != "loaded_frozen_canonical_bytes" or snapshot_manifest.get("fresh_render_equality_gate_used") is not False:
            raise M35RunnerError(f"EXACT_PLAN_SNAPSHOT_BINDING_INVALID:{probe.get('probe_id')}")
        if payload.get("probe", {}).get("probe_id") != probe.get("probe_id") or int(payload.get("probe", {}).get("step")) != int(probe.get("probe_step")):
            raise M35RunnerError(f"EXACT_PLAN_SNAPSHOT_IDENTITY_INVALID:{probe.get('probe_id')}")
        if hashes.get("raw_observation_sha256") != probe.get("raw_observation_sha256") or hashes.get("policy_rgb_224_sha256") != probe.get("policy_rgb_224_sha256") or hashes.get("policy_input_sha256") != probe.get("policy_input_sha256"):
            raise M35RunnerError(f"EXACT_PLAN_SNAPSHOT_INPUT_HASH_INVALID:{probe.get('probe_id')}")
        state = payload.get("full_simulator_state", {}).get("registered_flat_state")
        if state is None or hashlib.sha256(np.asarray(state).tobytes(order="C")).hexdigest() != probe.get("sim_state_sha256"):
            raise M35RunnerError(f"EXACT_PLAN_SNAPSHOT_STATE_HASH_INVALID:{probe.get('probe_id')}")
        if _sha_json(payload.get("clean_reference_action_window")) != probe.get("clean_reference_action_window_sha256"):
            raise M35RunnerError(f"EXACT_PLAN_SNAPSHOT_ACTION_WINDOW_INVALID:{probe.get('probe_id')}")
    normalized_probes = []
    for row in probes:
        normalized = dict(row)
        normalized["step"] = int(normalized["probe_step"])
        normalized_probes.append(normalized)
    return {"root": root, "manifest": manifest, "manifest_sha256": expected_manifest_sha256, "audit_sha256": _sha(audit_path), "parent": dict(parent), "probes": normalized_probes, "clean": clean, "clean_path": clean_path, "probe_plan_path": _inside(root, parent.get("probe_plan_path")), "canaries": [dict(row) for row in canaries]}


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
    try:
        authority = validate_formal_m4_v2_authority(
            protocol,
            protocol_path=protocol_path,
            split_path=split_path,
            source_commit=args.source_commit,
            source_tree=args.source_tree,
            authorization=authorization,
        )
    except M4GovernanceError as exc:
        raise M35RunnerError(str(exc)) from exc
    return _load(split_path), authority


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
    shutil.copy2(corridor_gate["manifest_path"], output / "M4_FINAL_PARENT_MANIFEST.json")
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
    exact = _load_exact_plan_authority(args.exact_plan_root, args.parent_key, args.exact_plan_manifest_sha256)
    if exact["parent"].get("split") != parent.get("split"):
        raise M35RunnerError("EXACT_PLAN_SPLIT_BINDING_INVALID")
    shutil.copy2(exact["clean_path"], output / "CLEAN_TRAJECTORY_V1_4.json")
    shutil.copy2(exact["probe_plan_path"], output / "PROBE_PLAN_V1_4.json")
    _write(output / "EXACT_PLAN_BINDING.json", {
        "schema": "STAGE_V_M4_EXACT_PLAN_RUNTIME_BINDING_V1",
        "status": "PASS_EXACT_FROZEN_AUTHORITY_LOADED",
        "exact_plan_root": str(exact["root"]),
        "exact_plan_manifest_sha256": exact["manifest_sha256"],
        "exact_plan_audit_sha256": exact["audit_sha256"],
        "canonical_parent_key": args.parent_key,
        "probe_count": len(exact["probes"]),
        "probe_selection_recomputed": False,
        "snapshots_loaded_frozen": True,
        "outcomes_read": False,
        "protected_counters": dict(COUNTERS),
    })
    clean_actions = [list(row["env_action"]) for row in exact["clean"]["rows"]]
    probes = exact["probes"]
    snapshot_rows = [{"probe_id": str(probe["probe_id"]), "step": int(probe["step"]), "path": str(probe["snapshot_path"]), "manifest_sha256": str(probe["snapshot_manifest_sha256"])} for probe in probes]
    binding_env, _binding_obs = _new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, init_state, args, output)
    try:
        _write_runtime_binding_receipt(args, binding_env, output)
    finally:
        binding_env.close()
    _write(output / "M4_CAUSAL_SNAPSHOT_CANARY.json", {"schema": "STAGE_V_M4_CAUSAL_SNAPSHOT_CANARY_V1", "status": "PASS", "source": "EXACT_40X24_PLAN_ONLY_INDEPENDENT_AUDIT", "snapshots": snapshot_rows, "canaries": exact["canaries"], "fresh_render_equality_gate_used": False, "intervention_executed": False, "protected_counters": dict(COUNTERS)})
    branches: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for probe in probes:
        snapshot_root = _inside(exact["root"], probe["snapshot_path"])
        control = _run_branch(snapshot_root=snapshot_root, gate_a_root=exact["root"], OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output, clean_actions=clean_actions, model=model, adapter=adapter, arm="CONTROL", dose=0, probe_id=str(probe["probe_id"]), repetition=0)
        control_record = _branch_record(args.parent_key, probe, "CONTROL", control)
        branches.append(control_record)
        for dose_name, dose_steps in DOSES.items():
            treatment = _run_branch(snapshot_root=snapshot_root, gate_a_root=exact["root"], OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output, clean_actions=clean_actions, model=model, adapter=adapter, arm=dose_name, dose=dose_steps, probe_id=str(probe["probe_id"]), repetition=0)
            pair = _pair_label(control, treatment, dose_steps)
            treatment_record = _branch_record(args.parent_key, probe, dose_name, treatment, pair=pair, control=control_record)
            branches.append(treatment_record)
            labels.append(_label(args.parent_key, probe, dose_name, treatment_record, control_record, pair))
            observations.append({"schema": "STAGE_V_M4_TREATMENT_OBSERVATION_V1", "canonical_parent_key": args.parent_key, "probe_id": probe["probe_id"], "dose": dose_name, **{key: pair[key] for key in ("label_class", "control_valid", "treatment_valid", "f_control", "f_open", "control_physical_class", "treatment_physical_class")}, "treatment_compliant": treatment.get("treatment_compliant") is True, "treatment_branch_id": treatment_record["branch_id"], "control_branch_id": control_record["branch_id"], "protected_counters": dict(COUNTERS)})
    _write_jsonl(output / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl", branches)
    _write_jsonl(output / "M4_TREATMENT_OBSERVATIONS_V1.jsonl", observations)
    _write_jsonl(output / "M4_V_PHYS_LABELS_V1.jsonl", labels)
    result = {"schema": "STAGE_V_M4_PARENT_RESULT_V1", "status": "PASS", "parent_atomic": True, "canonical_parent_key": args.parent_key, "suite": suite, "task_index": task_index, "state_index": state_index, "split": parent["split"], "source_commit": args.source_commit, "source_tree": args.source_tree, "runner_sha256": _sha(Path(__file__)), "protocol_sha256": _sha(protocol_path), "authorization_receipt_sha256": _sha(authorization_path), "exact_plan_manifest_sha256": exact["manifest_sha256"], "clean_success": True, "probe_count": len(probes), "branch_count": len(branches), "treatment_label_count": len(labels), "expected_physical_executions": 96, "expected_treatment_labels": 72, "primary_estimand": "V_phys@T5", "primary_window": "MATCHED_CANONICAL_ACTION_T_PLUS_H_PHYS", "native_policy_calls_in_primary_window": 0, "fresh_render_equality_gate_used": False, "fresh_render_primary_consumption": False, "selection_outcomes_read": False, "probe_selection_source": "EXACT_FROZEN_PLAN_MANIFEST", "probe_selection_recomputed": False, "causal_snapshot_canary_status": "PASS", "label_status": "VALID", "independent_audit_status": "PENDING", "protected_counters": dict(COUNTERS)}
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
    parser.add_argument("--exact-plan-root", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--exact-plan-manifest-sha256", required=True)
    parser.add_argument("--enable-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, KeyError, ValueError, CausalSnapshotError, M35RunnerError) as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
