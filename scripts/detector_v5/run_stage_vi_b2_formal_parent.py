"""Run one frozen Stage VI-B2 formal parent; no probe selection or model tuning."""
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
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.detector_v5 import run_stage_v_m4_matched_parent as m4  # noqa: E402


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root: Path, relative: Any) -> Path:
    path = (root / str(relative)).resolve()
    path.relative_to(root.resolve())
    return path


def sha_json(value: Any) -> str:
    return m4.canonical_sha256(m4.canonical_value(value))


def validate_authority(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path, str, dict[str, str]]:
    protocol = load(args.protocol.resolve())
    authority = load(args.authority.resolve())
    source_commit = args.source_commit
    source_tree = args.source_tree
    if protocol.get("schema") != "STAGE_VI_B2_FORMAL_M4_PROTOCOL_V1" or protocol.get("status") != "FROZEN_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise ValueError("B2_FORMAL_PROTOCOL_INVALID")
    if authority.get("schema") != "STAGE_VI_B2_FORMAL_M4_AUTHORITY_V1" or authority.get("status") != "PASS" or authority.get("formal_m4_authorized") is not True:
        raise ValueError("B2_FORMAL_AUTHORITY_INVALID")
    protocol_source = protocol.get("source_binding", {})
    authority_source = authority.get("source_binding", {})
    if protocol_source.get("runtime_commit") != source_commit or protocol_source.get("runtime_tree") != source_tree or authority_source.get("runtime_commit") != source_commit or authority_source.get("runtime_tree") != source_tree:
        raise ValueError("B2_FORMAL_SOURCE_BINDING")
    if authority.get("protocol_sha256") != sha(args.protocol.resolve()):
        raise ValueError("B2_FORMAL_PROTOCOL_HASH")
    provenance_path = args.authority.resolve().parent / "B2_RUNTIME_PROVENANCE.json"
    if not provenance_path.is_file() or sha(provenance_path) != authority.get("runtime_provenance_sha256"):
        raise ValueError("B2_FORMAL_PROVENANCE_HASH")
    provenance = load(provenance_path)
    if provenance.get("runner_sha256") != sha(Path(__file__)) or provenance.get("source_commit") != source_commit or provenance.get("source_tree") != source_tree:
        raise ValueError("B2_FORMAL_RUNNER_PROVENANCE")
    if protocol.get("protected_counters") != COUNTERS or authority.get("protected_counters") != COUNTERS:
        raise ValueError("B2_FORMAL_COUNTERS")
    inputs = protocol.get("inputs", {})
    plan_root = Path(str(inputs.get("exact_plan_root"))).resolve()
    plan_manifest = plan_root / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"
    plan_audit = plan_root / "B2_PLAN_INDEPENDENT_AUDIT.json"
    seal_path = plan_root / "SHA256SUMS.sha256"
    if not plan_root.is_dir() or not all(path.is_file() for path in (plan_manifest, plan_audit, seal_path, plan_root / "SHA256SUMS")):
        raise ValueError("B2_EXACT_PLAN_ROOT_INCOMPLETE")
    seal_tokens = seal_path.read_text(encoding="utf-8").split()
    if not seal_tokens or seal_tokens[0] != sha(plan_root / "SHA256SUMS") or seal_tokens[0] != inputs.get("exact_plan_root_seal_sha256"):
        raise ValueError("B2_EXACT_PLAN_ROOT_SEAL")
    manifest = load(plan_manifest)
    downstream = manifest.get("downstream_source", {})
    if sha(plan_manifest) != inputs.get("exact_plan_manifest_sha256") or manifest.get("status") != "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" or manifest.get("parent_count") != 16 or manifest.get("probe_count_total") != 384 or manifest.get("planned_branch_authority_count") != 1536 or downstream.get("commit") != source_commit or downstream.get("tree") != source_tree:
        raise ValueError("B2_EXACT_PLAN_BINDING")
    audit = load(plan_audit)
    if audit.get("status") != "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" or sha(plan_audit) != inputs.get("plan_audit_sha256"):
        raise ValueError("B2_EXACT_PLAN_AUDIT")
    split = load(Path(str(inputs["formal_parent_split_path"])))
    if sha(Path(str(inputs["formal_parent_split_path"]))) != inputs.get("formal_parent_split_sha256") or split.get("schema") != "STAGE_VI_B2_FORMAL_PARENT_SPLIT_V1" or split.get("status") != "FROZEN" or split.get("parent_count") != 16 or split.get("outcomes_read") is not False:
        raise ValueError("B2_FORMAL_SPLIT_BINDING")
    snapshot_source = inputs.get("snapshot_producer_source_binding")
    if not isinstance(snapshot_source, Mapping) or not snapshot_source.get("commit") or not snapshot_source.get("tree"):
        raise ValueError("B2_SNAPSHOT_PRODUCER_SOURCE_BINDING")
    snapshot_gate_runner_sha = inputs.get("snapshot_producer_gate_a_runner_sha256")
    if snapshot_gate_runner_sha and manifest.get("downstream_source", {}).get("gate_a_runner_sha256") != snapshot_gate_runner_sha:
        raise ValueError("B2_SNAPSHOT_PRODUCER_GATE_A_RUNNER")
    return protocol, authority, plan_root, sha(plan_manifest), {"commit": str(snapshot_source["commit"]), "tree": str(snapshot_source["tree"])}


def load_exact(plan_root: Path, parent_key: str, expected_manifest_sha: str, source_commit: str, source_tree: str, snapshot_source: Mapping[str, str]) -> dict[str, Any]:
    manifest_path = plan_root / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"
    manifest = load(manifest_path)
    if sha(manifest_path) != expected_manifest_sha:
        raise ValueError("B2_EXACT_MANIFEST_SHA")
    parent = next((row for row in manifest["parents"] if str(row.get("canonical_parent_key")) == parent_key), None)
    if not isinstance(parent, Mapping) or parent.get("status") != "PASS" or parent.get("probe_count") != 24:
        raise ValueError("B2_EXACT_PARENT")
    parent = dict(parent)
    parent_root = inside(plan_root, parent["output_dir"])
    clean_path = inside(plan_root, parent["clean_trajectory_path"])
    clean = load(clean_path)
    if sha(clean_path) != parent.get("clean_trajectory_sha256") or clean.get("outcomes_read") is not False:
        raise ValueError("B2_CLEAN_BINDING")
    actions = [{"step": int(row["step"]), "raw": row["raw_action"], "env": row["env_action"]} for row in clean["rows"]]
    if sha_json(actions) != parent.get("clean_reference_action_sequence_sha256"):
        raise ValueError("B2_CLEAN_ACTION_SHA")
    receipt = load(parent_root / "M3_5_V1_4_GATE_A_RECEIPT.json")
    if receipt.get("status") != "PASS" or receipt.get("source_commit") != snapshot_source["commit"] or receipt.get("source_tree") != snapshot_source["tree"] or receipt.get("outcomes_read") is not False or receipt.get("intervention_executed") is not False or receipt.get("protected_counters") != COUNTERS or len(receipt.get("canary_receipts", [])) != 24 or any(row.get("status") != "PASS" for row in receipt["canary_receipts"]):
        raise ValueError("B2_GATE_A_RECEIPT")
    probes = sorted([dict(row) for row in manifest["probe_authorities"] if str(row.get("canonical_parent_key")) == parent_key], key=lambda row: str(row["probe_id"]))
    if len(probes) != 24 or len({row["probe_id"] for row in probes}) != 24:
        raise ValueError("B2_PROBE_COUNT")
    for probe in probes:
        snapshot_root = inside(plan_root, probe["snapshot_path"])
        snapshot_manifest = snapshot_root / "CAUSAL_PROBE_SNAPSHOT_V2.json"
        if sha(snapshot_manifest) != probe.get("snapshot_manifest_sha256"):
            raise ValueError(f"B2_SNAPSHOT_SHA:{probe['probe_id']}")
        loaded = m4.load_snapshot(snapshot_root, materialize_torch=True)
        payload = loaded["payload"]
        hashes = m4.assert_primary_observation_exact(payload)
        binding = loaded["manifest"].get("binding", {})
        if binding.get("parent_key") != parent_key or binding.get("probe_id") != probe["probe_id"] or binding.get("source_commit") != snapshot_source["commit"] or binding.get("source_tree") != snapshot_source["tree"] or loaded["manifest"].get("primary_input_authority") != "loaded_frozen_canonical_bytes" or loaded["manifest"].get("fresh_render_equality_gate_used") is not False:
            raise ValueError(f"B2_SNAPSHOT_BINDING:{probe['probe_id']}")
        if hashes.get("raw_observation_sha256") != probe.get("raw_observation_sha256") or hashes.get("policy_rgb_224_sha256") != probe.get("policy_rgb_224_sha256") or hashes.get("policy_input_sha256") != probe.get("policy_input_sha256"):
            raise ValueError(f"B2_SNAPSHOT_INPUT:{probe['probe_id']}")
        state = payload.get("full_simulator_state", {}).get("registered_flat_state")
        if state is None or hashlib.sha256(__import__("numpy").asarray(state).tobytes(order="C")).hexdigest() != probe.get("sim_state_sha256"):
            raise ValueError(f"B2_SNAPSHOT_STATE:{probe['probe_id']}")
        if sha_json(payload.get("clean_reference_action_window")) != probe.get("clean_reference_action_window_sha256"):
            raise ValueError(f"B2_SNAPSHOT_ACTION:{probe['probe_id']}")
        probe["step"] = int(probe["probe_step"])
    return {"root": plan_root, "manifest": manifest, "parent": parent, "probes": probes, "canaries": [dict(row) for row in receipt["canary_receipts"]], "clean": clean, "clean_path": clean_path}


def run(args: argparse.Namespace) -> int:
    if not args.enable_runtime:
        raise ValueError("B2_FORMAL_RUNTIME_DISABLED")
    protocol, authority, plan_root, plan_sha, snapshot_source = validate_authority(args)
    parent_key = args.parent_key
    exact = load_exact(plan_root, parent_key, plan_sha, args.source_commit, args.source_tree, snapshot_source)
    parent = exact["parent"]
    output = args.output_dir.resolve()
    if output.exists():
        unexpected = [path.name for path in output.iterdir() if path.name not in {"CLAIM.json", "RESOURCE_PRE.json", "JOB.json", "SCIENCE_RUNNER.log"}]
        if unexpected:
            raise ValueError(f"B2_OUTPUT_NOT_NEW:{sorted(unexpected)}")
    else:
        output.mkdir(parents=True)
    shutil.copy2(args.protocol.resolve(), output / "M4_PROTOCOL.json")
    shutil.copy2(args.authority.resolve(), output / "M4_AUTHORITY.json")
    shutil.copy2(exact["clean_path"], output / "CLEAN_TRAJECTORY_V1_4.json")
    shutil.copy2(inside(plan_root, parent["probe_plan_path"]), output / "PROBE_PLAN_V1_4.json")
    m4._write(output / "EXACT_PLAN_BINDING.json", {"schema": "STAGE_VI_B2_EXACT_PLAN_RUNTIME_BINDING_V1", "status": "PASS_EXACT_FROZEN_AUTHORITY_LOADED", "exact_plan_root": str(plan_root), "exact_plan_manifest_sha256": plan_sha, "canonical_parent_key": parent_key, "probe_count": 24, "probe_selection_recomputed": False, "snapshots_loaded_frozen": True, "snapshot_producer_source_binding": snapshot_source, "outcomes_read": False, "protected_counters": COUNTERS})
    suite, task_part, state_part = parent_key.split("/")
    args.suite = suite
    args.parent_key = parent_key
    args.protocol = args.protocol.resolve()
    args.model_path = args.model_path.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
    get_image, get_processor, get_model, adapter_type, benchmark, libero_runtime = m4._load_external_modules(args.official_snapshot_root, args.upstream_root)
    get_libero_path, OffScreenRenderEnv = libero_runtime
    task_index, state_index = int(task_part.removeprefix("task_")), int(state_part.removeprefix("state_"))
    suite_obj = benchmark.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    init_state = copy.deepcopy(suite_obj.get_task_init_states(task_index)[state_index])
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    adapter, model, _processor, _unnorm_key = m4._load_policy(args, get_processor, get_model, adapter_type)
    horizon = int(m4.HORIZONS[suite])
    clean_actions = [list(row["env_action"]) for row in exact["clean"]["rows"]]
    snapshot_rows = [{"probe_id": str(probe["probe_id"]), "step": int(probe["step"]), "path": str(probe["snapshot_path"]), "manifest_sha256": str(probe["snapshot_manifest_sha256"])} for probe in exact["probes"]]
    binding_env, _ = m4._new_env(OffScreenRenderEnv, bddl, horizon, args.gpu, init_state, args, output)
    try:
        m4._write_runtime_binding_receipt(args, binding_env, output)
    finally:
        binding_env.close()
    m4._write(output / "M4_CAUSAL_SNAPSHOT_CANARY.json", {"schema": "STAGE_VI_B2_FORMAL_SNAPSHOT_CANARY_V1", "status": "PASS", "snapshots": snapshot_rows, "canaries": exact["canaries"], "fresh_render_equality_gate_used": False, "intervention_executed": False, "outcomes_read": False, "protected_counters": COUNTERS})
    branches, labels, observations = [], [], []
    for probe in exact["probes"]:
        snapshot_root = inside(plan_root, probe["snapshot_path"])
        control = m4._run_branch(snapshot_root=snapshot_root, gate_a_root=plan_root, OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output, clean_actions=clean_actions, model=model, adapter=adapter, arm="CONTROL", dose=0, probe_id=str(probe["probe_id"]), repetition=0, allow_horizon_censoring=True)
        control_record = m4._branch_record(parent_key, probe, "CONTROL", control)
        branches.append(control_record)
        for dose_name, dose_steps in m4.DOSES.items():
            treatment = m4._run_branch(snapshot_root=snapshot_root, gate_a_root=plan_root, OffScreenRenderEnv=OffScreenRenderEnv, bddl=bddl, horizon=horizon, init_state=init_state, args=args, output_dir=output, clean_actions=clean_actions, model=model, adapter=adapter, arm=dose_name, dose=dose_steps, probe_id=str(probe["probe_id"]), repetition=0, allow_horizon_censoring=True)
            pair = m4._pair_label(control, treatment, dose_steps)
            treatment_record = m4._branch_record(parent_key, probe, dose_name, treatment, pair=pair, control=control_record)
            branches.append(treatment_record)
            labels.append(m4._label(parent_key, probe, dose_name, treatment_record, control_record, pair))
            observations.append({"schema": "STAGE_VI_B2_M4_TREATMENT_OBSERVATION_V1", "horizon_contract": m4.HORIZON_CONTRACT, "canonical_parent_key": parent_key, "probe_id": probe["probe_id"], "dose": dose_name, **{key: pair[key] for key in ("label_class", "control_valid", "treatment_valid", "f_control", "f_open", "control_physical_class", "treatment_physical_class", "censoring_class")}, "treatment_compliant": treatment.get("treatment_compliant") is True, "treatment_branch_id": treatment_record["branch_id"], "control_branch_id": control_record["branch_id"], "protected_counters": COUNTERS})
    m4._write_jsonl(output / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl", branches)
    m4._write_jsonl(output / "M4_TREATMENT_OBSERVATIONS_V1.jsonl", observations)
    m4._write_jsonl(output / "M4_V_PHYS_LABELS_V1.jsonl", labels)
    result = {"schema": "STAGE_V_M4_PARENT_RESULT_V1", "stage_vi_b2": True, "horizon_contract": m4.HORIZON_CONTRACT, "status": "PASS", "parent_atomic": True, "canonical_parent_key": parent_key, "suite": suite, "task_index": task_index, "state_index": state_index, "split": parent.get("split"), "source_commit": args.source_commit, "source_tree": args.source_tree, "runner_sha256": sha(Path(__file__)), "protocol_sha256": sha(args.protocol), "authorization_receipt_sha256": sha(args.authority), "exact_plan_manifest_sha256": plan_sha, "probe_count": 24, "branch_count": len(branches), "treatment_label_count": len(labels), "expected_physical_executions": 96, "expected_treatment_labels": 72, "primary_estimand": "V_phys@T5", "primary_window": "MATCHED_CANONICAL_ACTION_T_PLUS_H_PHYS", "native_policy_calls_in_primary_window": 0, "fresh_render_equality_gate_used": False, "fresh_render_primary_consumption": False, "selection_outcomes_read": False, "probe_selection_source": "EXACT_FROZEN_B2_PLAN_MANIFEST", "probe_selection_recomputed": False, "causal_snapshot_canary_status": "PASS", "label_status": "VALID", "protected_counters": COUNTERS, "censored_branch_count": sum(1 for row in branches if row.get("branch", {}).get("horizon_censored") is True), "censored_label_count": sum(1 for row in labels if row.get("censoring_class") != "NONE"), "binary_label_count": sum(1 for row in labels if row.get("binary_label_consumable") is True), "abstention_map_frozen": True}
    m4._write(output / "PARENT_RESULT.json", result)
    m4._seal(output)
    audit = subprocess.run([str(args.python), str(REPO / "scripts/detector_v5/audit_stage_v_m4_matched_parent.py"), "--root", str(output), "--parent-key", parent_key, "--source-commit", args.source_commit, "--source-tree", args.source_tree], cwd=REPO, capture_output=True, text=True, check=False)
    audit_data = load(output / "M4_INDEPENDENT_AUDIT.json") if (output / "M4_INDEPENDENT_AUDIT.json").is_file() else {"status": "FAIL"}
    result["independent_audit_status"] = audit_data.get("status")
    result["independent_audit_sha256"] = sha(output / "M4_INDEPENDENT_AUDIT.json") if (output / "M4_INDEPENDENT_AUDIT.json").is_file() else None
    if audit_data.get("status") != "PASS_M4_PARENT_INDEPENDENT" or audit.returncode != 0:
        result["status"], result["label_status"] = "HOLD_SEALED", "HOLD"
    m4._write(output / "PARENT_RESULT.json", result)
    m4._seal(output)
    return 0 if result["status"] == "PASS" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("protocol", "authority", "plan_root", "output_dir", "official_snapshot_root", "upstream_root", "model_path", "python"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--enable-runtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
