"""Prepare the Stage VI-B2 16-parent zero-treatment plan and snapshots."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.stage_v_canonical_execution_core import canonical_sha256, canonical_value  # noqa: E402
from gripper_attack.stage_v_causal_observation_snapshot import load_snapshot  # noqa: E402


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
GATE_A = REPO / "scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_a.py"
MODEL_RELATIVE = {
    "libero_10": "libero-10/openvla-7b-finetuned-libero-10",
    "libero_goal": "libero-goal",
    "libero_object": "openvla-7b-finetuned-libero-object",
    "libero_spatial": "libero-spatial/spatial_c8f03f4_20260620",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inventory(requested: list[int]) -> dict[str, Any]:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    rows = []
    for line in gpu.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append({"gpu": int(parts[0]), "memory_free_mib": int(float(parts[1])), "memory_used_mib": int(float(parts[2])), "utilization_gpu_percent": int(float(parts[3]))})
        except ValueError:
            continue
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    eligible = [row["gpu"] for row in rows if row["gpu"] in requested and row["memory_free_mib"] > 20480]
    return {"status": "PASS" if gpu.returncode == 0 and eligible else "HOLD", "requested": requested, "eligible": eligible, "minimum_free_memory_mib": 20480, "strict_rule": "memory_free_mib > 20480", "gpu_rows": rows, "compute_apps_raw": apps.stdout.splitlines(), "foreign_workload_allowed": True, "foreign_process_interference": False}


def _stable_reserve_passes(paths: list[Path]) -> set[str]:
    passes: set[str] = set()
    for root in paths:
        for path in sorted(root.glob("parents/*/attempt_01/M4_CORRIDOR_PREFLIGHT.json")):
            data = load_json(path)
            if data.get("status") == "PASS" and data.get("reason") == "M4_CORRIDOR_24_EXACT":
                passes.add(str(data["canonical_parent_key"]))
    return passes


def select_parents(config: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = config["inputs"]
    post_path = Path(inputs["post_hold_manifest_path"])
    old_path = Path(inputs["stable_reserve_manifest_path"])
    prior_path = Path(inputs["prior_formal_split_path"])
    post, old, prior = load_json(post_path), load_json(old_path), load_json(prior_path)
    if sha(post_path) != inputs["post_hold_manifest_sha256"] or sha(old_path) != inputs["stable_reserve_manifest_sha256"] or sha(prior_path) != inputs["prior_formal_split_sha256"]:
        raise ValueError("B2_PARENT_SOURCE_SHA_MISMATCH")
    if post.get("status") != "FROZEN" or post.get("outcomes_read") is not False or old.get("status") != "FROZEN" or old.get("outcomes_read") is not False:
        raise ValueError("B2_PARENT_SOURCE_NOT_CLEAN_ONLY")
    prior_keys = {str(row["canonical_parent_key"]) for row in prior["parents"]}
    post_rows = {str(row["canonical_parent_key"]): dict(row) for row in post["parents"]}
    old_rows = {str(row["canonical_parent_key"]): dict(row) for row in old["parents"]}
    stable = _stable_reserve_passes([Path(item) for item in config["inputs"]["stable_reserve_replicates"]])
    expected_stable = {"libero_object/task_03/state_28", "libero_object/task_06/state_27", "libero_object/task_09/state_22"}
    if stable != expected_stable:
        raise ValueError(f"B2_STABLE_RESERVE_SET:{sorted(stable)}")
    selected = []
    for frozen in config["parents"]:
        key = str(frozen["canonical_parent_key"])
        source = str(frozen["source"])
        row = post_rows.get(key) if source == "post_hold" else old_rows.get(key)
        if row is None or (source == "post_hold" and key in prior_keys) or (source == "reserve_a_b_pass" and key not in stable):
            raise ValueError(f"B2_PARENT_INELIGIBLE:{key}")
        suite, task_part, state_part = key.split("/")
        selected.append({"ordinal": int(frozen["ordinal"]), "canonical_parent_key": key, "suite": suite, "task_index": int(task_part.removeprefix("task_")), "state_index": int(state_part.removeprefix("state_")), "split": "TEST", "source": source, "source_rank": frozen.get("source_rank"), "taxonomy_status": row.get("taxonomy_status", "SUPPORTED"), "bddl_path": row.get("bddl_path")})
    if len(selected) != 16 or len({row["canonical_parent_key"] for row in selected}) != 16:
        raise ValueError("B2_PARENT_SELECTION_COUNT")
    return selected


def _command(root: Path, parent: Mapping[str, Any], gpu: int, runtime: Mapping[str, str], source_commit: str, source_tree: str) -> list[str]:
    return [runtime["python"], str(GATE_A), "--protocol", str(root / "B2_GATE_A_PROTOCOL.json"), "--selection-manifest", str(root / "B2_SELECTION_MANIFEST.json"), "--parent-key", str(parent["canonical_parent_key"]), "--output-dir", str(root / "parents" / f"{int(parent['ordinal']):02d}_{str(parent['canonical_parent_key']).replace('/', '__')}"), "--official-snapshot-root", runtime["official_snapshot_root"], "--upstream-root", runtime["upstream_root"], "--model-path", str(Path(runtime["model_root"]) / MODEL_RELATIVE[str(parent["suite"])]), "--gpu", str(gpu), "--source-commit", source_commit, "--source-tree", source_tree, "--enable-runtime"]


def run_clean(root: Path, parents: list[dict[str, Any]], runtime: dict[str, str], gpus: list[int], source_commit: str, source_tree: str) -> list[dict[str, Any]]:
    log_root = root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=min(8, len(gpus))) as pool:
        pending = iter(parents)
        active = {}
        for gpu in gpus[:8]:
            try:
                parent = next(pending)
            except StopIteration:
                break
            active[pool.submit(_run_one, root, parent, gpu, runtime, source_commit, source_tree, log_root)] = (gpu, parent)
        while active:
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                gpu, _parent = active.pop(future)
                try:
                    parent = next(pending)
                except StopIteration:
                    continue
                active[pool.submit(_run_one, root, parent, gpu, runtime, source_commit, source_tree, log_root)] = (gpu, parent)
    return sorted(results, key=lambda row: int(row["ordinal"]))


def _run_one(root: Path, parent: Mapping[str, Any], gpu: int, runtime: Mapping[str, str], source_commit: str, source_tree: str, log_root: Path) -> dict[str, Any]:
    command = _command(root, parent, gpu, runtime, source_commit, source_tree)
    slug = f"{int(parent['ordinal']):02d}_{str(parent['canonical_parent_key']).replace('/', '__')}"
    log = log_root / f"{slug}.log"
    write(log_root / f"{slug}.COMMAND.json", {"schema": "STAGE_VI_B2_ZERO_TREATMENT_COMMAND_V1", "parent": dict(parent), "gpu": gpu, "command": command, "intervention_executed": False, "outcomes_read": False, "protected_counters": COUNTERS})
    started = datetime.now(timezone.utc).isoformat()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=False, text=True)
    return {"ordinal": parent["ordinal"], "canonical_parent_key": parent["canonical_parent_key"], "gpu": gpu, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "return_code": process.returncode, "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS}


def build_exact_manifest(root: Path, parents: list[dict[str, Any]], source_commit: str, source_tree: str, runtime: Mapping[str, str]) -> dict[str, Any]:
    entries, probes, branches, errors = [], [], [], []
    for parent in parents:
        key = str(parent["canonical_parent_key"])
        parent_root = root / "parents" / f"{int(parent['ordinal']):02d}_{key.replace('/', '__')}"
        entry = {**parent, "output_dir": parent_root.relative_to(root).as_posix(), "status": "HOLD"}
        try:
            receipt = load_json(parent_root / "M3_5_V1_4_GATE_A_RECEIPT.json")
            audit = load_json(parent_root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json")
            clean = load_json(parent_root / "CLEAN_TRAJECTORY_V1_4.json")
            plan = load_json(parent_root / "PROBE_PLAN_V1_4.json")
            taxonomy = load_json(parent_root / "TAXONOMY_BINDING.json")
            if receipt.get("status") != "PASS" or audit.get("status") != "PASS" or receipt.get("snapshot_count") != 24 or receipt.get("intervention_executed") is not False or receipt.get("outcomes_read") is not False or receipt.get("protected_counters") != COUNTERS or taxonomy.get("status") != "PASS":
                raise ValueError("GATE_A_NOT_PASS")
            if clean.get("outcomes_read") is not False or plan.get("outcomes_read") is not False or len(plan.get("probe_steps", [])) != 24:
                raise ValueError("GATE_A_CLEAN_PLAN_INVALID")
            actions = [{"step": int(row["step"]), "raw": row["raw_action"], "env": row["env_action"]} for row in clean["rows"]]
            action_sha = canonical_sha256(canonical_value(actions))
            entry.update({"status": "PASS", "receipt_sha256": sha(parent_root / "M3_5_V1_4_GATE_A_RECEIPT.json"), "audit_sha256": sha(parent_root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json"), "clean_trajectory_path": parent_root.joinpath("CLEAN_TRAJECTORY_V1_4.json").relative_to(root).as_posix(), "clean_trajectory_sha256": sha(parent_root / "CLEAN_TRAJECTORY_V1_4.json"), "clean_reference_action_sequence_sha256": action_sha, "probe_plan_path": parent_root.joinpath("PROBE_PLAN_V1_4.json").relative_to(root).as_posix(), "probe_plan_sha256": sha(parent_root / "PROBE_PLAN_V1_4.json"), "taxonomy_binding_path": parent_root.joinpath("TAXONOMY_BINDING.json").relative_to(root).as_posix(), "taxonomy_binding_sha256": sha(parent_root / "TAXONOMY_BINDING.json"), "probe_count": 24, "intervention_executed": False, "outcomes_read": False, "model_path": str(Path(runtime["model_root"]) / MODEL_RELATIVE[str(parent["suite"])] )})
            snapshot_by_probe = {str(row["probe_id"]): row for row in receipt["snapshots"]}
            for probe in plan["probe_steps"]:
                probe_id = str(probe["probe_id"])
                snapshot_row = snapshot_by_probe[probe_id]
                snapshot_root = parent_root / str(snapshot_row["path"])
                loaded = load_snapshot(snapshot_root, materialize_torch=True)
                payload = loaded["payload"]
                probes.append({"canonical_parent_key": key, "probe_id": probe_id, "probe_step": int(probe["step"]), "sim_state_sha256": probe.get("state_sha256"), "raw_observation_sha256": payload.get("raw_observation_sha256"), "policy_rgb_224_sha256": probe.get("policy_rgb_224_sha256"), "policy_input_sha256": probe.get("policy_input_sha256"), "snapshot_path": snapshot_root.relative_to(root).as_posix(), "snapshot_manifest_sha256": snapshot_row["manifest_sha256"], "clean_reference_action_sequence_sha256": action_sha, "clean_reference_action_window_sha256": canonical_sha256(canonical_value(payload.get("clean_reference_action_window"))), "H_phys": 10, "intervention_executed": False, "outcomes_read": False})
                for arm in ("CONTROL", "T3", "T5", "T10"):
                    branches.append({"canonical_parent_key": key, "probe_id": probe_id, "probe_step": int(probe["step"]), "arm": arm, "branch_id": "b2-plan-" + hashlib.sha256(f"B2_PLAN::{key}::{probe_id}::{arm}".encode()).hexdigest(), "snapshot_manifest_sha256": snapshot_row["manifest_sha256"], "execution_status": "PLANNED_NOT_EXECUTED", "outcomes_read": False, "protected_counters": COUNTERS})
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}:{exc}"
            errors.append(f"{key}:{entry['error']}")
        entries.append(entry)
    return {"schema": "STAGE_VI_B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST_V1", "status": "FROZEN_PLAN_ONLY_PENDING_INDEPENDENT_AUDIT", "sealed": False, "downstream_source": {"commit": source_commit, "tree": source_tree, "gate_a_runner_sha256": sha(GATE_A)}, "parent_count": 16, "probe_count_per_parent": 24, "probe_count_total": len(probes), "planned_branch_authority_count": len(branches), "planned_branch_authority_expected": 1536, "parents": entries, "probe_authorities": probes, "branch_authorities": branches, "selection_outcomes_read": False, "intervention_executed": False, "v_phys_generated": False, "teacher_predictions_read": False, "student_predictions_read": False, "protected_counters": COUNTERS, "errors": sorted(errors), "failure_action": "HOLD_SEALED_NO_RESERVE_SUBSTITUTION_NO_RERUN"}


def seal(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append(f"{sha(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args(argv)
    try:
        config = load_json(args.parent_manifest.resolve())
        parents = select_parents(config)
        source_commit, source_tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
        root = args.output_root.resolve()
        if root.exists():
            raise ValueError(f"REFUSE_OVERWRITE:{root}")
        root.mkdir(parents=True)
        runtime = {"python": str(args.python_executable), "official_snapshot_root": str(args.official_snapshot_root), "upstream_root": str(args.upstream_root), "model_root": str(args.model_root)}
        write(root / "B2_SELECTION_MANIFEST.json", {"schema": "STAGE_VI_B2_ZERO_TREATMENT_SELECTION_MANIFEST_V1", "status": "FROZEN_BEFORE_PLAN_RUNTIME", "selection_reads": {"branch_results_read": False, "counterfactual_outcomes_read": False, "v_phys_read": False}, "parent_count": 16, "selected_parents": parents, "source_parent_manifest": str(args.parent_manifest.resolve()), "source_parent_manifest_sha256": sha(args.parent_manifest.resolve()), "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS})
        split = {"schema": "STAGE_VI_B2_FORMAL_PARENT_SPLIT_V1", "status": "FROZEN", "parent_count": 16, "counts": {"TRAIN": 0, "VAL": 0, "TEST": 16}, "parents": parents, "formal_m4_authorized": False, "outcomes_read": False, "protected_counters": COUNTERS}
        write(root / "B2_PARENT_SPLIT.json", split)
        write(root / "B2_PLAN_PROTOCOL.json", {"schema": "STAGE_VI_B2_ZERO_TREATMENT_PLAN_PROTOCOL_V1", "status": "FROZEN_RUNTIME_AUTHORIZED", "runtime_authorized": True, "source_binding": {"runtime_commit": source_commit, "runtime_tree": source_tree}, "probe_plan_selection_version": "STAGE_V_M3_5_CORRIDOR_QUANTILES_V1", "matrix": {"parents": 16, "probes_per_parent": 24, "probe_count_total": 384, "planned_branch_authorities_total": 1536}, "operation": {"clean_rollout": True, "snapshot_capture": True, "snapshot_restore_canary": True, "intervention_executed": False, "outcomes_read": False, "v_phys_generated": False, "teacher_predictions_read": False, "student_predictions_read": False, "eval160_reads": False, "fresh_render_primary_consumption": "HARD_STOP"}, "protected_counters": COUNTERS, "resource_contract": {"minimum_free_memory_mib": 20480, "strict_comparison": "memory_free_mib > 20480", "foreign_workload_allowed": True, "foreign_process_interference": False, "one_project_worker_per_gpu": True, "max_project_workers": 8}})
        write(root / "B2_GATE_A_PROTOCOL.json", {"schema": "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_A", "version": "V1.4.2-GATE-A", "status": "FROZEN_RUNTIME_AUTHORIZED", "runtime_authorized": True, "source_binding": {"runtime_commit": source_commit, "runtime_tree": source_tree}, "probe_plan_selection_version": "STAGE_V_M3_5_CORRIDOR_QUANTILES_V1", "operation": {"fresh_render_primary_consumption": "HARD_STOP", "fresh_render_equality_gate_used": False, "intervention_executed": False}, "protected_counters": COUNTERS})
        write(root / "B2_PLAN_AUTHORIZATION.json", {"schema": "STAGE_VI_B2_ZERO_TREATMENT_PLAN_AUTHORIZATION_V1", "status": "PASS_PRELAUNCH", "source_commit": source_commit, "source_tree": source_tree, "protocol_sha256": sha(root / "B2_PLAN_PROTOCOL.json"), "selection_manifest_sha256": sha(root / "B2_SELECTION_MANIFEST.json"), "parent_split_sha256": sha(root / "B2_PARENT_SPLIT.json"), "intervention_executed": False, "outcomes_read": False, "protected_counters": COUNTERS})
        requested = sorted({int(item) for item in args.gpus.split(",") if item.strip()})
        resource = inventory(requested)
        write(root / "RESOURCE_PRELAUNCH.json", resource)
        if not resource["eligible"]:
            raise ValueError("NO_GPU_FREE_MEMORY_ABOVE_20G")
        results = run_clean(root, parents, runtime, resource["eligible"], source_commit, source_tree)
        write(root / "PARENT_RUN_REGISTRY.json", {"schema": "STAGE_VI_B2_ZERO_TREATMENT_PARENT_RUN_REGISTRY_V1", "status": "COMPLETE" if all(row["return_code"] == 0 for row in results) else "HOLD", "parent_count": 16, "results": results, "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS})
        exact = build_exact_manifest(root, parents, source_commit, source_tree, runtime)
        write(root / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json", exact)
        audit_process = subprocess.run([str(args.python_executable), str(REPO / "scripts/detector_v5/audit_stage_vi_b2_formal_plan.py"), "--root", str(root), "--source-commit", source_commit, "--source-tree", source_tree], cwd=REPO, capture_output=True, text=True, check=False)
        audit = load_json(root / "B2_PLAN_INDEPENDENT_AUDIT.json") if (root / "B2_PLAN_INDEPENDENT_AUDIT.json").is_file() else {"status": "FAIL"}
        if audit.get("status") == "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" and audit_process.returncode == 0:
            exact["status"], exact["sealed"], exact["independent_audit_sha256"] = "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN", True, sha(root / "B2_PLAN_INDEPENDENT_AUDIT.json")
        else:
            exact["status"], exact["sealed"] = "HOLD_SEALED_STAGE_VI_B2_ZERO_TREATMENT_PLAN", True
        write(root / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json", exact)
        write(root / "B2_PLAN_RESULT.json", {"schema": "STAGE_VI_B2_ZERO_TREATMENT_PLAN_RESULT_V1", "status": audit.get("status", "FAIL"), "parent_count": 16, "probe_count_total": exact["probe_count_total"], "planned_branch_authority_count": exact["planned_branch_authority_count"], "intervention_executed": False, "outcomes_read": False, "protected_counters": COUNTERS})
        seal(root)
        print(json.dumps({"status": audit.get("status", "FAIL"), "root": str(root), "source_commit": source_commit, "source_tree": source_tree, "parents": 16, "probes": exact["probe_count_total"], "planned_branches": exact["planned_branch_authority_count"]}, sort_keys=True))
        return 0 if audit.get("status") == "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" and audit_process.returncode == 0 else 2
    except Exception as exc:
        print(json.dumps({"status": "HOLD_SEALED", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
