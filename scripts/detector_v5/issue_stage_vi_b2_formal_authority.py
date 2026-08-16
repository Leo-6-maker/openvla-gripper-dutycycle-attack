"""Issue the append-only Stage VI-B2 formal authority after zero-treatment PASS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
ARMS = ["CONTROL", "T3", "T5", "T10"]


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def seal(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append(f"{sha(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def check_clean(source: Path) -> list[str]:
    return [line for line in git(source, "status", "--porcelain", "--untracked-files=all").splitlines() if line]


def run(args: argparse.Namespace) -> int:
    source = args.source_worktree.resolve()
    plan_root = args.plan_root.resolve()
    parent_manifest = args.parent_manifest.resolve()
    lock_path = args.lock.resolve()
    plan_manifest = plan_root / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"
    plan_audit = plan_root / "B2_PLAN_INDEPENDENT_AUDIT.json"
    plan_sums = plan_root / "SHA256SUMS"
    plan_sums_sha = plan_root / "SHA256SUMS.sha256"
    if not source.is_dir() or check_clean(source):
        raise ValueError("SOURCE_WORKTREE_NOT_CLEAN")
    source_commit = git(source, "rev-parse", "HEAD")
    source_tree = git(source, "rev-parse", "HEAD^{tree}")
    if args.source_commit != source_commit or args.source_tree != source_tree:
        raise ValueError("SOURCE_BINDING_MISMATCH")
    if args.ci_status != "PASS":
        raise ValueError("CI_NOT_PASS")
    if not git(source, "cat-file", "-e", f"{args.github_ref}^{{commit}}"):
        raise ValueError("GITHUB_REF_UNRESOLVED")
    if subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", source_commit, args.github_ref], check=False).returncode != 0:
        raise ValueError("SOURCE_NOT_REACHABLE_FROM_GITHUB_REF")
    lock = load(lock_path)
    manifest = load(parent_manifest)
    audit = load(plan_audit)
    exact = load(plan_manifest)
    if lock.get("status") != "FROZEN_PRE_HOLDOUT_LOCK" or lock.get("source_bindings", {}).get("future_formal_source_commit") != "PENDING_SUCCESSOR_SOURCE_FREEZE":
        raise ValueError("PRE_HOLDOUT_LOCK_INVALID")
    if sha(parent_manifest) != lock.get("formal_m4_contract", {}).get("parent_manifest_sha256"):
        raise ValueError("PARENT_MANIFEST_LOCK_HASH")
    if manifest.get("status") != "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" or audit.get("status") != "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN":
        raise ValueError("ZERO_TREATMENT_PLAN_NOT_PASS")
    if not all(path.is_file() for path in (plan_sums, plan_sums_sha)) or plan_sums_sha.read_text(encoding="utf-8").split()[0] != sha(plan_sums):
        raise ValueError("ZERO_TREATMENT_PLAN_SEAL")
    if exact.get("independent_audit_sha256") != sha(plan_audit):
        raise ValueError("ZERO_TREATMENT_MANIFEST_AUDIT_BINDING")
    if exact.get("parent_count") != 16 or exact.get("probe_count_total") != 384 or exact.get("planned_branch_authority_count") != 1536 or exact.get("protected_counters") != COUNTERS or exact.get("intervention_executed") is not False or exact.get("outcomes_read") is not False:
        raise ValueError("ZERO_TREATMENT_PLAN_MATRIX_OR_BOUNDARY")
    output = args.output_root.resolve()
    if output.exists():
        raise ValueError(f"REFUSE_OVERWRITE:{output}")
    output.mkdir(parents=True)
    protocol = {
        "schema": "STAGE_VI_B2_FORMAL_M4_PROTOCOL_V1",
        "status": "FROZEN_AUTHORIZED",
        "runtime_authorized": True,
        "successor_execution_id": args.successor_execution_id,
        "attempt_ordinal": 1,
        "source_binding": {"runtime_commit": source_commit, "runtime_tree": source_tree, "github_ref": args.github_ref},
        "scientific_estimand": {"primary": "V_phys@T5", "secondary": ["V_phys@T3", "V_phys@T10"], "teacher_student_frozen": True, "selected_detector": "B2-C_SOFT_TV_DISTILL_DIRECT_VPHYS", "model_or_threshold_changes_after_lock": False},
        "matrix": {"parents": 16, "probes_per_parent": 24, "arms_per_probe": ARMS, "physical_executions": 1536, "treatment_labels": 1152, "H_phys": 10},
        "operation": {"clean_reference_actions": "EXACT_FROZEN_GATE_A_BYTES", "primary_input_authority": "loaded_frozen_canonical_bytes", "matched_action_replay": True, "fresh_render_primary_consumption": False, "allow_horizon_censoring": True, "censoring_policy": "preserve_abstains_and_exclude_from_binary_V_phys", "selection_outcomes_read": False},
        "inputs": {"formal_parent_manifest_path": str(parent_manifest), "formal_parent_manifest_sha256": sha(parent_manifest), "formal_parent_split_path": str(plan_root / "B2_PARENT_SPLIT.json"), "formal_parent_split_sha256": sha(plan_root / "B2_PARENT_SPLIT.json"), "exact_plan_root": str(plan_root), "exact_plan_root_seal_sha256": plan_sums_sha.read_text(encoding="utf-8").split()[0], "exact_plan_manifest_sha256": sha(plan_manifest), "plan_audit_sha256": sha(plan_audit), "pre_holdout_lock_path": str(lock_path), "pre_holdout_lock_sha256": sha(lock_path), "official_environment": str(args.official_environment), "official_snapshot_root": str(args.official_snapshot_root), "upstream_root": str(args.upstream_root), "model_root": str(args.model_root), "selected_student_model_sha256": lock.get("development_evidence", {}).get("selected_student_model_sha256")},
        "resource_contract": {"minimum_free_memory_mib": 20480, "strict_rule": "memory_free_mib > 20480", "one_project_worker_per_physical_gpu": True, "max_project_workers": 8, "foreign_workload_interference": False},
        "protected_counters": COUNTERS,
    }
    write(output / "B2_FORMAL_PROTOCOL.json", protocol)
    provenance = {"schema": "STAGE_VI_B2_RUNTIME_PROVENANCE_V1", "status": "PASS", "source_commit": source_commit, "source_tree": source_tree, "github_ref": args.github_ref, "ci_status": args.ci_status, "source_worktree": str(source), "official_environment": str(args.official_environment), "official_snapshot_root": str(args.official_snapshot_root), "upstream_root": str(args.upstream_root), "model_root": str(args.model_root), "runner_relative_path": "scripts/detector_v5/run_stage_vi_b2_formal_parent.py", "runner_sha256": sha(source / "scripts/detector_v5/run_stage_vi_b2_formal_parent.py"), "scheduler_relative_path": "scripts/detector_v5/run_stage_vi_b2_formal_scheduler.py", "scheduler_sha256": sha(source / "scripts/detector_v5/run_stage_vi_b2_formal_scheduler.py"), "protected_counters": COUNTERS}
    write(output / "B2_RUNTIME_PROVENANCE.json", provenance)
    authority = {"schema": "STAGE_VI_B2_FORMAL_M4_AUTHORITY_V1", "status": "PASS", "formal_m4_authorized": True, "runtime_authorized": True, "successor_execution_id": args.successor_execution_id, "attempt_ordinal": 1, "source_binding": {"runtime_commit": source_commit, "runtime_tree": source_tree, "github_ref": args.github_ref}, "protocol_sha256": sha(output / "B2_FORMAL_PROTOCOL.json"), "runtime_provenance_sha256": sha(output / "B2_RUNTIME_PROVENANCE.json"), "parent_manifest_sha256": sha(parent_manifest), "exact_plan_manifest_sha256": sha(plan_manifest), "exact_plan_audit_sha256": sha(plan_audit), "pre_holdout_lock_sha256": sha(lock_path), "historical_stage_v_attempts": "NONCONSUMABLE", "historical_mismatch": "NON_REPRODUCED_UNRESOLVED", "scientific_estimand_unchanged": True, "final40_unchanged": True, "teacher_student_unchanged": True, "thresholds_unchanged": True, "owner_authorization": "Stage VI-B2 direct vulnerability detector successor after frozen development gate and zero-treatment plan PASS", "protected_counters": COUNTERS}
    write(output / "B2_FORMAL_AUTHORITY.json", authority)
    seal(output)
    result = {"status": "PASS_STAGE_VI_B2_FORMAL_AUTHORITY", "root": str(output), "source_commit": source_commit, "source_tree": source_tree, "github_ref": args.github_ref, "successor_execution_id": args.successor_execution_id, "protected_counters": COUNTERS}
    write(output / "B2_AUTHORITY_RESULT.json", result)
    seal(output)
    print(json.dumps(result, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--github-ref", required=True)
    parser.add_argument("--ci-status", required=True)
    parser.add_argument("--successor-execution-id", required=True)
    parser.add_argument("--official-environment", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "HOLD_STAGE_VI_B2_FORMAL_AUTHORITY", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
