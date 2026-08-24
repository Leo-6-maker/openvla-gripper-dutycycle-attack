"""Run outcome-blind clean-only Gate-A over the frozen spatial order."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
GATE_A = Path(__file__).resolve().with_name("run_stage_v_m3_5_v1_4_gate_a.py")
MODEL_RELATIVE = "libero-spatial/spatial_c8f03f4_20260620"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seal(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append(f"{_sha(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{_sha(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def _gpu(gpu: int) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            rows.append({"gpu": int(fields[0]), "memory_free_mib": int(float(fields[1])), "memory_used_mib": int(float(fields[2])), "utilization_gpu_percent": int(float(fields[3]))})
        except ValueError:
            continue
    row = next((item for item in rows if item["gpu"] == gpu), None)
    return {"status": "PASS" if result.returncode == 0 and row and row["memory_free_mib"] > 20480 else "HOLD", "requested_gpu": gpu, "gpu": row, "minimum_free_memory_mib": 20480, "strict_rule": "memory_free_mib > 20480", "foreign_workload_allowed": True, "foreign_process_interference": False, "captured_utc": _now()}


def _gate_a_status(root: Path, key: str) -> tuple[str, dict[str, Any]]:
    receipt_path = root / "M3_5_V1_4_GATE_A_RECEIPT.json"
    audit_path = root / "M3_5_V1_4_GATE_A_INDEPENDENT_AUDIT.json"
    if not receipt_path.is_file() or not audit_path.is_file():
        return "HOLD", {"reason": "GATE_A_FILES_MISSING", "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS}
    receipt, audit = _load(receipt_path), _load(audit_path)
    for value in (receipt, audit):
        if value.get("outcomes_read") is True or value.get("intervention_executed") is True or value.get("protected_counters") != COUNTERS:
            raise ValueError(f"GATE_A_BOUNDARY_VIOLATION:{key}")
    forbidden = [path.name for path in root.rglob("*") if path.is_file() and path.name in {"M4_COUNTERFACTUAL_BRANCHES_V1.jsonl", "M4_TREATMENT_OBSERVATIONS_V1.jsonl", "M4_V_PHYS_LABELS_V1.jsonl"}]
    if forbidden:
        raise ValueError(f"GATE_A_INTERVENTION_ARTIFACT:{key}:{forbidden[0]}")
    passed = receipt.get("status") == "PASS" and audit.get("status") == "PASS" and receipt.get("snapshot_count") == 24
    return ("PASS" if passed else "HOLD"), {"receipt_sha256": _sha(receipt_path), "audit_sha256": _sha(audit_path), "snapshot_count": receipt.get("snapshot_count"), "receipt_status": receipt.get("status"), "audit_status": audit.get("status"), "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS}


def run(args: argparse.Namespace) -> int:
    freeze_root = args.freeze_root.resolve()
    freeze_path = freeze_root / "FRESH_SPATIAL_POPULATION_MANIFEST.json"
    freeze = _load(freeze_path)
    if freeze.get("status") != "PASS_FROZEN_FRESH_SPATIAL_UNIVERSE" or freeze.get("fresh_candidate_count", 0) < 2 or freeze.get("selection", {}).get("clean_rollouts_started") is not False:
        raise ValueError("FRESH_SPATIAL_FREEZE_INVALID")
    if freeze.get("protected_counters") != COUNTERS or freeze.get("outcomes_read") is not False or freeze.get("intervention_executed") is not False:
        raise ValueError("FRESH_SPATIAL_FREEZE_BOUNDARY_INVALID")
    if freeze.get("source_binding") != {"commit": args.source_commit, "tree": args.source_tree}:
        raise ValueError("FRESH_SPATIAL_FREEZE_SOURCE_MISMATCH")
    freeze_seal = freeze_root / "SHA256SUMS.sha256"
    if not freeze_seal.is_file() or freeze_seal.read_text(encoding="utf-8").split()[0] != _sha(freeze_root / "SHA256SUMS"):
        raise ValueError("FRESH_SPATIAL_FREEZE_SEAL_INVALID")
    output = args.output_root.resolve()
    if output.exists():
        raise ValueError(f"REFUSE_OVERWRITE:{output}")
    output.mkdir(parents=True)
    rows = []
    for item in freeze["fresh_candidate_order"]:
        key = str(item["canonical_parent_key"])
        suite, task, state = key.split("/")
        rows.append({"ordinal": int(item["order"]), "canonical_parent_key": key, "suite": suite, "task_index": int(task.removeprefix("task_")), "state_index": int(state.removeprefix("state_")), "split": "FRESH_STATIC_CANDIDATE", "source": "fresh_spatial_v3", "selection_rank_sha256": str(item["rank_sha256"])})
    selection = {"schema": "STAGE_VI_B2_FRESH_SPATIAL_GATE_A_SELECTION_V1", "status": "FROZEN_BEFORE_CLEAN_ROLLOUT", "selection_rule": "frozen fresh spatial order; first two complete clean-only Gate-A PASS identities", "selection_reads": {"branch_results_read": False, "counterfactual_outcomes_read": False, "v_phys_read": False}, "freeze_manifest_path": str(freeze_path), "freeze_manifest_sha256": _sha(freeze_path), "freeze_seal_sha256": freeze_seal.read_text(encoding="utf-8").split()[0], "parent_count": len(rows), "selected_parents": rows, "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS}
    _write(output / "FRESH_GATE_A_SELECTION_MANIFEST.json", selection)
    protocol = {"schema": "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_4_GATE_A", "version": "V1.4.2-GATE-A", "status": "FROZEN_RUNTIME_AUTHORIZED", "runtime_authorized": True, "source_binding": {"runtime_commit": args.source_commit, "runtime_tree": args.source_tree}, "probe_plan_selection_version": "STAGE_V_M3_5_CORRIDOR_QUANTILES_V1", "operation": {"fresh_render_primary_consumption": "HARD_STOP", "fresh_render_equality_gate_used": False, "intervention_executed": False, "outcomes_read": False}, "fresh_population_binding": {"freeze_manifest_sha256": selection["freeze_manifest_sha256"], "freeze_seal_sha256": selection["freeze_seal_sha256"], "candidate_count": len(rows), "candidate_outcomes_read": False}, "protected_counters": COUNTERS}
    _write(output / "FRESH_GATE_A_PROTOCOL.json", protocol)
    scan = []
    passed: list[dict[str, Any]] = []
    for row in rows:
        if len(passed) >= 2:
            break
        gpu = _gpu(args.gpu)
        if gpu["status"] != "PASS":
            raise ValueError(f"GPU_RESOURCE_HOLD:{gpu}")
        key = str(row["canonical_parent_key"])
        slug = f"{int(row['ordinal']):03d}_{key.replace('/', '__')}"
        candidate_root = output / "candidates" / slug
        log_path = candidate_root.with_suffix(".log")
        command = [str(args.python), str(GATE_A), "--protocol", str(output / "FRESH_GATE_A_PROTOCOL.json"), "--selection-manifest", str(output / "FRESH_GATE_A_SELECTION_MANIFEST.json"), "--parent-key", key, "--output-dir", str(candidate_root), "--official-snapshot-root", str(args.official_snapshot_root), "--upstream-root", str(args.upstream_root), "--model-path", str(Path(args.model_root) / MODEL_RELATIVE), "--gpu", str(args.gpu), "--source-commit", args.source_commit, "--source-tree", args.source_tree, "--enable-runtime"]
        candidate_root.parent.mkdir(parents=True, exist_ok=True)
        started = _now()
        with log_path.open("w", encoding="utf-8") as handle:
            return_code = subprocess.run(command, cwd=GATE_A.parents[2], stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
        status, evidence = _gate_a_status(candidate_root, key)
        result = {"ordinal": row["ordinal"], "canonical_parent_key": key, "gpu": args.gpu, "started_utc": started, "finished_utc": _now(), "return_code": return_code, "status": status, "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS, "gate_a": evidence}
        scan.append(result)
        if status == "PASS" and return_code == 0:
            passed.append({**row, "clean_gate_a_root": str(candidate_root), "clean_gate_a_status": "PASS", "replacement_slot": 12 if len(passed) == 0 else 14})
    result = {"schema": "STAGE_VI_B2_FRESH_SPATIAL_GATE_A_SCAN_V1", "status": "PASS_TWO_FRESH_GATE_A_PARENTS" if len(passed) == 2 else "HOLD_NO_TWO_FRESH_GATE_A_PARENTS", "freeze_manifest_sha256": selection["freeze_manifest_sha256"], "freeze_seal_sha256": selection["freeze_seal_sha256"], "candidate_count": len(rows), "scanned_count": len(scan), "passed_count": len(passed), "selected_parents": passed, "scan": scan, "selection_outcomes_read": False, "outcomes_read": False, "intervention_executed": False, "protected_counters": COUNTERS, "generated_utc": _now()}
    _write(output / "FRESH_GATE_A_SCAN_RESULT.json", result)
    _seal(output)
    print(json.dumps({"status": result["status"], "root": str(output), "scanned": len(scan), "passed": len(passed)}, sort_keys=True))
    return 0 if result["status"] == "PASS_TWO_FRESH_GATE_A_PARENTS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--official-snapshot-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "HOLD_FRESH_SPATIAL_GATE_A", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
