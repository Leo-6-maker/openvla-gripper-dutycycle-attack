"""Run the frozen Stage VI-B2 parent queue with one worker per eligible GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
REPO = Path(__file__).resolve().parents[2]


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory(requested: list[int]) -> dict[str, Any]:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    rows = []
    for line in gpu.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append({"gpu": int(parts[0]), "uuid": parts[1], "memory_free_mib": int(float(parts[2])), "memory_used_mib": int(float(parts[3])), "utilization_gpu_percent": int(float(parts[4]))})
        except ValueError:
            continue
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    eligible = [row["gpu"] for row in rows if row["gpu"] in requested and row["memory_free_mib"] > 20480]
    return {"status": "PASS" if gpu.returncode == 0 else "HOLD", "requested": requested, "eligible": eligible, "minimum_free_memory_mib": 20480, "strict_rule": "memory_free_mib > 20480", "gpu_rows": rows, "compute_apps_raw": apps.stdout.splitlines(), "foreign_workload_allowed": True, "foreign_process_interference": False, "captured_utc": now()}


def parent_slug(row: Mapping[str, Any]) -> str:
    return f"{int(row['ordinal']):02d}_{str(row['canonical_parent_key']).replace('/', '__')}"


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = load(args.protocol.resolve())
    authority = load(args.authority.resolve())
    plan_manifest = load(args.plan_root.resolve() / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json")
    plan_audit = load(args.plan_root.resolve() / "B2_PLAN_INDEPENDENT_AUDIT.json")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO, text=True).strip()
    if protocol.get("schema") != "STAGE_VI_B2_FORMAL_M4_PROTOCOL_V1" or protocol.get("status") != "FROZEN_AUTHORIZED" or protocol.get("runtime_authorized") is not True:
        raise ValueError("PROTOCOL_INVALID")
    if authority.get("schema") != "STAGE_VI_B2_FORMAL_M4_AUTHORITY_V1" or authority.get("status") != "PASS" or authority.get("formal_m4_authorized") is not True:
        raise ValueError("AUTHORITY_INVALID")
    for value in (protocol, authority):
        source = value.get("source_binding", {})
        if source.get("runtime_commit") != source_commit or source.get("runtime_tree") != source_tree:
            raise ValueError("SOURCE_BINDING_MISMATCH")
        if value.get("protected_counters") != COUNTERS:
            raise ValueError("PROTECTED_COUNTERS")
    if plan_audit.get("status") != "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" or plan_manifest.get("status") != "PASS_STAGE_VI_B2_ZERO_TREATMENT_PLAN" or sha(args.plan_root.resolve() / "B2_EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json") != authority.get("exact_plan_manifest_sha256"):
        raise ValueError("ZERO_TREATMENT_PLAN_NOT_BOUND")
    parents = [dict(row) for row in plan_manifest.get("parents", []) if isinstance(row, Mapping)]
    if len(parents) != 16 or [int(row.get("ordinal", 0)) for row in parents] != list(range(1, 17)) or any(row.get("status") != "PASS" or row.get("outcomes_read") is not False for row in parents):
        raise ValueError("PARENT_PLAN_INVALID")
    return protocol, authority, plan_manifest, parents


def launch(args: argparse.Namespace, parent: Mapping[str, Any], gpu: Mapping[str, Any], output: Path, protocol: Mapping[str, Any], authority: Mapping[str, Any], plan_manifest: Mapping[str, Any], active: dict[int, dict[str, Any]]) -> dict[str, Any]:
    parent_dir = output / "parents" / parent_slug(parent)
    parent_dir.mkdir(parents=True, exist_ok=False)
    job = {"schema": "STAGE_VI_B2_FORMAL_PARENT_JOB_V1", "status": "CLAIMED", "canonical_parent_key": parent["canonical_parent_key"], "ordinal": parent["ordinal"], "gpu": gpu["gpu"], "gpu_uuid": gpu["uuid"], "started_utc": now(), "source_commit": protocol["source_binding"]["runtime_commit"], "source_tree": protocol["source_binding"]["runtime_tree"], "authority_sha256": sha(args.authority.resolve()), "protocol_sha256": sha(args.protocol.resolve()), "outcomes_read": False, "protected_counters": COUNTERS}
    write(parent_dir / "RESOURCE_PRE.json", {"schema": "STAGE_VI_B2_RESOURCE_PRE_V1", "status": "PASS", "gpu": gpu["gpu"], "gpu_uuid": gpu["uuid"], "memory_free_mib": gpu["memory_free_mib"], "strict_rule": "memory_free_mib > 20480", "foreign_workload_allowed": True, "foreign_process_interference": False, "captured_utc": now(), "protected_counters": COUNTERS})
    write(parent_dir / "JOB.json", job)
    command = [str(args.python), str(args.runner), "--protocol", str(args.protocol.resolve()), "--authority", str(args.authority.resolve()), "--plan-root", str(args.plan_root.resolve()), "--output-dir", str(parent_dir), "--official-snapshot-root", str(protocol["inputs"]["official_snapshot_root"]), "--upstream-root", str(protocol["inputs"]["upstream_root"]), "--model-path", str(parent["model_path"]), "--python", str(args.python), "--parent-key", str(parent["canonical_parent_key"]), "--gpu", str(gpu["gpu"]), "--source-commit", str(protocol["source_binding"]["runtime_commit"]), "--source-tree", str(protocol["source_binding"]["runtime_tree"]), "--enable-runtime"]
    log_path = parent_dir / "SCIENCE_RUNNER.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu["gpu"])
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, env=env, text=True)
    handle.close()
    claim = {"schema": "STAGE_VI_B2_ATOMIC_PARENT_CLAIM_V1", "status": "ACTIVE", "canonical_parent_key": parent["canonical_parent_key"], "ordinal": parent["ordinal"], "worker_id": f"stage_vi_b2_gpu_{gpu['gpu']}", "worker_pid": process.pid, "gpu": gpu["gpu"], "gpu_uuid": gpu["uuid"], "cuda_visible_devices": str(gpu["gpu"]), "source_commit": protocol["source_binding"]["runtime_commit"], "source_tree": protocol["source_binding"]["runtime_tree"], "authority_sha256": sha(args.authority.resolve()), "protocol_sha256": sha(args.protocol.resolve()), "attempt_ordinal": 1, "outcome_selection": False, "outcomes_read": False, "protected_counters": COUNTERS, "claimed_utc": now()}
    write(parent_dir / "CLAIM.json", claim)
    active[gpu["gpu"]] = {"process": process, "parent": dict(parent), "gpu": dict(gpu), "output": parent_dir, "log": str(log_path), "handle_closed": True}
    return claim


def finish_worker(worker: dict[str, Any]) -> dict[str, Any]:
    process = worker["process"]
    return_code = process.poll()
    if return_code is None:
        return {"done": False}
    parent_dir = worker["output"]
    result = load(parent_dir / "PARENT_RESULT.json") if (parent_dir / "PARENT_RESULT.json").is_file() else {}
    audit = load(parent_dir / "M4_INDEPENDENT_AUDIT.json") if (parent_dir / "M4_INDEPENDENT_AUDIT.json").is_file() else {}
    status = "PASS" if return_code == 0 and result.get("status") == "PASS" and audit.get("status") == "PASS_M4_PARENT_INDEPENDENT" else "HOLD"
    return {"done": True, "return_code": return_code, "status": status, "ordinal": worker["parent"]["ordinal"], "canonical_parent_key": worker["parent"]["canonical_parent_key"], "gpu": worker["gpu"]["gpu"], "gpu_uuid": worker["gpu"]["uuid"], "finished_utc": now(), "parent_result_sha256": sha(parent_dir / "PARENT_RESULT.json") if (parent_dir / "PARENT_RESULT.json").is_file() else None, "audit_sha256": sha(parent_dir / "M4_INDEPENDENT_AUDIT.json") if (parent_dir / "M4_INDEPENDENT_AUDIT.json").is_file() else None, "protected_counters": COUNTERS}


def run(args: argparse.Namespace) -> int:
    protocol, authority, plan_manifest, parents = validate(args)
    output = args.output_root.resolve()
    if output.exists():
        raise ValueError(f"REFUSE_OVERWRITE:{output}")
    output.mkdir(parents=True)
    requested = sorted({int(item.strip()) for item in args.gpus.split(",") if item.strip()})[:8]
    if not requested:
        raise ValueError("NO_REQUESTED_GPUS")
    write(output / "SCHEDULER_PROTOCOL.json", {"schema": "STAGE_VI_B2_FORMAL_SCHEDULER_PROTOCOL_V1", "status": "FROZEN", "selection_order": "canonical_parent_manifest_ordinal", "parent_count": 16, "one_project_worker_per_gpu": True, "max_workers": 8, "resource_rule": "memory_free_mib > 20480", "foreign_workload_interference": False, "source_commit": protocol["source_binding"]["runtime_commit"], "source_tree": protocol["source_binding"]["runtime_tree"], "authority_sha256": sha(args.authority.resolve()), "protocol_sha256": sha(args.protocol.resolve()), "outcomes_read": False, "protected_counters": COUNTERS})
    queue = {"schema": "STAGE_VI_B2_FORMAL_PARENT_QUEUE_V1", "status": "RUNNING", "parent_count": 16, "parents": [{"ordinal": row["ordinal"], "canonical_parent_key": row["canonical_parent_key"], "status": "PENDING"} for row in parents], "selection_source": "FROZEN_ZERO_TREATMENT_PLAN_MANIFEST", "outcomes_read": False, "protected_counters": COUNTERS}
    write(output / "FORMAL_PARENT_QUEUE.json", queue)
    write(output / "RESOURCE_PRELAUNCH.json", inventory(requested))
    pending = list(parents)
    active: dict[int, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    hold: dict[str, Any] | None = None
    while pending or active:
        for gpu_id, worker in list(active.items()):
            result = finish_worker(worker)
            if not result["done"]:
                continue
            active.pop(gpu_id)
            completed.append(result)
            for row in queue["parents"]:
                if row["ordinal"] == result["ordinal"]:
                    row.update({"status": result["status"], "gpu": result["gpu"], "finished_utc": result["finished_utc"], "return_code": result["return_code"]})
            write(output / "FORMAL_PARENT_QUEUE.json", queue)
            if result["status"] != "PASS":
                hold = {"schema": "STAGE_VI_B2_FORMAL_GLOBAL_HOLD_V1", "status": "HOLD_SHADOW_OR_FORMAL_PARENT", "reason": "PARENT_STRUCTURAL_AUDIT_NOT_PASS", "failed_parent": result, "outcomes_read": False, "protected_counters": COUNTERS}
                write(output / "GLOBAL_HOLD.json", hold)
                pending.clear()
                break
        if hold:
            break
        if pending:
            snapshot = inventory(requested)
            used = set(active)
            free = [row for row in snapshot["gpu_rows"] if row["gpu"] in snapshot["eligible"] and row["gpu"] not in used]
            while pending and free and len(active) < 8:
                parent = pending.pop(0)
                gpu = free.pop(0)
                launch(args, parent, gpu, output, protocol, authority, plan_manifest, active)
                for row in queue["parents"]:
                    if row["ordinal"] == parent["ordinal"]:
                        row.update({"status": "ACTIVE", "gpu": gpu["gpu"], "started_utc": now()})
                write(output / "FORMAL_PARENT_QUEUE.json", queue)
        if pending or active:
            time.sleep(5)
    if hold:
        queue["status"] = "HOLD"
    else:
        queue["status"] = "PASS"
    queue["completed"] = sorted(completed, key=lambda row: int(row["ordinal"]))
    write(output / "FORMAL_PARENT_QUEUE.json", queue)
    status = "HOLD" if hold else "PASS_STAGE_VI_B2_FORMAL_M4_SCHEDULER"
    write(output / "SCHEDULER_STATUS.json", {"schema": "STAGE_VI_B2_FORMAL_SCHEDULER_STATUS_V1", "status": status, "parent_count": len(completed), "pass_parent_count": sum(row["status"] == "PASS" for row in completed), "planned_parent_count": 16, "outcomes_read": False, "protected_counters": COUNTERS, "finished_utc": now()})
    print(json.dumps({"status": status, "root": str(output), "completed": len(completed), "passed": sum(row["status"] == "PASS" for row in completed)}, sort_keys=True))
    return 0 if not hold and len(completed) == 16 and all(row["status"] == "PASS" for row in completed) else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("protocol", "authority", "plan_root", "runner", "python", "output_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "HOLD_STAGE_VI_B2_FORMAL_SCHEDULER", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
