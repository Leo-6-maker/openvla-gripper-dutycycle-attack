"""Run the Stage V dynamic queue with one worker per approved GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

try:
    from .stage_v_dynamic_common import atomic_write_json, gpu_preflight, load_rows, normalize_parent, project_queue, sha256_file, terminate_process_group, utc_now
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, gpu_preflight, load_rows, normalize_parent, project_queue, sha256_file, terminate_process_group, utc_now

try:
    from scripts.fec.atomic_task_queue import AtomicTaskQueue
except ModuleNotFoundError:  # direct server execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.fec.atomic_task_queue import AtomicTaskQueue


class Dispatcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.run_root.resolve()
        self.processes: list[subprocess.Popen[Any]] = []
        self.stop_requested = False
        self.run_id = args.run_id
        self.queue = AtomicTaskQueue(str(args.queue_db), run_id=self.run_id)

    def _signal(self, *_args: Any) -> None:
        self.stop_requested = True
        for process in self.processes:
            if process.poll() is None:
                terminate_process_group(process, grace_seconds=5)

    def _preflight(self) -> dict[str, Any]:
        if self.args.preflight_file:
            value = json.loads(self.args.preflight_file.read_text(encoding="utf-8"))
            safe = sorted({int(gpu) for gpu in value.get("safe_gpus", []) if int(gpu) not in self.args.excluded_gpus})
            if value.get("status") != "PASS":
                return value
            if len(safe) < self.args.required_workers:
                value = dict(value)
                value["status"] = "PRELAUNCH_WAITING_FOR_8_GPUS"
            else:
                value = dict(value)
                value["all_safe_gpus"] = safe
                value["safe_gpus"] = safe[:self.args.required_workers]
                value["safe_gpu_count"] = len(safe)
                value["selected_gpu_count"] = len(value["safe_gpus"])
            return value
        return gpu_preflight(
            required_count=self.args.required_workers,
            excluded_gpus=self.args.excluded_gpus,
            canary_peak_mib=self.args.canary_peak_mib,
            protected_pids=self.args.protected_pids,
            gpu_query_command=self.args.gpu_query_command,
        )

    def _prepare(self) -> None:
        preflight = self._preflight()
        atomic_write_json(self.args.preflight_output, preflight)
        if preflight.get("status") != "PASS":
            raise RuntimeError("PRELAUNCH_WAITING_FOR_8_GPUS")
        approved = sorted(int(gpu) for gpu in preflight.get("safe_gpus", []))
        if len(approved) != self.args.required_workers or any(gpu in self.args.excluded_gpus for gpu in approved):
            raise RuntimeError("GPU_PREFLIGHT_POLICY_FAIL")
        rows = [normalize_parent(row) for row in load_rows(self.args.parent_manifest)]
        if len(rows) != self.args.expected_parent_count:
            raise RuntimeError(f"PARENT_MANIFEST_COUNT:{len(rows)}/{self.args.expected_parent_count}")
        keys = [row["canonical_parent_key"] for row in rows]
        if len(set(keys)) != len(keys):
            raise RuntimeError("DUPLICATE_PARENT_KEYS")
        science_manifest_sha = None
        if self.args.science_runner and not self.args.science_parent_manifest:
            raise RuntimeError("SCIENCE_PARENT_MANIFEST_MISSING")
        if self.args.science_parent_manifest:
            science_path = self.args.science_parent_manifest.resolve()
            try:
                science_value = json.loads(science_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"SCIENCE_PARENT_MANIFEST_INVALID:{exc}") from exc
            science_rows = science_value.get("selected_parents") if isinstance(science_value, dict) else None
            science_rows_are_objects = isinstance(science_rows, list) and all(isinstance(row, dict) for row in science_rows)
            science_keys = [str(row.get("canonical_parent_key")) for row in science_rows] if science_rows_are_objects else []
            if (
                not isinstance(science_value, dict)
                or science_value.get("schema") != "STAGE_V_FORMAL_PARENT_MANIFEST_V1"
                or science_value.get("status") != "FROZEN"
                or not science_rows_are_objects
                or len(science_rows or []) != self.args.expected_parent_count
                or len(set(science_keys)) != len(science_keys)
                or any(row.get("old_artifacts_reused") is not False or row.get("source_artifact_read") is not False for row in (science_rows or []) if isinstance(row, dict))
                or science_value.get("old_artifacts_reused") is not False
                or science_value.get("source_artifacts_modified") is not False
            ):
                raise RuntimeError("SCIENCE_PARENT_MANIFEST_BINDING_FAIL")
            if set(science_keys) != set(keys):
                raise RuntimeError("SCIENCE_PARENT_MANIFEST_PARENT_SET_FAIL")
            science_manifest_sha = sha256_file(science_path)
        if self.root.exists() and any(self.root.iterdir()):
            if any((self.root / name).exists() for name in ("DISPATCHER_COMPLETE.json", "ABORTED_INCOMPLETE.json")):
                raise RuntimeError("COMPLETED_OR_ABORTED_ROOT_REUSE")
        self.root.mkdir(parents=True, exist_ok=True)
        self.args.queue_db.parent.mkdir(parents=True, exist_ok=True)
        manifest_sha = sha256_file(self.args.parent_manifest)
        source_sha = f"{self.args.source_commit}:{self.args.source_tree}"
        self.queue.init_run(
            state="ACTIVE", manifest_sha=manifest_sha, source_sha=source_sha,
            config_sha=self.args.config_sha256,
            capacity_policy={"required_workers": self.args.required_workers, "approved_gpus": approved, "gpu5_excluded": 5 in self.args.excluded_gpus},
        )
        self.queue.register_tasks([
            {
                "cell_id": row["canonical_parent_key"], "parent_id": row["canonical_parent_key"],
                "suite": row["suite"], "task_index": row["task_index"], "state_index": row["state_index"],
                "arm": "PARENT", "task_kind": "STAGE_V_PARENT", "priority": index,
            }
            for index, row in enumerate(rows)
        ])
        project_queue(self.root, self.queue.list_tasks())
        atomic_write_json(self.root / "RUN_MANIFEST.json", {
            "schema": "STAGE_V_R2_DYNAMIC_RUN_MANIFEST_V2",
            "run_id": self.run_id,
            "source_commit": self.args.source_commit,
            "source_tree": self.args.source_tree,
            "science_source_commit": self.args.science_source_commit,
            "science_source_tree": self.args.science_source_tree,
            "science_provenance": str(self.args.science_provenance) if self.args.science_provenance else None,
            "science_parent_manifest": str(self.args.science_parent_manifest.resolve()) if self.args.science_parent_manifest else None,
            "science_parent_manifest_sha256": science_manifest_sha,
            "parent_manifest": str(self.args.parent_manifest),
            "parent_manifest_sha256": manifest_sha,
            "planned_parents": len(rows),
            "approved_gpus": approved,
            "gpu5_used": 5 in approved,
            "workers": len(approved),
            "dispatcher_pid": os.getpid(),
            "dispatcher_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "dynamic_claims": True,
            "one_project_worker_per_gpu": True,
            "old_artifacts_reused": False,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "vis_pgd_attack_rollouts": 0,
            "created_utc": utc_now(),
        })
        atomic_write_json(self.root / "DISPATCHER_START.json", {
            "schema": "STAGE_V_DYNAMIC_DISPATCHER_START_V2",
            "dispatcher_pid": os.getpid(),
            "dispatcher_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "run_id": self.run_id,
            "approved_gpus": approved,
            "planned_parents": len(rows),
            "started_utc": utc_now(),
        })

    def _spawn(self, gpu_id: int) -> None:
        worker = Path(__file__).with_name("run_stage_v_dynamic_worker.py")
        command = [
            sys.executable, str(worker),
            "--run-root", str(self.root), "--repo-root", str(self.args.repo_root),
            "--queue-db", str(self.args.queue_db), "--run-id", self.run_id,
            "--manifest-sha", sha256_file(self.args.parent_manifest),
            "--parent-manifest", str(self.args.parent_manifest),
            "--source-commit", self.args.source_commit, "--source-tree", self.args.source_tree,
            "--gpu-id", str(gpu_id), "--worker-id", f"stage-v-r2-gpu{gpu_id}",
            "--heartbeat-seconds", str(self.args.worker_heartbeat_seconds),
            "--max-attempts", str(self.args.max_attempts), "--probe-limit", str(self.args.probe_limit),
            "--science-source-commit", self.args.science_source_commit,
            "--science-source-tree", self.args.science_source_tree,
        ]
        if self.args.science_provenance:
            command += ["--science-provenance", str(self.args.science_provenance)]
        if self.args.science_runner:
            command += ["--science-runner", str(self.args.science_runner)]
        if self.args.science_repo_root:
            command += ["--science-repo-root", str(self.args.science_repo_root)]
        if self.args.science_parent_manifest:
            command += ["--science-parent-manifest", str(self.args.science_parent_manifest)]
        if self.args.worker_command:
            command += ["--worker-command", self.args.worker_command]
        log = (self.root / f"worker_gpu{gpu_id}.log").open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(command, cwd=str(self.args.repo_root), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
        self.processes.append(process)
        atomic_write_json(self.root / f"worker_gpu{gpu_id}.pid.json", {"pid": process.pid, "gpu_id": gpu_id, "started_utc": utc_now()})

    def run(self) -> int:
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._signal)
            signal.signal(signal.SIGINT, self._signal)
        try:
            self._prepare()
            preflight = json.loads(self.args.preflight_output.read_text(encoding="utf-8"))
            for gpu_id in sorted(preflight["safe_gpus"]):
                self._spawn(int(gpu_id))
            while True:
                if self.stop_requested:
                    raise RuntimeError("DISPATCHER_STOP_REQUESTED")
                alive = 0
                for process in self.processes:
                    code = process.poll()
                    if code is None:
                        alive += 1
                    elif code != 0:
                        raise RuntimeError(f"WORKER_EXIT:{process.pid}:{code}")
                project_queue(self.root, self.queue.list_tasks())
                tasks = self.queue.list_tasks()
                if alive == 0:
                    fatal = [task for task in tasks if task["state"] not in {"DONE_VALID", "DONE", "DONE_CLASSIFIED_TC"}]
                    if fatal:
                        raise RuntimeError("QUEUE_NOT_COMPLETE")
                    atomic_write_json(self.root / "QUEUE_STATE.json", {"schema": "STAGE_V_QUEUE_STATE_V2", "tasks": tasks, "updated_utc": utc_now()})
                    atomic_write_json(self.root / "DISPATCHER_COMPLETE.json", {
                        "schema": "STAGE_V_DYNAMIC_DISPATCHER_COMPLETE_V2", "status": "PASS",
                        "dispatcher_pid": os.getpid(), "run_id": self.run_id,
                        "planned_parents": len(tasks), "completed_parents": sum(task["state"] == "DONE_VALID" for task in tasks),
                        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
                        "completed_utc": utc_now(),
                    })
                    return 0
                time.sleep(self.args.poll_seconds)
        except Exception as exc:
            for process in self.processes:
                if process.poll() is None:
                    terminate_process_group(process, grace_seconds=5)
            atomic_write_json(self.root / "DISPATCHER_FAILURE.json", {
                "schema": "STAGE_V_DYNAMIC_DISPATCHER_FAILURE_V2", "status": "FAIL",
                "reason": f"{type(exc).__name__}:{exc}", "dispatcher_pid": os.getpid(), "updated_utc": utc_now(),
            })
            return 1
        finally:
            for process in self.processes:
                if process.poll() is None:
                    terminate_process_group(process, grace_seconds=5)
                else:
                    process.wait()
            self.queue.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--queue-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--required-workers", type=int, default=8)
    parser.add_argument("--excluded-gpus", type=lambda value: [int(item) for item in value.split(",") if item], default=[5])
    parser.add_argument("--protected-pids", type=lambda value: [int(item) for item in value.split(",") if item], default=[])
    parser.add_argument("--canary-peak-mib", type=float, default=0.0)
    parser.add_argument("--gpu-query-command", default="nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits")
    parser.add_argument("--preflight-file", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    parser.add_argument("--science-runner", type=Path)
    parser.add_argument("--science-repo-root", type=Path)
    parser.add_argument("--science-parent-manifest", type=Path)
    parser.add_argument("--science-provenance", type=Path)
    parser.add_argument("--science-source-commit", default="")
    parser.add_argument("--science-source-tree", default="")
    parser.add_argument("--worker-command", default="")
    parser.add_argument("--probe-limit", type=int, default=24)
    parser.add_argument("--worker-heartbeat-seconds", type=float, default=30)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--config-sha256", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.required_workers != 8:
        raise SystemExit("Stage V R2 requires exactly 8 workers")
    return Dispatcher(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
