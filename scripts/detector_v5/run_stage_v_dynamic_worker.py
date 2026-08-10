"""Claim and execute Stage V parents, one at a time, on one physical GPU."""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading
import time
from typing import Any

try:
    from .stage_v_dynamic_common import (
        atomic_write_json, attempt_dir, canonical_parent_key, project_queue, read_json,
        science_artifact_status, sha256_file, sha256_json, load_rows, terminate_process_group, utc_now,
    )
except ImportError:  # direct server execution
    from stage_v_dynamic_common import (
        atomic_write_json, attempt_dir, canonical_parent_key, project_queue, read_json,
        science_artifact_status, sha256_file, sha256_json, load_rows, terminate_process_group, utc_now,
    )

try:
    from scripts.fec.atomic_task_queue import AtomicTaskQueue
except ModuleNotFoundError:  # direct server execution from scripts/detector_v5
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.fec.atomic_task_queue import AtomicTaskQueue

try:
    from .stage_v_gpu_resource_contract import (
        GpuLeaseStore, MODE_M35, ResourceContractError, admit_mode_b_or_c, query_inventory,
        verify_recheck, write_resource_receipt,
    )
except ImportError:  # direct server execution
    from stage_v_gpu_resource_contract import (
        GpuLeaseStore, MODE_M35, ResourceContractError, admit_mode_b_or_c, query_inventory,
        verify_recheck, write_resource_receipt,
    )


def _gpu_row(gpu_id: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=utilization.gpu,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        fields = [item.strip() for item in completed.stdout.strip().split(",")]
        return {
            "gpu_utilization_percent": float(fields[0]) if fields and fields[0] else None,
            "gpu_memory_used_mib": float(fields[1]) if len(fields) > 1 and fields[1] else None,
            "gpu_memory_free_mib": float(fields[2]) if len(fields) > 2 and fields[2] else None,
        }
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"gpu_utilization_percent": None, "gpu_memory_used_mib": None, "gpu_memory_free_mib": None}


def _proc_cpu_seconds(pid: int | None) -> float | None:
    if not pid or os.name != "posix":
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        ticks = int(os.sysconf("SC_CLK_TCK"))
        return (int(fields[11]) + int(fields[12])) / ticks
    except (OSError, IndexError, ValueError):
        return None


class Worker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.run_root.resolve()
        self.worker_root = self.root / f"worker_gpu{args.gpu_id}"
        self.worker_root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.worker_root / "WORKER_STATUS.json"
        self.heartbeat_path = self.worker_root / "WORKER_HEARTBEAT.json"
        self.queue = AtomicTaskQueue(str(args.queue_db), run_id=args.run_id)
        self.manifest_rows = {
            canonical_parent_key(row): row for row in load_rows(args.parent_manifest)
        }
        self.current: dict[str, Any] | None = None
        self.child: subprocess.Popen[Any] | None = None
        self.child_pgid: int | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.progress_lock = threading.Lock()
        self.sequence = 0
        self.last_progress_epoch = time.time()
        self.last_artifact_epoch = time.time()
        self.last_cpu_progress_epoch = time.time()
        self.last_simulator_progress_epoch = self.last_progress_epoch
        self.last_branch_progress_epoch = self.last_progress_epoch
        self.last_simulator_step: int | None = None
        self.last_branch_progress: int | None = None
        self.current_branch: str | None = None
        self.branch_started_epoch: float | None = None
        self.last_child_cpu_seconds = 0.0
        self.parent_started_epoch: float | None = None
        self.resource_lease: dict[str, Any] | None = None
        self.resource_store: GpuLeaseStore | None = None
        if self.args.resource_mode in {"MODE_B_THROUGHPUT_SCIENCE", "MODE_C_TRAINING", MODE_M35}:
            self.resource_store = GpuLeaseStore(self.args.lease_db or (self.root / "GPU_LEASES.sqlite"))

    @property
    def resource_enabled(self) -> bool:
        return self.args.resource_mode in {"MODE_B_THROUGHPUT_SCIENCE", "MODE_C_TRAINING", MODE_M35}

    def _resource_row(self) -> dict[str, Any]:
        inventory, error = query_inventory()
        if error:
            raise ResourceContractError(error)
        for row in inventory:
            if int(row.get("gpu_id", -1)) == self.args.gpu_id:
                return row
        raise ResourceContractError(f"GPU_NOT_IN_INVENTORY:{self.args.gpu_id}")

    def _acquire_resource_lease(self) -> None:
        if not self.resource_enabled:
            return
        assert self.resource_store is not None
        inventory, error = query_inventory()
        if error:
            raise ResourceContractError(error)
        active = self.resource_store.active()
        admission = admit_mode_b_or_c(
            inventory,
            mode=self.args.resource_mode,
            leased_gpu_ids=[row["gpu_id"] for row in active],
            project_pids=[row["worker_pid"] for row in active],
            project_process_tokens=(str(self.root), "run_stage_v_dynamic_worker.py"),
            minimum_free_mib=self.args.minimum_free_mib,
        )
        atomic_write_json(self.worker_root / "RESOURCE_ADMISSION.json", admission)
        decision = next((row for row in admission["gpu_decisions"] if int(row["gpu_id"]) == self.args.gpu_id), None)
        if not decision or not decision["safe"]:
            raise ResourceContractError(f"GPU_NOT_ELIGIBLE:{self.args.gpu_id}:{decision and decision['reasons']}")
        lease = self.resource_store.acquire(
            gpu_id=self.args.gpu_id,
            gpu_uuid=decision["gpu_uuid"],
            worker_id=self.args.worker_id,
            worker_pid=os.getpid(),
            stage=self.args.stage,
            atomic_job_id=self.args.run_id,
            source_commit=self.args.source_commit,
            source_tree=self.args.source_tree,
            runtime_root=self.root,
            launch_snapshot=decision,
        )
        try:
            rechecked = self._resource_row()
            verify_recheck(rechecked, expected_gpu_id=self.args.gpu_id,
                           expected_gpu_uuid=decision["gpu_uuid"],
                           minimum_free_mib=self.args.minimum_free_mib)
        except Exception:
            self.resource_store.release(lease, reason="RECHECK_FAILED")
            raise
        self.resource_lease = lease
        write_resource_receipt(self.worker_root / "RESOURCE_LEASE.json", phase="LEASE_ACQUIRED",
                               gpu_snapshot=rechecked, lease=lease, atomic_job_id=self.args.run_id)

    def _write_job_resource_receipt(self, output_dir: Path, phase: str, atomic_job_id: str) -> None:
        if not self.resource_enabled or self.resource_lease is None:
            return
        snapshot = self._resource_row()
        write_resource_receipt(output_dir / f"RESOURCE_{phase}.json", phase=phase,
                               gpu_snapshot=snapshot, lease=self.resource_lease,
                               atomic_job_id=atomic_job_id)

    def _status(self, state: str, *, child_pid: int | None = None, error: str | None = None) -> dict[str, Any]:
        with self.progress_lock:
            parent = self.current or {}
            progress = self._progress_snapshot(parent.get("output_dir")) if parent else {}
            if parent:
                branch = progress.get("current_branch")
                if branch != self.current_branch:
                    self.current_branch = str(branch) if branch else None
                    self.branch_started_epoch = time.time() if self.current_branch else None
            gpu = _gpu_row(self.args.gpu_id)
            payload = {
                "schema": "STAGE_V_DYNAMIC_WORKER_STATUS_V2",
                "worker_id": self.args.worker_id,
                "worker_pid": os.getpid(),
                "worker_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
                "gpu_id": self.args.gpu_id,
                "state": state,
                "current_parent": parent.get("canonical_parent_key"),
                "current_output_dir": parent.get("output_dir"),
                "current_branch": progress.get("current_branch"),
                "simulator_step": progress.get("simulator_step", 0),
                "branch_progress": progress.get("branch_progress", 0),
                "last_progress_utc": progress.get("last_progress_utc") or utc_now(),
                "last_artifact_utc": progress.get("last_artifact_utc"),
                "last_progress_epoch": self.last_progress_epoch,
                "last_artifact_epoch": self.last_artifact_epoch,
                "last_simulator_progress_epoch": self.last_simulator_progress_epoch,
                "last_branch_progress_epoch": self.last_branch_progress_epoch,
                "branch_started_epoch": self.branch_started_epoch,
                "parent_started_epoch": self.parent_started_epoch,
                "child_pid": child_pid,
                "child_pgid": self.child_pgid,
                "child_cpu_seconds": progress.get("child_cpu_seconds"),
                "last_cpu_progress_epoch": self.last_cpu_progress_epoch,
                "gpu_utilization_percent": gpu.get("gpu_utilization_percent"),
                "gpu_memory_used_mib": gpu.get("gpu_memory_used_mib"),
                "gpu_memory_free_mib": gpu.get("gpu_memory_free_mib"),
                "heartbeat_sequence": self.sequence,
                "error": error,
                "updated_utc": utc_now(),
            }
            atomic_write_json(self.status_path, payload)
            atomic_write_json(self.heartbeat_path, payload)
            return payload

    def _progress_snapshot(self, output_dir: str | None) -> dict[str, Any]:
        if not output_dir:
            return {}
        root = Path(output_dir)
        files = list(root.rglob("*")) if root.exists() else []
        branch_files = [path for path in files if path.name == "COUNTERFACTUAL_BRANCHES.jsonl"]
        branch_progress = 0
        newest = None
        for path in branch_files:
            try:
                branch_progress += sum(1 for _ in path.open("r", encoding="utf-8"))
                newest = max(newest or 0.0, path.stat().st_mtime)
            except OSError:
                pass
        progress_file = root / "PROGRESS.json"
        progress = read_json(progress_file, {}) if progress_file.is_file() else {}
        if not isinstance(progress, dict):
            progress = {}
        now = time.time()
        simulator_step = int(progress.get("simulator_step", 0) or 0)
        branch_progress = int(progress.get("branch_progress", 0) or 0)
        if self.last_simulator_step is None:
            self.last_simulator_step = simulator_step
        elif simulator_step != self.last_simulator_step:
            self.last_simulator_step = simulator_step
            self.last_simulator_progress_epoch = now
        if self.last_branch_progress is None:
            self.last_branch_progress = branch_progress
        elif branch_progress != self.last_branch_progress:
            self.last_branch_progress = branch_progress
            self.last_branch_progress_epoch = now
        child_cpu_seconds = _proc_cpu_seconds(self.child.pid if self.child else None)
        if child_cpu_seconds is not None:
            if child_cpu_seconds > self.last_child_cpu_seconds:
                self.last_cpu_progress_epoch = now
            self.last_child_cpu_seconds = max(self.last_child_cpu_seconds, child_cpu_seconds)
        progress_updated = progress.get("updated_epoch")
        if progress_updated is None and progress.get("updated_utc"):
            try:
                progress_updated = _datetime.datetime.fromisoformat(str(progress["updated_utc"]).replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError):
                progress_updated = None
        if isinstance(progress_updated, (int, float)) and float(progress_updated) > self.last_progress_epoch:
            self.last_progress_epoch = float(progress_updated)
        if newest and newest > self.last_artifact_epoch:
            self.last_artifact_epoch = newest
            self.last_progress_epoch = newest
        self.last_progress_epoch = max(
            self.last_progress_epoch, self.last_simulator_progress_epoch,
            self.last_branch_progress_epoch,
        )
        return {
            "branch_progress": branch_progress,
            "simulator_step": simulator_step,
            "current_branch": progress.get("current_branch"),
            "last_progress_utc": progress.get("updated_utc"),
            "last_artifact_utc": progress.get("last_artifact_utc"),
            "child_cpu_seconds": child_cpu_seconds,
        }

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(self.args.heartbeat_seconds):
            self.sequence += 1
            self._status("RUNNING", child_pid=self.child.pid if self.child else None)
            if self.current:
                task = self.current
                self.queue.heartbeat(
                    task["cell_id"], task["attempt_id"], self.args.worker_id,
                    task["lease_token"], task["lease_epoch"],
                )
                project_queue(self.root, self.queue.list_tasks())

    def _write_exit(self, exit_code: int, reason: str) -> int:
        atomic_write_json(self.worker_root / "WORKER_EXIT.json", {
            "schema": "STAGE_V_DYNAMIC_WORKER_EXIT_V1",
            "worker_id": self.args.worker_id,
            "worker_pid": os.getpid(),
            "gpu_id": self.args.gpu_id,
            "exit_code": exit_code,
            "reason": reason,
            "updated_utc": utc_now(),
        })
        return exit_code

    def _command(self, task: dict[str, Any], output_dir: Path) -> list[str]:
        if self.args.m35_launcher:
            required = (
                self.args.m35_runner, self.args.m35_protocol, self.args.m35_authorization_receipt,
                self.args.m35_official_snapshot_root, self.args.m35_upstream_root,
                self.args.m35_model_root, self.args.m35_source_commit, self.args.m35_source_tree,
            )
            if any(value is None for value in required):
                raise RuntimeError("M35_LAUNCHER_CONFIGURATION_INCOMPLETE")
            return [
                sys.executable, str(self.args.m35_launcher),
                "--runner", str(self.args.m35_runner), "--protocol", str(self.args.m35_protocol),
                "--authorization-receipt", str(self.args.m35_authorization_receipt),
                "--official-snapshot-root", str(self.args.m35_official_snapshot_root),
                "--upstream-root", str(self.args.m35_upstream_root),
                "--model-root", str(self.args.m35_model_root),
                "--parent-key", task["canonical_parent_key"], "--output-dir", str(output_dir),
                "--gpu", str(self.args.gpu_id), "--source-commit", str(self.args.m35_source_commit),
                "--source-tree", str(self.args.m35_source_tree),
            ]
        if self.args.worker_command:
            text = self.args.worker_command.format(
                parent_key=task["canonical_parent_key"], output_dir=str(output_dir),
                gpu_id=self.args.gpu_id, attempt=task["attempt_count"],
            )
            tokens = shlex.split(text, posix=(os.name != "nt"))
            if os.name == "nt":
                tokens = [token.replace("\\\\", "\\") for token in tokens]
            return tokens
        if not self.args.science_runner or not self.args.science_parent_manifest or not self.args.science_repo_root:
            raise RuntimeError("science runner configuration is incomplete")
        return [
            sys.executable, str(self.args.science_runner),
            "--gpu-id", str(self.args.gpu_id),
            "--repo-root", str(self.args.science_repo_root),
            "--parent-manifest", str(self.args.science_parent_manifest),
            "--output-root", str(output_dir),
            "--parent-keys", task["canonical_parent_key"],
            "--probe-limit", str(self.args.probe_limit),
        ]

    def _run_task(self, task: dict[str, Any]) -> bool:
        self.current = task
        self.parent_started_epoch = time.time()
        self.last_progress_epoch = self.parent_started_epoch
        self.last_artifact_epoch = self.parent_started_epoch
        self.last_cpu_progress_epoch = self.parent_started_epoch
        self.last_simulator_progress_epoch = self.parent_started_epoch
        self.last_branch_progress_epoch = self.parent_started_epoch
        self.last_simulator_step = None
        self.last_branch_progress = None
        self.current_branch = None
        self.branch_started_epoch = None
        output_dir = attempt_dir(self.root, task["canonical_parent_key"], int(task["attempt_count"]))
        output_dir.mkdir(parents=True, exist_ok=False)
        log_path = output_dir / "SCIENCE_RUNNER.log"
        job = {
            "schema": "STAGE_V_DYNAMIC_JOB_V2",
            "canonical_parent_key": task["canonical_parent_key"],
            "manifest_row_sha256": task.get("manifest_row_sha256"),
            "attempt": task["attempt_count"],
            "worker_pid": os.getpid(),
            "worker_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "gpu_id": self.args.gpu_id,
            "claim_utc": utc_now(),
            "start_utc": utc_now(),
            "output_dir": str(output_dir),
            "state": "RUNNING",
        }
        job["manifest_row_sha256"] = sha256_json(self.manifest_rows.get(task["canonical_parent_key"], {"canonical_parent_key": task["canonical_parent_key"]}))
        atomic_write_json(output_dir / "JOB.json", job)
        task["output_dir"] = str(output_dir)
        self._status("RUNNING")
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(self.args.gpu_id),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        code = 127
        resource_error: str | None = None
        try:
            self._write_job_resource_receipt(output_dir, "PRE", task["attempt_id"])
            command = self._command(task, output_dir)
            with log_path.open("w", encoding="utf-8") as log:
                self.child = subprocess.Popen(command, cwd=str(self.args.repo_root), env=environment,
                                              stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=(os.name == "posix"))
                job["child_pid"] = self.child.pid
                self.child_pgid = os.getpgid(self.child.pid) if hasattr(os, "getpgid") else self.child.pid
                job["child_pgid"] = self.child_pgid
                atomic_write_json(output_dir / "JOB.json", job)
                self._status("RUNNING", child_pid=self.child.pid)
                code = self.child.wait()
            self.child = None
            self.child_pgid = None
        except Exception as exc:
            if isinstance(exc, ResourceContractError):
                resource_error = str(exc)
            atomic_write_json(output_dir / "WORKER_EXCEPTION.json", {"error": f"{type(exc).__name__}: {exc}", "utc": utc_now()})
        finally:
            if self.resource_enabled:
                try:
                    self._write_job_resource_receipt(output_dir, "POST", task["attempt_id"])
                except Exception as exc:
                    resource_error = resource_error or f"RESOURCE_POST_RECEIPT:{type(exc).__name__}:{exc}"
        resource_receipts = all((output_dir / name).is_file() for name in ("RESOURCE_PRE.json", "RESOURCE_POST.json")) if self.resource_enabled else True
        if self.resource_enabled and not resource_receipts:
            resource_error = resource_error or "RESOURCE_RECEIPT_MISSING"
        artifact = science_artifact_status(
            output_dir, task["canonical_parent_key"],
            expected_source_commit=self.args.science_source_commit or None,
            expected_source_tree=self.args.science_source_tree or None,
            expected_row=self.manifest_rows.get(task["canonical_parent_key"])
            if (self.args.science_source_commit or self.args.science_source_tree) else None,
            artifact_schema=self.args.artifact_schema,
        )
        validation = {
            "schema": "STAGE_V_PARENT_VALIDATION_V2",
            "canonical_parent_key": task["canonical_parent_key"],
            "source_commit": self.args.source_commit,
            "source_tree": self.args.source_tree,
            "science_source_commit": self.args.science_source_commit,
            "science_source_tree": self.args.science_source_tree,
            "artifact_schema": self.args.artifact_schema,
            "science_provenance": str(self.args.science_provenance) if self.args.science_provenance else None,
            "exit_code": code,
            "artifact_audit_verdict": "PASS" if artifact["valid"] and code == 0 else "FAIL",
            "label_status": artifact.get("label_status") if artifact["valid"] and code == 0 else "INVALID",
            "artifact_path": artifact.get("path"),
            "artifact_sha256": artifact.get("artifact_sha256"),
            "reason": artifact.get("reason"),
            "resource_contract_mode": self.args.resource_mode,
            "resource_receipts": resource_receipts,
            "resource_error": resource_error,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "vis_pgd_attack_rollouts": 0,
            "validated_utc": utc_now(),
        }
        atomic_write_json(output_dir / "PARENT_VALIDATION.json", validation)
        if resource_error:
            outcome = "HOLD_RESOURCE_CONTRACT"
            error_class = resource_error
        elif code == 0 and artifact["valid"]:
            outcome = "DONE_VALID"
            error_class = None
        elif code != 0 and not artifact["result"] and int(task["attempt_count"]) < self.args.max_attempts:
            outcome = "FAILED_RETRYABLE_INFRA"
            error_class = "PRE_SIMULATOR_OR_TRANSIENT_EXIT"
        else:
            outcome = "FAILED_FATAL_POST_ACTION"
            error_class = "INVALID_OR_PARTIAL_SCIENCE_RESULT"
        receipt_sha = sha256_file(output_dir / "PARENT_VALIDATION.json")
        committed = self.queue.commit_result(
            task["cell_id"], task["attempt_id"], self.args.worker_id,
            task["lease_token"], task["lease_epoch"], exit_code=code,
            error_class=error_class, exposure_status="DIRECT_OPEN_COUNTERFACTUAL_ONLY",
            task_outcome=outcome, output_dir=str(output_dir), receipt_sha=receipt_sha,
        )
        if not committed:
            outcome = "FAILED_FATAL_POST_ACTION"
        self.current = None
        self.parent_started_epoch = None
        self.current_branch = None
        self.branch_started_epoch = None
        self._status("IDLE")
        job.update({
            "state": outcome,
            "exit_code": code,
            "complete_utc": utc_now(),
            "artifact_sha256": validation.get("artifact_sha256"),
            "validation": validation,
        })
        atomic_write_json(output_dir / "JOB.json", job)
        project_queue(self.root, self.queue.list_tasks())
        return outcome in {"DONE_VALID", "FAILED_RETRYABLE_INFRA"}

    def run(self) -> int:
        try:
            self._acquire_resource_lease()
        except ResourceContractError as exc:
            self._status("HOLD", error=str(exc))
            return self._write_exit(1, f"RESOURCE_CONTRACT_HOLD:{exc}")
        self._status("STARTING")
        self.thread = threading.Thread(target=self._heartbeat_loop, name="stage-v-heartbeat", daemon=True)
        self.thread.start()
        try:
            while not self.stop_event.is_set():
                task = self.queue.claim_task(
                    self.args.worker_id, hostname=socket.gethostname(), pid=os.getpid(),
                    gpu_id=self.args.gpu_id, expected_manifest_sha=self.args.manifest_sha,
                    expected_source_sha=f"{self.args.source_commit}:{self.args.source_tree}",
                )
                project_queue(self.root, self.queue.list_tasks())
                if task is None:
                    self._status("IDLE")
                    fatal = any(item["state"] in {"FAILED_FATAL_POST_ACTION", "HOLD"} for item in self.queue.list_tasks())
                    return self._write_exit(1 if fatal else 0, "FATAL_QUEUE_STATE" if fatal else "QUEUE_DRAINED")
                task["canonical_parent_key"] = task["parent_id"]
                if not self._run_task(task):
                    self._status("FAILED", error="FATAL_PARENT_RESULT")
                    return self._write_exit(1, "FATAL_PARENT_RESULT")
            return self._write_exit(1, "STOP_REQUESTED")
        finally:
            self.stop_event.set()
            if self.child is not None and self.child.poll() is None:
                terminate_process_group(self.child, grace_seconds=10)
            if self.thread:
                self.thread.join(timeout=max(1.0, self.args.heartbeat_seconds + 1))
            self._status("STOPPED" if self.stop_event.is_set() else "EXITED", child_pid=None)
            if self.resource_lease is not None and self.resource_store is not None:
                self.resource_store.release(self.resource_lease)
            self.queue.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--queue-db", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=30)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--probe-limit", type=int, default=24)
    parser.add_argument("--science-runner", type=Path)
    parser.add_argument("--science-repo-root", type=Path)
    parser.add_argument("--science-parent-manifest", type=Path)
    parser.add_argument("--science-source-commit", default="")
    parser.add_argument("--science-source-tree", default="")
    parser.add_argument("--science-provenance", type=Path)
    parser.add_argument("--worker-command", default="")
    parser.add_argument("--resource-mode", default="LEGACY")
    parser.add_argument("--lease-db", type=Path)
    parser.add_argument("--stage", default="STAGE_V")
    parser.add_argument("--minimum-free-mib", type=int, default=20_480)
    parser.add_argument("--artifact-schema", default="STAGE_V_PARENT_RESULT_V2")
    parser.add_argument("--m35-launcher", type=Path)
    parser.add_argument("--m35-runner", type=Path)
    parser.add_argument("--m35-protocol", type=Path)
    parser.add_argument("--m35-authorization-receipt", type=Path)
    parser.add_argument("--m35-official-snapshot-root", type=Path)
    parser.add_argument("--m35-upstream-root", type=Path)
    parser.add_argument("--m35-model-root", type=Path)
    parser.add_argument("--m35-source-commit")
    parser.add_argument("--m35-source-tree")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.heartbeat_seconds <= 0 or args.max_attempts < 1:
        raise SystemExit("invalid heartbeat/max-attempts")
    return Worker(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
