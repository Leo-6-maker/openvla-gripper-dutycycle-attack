"""Server-local receipt controller for the Stage V R2 pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping

try:
    from .materialize_stage_v_r2_next_plan import append_registry, build_c0_plan
    from .run_stage_v_r2_mainline_orchestrator import FileLock, OrchestratorError, pid_alive, source_binding, verify_registry_chain
    from ..detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now
except ImportError:  # direct server execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.monitoring.materialize_stage_v_r2_next_plan import append_registry, build_c0_plan
    from scripts.monitoring.run_stage_v_r2_mainline_orchestrator import FileLock, OrchestratorError, pid_alive, source_binding, verify_registry_chain
    from scripts.detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now


WAIT_Q = "WAIT_QUALIFICATION"
WAIT_INPUT = "WAIT_NEXT_STAGE_INPUT"
HARD_STOP = "HARD_STOP"


class ControllerError(RuntimeError):
    pass


def _q_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"states": {}, "events": 0}
    try:
        con = sqlite3.connect(path)
        states = {str(state): int(count) for state, count in con.execute("select state,count(1) from tasks group by state")}
        events = int(con.execute("select count(1) from events").fetchone()[0])
        con.close()
        return {"states": states, "events": events}
    except sqlite3.Error as exc:
        return {"states": {}, "events": 0, "error": f"{type(exc).__name__}:{exc}"}


class PlanController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.state_root = Path(args.state_root).resolve()
        self.qualification_root = Path(args.qualification_root).resolve()
        self.candidate_manifest = Path(args.candidate_manifest).resolve()
        self.science_provenance = Path(args.science_provenance).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_root / "STAGE_V_R2_PLAN_CONTROLLER_STATE.json"
        self.heartbeat_path = self.state_root / "STAGE_V_R2_PLAN_CONTROLLER_HEARTBEAT.json"
        self.events_path = self.state_root / "STAGE_V_R2_PLAN_CONTROLLER_EVENTS.jsonl"
        self.count = 0
        self.source = source_binding(self.repo_root)
        self.lock = FileLock(Path(args.lock_path), self.state_root, {
            "schema": "STAGE_V_R2_PLAN_CONTROLLER_LOCK_V1", "pid": os.getpid(),
            "pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(), "started_utc": utc_now(),
        })

    def _q_status(self) -> tuple[bool, str]:
        q2_report_path = self.qualification_root / "Q2_CONTROL_QUALIFICATION_REPORT.json"
        if q2_report_path.is_file():
            audit_path = self.qualification_root / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json"
            manifest_path = self.qualification_root / "Q2_PARENT_MANIFEST_A.json"
            if not audit_path.is_file() or not manifest_path.is_file():
                return False, "Q2_QUALIFICATION_INCOMPLETE"
            report = read_json(q2_report_path, {})
            audit = read_json(audit_path, {})
            manifest = read_json(manifest_path, {})
            if not isinstance(report, Mapping) or report.get("status") != "PASS":
                return False, "Q2_QUALIFICATION_REPORT_FAIL"
            if not isinstance(audit, Mapping) or audit.get("verdict") != "PASS":
                return False, "Q2_QUALIFICATION_AUDIT_FAIL"
            if not isinstance(manifest, Mapping) or manifest.get("status") != "PASS" or int(manifest.get("selected_count", -1)) != 40:
                return False, "Q2_QUALIFICATION_MANIFEST_FAIL"
            try:
                counts = {suite: int((report.get("qualified_by_suite") or {}).get(suite, -1)) for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")}
                boundary_counts = [int(report.get(field, -1)) for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts", "attack_rollouts")]
            except (TypeError, ValueError):
                return False, "Q2_QUALIFICATION_RECEIPT_INVALID"
            if counts != {suite: 10 for suite in counts} or any(count != 0 for count in boundary_counts):
                return False, "Q2_QUALIFICATION_CLOSURE_COUNT_FAIL"
            return True, "PASS"
        failure_names = ("CONTROL_QUALIFICATION_FAILURE.json", "ABORTED_INCOMPLETE.json", "QUALIFICATION_FAILURE.json")
        if any((self.qualification_root / name).is_file() for name in failure_names):
            return False, "QUALIFICATION_FAILED"
        report_path = self.qualification_root / "CONTROL_QUALIFICATION_REPORT.json"
        audit_path = self.qualification_root / "CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json"
        if not report_path.is_file() or not audit_path.is_file():
            return False, "QUALIFICATION_INCOMPLETE"
        report = read_json(report_path, {})
        audit = read_json(audit_path, {})
        if not isinstance(report, Mapping) or report.get("status") != "PASS":
            return False, "QUALIFICATION_REPORT_FAIL"
        if not isinstance(audit, Mapping) or audit.get("verdict") != "PASS":
            return False, "QUALIFICATION_AUDIT_FAIL"
        try:
            evaluated_rows = int(report.get("evaluated_rows", 0))
            boundary_counts = [int(report.get(field, -1)) for field in ("eval160_reads", "protected_eval_reads", "vis_pgd_attack_rollouts")]
        except (TypeError, ValueError):
            return False, "QUALIFICATION_RECEIPT_INVALID"
        if evaluated_rows < 160 or any(count != 0 for count in boundary_counts):
            return False, "QUALIFICATION_CLOSURE_COUNT_FAIL"
        return True, "PASS"

    def _write(self, status: str, reason: str, *, q_reason: str = "", next_stage: str | None = None) -> None:
        self.count += 1
        payload = {
            "schema": "STAGE_V_R2_PLAN_CONTROLLER_V1", "status": status, "reason": reason,
            "next_stage": next_stage, "heartbeat_count": self.count, "pid": os.getpid(),
            "pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "repo_root": str(self.repo_root), "source_commit": self.source["commit"], "source_tree": self.source["tree"],
            "source_status_porcelain": self.source["status_porcelain"], "qualification_root": str(self.qualification_root),
            "qualification_reason": q_reason, "qualification_progress": _q_progress(self.qualification_root / ("Q2_CONTROL_QUALIFICATION.sqlite" if (self.qualification_root / "Q2_CONTROL_QUALIFICATION.sqlite").is_file() else "CONTROL_QUALIFICATION.sqlite")),
            "registry_chain_mode": "APPEND_ONLY_VERSIONED", "server_pipeline_autonomy_ready": self._q2_ready(),
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
            "external_root_process_present": pid_alive(int(self.args.external_pid or 0)), "external_root_process_terminated": False,
            "gpu5_touched": False, "updated_utc": utc_now(),
        }
        atomic_write_json(self.state_path, payload)
        atomic_write_json(self.heartbeat_path, payload)

    def _q2_ready(self) -> bool:
        return (self.qualification_root / "Q2_PARENT_MANIFEST_A.json").is_file() and (self.qualification_root / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json").is_file()

    def tick(self) -> str:
        current = source_binding(self.repo_root)
        if current != self.source:
            self._write(HARD_STOP, "SOURCE_BINDING_DRIFT")
            return HARD_STOP
        if self.source["status_porcelain"]:
            self._write(HARD_STOP, "SOURCE_WORKTREE_DIRTY")
            return HARD_STOP
        ok, reason = self._q_status()
        if reason == "QUALIFICATION_FAILED" or reason.endswith("_FAIL") or reason.endswith("_FAILURE") or reason.endswith("_INVALID"):
            self._write(HARD_STOP, reason, q_reason=reason)
            return HARD_STOP
        if not ok:
            self._write(WAIT_Q, reason, q_reason=reason, next_stage="C0")
            return WAIT_Q
        registry_paths = sorted(self.state_root.glob("PLAN_REGISTRY_V*.json"))
        registered_stages: set[str] = set()
        if registry_paths:
            latest = max(registry_paths, key=lambda path: int(path.stem.removeprefix("PLAN_REGISTRY_V")))
            try:
                registered_plans, _, _, _ = verify_registry_chain(latest, source=self.source)
            except (OSError, OrchestratorError) as exc:
                self._write(HARD_STOP, f"PLAN_REGISTRY_INVALID:{type(exc).__name__}:{exc}", q_reason=reason)
                return HARD_STOP
            registered_stages = set(registered_plans)
        if "C0" not in registered_stages:
            try:
                plan, plan_path, _ = build_c0_plan(
                    repo_root=self.repo_root, state_root=self.state_root,
                    qualification_root=self.qualification_root, candidate_manifest=self.candidate_manifest,
                    science_provenance=self.science_provenance, source_commit=self.source["commit"],
                    source_tree=self.source["tree"], python_executable=self.args.python_executable,
                    external_pid=self.args.external_pid,
                )
                receipts = list(plan.get("input_receipts", []))
                registry_path = append_registry(
                    state_root=self.state_root, source_commit=self.source["commit"], source_tree=self.source["tree"],
                    stage="C0", plan_path=plan_path, upstream_receipts=receipts,
                )
                atomic_write_json(self.state_root / "C0_PLAN_MATERIALIZED.json", {
                    "schema": "STAGE_V_R2_PLAN_MATERIALIZED_V1", "stage": "C0", "plan_path": str(plan_path),
                    "plan_sha256": sha256_file(plan_path), "registry_path": str(registry_path),
                    "registry_sha256": sha256_file(registry_path), "updated_utc": utc_now(),
                })
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self._write(HARD_STOP, f"C0_PLAN_MATERIALIZATION_FAIL:{type(exc).__name__}:{exc}", q_reason=reason)
                return HARD_STOP
        c0_audit_path = self.state_root / "C0_AUDIT.json"
        if c0_audit_path.is_file():
            c0_audit = read_json(c0_audit_path, {})
            if not isinstance(c0_audit, Mapping) or c0_audit.get("status") != "PASS":
                self._write(HARD_STOP, "C0_AUDIT_FAIL", q_reason=reason, next_stage="R2A")
                return HARD_STOP
            self._write(WAIT_INPUT, "R2A_RUNNER_SPEC_REQUIRED", q_reason=reason, next_stage="R2A")
            return WAIT_INPUT
        self._write("C0_PLAN_READY", "WAITING_FOR_ORCHESTRATOR_C0", q_reason=reason, next_stage="C0")
        return "C0_PLAN_READY"

    def run(self) -> int:
        self.lock.acquire()
        try:
            while True:
                status = self.tick()
                if self.args.once or status == HARD_STOP:
                    return 1 if status == HARD_STOP else 0
                time.sleep(max(1.0, float(self.args.poll_seconds)))
        finally:
            self.lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--science-provenance", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--external-pid", type=int, default=1895889)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(PlanController(build_parser().parse_args()).run())
