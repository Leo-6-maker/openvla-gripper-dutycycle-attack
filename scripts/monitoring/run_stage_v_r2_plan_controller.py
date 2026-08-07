"""Server-local receipt controller for the Stage V R2 pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping

try:
    from .materialize_stage_v_r2_next_plan import append_registry, build_c0_plan, build_stage_plan_from_spec
    from .run_stage_v_r2_mainline_orchestrator import FileLock, OrchestratorError, pid_alive, source_binding, verify_registry_chain
    from ..detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now
except ImportError:  # direct server execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.monitoring.materialize_stage_v_r2_next_plan import append_registry, build_c0_plan, build_stage_plan_from_spec
    from scripts.monitoring.run_stage_v_r2_mainline_orchestrator import FileLock, OrchestratorError, pid_alive, source_binding, verify_registry_chain
    from scripts.detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now


WAIT_Q = "WAIT_QUALIFICATION"
WAIT_INPUT = "WAIT_NEXT_STAGE_INPUT"
HARD_STOP = "HARD_STOP"
PIPELINE_COMPLETE_VIS = "GPU_PIPELINE_COMPLETE_VIS"
PIPELINE_COMPLETE_NO_VIS = "GPU_PIPELINE_COMPLETE_NO_VIS"
STAGE_CHAIN = ("R2A", "R2B_DECISION", "R2B", "STAGE_V2", "STAGE_O", "STUDENT_FREEZE", "PILOT_QUALIFICATION", "DIRECT_OPEN_PILOT")


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

    def _load_registered(self) -> dict[str, dict[str, Any]]:
        paths = sorted(self.state_root.glob("PLAN_REGISTRY_V*.json"))
        if not paths:
            return {}
        latest = max(paths, key=lambda path: int(path.stem.removeprefix("PLAN_REGISTRY_V")))
        plans, _, _, _ = verify_registry_chain(latest, source=self.source)
        return plans

    def _audit_status(self, stage: str) -> str | None:
        path = self.state_root / f"{stage}_AUDIT.json"
        if not path.is_file():
            return None
        value = read_json(path, {})
        return str(value.get("status")) if isinstance(value, Mapping) else "INVALID"

    def _launch_root(self, stage: str) -> Path | None:
        value = read_json(self.state_root / f"{stage}_LAUNCH.json", {})
        if not isinstance(value, Mapping) or not value.get("output_root"):
            return None
        root = Path(str(value["output_root"])).resolve()
        return root if root.is_dir() else None

    def _stage_receipt(self, stage: str, plan: Mapping[str, Any]) -> Mapping[str, Any] | None:
        root = self._launch_root(stage)
        if root is None:
            return None
        names = list(plan.get("decision_receipt_names") or [])
        if not names:
            names = {
                "R2B_DECISION": ["STAGE_V_R2B_DECISION.json", "R2B_DECISION.json"],
                "DIRECT_OPEN_PILOT": ["DIRECT_OPEN_TIMING_REPORT.json"],
            }.get(stage, [])
        for name in names:
            path = Path(str(name))
            if not path.is_absolute():
                path = root / path
            value = read_json(path, {})
            if isinstance(value, Mapping):
                return value
        return None

    def _materialize_spec_stage(self, stage: str, *, q_reason: str) -> str:
        spec_path = self.state_root / f"{stage}_SPEC.json"
        if not spec_path.is_file():
            self._write(WAIT_INPUT, f"{stage}_SPEC_REQUIRED", q_reason=q_reason, next_stage=stage)
            return WAIT_INPUT
        try:
            plan, plan_path = build_stage_plan_from_spec(
                stage=stage, spec_path=spec_path, state_root=self.state_root,
                source_commit=self.source["commit"], source_tree=self.source["tree"],
            )
            registry_path = append_registry(
                state_root=self.state_root, source_commit=self.source["commit"], source_tree=self.source["tree"],
                stage=stage, plan_path=plan_path, upstream_receipts=list(plan.get("input_receipts", [])),
            )
            atomic_write_json(self.state_root / f"{stage}_PLAN_MATERIALIZED.json", {
                "schema": "STAGE_V_R2_PLAN_MATERIALIZED_V1", "stage": stage,
                "plan_path": str(plan_path), "plan_sha256": sha256_file(plan_path),
                "registry_path": str(registry_path), "registry_sha256": sha256_file(registry_path),
                "updated_utc": utc_now(),
            })
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._write(HARD_STOP, f"{stage}_PLAN_MATERIALIZATION_FAIL:{type(exc).__name__}:{exc}", q_reason=q_reason, next_stage=stage)
            return HARD_STOP
        self._write(f"{stage}_PLAN_READY", f"{stage}_PLAN_MATERIALIZED", q_reason=q_reason, next_stage=stage)
        return f"{stage}_PLAN_READY"

    def _require_audit(self, stage: str, *, q_reason: str) -> str | None:
        status = self._audit_status(stage)
        if status is None:
            self._write(f"WAITING_FOR_ORCHESTRATOR_{stage}", f"{stage}_AUDIT_PENDING", q_reason=q_reason, next_stage=stage)
            return f"WAITING_FOR_ORCHESTRATOR_{stage}"
        if status != "PASS":
            self._write(HARD_STOP, f"{stage}_AUDIT_FAIL", q_reason=q_reason, next_stage=stage)
            return HARD_STOP
        return None

    def _ensure_stage(self, stage: str, plans: Mapping[str, Any], *, q_reason: str) -> tuple[str | None, dict[str, dict[str, Any]]]:
        if stage not in plans:
            status = self._materialize_spec_stage(stage, q_reason=q_reason)
            if status == HARD_STOP:
                return HARD_STOP, dict(plans)
            return status, self._load_registered()
        return None, dict(plans)

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
        try:
            registered_plans = self._load_registered()
        except (OSError, OrchestratorError, ValueError, TypeError) as exc:
            self._write(HARD_STOP, f"PLAN_REGISTRY_INVALID:{type(exc).__name__}:{exc}", q_reason=reason)
            return HARD_STOP
        if "C0" not in registered_plans:
            try:
                plan, plan_path, _ = build_c0_plan(
                    repo_root=self.repo_root, state_root=self.state_root,
                    qualification_root=self.qualification_root, candidate_manifest=self.candidate_manifest,
                    science_provenance=self.science_provenance, source_commit=self.source["commit"],
                    source_tree=self.source["tree"], python_executable=self.args.python_executable,
                    external_pid=self.args.external_pid, allow_gpu5=bool(getattr(self.args, "allow_gpu5", False)),
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
            self._write("C0_PLAN_READY", "C0_PLAN_MATERIALIZED", q_reason=reason, next_stage="C0")
            return "C0_PLAN_READY"
        wait = self._require_audit("C0", q_reason=reason)
        if wait:
            return wait

        wait, registered_plans = self._ensure_stage("R2A", registered_plans, q_reason=reason)
        if wait:
            return wait
        wait = self._require_audit("R2A", q_reason=reason)
        if wait:
            return wait

        wait, registered_plans = self._ensure_stage("R2B_DECISION", registered_plans, q_reason=reason)
        if wait:
            return wait
        wait = self._require_audit("R2B_DECISION", q_reason=reason)
        if wait:
            return wait
        decision = self._stage_receipt("R2B_DECISION", registered_plans["R2B_DECISION"])
        if decision is None:
            self._write(WAIT_INPUT, "R2B_DECISION_RECEIPT_REQUIRED", q_reason=reason, next_stage="R2B_DECISION")
            return WAIT_INPUT
        decision_status = str(decision.get("status", decision.get("verdict", "")))
        if decision_status not in {"R2B_REQUIRED", "R2B_NOT_REQUIRED"}:
            self._write(HARD_STOP, "R2B_DECISION_INVALID", q_reason=reason, next_stage="R2B_DECISION")
            return HARD_STOP

        if decision_status == "R2B_REQUIRED":
            wait, registered_plans = self._ensure_stage("R2B", registered_plans, q_reason=reason)
            if wait:
                return wait
            wait = self._require_audit("R2B", q_reason=reason)
            if wait:
                return wait
        elif "R2B" in registered_plans:
            self._write(HARD_STOP, "R2B_PLAN_PRESENT_WHEN_NOT_REQUIRED", q_reason=reason, next_stage="R2B_DECISION")
            return HARD_STOP

        for stage in ("STAGE_V2", "STAGE_O", "STUDENT_FREEZE", "PILOT_QUALIFICATION", "DIRECT_OPEN_PILOT"):
            wait, registered_plans = self._ensure_stage(stage, registered_plans, q_reason=reason)
            if wait:
                return wait
            wait = self._require_audit(stage, q_reason=reason)
            if wait:
                return wait

        direct = self._stage_receipt("DIRECT_OPEN_PILOT", registered_plans["DIRECT_OPEN_PILOT"])
        if direct is None:
            self._write(WAIT_INPUT, "DIRECT_OPEN_DECISION_RECEIPT_REQUIRED", q_reason=reason, next_stage="DIRECT_OPEN_PILOT")
            return WAIT_INPUT
        direct_status = str(direct.get("status", direct.get("verdict", "")))
        if direct_status == "NO_GO":
            self._write(PIPELINE_COMPLETE_NO_VIS, "DIRECT_OPEN_NO_GO", q_reason=reason, next_stage=None)
            return PIPELINE_COMPLETE_NO_VIS
        if direct_status != "GO":
            self._write(HARD_STOP, "DIRECT_OPEN_DECISION_INVALID", q_reason=reason, next_stage="DIRECT_OPEN_PILOT")
            return HARD_STOP

        wait, registered_plans = self._ensure_stage("VIS_SMALL_MATRIX", registered_plans, q_reason=reason)
        if wait:
            return wait
        wait = self._require_audit("VIS_SMALL_MATRIX", q_reason=reason)
        if wait:
            return wait
        self._write(PIPELINE_COMPLETE_VIS, "VIS_SMALL_MATRIX_PASS", q_reason=reason, next_stage=None)
        return PIPELINE_COMPLETE_VIS

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
    parser.add_argument("--allow-gpu5", action="store_true", help="Authorize GPU5 for fresh downstream plans; never hot-adds it to an existing run")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(PlanController(build_parser().parse_args()).run())
