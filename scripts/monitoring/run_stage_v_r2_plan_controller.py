"""Server-local receipt controller for the Stage V R2 pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any, Mapping

try:
    from .materialize_stage_v_r2_next_plan import append_registry, build_c0_plan, build_stage_plan_from_spec
    from .run_stage_v_r2_mainline_orchestrator import FileLock, OrchestratorError, pid_alive, source_binding, verify_registry_chain
    from ..detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now
    from ..detector_v5.stage_v_science_core_provenance import build as build_science_provenance, verify as verify_science_provenance
except ImportError:  # direct server execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.monitoring.materialize_stage_v_r2_next_plan import append_registry, build_c0_plan, build_stage_plan_from_spec
    from scripts.monitoring.run_stage_v_r2_mainline_orchestrator import FileLock, OrchestratorError, pid_alive, source_binding, verify_registry_chain
    from scripts.detector_v5.stage_v_dynamic_common import atomic_write_json, read_json, sha256_file, utc_now
    from scripts.detector_v5.stage_v_science_core_provenance import build as build_science_provenance, verify as verify_science_provenance


WAIT_Q = "WAIT_QUALIFICATION"
WAIT_INPUT = "WAIT_NEXT_STAGE_INPUT"
HARD_STOP = "HARD_STOP"
PIPELINE_COMPLETE_VIS = "GPU_PIPELINE_COMPLETE_VIS"
PIPELINE_COMPLETE_NO_VIS = "GPU_PIPELINE_COMPLETE_NO_VIS"
STAGE_CHAIN = ("R2A", "R2B_DECISION", "R2B", "STAGE_V2", "STAGE_O", "STUDENT_FREEZE", "PILOT_QUALIFICATION", "DIRECT_OPEN_PILOT")


class ControllerError(RuntimeError):
    pass


def _receipt(name: str, path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{name}_MISSING:{path}")
    return {"name": name, "path": str(path), "sha256": sha256_file(path)}


def _ensure_science_parent_manifest(formal_manifest: Path, state_root: Path) -> Path:
    value = read_json(formal_manifest, {})
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "STAGE_V_FORMAL_PARENT_MANIFEST_V1"
        or value.get("status") != "FROZEN"
        or int(value.get("selected_count", -1)) != 40
        or not isinstance(value.get("selected_parents"), list)
    ):
        raise ValueError("R2A_FORMAL_PARENT_MANIFEST_INVALID")
    derived = dict(value)
    derived["schema"] = "D8_STAGE_V_CLEAN_SUCCESS_PARENT_MANIFEST_V1"
    derived["adapter_source_manifest_sha256"] = sha256_file(formal_manifest)
    destination = state_root / "D8_STAGE_V_CLEAN_SUCCESS_PARENT_MANIFEST_V1.json"
    if destination.is_file():
        if read_json(destination, {}) != derived:
            raise ValueError("R2A_SCIENCE_PARENT_MANIFEST_ALREADY_EXISTS_DIFFERENT")
    else:
        atomic_write_json(destination, derived)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    expected_sidecar = f"{sha256_file(destination)}  {destination.name}\n"
    if sidecar.is_file() and sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise ValueError("R2A_SCIENCE_PARENT_MANIFEST_SHA256_SIDECAR_MISMATCH")
    if not sidecar.is_file():
        sidecar.write_text(expected_sidecar, encoding="utf-8")
    return destination


def _git_binding(repo_root: Path) -> dict[str, str]:
    def git(*parts: str) -> str:
        result = subprocess.run(["git", "-C", str(repo_root), *parts], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ValueError(f"SCIENCE_GIT_QUERY_FAIL:{result.stderr[-200:]}")
        return result.stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"), "status": git("status", "--porcelain")}


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
        try:
            plans, _, _, _ = verify_registry_chain(latest, source=self.source)
            return plans
        except OrchestratorError as exc:
            if str(exc) not in {"PLAN_REGISTRY_SOURCE_MISMATCH", "PLAN_REGISTRY_TREE_MISMATCH"}:
                raise
            registry = read_json(latest, {})
            if not isinstance(registry, Mapping):
                raise
            previous_source = {
                "commit": str(registry.get("source_commit", "")),
                "tree": str(registry.get("source_tree", "")),
            }
            if (
                not previous_source["commit"]
                or not previous_source["tree"]
                or previous_source["commit"] == self.source["commit"]
                or "C0" not in verify_registry_chain(latest, source=previous_source)[0]
            ):
                raise
            ancestry = subprocess.run(
                ["git", "-C", str(self.repo_root), "merge-base", "--is-ancestor", previous_source["commit"], self.source["commit"]],
                capture_output=True,
                check=False,
            )
            if ancestry.returncode != 0:
                raise OrchestratorError("PLAN_REGISTRY_SOURCE_UPGRADE_NOT_DESCENDANT") from exc
            return verify_registry_chain(latest, source=previous_source)[0]

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

    def _ensure_r2a_spec(self, *, q_reason: str) -> str:
        """JIT-bind the fresh Q2 manifest to the frozen external R2A core."""
        spec_path = self.state_root / "R2A_SPEC.json"
        if spec_path.is_file():
            return self._materialize_spec_stage("R2A", q_reason=q_reason)
        required = (
            "r2a_science_runner", "r2a_science_auditor", "r2a_science_repo_root",
            "r2a_science_source_commit", "r2a_science_source_tree", "r2a_timeout_policy",
            "r2a_science_runner_sha256", "r2a_science_auditor_sha256",
        )
        if any(not str(getattr(self.args, name, "")) for name in required):
            self._write(WAIT_INPUT, "R2A_SCIENCE_BINDING_REQUIRED", q_reason=q_reason, next_stage="R2A")
            return WAIT_INPUT
        runner = Path(str(self.args.r2a_science_runner)).resolve()
        science_auditor = Path(str(self.args.r2a_science_auditor)).resolve()
        science_repo = Path(str(self.args.r2a_science_repo_root)).resolve()
        timeout_policy = Path(str(self.args.r2a_timeout_policy)).resolve()
        if not all(path.is_file() for path in (runner, science_auditor, timeout_policy)) or not science_repo.is_dir():
            self._write(HARD_STOP, "R2A_SCIENCE_BINDING_PATH_INVALID", q_reason=q_reason, next_stage="R2A")
            return HARD_STOP
        if sha256_file(runner) != str(self.args.r2a_science_runner_sha256) or sha256_file(science_auditor) != str(self.args.r2a_science_auditor_sha256):
            self._write(HARD_STOP, "R2A_SCIENCE_SNAPSHOT_SHA256_MISMATCH", q_reason=q_reason, next_stage="R2A")
            return HARD_STOP
        try:
            science_source = _git_binding(science_repo)
        except ValueError as exc:
            self._write(HARD_STOP, str(exc), q_reason=q_reason, next_stage="R2A")
            return HARD_STOP
        if (
            science_source["commit"] != str(self.args.r2a_science_source_commit)
            or science_source["tree"] != str(self.args.r2a_science_source_tree)
            or science_source["status"]
        ):
            self._write(HARD_STOP, "R2A_SCIENCE_SOURCE_BINDING_INVALID", q_reason=q_reason, next_stage="R2A")
            return HARD_STOP
        q2_report = self.qualification_root / "Q2_CONTROL_QUALIFICATION_REPORT.json"
        q2_audit = self.qualification_root / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json"
        q2_manifest = self.qualification_root / "Q2_PARENT_MANIFEST_A.json"
        formal_manifest = self.qualification_root / "STAGE_V_FORMAL_PARENT_MANIFEST_V1.json"
        for path in (q2_report, q2_audit, q2_manifest, formal_manifest, self.candidate_manifest, self.science_provenance):
            if not path.is_file():
                self._write(WAIT_INPUT, f"R2A_INPUT_MISSING:{path.name}", q_reason=q_reason, next_stage="R2A")
                return WAIT_INPUT
        try:
            science_manifest = _ensure_science_parent_manifest(formal_manifest, self.state_root)
        except (OSError, TypeError, ValueError) as exc:
            self._write(HARD_STOP, f"R2A_SCIENCE_PARENT_MANIFEST_FAIL:{type(exc).__name__}:{exc}", q_reason=q_reason, next_stage="R2A")
            return HARD_STOP
        provenance_path = self.state_root / "R2A_SCIENCE_CORE_PROVENANCE.json"
        if provenance_path.is_file():
            provenance_ok, provenance_errors = verify_science_provenance(
                provenance_path, expected_commit=str(self.args.r2a_science_source_commit), expected_tree=str(self.args.r2a_science_source_tree),
            )
            if not provenance_ok:
                self._write(HARD_STOP, "R2A_SCIENCE_CORE_PROVENANCE_FAIL:" + ";".join(provenance_errors), q_reason=q_reason, next_stage="R2A")
                return HARD_STOP
        else:
            atomic_write_json(provenance_path, build_science_provenance(
                [runner, science_auditor], source_commit=str(self.args.r2a_science_source_commit), source_tree=str(self.args.r2a_science_source_tree),
            ))
        supervisor = self.repo_root / "scripts/detector_v5/run_stage_v_parent_aware_supervisor.py"
        dispatcher = self.repo_root / "scripts/detector_v5/run_stage_v_dynamic_dispatcher.py"
        auditor = self.repo_root / "scripts/detector_v5/audit_stage_v_dynamic_queue.py"
        for path in (supervisor, dispatcher, auditor):
            if not path.is_file():
                self._write(HARD_STOP, f"R2A_CONTROL_TOOL_MISSING:{path.name}", q_reason=q_reason, next_stage="R2A")
                return HARD_STOP
        config_path = self.state_root / "R2A_CONFIG.json"
        runner_provenance = build_science_provenance([runner], source_commit="x", source_tree="y")["files"][0]
        auditor_provenance = build_science_provenance([science_auditor], source_commit="x", source_tree="y")["files"][0]
        config = {
            "schema": "STAGE_V_R2A_CONFIG_V1", "stage": "R2A", "status": "FROZEN",
            "control_source_commit": self.source["commit"], "control_source_tree": self.source["tree"],
            "science_source_commit": str(self.args.r2a_science_source_commit), "science_source_tree": str(self.args.r2a_science_source_tree),
            "science_repo_root": str(science_repo), "science_runner": str(runner), "science_runner_sha256": sha256_file(runner),
            "science_runner_git_blob_sha1": runner_provenance["git_blob_sha1"], "science_auditor": str(science_auditor),
            "science_auditor_sha256": sha256_file(science_auditor), "science_auditor_git_blob_sha1": auditor_provenance["git_blob_sha1"],
            "science_parent_manifest": _receipt("science_parent_manifest", science_manifest),
            "timeout_policy": _receipt("timeout_policy", timeout_policy), "probe_limit": 24, "expected_branch_count": 72,
            "planned_parents": 40, "approved_gpus": list(range(8)), "gpu5_authorized": True,
            "old_artifacts_reused": False, "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
        }
        if config_path.is_file():
            if read_json(config_path, {}) != config:
                self._write(HARD_STOP, "R2A_CONFIG_ALREADY_EXISTS_DIFFERENT", q_reason=q_reason, next_stage="R2A")
                return HARD_STOP
        else:
            atomic_write_json(config_path, config)
        formal_manifest_binding = _receipt("formal_parent_manifest", formal_manifest)
        science_manifest_binding = _receipt("science_parent_manifest", science_manifest)
        command = [
            str(self.args.python_executable), str(supervisor), "--run-root", "{output_root}", "--repo-root", str(self.repo_root),
            "--parent-manifest", str(formal_manifest), "--parent-manifest-sha256", formal_manifest_binding["sha256"],
            "--queue-db", "{output_root}/STAGE_V_R2A.sqlite", "--run-id", "stage-v-r2a-{source_commit}",
            "--expected-parent-count", "40", "--expected-source-commit", "{source_commit}", "--expected-source-tree", "{source_tree}",
            "--lock-path", str(self.state_root.parent / ".stage_v_r2a.lock"), "--approved-gpus", "{approved_gpus}",
            "--external-pid", str(self.args.external_pid), "--preflight-file", "{output_root}/GPU_PREFLIGHT.json",
            "--timeout-policy", str(timeout_policy), "--dispatcher-script", str(dispatcher), "--auditor-script", str(auditor),
            "--science-runner", str(runner), "--science-provenance", str(provenance_path),
            "--science-source-commit", str(self.args.r2a_science_source_commit), "--science-source-tree", str(self.args.r2a_science_source_tree),
            "--science-repo-root", str(science_repo), "--science-parent-manifest", str(science_manifest),
            "--probe-limit", "24", "--max-attempts", "1", "--allow-gpu5",
        ]
        audit_command = [
            str(self.args.python_executable), str(auditor), "--run-root", "{output_root}", "--parent-manifest", str(formal_manifest),
            "--queue-db", "{output_root}/STAGE_V_R2A.sqlite", "--run-id", "stage-v-r2a-{source_commit}",
            "--expected-parent-count", "40", "--expected-branch-count", "72", "--expected-source-commit", "{source_commit}",
            "--expected-source-tree", "{source_tree}", "--science-source-commit", str(self.args.r2a_science_source_commit),
            "--science-source-tree", str(self.args.r2a_science_source_tree), "--science-provenance", str(provenance_path),
            "--science-parent-manifest", str(science_manifest), "--allow-gpu5",
        ]
        spec = {
            "schema": "STAGE_V_R2_STAGE_SPEC_V1", "stage": "R2A", "source_commit": self.source["commit"], "source_tree": self.source["tree"],
            "runner_path": str(supervisor), "runner_path_sha256": sha256_file(supervisor), "auditor_path": str(auditor), "auditor_path_sha256": sha256_file(auditor),
            "config_path": str(config_path), "config_path_sha256": sha256_file(config_path), "cwd": str(self.repo_root),
            "python_executable": str(self.args.python_executable), "parent_manifest": formal_manifest_binding,
            "input_receipts": [_receipt("q2_report", q2_report), _receipt("q2_audit", q2_audit), _receipt("q2_parent_manifest", q2_manifest),
                               formal_manifest_binding, science_manifest_binding, _receipt("candidate_manifest", self.candidate_manifest), _receipt("science_provenance", provenance_path),
                               _receipt("science_runner", runner), _receipt("science_auditor", science_auditor), _receipt("timeout_policy", timeout_policy), _receipt("r2a_config", config_path)],
            "output_root_template": str(self.state_root.parent / "STAGE_V_R2A_COUNTERFACTUAL_MAP_{commit8}_{utc}"),
            "command_template": command, "audit_command_template": audit_command,
            "completion_receipts": ["SUPERVISOR_COMPLETE.json", "STAGE_V_CLOSURE_RECEIPT.json"],
            "resource_policy": {"resource_kind": "GPU", "required_gpu_count": 8, "minimum_gpu_count": 8, "maximum_gpu_count": 8, "strict_gpu_count": True,
                                 "excluded_gpus": [], "gpu5_authorized": True, "protected_pids": [int(self.args.external_pid)], "canary_peak_mib": 0},
            "gpu_policy": {"required_count": 8, "excluded_gpus": [], "gpu5_authorized": True, "protected_pids": [int(self.args.external_pid)]},
            "lock_path": str(self.state_root.parent / ".stage_v_r2a.lock"),
            "forbidden_boundary_contract": {"eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0},
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "created_utc": utc_now(),
        }
        atomic_write_json(spec_path, spec)
        return self._materialize_spec_stage("R2A", q_reason=q_reason)

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
            status = self._ensure_r2a_spec(q_reason=q_reason) if stage == "R2A" else self._materialize_spec_stage(stage, q_reason=q_reason)
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
    parser.add_argument("--r2a-science-runner", type=Path)
    parser.add_argument("--r2a-science-auditor", type=Path)
    parser.add_argument("--r2a-science-repo-root", type=Path)
    parser.add_argument("--r2a-science-source-commit", default="")
    parser.add_argument("--r2a-science-source-tree", default="")
    parser.add_argument("--r2a-science-runner-sha256", default="")
    parser.add_argument("--r2a-science-auditor-sha256", default="")
    parser.add_argument("--r2a-timeout-policy", type=Path)
    parser.add_argument("--allow-gpu5", action="store_true", help="Authorize GPU5 for fresh downstream plans; never hot-adds it to an existing run")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(PlanController(build_parser().parse_args()).run())
