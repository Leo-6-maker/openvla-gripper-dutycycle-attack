"""Local, receipt-driven monitor for the Stage V R2 mainline.

This monitor owns no science launcher.  It verifies bindings, records the
resource gate, and advances only from complete, SHA-bound receipts.  Missing
command plans or fewer than eight safe GPUs are ordinary wait states; they
never create a formal root or guess a command.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

try:
    from .monitor_stage_v_goal import (
        ExclusiveMonitorLock,
        MonitorError,
        atomic_write_json,
        parse_json,
        read_gpu_snapshot,
        read_meminfo,
        read_vmstat,
        read_xid_status,
    )
except ImportError:  # direct server execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.monitoring.monitor_stage_v_goal import (
        ExclusiveMonitorLock,
        MonitorError,
        atomic_write_json,
        parse_json,
        read_gpu_snapshot,
        read_meminfo,
        read_vmstat,
        read_xid_status,
    )


SCHEMA = "STAGE_V_R2_MAINLINE_MONITOR_V1"
WAITING = "WAITING_FOR_8_SAFE_GPUS"
STATES = {
    "PREPARATION", WAITING, "QUALIFICATION_RUNNING", "DYNAMIC8_CANARY",
    "STAGE_V_R2A_RUNNING", "STAGE_V_R2A_AUDIT", "STAGE_V_R2B_RUNNING",
    "STAGE_V_CLOSED", "STAGE_V2_RUNNING", "STAGE_O_RUNNING",
    "DIRECT_OPEN_PILOT", "VIS_SMALL_MATRIX", "GOAL_COMPLETE", "HARD_STOP",
}
FORMAL_PREFIX = "STAGE_V_R2A_COUNTERFACTUAL_MAP_"


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GIT_QUERY_FAIL:{result.stderr[-200:]}")
    return result.stdout.strip()


def source_binding(repo_root: Path) -> dict[str, Any]:
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "status_porcelain": _git(repo_root, "status", "--porcelain"),
    }


def _mem_snapshot() -> dict[str, Any]:
    mem = read_meminfo()
    vm = read_vmstat()
    available = mem.get("MemAvailable")
    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", 0)
    return {
        "available_ram_gib": round(available / (1024 * 1024), 3) if available is not None else None,
        "swap_used_bytes": max(0, swap_total - swap_free) * 1024,
        "swap_in": vm.get("pswpin"),
        "swap_out": vm.get("pswpout"),
        "oom_kill": vm.get("oom_kill"),
    }


def _compute_pids() -> set[int] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}


def resource_gate(
    *,
    gpu_rows: list[Mapping[str, Any]],
    gpu_error: str | None,
    memory: Mapping[str, Any],
    xid_status: str,
    xid_error: str | None,
    baseline_oom: int | None,
    required_gpus: int,
    excluded_gpus: set[int],
    protected_pids: set[int],
    canary_peak_mib: float,
    swap_bad_streak: int,
    minimum_ram_gib: float,
) -> dict[str, Any]:
    hard: list[str] = []
    waiting: list[str] = []
    current_oom = memory.get("oom_kill")
    if baseline_oom is not None and isinstance(current_oom, int) and current_oom > baseline_oom:
        hard.append("OOM_KILL_COUNTER_INCREASED")
    available = memory.get("available_ram_gib")
    if available is None:
        waiting.append("AVAILABLE_RAM_UNKNOWN")
    elif float(available) < minimum_ram_gib:
        hard.append("AVAILABLE_RAM_BELOW_HARD_STOP")
    if swap_bad_streak >= 2:
        hard.append("SWAP_NONZERO_TWO_SAMPLES")
    elif memory.get("swap_used_bytes", 0) or memory.get("swap_in") or memory.get("swap_out"):
        waiting.append("SWAP_NOT_CLEAR")
    if xid_status == "XID_DETECTED":
        hard.append("NVIDIA_XID")
    elif xid_status != "CLEAR" or xid_error:
        waiting.append("GPU_XID_STATUS_UNKNOWN")
    if gpu_error:
        waiting.append(gpu_error)
    if canary_peak_mib <= 0:
        waiting.append("CANARY_PEAK_MEMORY_MISSING")
    minimum_free = canary_peak_mib * 1.5 + 4096.0
    compute_pids = _compute_pids()
    if compute_pids is None:
        waiting.append("COMPUTE_PROCESS_QUERY_UNKNOWN")
    elif protected_pids & compute_pids:
        waiting.append("PROTECTED_PROCESS_PRESENT")
    safe: list[int] = []
    decisions: list[dict[str, Any]] = []
    for raw in gpu_rows:
        gpu = int(raw.get("gpu_id", raw.get("index", -1)))
        free = raw.get("memory_free_mib")
        if free is None:
            free = raw.get("memory_free", -1)
        reasons: list[str] = []
        if gpu in excluded_gpus:
            reasons.append("EXCLUDED_GPU")
        if canary_peak_mib <= 0 or float(free) < minimum_free:
            reasons.append("INSUFFICIENT_FREE_MEMORY")
        if reasons:
            decisions.append({**dict(raw), "gpu_id": gpu, "safe": False, "reasons": reasons})
        else:
            safe.append(gpu)
            decisions.append({**dict(raw), "gpu_id": gpu, "safe": True, "reasons": []})
    safe = sorted(set(safe))
    if len(safe) < required_gpus:
        waiting.append(f"SAFE_GPU_COUNT:{len(safe)}/{required_gpus}")
    return {
        "verdict": "HARD_STOP" if hard else "PASS" if not waiting else WAITING,
        "hard_stop_reasons": sorted(set(hard)),
        "waiting_reasons": sorted(set(waiting)),
        "safe_gpus": safe[:required_gpus],
        "all_safe_gpus": safe,
        "required_gpus": required_gpus,
        "excluded_gpus": sorted(excluded_gpus),
        "minimum_free_memory_mib": minimum_free,
        "gpu_rows": decisions,
        "protected_pids": sorted(protected_pids),
        "updated_utc": utc_now(),
    }


def _all_pass_receipts(root: Path | None, names: tuple[str, ...]) -> bool:
    if root is None or not root.is_dir():
        return False
    for name in names:
        value = parse_json(root / name)
        if not isinstance(value, Mapping) or value.get("verdict", value.get("status")) not in {
            "PASS", "DONE", "STAGE_V_FORMAL_MAP_CLOSED", "STAGE_O_PASS", "STAGE_V2_PASS",
        }:
            return False
    return True


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except (AttributeError, OSError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class MainlineMonitor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.monitor_root = Path(args.monitor_root).resolve()
        self.monitor_root.mkdir(parents=True, exist_ok=True)
        self.start_utc = utc_now()
        baseline_path = self.monitor_root / "OOM_KILL_BASELINE.json"
        baseline = parse_json(baseline_path)
        if not isinstance(baseline, Mapping):
            baseline = {"oom_kill": _mem_snapshot().get("oom_kill"), "created_utc": self.start_utc}
            atomic_write_json(baseline_path, dict(baseline))
        self.baseline_oom = baseline.get("oom_kill") if isinstance(baseline.get("oom_kill"), int) else None
        self.swap_bad_streak = 0
        self.heartbeat_count = 0
        self.lock = ExclusiveMonitorLock(
            Path(args.lock_path).resolve(),
            {"schema": "STAGE_V_R2_MAINLINE_LOCK_V1", "monitor_pid": os.getpid(), "started_utc": self.start_utc},
        )

    def _verify_bindings(self) -> tuple[list[str], list[str]]:
        hard: list[str] = []
        missing: list[str] = []
        try:
            source = source_binding(self.repo_root)
        except RuntimeError as exc:
            return [str(exc)], []
        if source["commit"] != self.args.expected_source_commit or source["tree"] != self.args.expected_source_tree:
            hard.append("SOURCE_OR_TREE_MISMATCH")
        if source["status_porcelain"]:
            hard.append("SOURCE_WORKTREE_DIRTY")
        abort = parse_json(Path(self.args.old_root) / "ABORTED_INCOMPLETE.json")
        if not isinstance(abort, Mapping) or abort.get("status") != "ABORTED_INCOMPLETE":
            hard.append("OLD_ROOT_NOT_ABORTED_INCOMPLETE")
        elif abort.get("accepted_parent_results", 0) != 0 or abort.get("scientific_validity", 0) != 0:
            hard.append("OLD_ROOT_ACCEPTED_RESULT_NONZERO")
        if self.args.old_manifest:
            path = Path(self.args.old_manifest)
            if not path.is_file():
                hard.append("OLD_MANIFEST_MISSING")
            elif not self.args.old_manifest_sha256:
                missing.append("OLD_MANIFEST_SHA256_UNBOUND")
            elif self.args.old_manifest_sha256 and sha256_file(path) != self.args.old_manifest_sha256:
                hard.append("OLD_MANIFEST_SHA256_MISMATCH")
        for name, expected in (("candidate_manifest", self.args.candidate_manifest_sha256), ("postmortem", self.args.postmortem_sha256), ("timeout_policy", self.args.timeout_policy_sha256), ("science_provenance", self.args.science_provenance_sha256)):
            raw = getattr(self.args, name)
            if not raw:
                missing.append(name.upper())
                continue
            path = Path(raw)
            if not path.is_file():
                missing.append(f"{name.upper()}_MISSING")
            elif not expected:
                missing.append(f"{name.upper()}_SHA256_UNBOUND")
            elif expected and sha256_file(path) != expected:
                hard.append(f"{name.upper()}_SHA256_MISMATCH")
            if path.is_file() and _under(path, Path(self.args.old_root)):
                hard.append(f"{name.upper()}_INSIDE_OLD_ROOT")
        if _under(Path(self.args.formal_root_parent), Path(self.args.old_root)):
            hard.append("FORMAL_ROOT_UNDER_OLD_ROOT")
        formal_parent = Path(self.args.formal_root_parent)
        if formal_parent.is_dir():
            formal_roots = [path.resolve() for path in formal_parent.glob(f"{FORMAL_PREFIX}*") if path.is_dir()]
            registered = Path(self.args.r2a_root).resolve() if self.args.r2a_root else None
            if len(formal_roots) > 1:
                hard.append("MULTIPLE_FORMAL_ROOTS")
            if formal_roots and registered not in formal_roots:
                hard.append("UNREGISTERED_FORMAL_ROOT")
        if self.args.candidate_manifest:
            value = parse_json(Path(self.args.candidate_manifest))
            rows = value.get("parents", value.get("selected_parents", [])) if isinstance(value, Mapping) else value
            if isinstance(value, Mapping) and value.get("old_artifacts_reused") is True:
                hard.append("CANDIDATE_OLD_ARTIFACT_REUSE")
            if isinstance(rows, list) and any(isinstance(row, Mapping) and row.get("old_artifacts_reused") is True for row in rows):
                hard.append("CANDIDATE_ROW_OLD_ARTIFACT_REUSE")
        return sorted(set(hard)), sorted(set(missing))

    def _resources(self) -> dict[str, Any]:
        memory = _mem_snapshot()
        gpu_rows, gpu_error = read_gpu_snapshot()
        xid_status, xid_error = read_xid_status(self.start_utc)
        if memory.get("swap_used_bytes", 0) or memory.get("swap_in") or memory.get("swap_out"):
            self.swap_bad_streak += 1
        else:
            self.swap_bad_streak = 0
        result = resource_gate(
            gpu_rows=gpu_rows, gpu_error=gpu_error, memory=memory, xid_status=xid_status,
            xid_error=xid_error, baseline_oom=self.baseline_oom, required_gpus=self.args.required_gpus,
            excluded_gpus=set(self.args.excluded_gpus), protected_pids=set(self.args.protected_pids),
            canary_peak_mib=self.args.canary_peak_mib, swap_bad_streak=self.swap_bad_streak,
            minimum_ram_gib=self.args.minimum_ram_gib,
        )
        return {**memory, "gpu_xid_status": xid_status, "gpu_xid_error": xid_error, "gpu_query_error": gpu_error, **result}

    def _phase(self) -> str:
        if _all_pass_receipts(Path(self.args.qualification_root) if self.args.qualification_root else None, ("CONTROL_QUALIFICATION_REPORT.json", "CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json")):
            if not _all_pass_receipts(Path(self.args.canary_root) if self.args.canary_root else None, ("DYNAMIC8_CONTROL_CANARY_REPORT.json", "DYNAMIC8_CONTROL_CANARY_AUDIT.json")):
                return "DYNAMIC8_CANARY" if self.args.canary_root and Path(self.args.canary_root).exists() else "PREPARATION"
            if not _all_pass_receipts(Path(self.args.r2a_root) if self.args.r2a_root else None, ("STAGE_V_CLOSURE_RECEIPT.json", "STAGE_V_COUNTERFACTUAL_AUDIT.json")):
                return "STAGE_V_R2A_AUDIT" if self.args.r2a_root and Path(self.args.r2a_root).exists() else "STAGE_V_R2A_RUNNING"
            if self.args.v2_root and not _all_pass_receipts(Path(self.args.v2_root), ("STAGE_V2_INDEPENDENT_AUDIT.json", "STAGE_V2_COMPLETE.json")):
                return "STAGE_V2_RUNNING" if Path(self.args.v2_root).exists() else "STAGE_V_CLOSED"
            if self.args.stage_o_root and not _all_pass_receipts(Path(self.args.stage_o_root), ("STAGE_O_INDEPENDENT_AUDIT.json", "STAGE_O_COMPLETE.json")):
                return "STAGE_O_RUNNING" if Path(self.args.stage_o_root).exists() else "STAGE_V2_RUNNING"
            if self.args.stage_o_root and _all_pass_receipts(Path(self.args.stage_o_root), ("STAGE_O_INDEPENDENT_AUDIT.json", "STAGE_O_COMPLETE.json")):
                return "GOAL_COMPLETE"
            return "STAGE_V2_RUNNING"
        return "QUALIFICATION_RUNNING" if self.args.qualification_root and Path(self.args.qualification_root).exists() else "PREPARATION"

    def _write(self, status: str, *, resource: Mapping[str, Any], hard: list[str], missing: list[str], phase: str) -> None:
        if status not in STATES or phase not in STATES:
            raise ValueError(f"unknown mainline monitor state: {status}/{phase}")
        self.heartbeat_count += 1
        formal_roots = []
        parent = Path(self.args.formal_root_parent).resolve()
        if parent.is_dir():
            formal_roots = sorted(str(path) for path in parent.glob(f"{FORMAL_PREFIX}*") if path.is_dir() and path != Path(self.args.old_root).resolve())
        payload = {
            "schema": SCHEMA, "status": status, "phase": phase, "monitor_pid": os.getpid(),
            "monitor_pgid": os.getpgid(0) if hasattr(os, "getpgid") else os.getpid(),
            "control_plane_mode": "LOCAL_AUTONOMOUS", "ssh_is_hard_stop": False,
            "source_commit": self.args.expected_source_commit, "source_tree": self.args.expected_source_tree,
            "old_root": str(Path(self.args.old_root).resolve()), "old_roots_reused": False,
            "formal_root_parent": str(parent), "formal_roots_observed": formal_roots,
            "formal_root_created": bool(formal_roots), "planned_parents": 40, "planned_branches": 2880,
            "excluded_gpus": sorted(self.args.excluded_gpus), "approved_gpus": resource.get("safe_gpus", []),
            "external_root_process_present": pid_alive(self.args.external_pid), "external_root_process_pid": self.args.external_pid,
            "external_root_process_terminated": False, "gpu5_touched": False,
            "hard_stop_reasons": hard, "missing_preparation_inputs": missing,
            "resource": dict(resource), "heartbeat_count": self.heartbeat_count,
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
            "updated_utc": utc_now(),
        }
        atomic_write_json(self.monitor_root / "STAGE_V_R2_MAINLINE_STATE.json", payload)
        atomic_write_json(self.monitor_root / "STAGE_V_R2_MAINLINE_HEARTBEAT.json", payload)

    def tick(self) -> str:
        hard, missing = self._verify_bindings()
        resource = self._resources()
        hard.extend(resource.get("hard_stop_reasons", []))
        if hard:
            self._write("HARD_STOP", resource=resource, hard=sorted(set(hard)), missing=missing, phase="HARD_STOP")
            return "HARD_STOP"
        if missing:
            self._write("PREPARATION", resource=resource, hard=[], missing=missing, phase="PREPARATION")
            return "PREPARATION"
        if resource.get("verdict") != "PASS":
            self._write(WAITING, resource=resource, hard=[], missing=[], phase=WAITING)
            atomic_write_json(self.monitor_root / "PRELAUNCH_WAITING_FOR_8_SAFE_GPUS.json", resource)
            return WAITING
        phase = self._phase()
        self._write(phase, resource=resource, hard=[], missing=[], phase=phase)
        return phase

    def run(self) -> int:
        self.lock.acquire(self.monitor_root)
        try:
            while True:
                status = self.tick()
                if self.args.once or status in {"HARD_STOP", "GOAL_COMPLETE"}:
                    return 1 if status == "HARD_STOP" else 0
                time.sleep(max(1.0, self.args.poll_seconds))
        finally:
            self.lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--monitor-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--formal-root-parent", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path)
    parser.add_argument("--old-manifest-sha256", default="")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--candidate-manifest-sha256", default="")
    parser.add_argument("--postmortem", type=Path)
    parser.add_argument("--postmortem-sha256", default="")
    parser.add_argument("--timeout-policy", type=Path)
    parser.add_argument("--timeout-policy-sha256", default="")
    parser.add_argument("--science-provenance", type=Path)
    parser.add_argument("--science-provenance-sha256", default="")
    parser.add_argument("--qualification-root", type=Path)
    parser.add_argument("--canary-root", type=Path)
    parser.add_argument("--r2a-root", type=Path)
    parser.add_argument("--v2-root", type=Path)
    parser.add_argument("--stage-o-root", type=Path)
    parser.add_argument("--required-gpus", type=int, default=8)
    parser.add_argument("--excluded-gpus", type=lambda value: [int(item) for item in value.split(",") if item], default=[5])
    parser.add_argument("--protected-pids", type=lambda value: [int(item) for item in value.split(",") if item], default=[])
    parser.add_argument("--external-pid", type=int, default=1895889)
    parser.add_argument("--canary-peak-mib", type=float, default=0)
    parser.add_argument("--minimum-ram-gib", type=float, default=128)
    parser.add_argument("--poll-seconds", type=float, default=300)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.required_gpus != 8 or 5 not in args.excluded_gpus:
        raise SystemExit("Stage V R2 requires exactly eight GPUs and GPU5 exclusion")
    if len(set(args.excluded_gpus)) != len(args.excluded_gpus):
        raise SystemExit("duplicate excluded GPU")
    return MainlineMonitor(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
