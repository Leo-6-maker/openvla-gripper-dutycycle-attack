#!/usr/bin/env python3
"""Run frozen post-HOLD V1.1 queues through the immutable corridor runner."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_spatial")
REPLICATES = ("A", "B")


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(f"{_sha(path)}  {path.name}\n", encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _safe_key(key: str) -> str:
    return key.replace("/", "__")


def _binding(value: Mapping[str, Any]) -> Path:
    path = Path(str(value.get("path", ""))).resolve()
    if not path.is_file() or _sha(path) != value.get("sha256"):
        raise ValueError(f"BOUND_FILE_INVALID:{path}")
    return path


def _gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=20,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GPU_QUERY_FAILED:{result.stderr.strip()}")
    rows = []
    for raw in result.stdout.splitlines():
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) != 3:
            raise RuntimeError(f"GPU_QUERY_ROW_INVALID:{raw}")
        rows.append({"index": int(parts[0]), "uuid": parts[1].lower().removeprefix("gpu-"), "memory_free_mib": int(parts[2])})
    return rows


def _compute_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=20,
    )
    return result.stdout.splitlines() if result.returncode == 0 else [f"QUERY_ERROR:{result.stderr.strip()}"]


def _status_pair(receipts: Mapping[str, Mapping[str, Any]]) -> tuple[str, bool]:
    statuses = tuple(str(receipts[rep].get("status", "MISSING")) for rep in REPLICATES)
    return "/".join(statuses), statuses == ("PASS", "PASS")


def _self_check() -> None:
    pair, stable = _status_pair({"A": {"status": "PASS"}, "B": {"status": "PASS"}})
    assert pair == "PASS/PASS" and stable
    pair, stable = _status_pair({"A": {"status": "PASS"}, "B": {"status": "CLEAN_FAILURE"}})
    assert pair == "PASS/CLEAN_FAILURE" and not stable
    print(json.dumps({"status": "PASS_SELF_CHECK", "rules": ["PASS_PAIR", "NONPASS_PAIR"]}, sort_keys=True))


class Dispatcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.outer_path = args.outer_protocol.resolve()
        self.inner_path = args.inner_protocol.resolve()
        self.auth_path = args.authorization.resolve()
        self.compat_path = args.compatibility_audit.resolve()
        self.outer, self.inner = _load(self.outer_path), _load(self.inner_path)
        self.auth, self.compat = _load(self.auth_path), _load(self.compat_path)
        self.lock = threading.Lock()
        self.gpu_condition = threading.Condition(self.lock)
        self.leased: set[int] = set()
        self.stop = threading.Event()
        self.root = args.output_root.resolve()
        self.results: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.stable = {suite: 0 for suite in SUITES}
        self.targets: dict[str, int] = {}
        self.queues: dict[str, list[dict[str, Any]]] = {}
        self.runtime_attempts: dict[str, Any] = {}
        self.launch: dict[str, Any] = {}
        self._validate()

    def _validate(self) -> None:
        if self.outer.get("schema") != "STAGE_V_M4_CORRIDOR_REPLENISHMENT_POST_32_OF_40_HOLD_V1_1" or self.outer.get("status") != "FROZEN_RUNTIME_AUTHORIZED" or self.outer.get("runtime_authorized") is not True:
            raise ValueError("OUTER_PROTOCOL_NOT_FROZEN_AUTHORIZED")
        if self.args.owner_basis != self.outer.get("owner_authorization_basis"):
            raise ValueError("OWNER_AUTHORIZATION_BASIS_MISMATCH")
        if self.outer.get("protected_counters") != COUNTERS or self.outer.get("operation", {}).get("outcomes_read") is not False:
            raise ValueError("OUTER_PROTECTED_BOUNDARY_INVALID")
        governance = self.outer.get("governance", {})
        expected_paths = {
            "inner_protocol_path": self.inner_path,
            "inner_authorization_path": self.auth_path,
            "predecessor_compatibility_audit_path": self.compat_path,
        }
        if any(Path(str(governance.get(key, ""))).resolve() != path for key, path in expected_paths.items()):
            raise ValueError("GOVERNANCE_PATH_MISMATCH")
        if self.inner.get("outer_protocol", {}).get("sha256") != _sha(self.outer_path) or Path(str(self.inner.get("outer_protocol", {}).get("path", ""))).resolve() != self.outer_path:
            raise ValueError("INNER_OUTER_BINDING_INVALID")
        if self.auth.get("status") != "PASS" or self.auth.get("protocol_sha256") != _sha(self.inner_path) or self.auth.get("protected_counters") != COUNTERS:
            raise ValueError("INNER_AUTHORIZATION_INVALID")
        if self.compat.get("status") != "PASS_PREDECESSOR_32_COMPATIBLE_WITH_POST_HOLD_V1_1" or self.compat.get("outer_protocol_sha256") != _sha(self.outer_path) or self.compat.get("new_protocol_sha256") != _sha(self.inner_path) or self.compat.get("new_authorization_sha256") != _sha(self.auth_path) or self.compat.get("failure_count") != 0 or self.compat.get("protected_counters") != COUNTERS:
            raise ValueError("PREDECESSOR_COMPATIBILITY_NOT_ESTABLISHED")

        sources = self.outer.get("reserve_sources", {})
        manifest_path = _binding(sources.get("candidate_manifest", {}))
        taxonomy_path = _binding(sources.get("taxonomy_audit", {}))
        attempt_path = _binding(sources.get("attempt_registry", {}))
        _binding(sources.get("compact_qualified_rows", {}))
        _binding(self.outer.get("predecessor", {}).get("terminal_report", {}))
        _binding(self.outer.get("predecessor", {}).get("pass_pass_inventory", {}))
        _binding({"path": self.outer["invalid_v1_launch"]["hold_report_path"], "sha256": self.outer["invalid_v1_launch"]["hold_report_sha256"]})
        manifest, taxonomy, attempts = _load(manifest_path), _load(taxonomy_path), _load(attempt_path)
        if manifest.get("schema") != "STAGE_V_M4_CORRIDOR_RESERVE_PARENT_MANIFEST_V1" or manifest.get("status") != "FROZEN" or manifest.get("candidate_count") != 22:
            raise ValueError("CANDIDATE_MANIFEST_INVALID")
        if taxonomy.get("status") != "PASS_STATIC_SUPPORTED_POOL" or taxonomy.get("counts") != {"ABSTAIN_STATIC": 0, "SUPPORTED": 25, "UNSUPPORTED_TAXONOMY": 1}:
            raise ValueError("STATIC_TAXONOMY_AUDIT_INVALID")
        if attempts.get("status") != "FROZEN_EXACT_43" or attempts.get("attempted_identity_count") != 43:
            raise ValueError("ATTEMPT_REGISTRY_INVALID")
        attempted = {str(row.get("canonical_parent_key")) for row in attempts.get("attempted_identities", [])}
        parents = manifest.get("parents", [])
        keys = [str(row.get("canonical_parent_key")) for row in parents]
        if len(keys) != 22 or len(set(keys)) != 22 or set(keys) & attempted:
            raise ValueError("CANDIDATE_ATTEMPT_FIREWALL_INVALID")
        qualification = self.outer.get("qualification", {})
        self.targets = {suite: int(qualification.get("target_stable_by_suite", {}).get(suite, -1)) for suite in SUITES}
        self.queues = {suite: [row for row in parents if row.get("suite") == suite] for suite in SUITES}
        if any([row["canonical_parent_key"] for row in self.queues[suite]] != qualification.get("queues", {}).get(suite) for suite in SUITES):
            raise ValueError("FROZEN_QUEUE_ORDER_MISMATCH")
        if any(self.targets[suite] < 1 or self.targets[suite] > len(self.queues[suite]) for suite in SUITES):
            raise ValueError("FROZEN_TARGET_INVALID")

        source = self.outer.get("source_binding", {})
        self.science_root = Path(str(source.get("science_worktree", ""))).resolve()
        self.runner = self.science_root / str(source.get("corridor_runner_path", ""))
        if _git(self.science_root, "rev-parse", "HEAD") != source.get("science_commit") or _git(self.science_root, "rev-parse", "HEAD^{tree}") != source.get("science_tree") or _git(self.science_root, "status", "--porcelain"):
            raise ValueError("SCIENCE_WORKTREE_BINDING_INVALID")
        if _sha(self.runner) != source.get("corridor_runner_sha256"):
            raise ValueError("SCIENCE_RUNNER_HASH_MISMATCH")
        if any(_sha(self.science_root / path) != expected for path, expected in source.get("science_files", {}).items()):
            raise ValueError("SCIENCE_IMPORT_CLOSURE_MISMATCH")
        runtime = self.outer.get("runtime_binding", {})
        self.python = Path(str(runtime.get("python_path", "")))
        self.snapshot = Path(str(runtime.get("official_snapshot", {}).get("path", "")))
        self.upstream = Path(str(runtime.get("upstream", {}).get("path", "")))
        self.model_root = Path(str(runtime.get("model_root", "")))
        if not self.python.is_file() or str(self.python.resolve()) != runtime.get("python_resolved_path") or _sha(self.python) != runtime.get("python_executable_sha256"):
            raise ValueError("OFFICIAL_PYTHON_BINDING_INVALID")
        for name, root, binding in (("snapshot", self.snapshot, runtime.get("official_snapshot", {})), ("upstream", self.upstream, runtime.get("upstream", {}))):
            if _git(root, "rev-parse", "HEAD") != binding.get("git_commit") or _git(root, "rev-parse", "HEAD^{tree}") != binding.get("git_tree") or _git(root, "status", "--porcelain"):
                raise ValueError(f"{name.upper()}_BINDING_INVALID")
        if any(not Path(str(binding.get("path", ""))).is_dir() for binding in runtime.get("models", {}).values()):
            raise ValueError("MODEL_PATH_BINDING_INVALID")
        if self.root != Path(str(self.outer.get("output", {}).get("root", ""))).resolve() or self.root.exists():
            raise ValueError(f"OUTPUT_ROOT_NOT_NEW_OR_MISMATCH:{self.root}")
        self.gpu_contract = self.outer.get("resource_contract", {})
        self.gpu_pool = [int(value) for value in self.gpu_contract.get("admitted_gpu_pool", [])]
        if self.gpu_pool != list(range(8)) or self.gpu_contract.get("free_memory_rule") != "STRICTLY_GREATER_THAN_20480_MIB_AT_LEASE":
            raise ValueError("GPU_RESOURCE_CONTRACT_INVALID")

    def eligible_gpus(self) -> list[dict[str, Any]]:
        threshold = int(self.gpu_contract["minimum_free_memory_mib_exclusive"])
        expected = {int(key): str(value).lower().removeprefix("gpu-") for key, value in self.gpu_contract["gpu_uuid_by_index"].items()}
        rows = [row for row in _gpu_inventory() if row["index"] in self.gpu_pool and row["uuid"] == expected.get(row["index"]) and row["memory_free_mib"] > threshold]
        return sorted(rows, key=lambda row: row["index"])

    def plan(self) -> dict[str, Any]:
        eligible = self.eligible_gpus()
        return {
            "schema": "STAGE_V_M4_POST_HOLD_V1_1_PLAN_V1",
            "status": "PASS_PLAN_ONLY" if len(eligible) >= 2 else "HOLD_RESOURCE_FEWER_THAN_TWO_ELIGIBLE_GPUS",
            "outer_protocol_sha256": _sha(self.outer_path),
            "inner_protocol_sha256": _sha(self.inner_path),
            "authorization_sha256": _sha(self.auth_path),
            "compatibility_audit_sha256": _sha(self.compat_path),
            "output_root": str(self.root),
            "queues": {suite: [row["canonical_parent_key"] for row in self.queues[suite]] for suite in SUITES},
            "targets": self.targets,
            "eligible_gpu_rows": eligible,
            "compute_processes_observed_non_gate_affecting": _compute_processes(),
            "maximum_concurrent_replicates": 6,
            "outcomes_read": False,
            "protected_counters": dict(COUNTERS),
        }

    def _acquire_pair(self) -> list[int] | None:
        while not self.stop.is_set():
            with self.gpu_condition:
                available = [row["index"] for row in self.eligible_gpus() if row["index"] not in self.leased]
                if len(available) >= 2:
                    pair = available[:2]
                    self.leased.update(pair)
                    return pair
                self.gpu_condition.wait(timeout=5)
        return None

    def _release_pair(self, gpus: list[int]) -> None:
        with self.gpu_condition:
            self.leased.difference_update(gpus)
            self.gpu_condition.notify_all()

    def _mark_attempt(self, parent: Mapping[str, Any], gpus: list[int]) -> None:
        key = str(parent["canonical_parent_key"])
        with self.lock:
            self.runtime_attempts["attempted_identities"].append({
                "canonical_parent_key": key,
                "suite": parent["suite"],
                "selection_rank": parent["selection_rank"],
                "status": "CORRIDOR_ATTEMPTED",
                "replicate_gpus": {"A": gpus[0], "B": gpus[1]},
                "marked_utc": _utc(),
            })
            self.runtime_attempts["attempted_identity_count"] = len(self.runtime_attempts["attempted_identities"])
            _write(self.root / "CORRIDOR_ATTEMPT_REGISTRY_RUNTIME.json", self.runtime_attempts)

    def _receipt(self, path: Path, key: str, replicate: str) -> dict[str, Any]:
        value = _load(path)
        source = self.outer["source_binding"]
        if value.get("schema") != "STAGE_V_M4_CORRIDOR_PREFLIGHT_V1" or value.get("canonical_parent_key") != key or value.get("replicate") != replicate:
            raise ValueError(f"RECEIPT_IDENTITY_INVALID:{key}:{replicate}")
        if value.get("source_commit") != source["science_commit"] or value.get("source_tree") != source["science_tree"] or value.get("protected_counters") != COUNTERS or value.get("outcomes_read") is not False or value.get("source_artifacts_modified") is not False:
            raise ValueError(f"RECEIPT_BOUNDARY_INVALID:{key}:{replicate}")
        if value.get("status") not in {"PASS", "CLEAN_FAILURE", "INELIGIBLE"}:
            raise ValueError(f"RECEIPT_STATUS_INVALID:{key}:{replicate}")
        return value

    def _run_pair(self, parent: Mapping[str, Any], gpus: list[int]) -> dict[str, Any]:
        key = str(parent["canonical_parent_key"])
        base = self.root / "parents" / _safe_key(key)
        processes: dict[str, tuple[subprocess.Popen[str], Any, Path, Path, list[str]]] = {}
        rows: dict[str, dict[str, Any]] = {}
        for replicate, gpu in zip(REPLICATES, gpus):
            output = base / replicate
            log = self.root / "logs" / f"{_safe_key(key)}__{replicate}.log"
            output.mkdir(parents=True, exist_ok=False)
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(self.python), str(self.runner), "--protocol", str(self.inner_path), "--authorization", str(self.auth_path),
                "--parent-key", key, "--output-dir", str(output), "--official-snapshot-root", str(self.snapshot),
                "--upstream-root", str(self.upstream), "--model-root", str(self.model_root), "--gpu", str(gpu),
                "--source-commit", self.outer["source_binding"]["science_commit"], "--source-tree", self.outer["source_binding"]["science_tree"],
                "--replicate", replicate,
            ]
            handle = log.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(command, cwd=self.science_root, stdout=handle, stderr=subprocess.STDOUT, text=True)
                processes[replicate] = (process, handle, output, log, command)
            except OSError as exc:
                handle.close()
                rows[replicate] = {"status": "RUNNER_ERROR", "reason": f"{type(exc).__name__}:{exc}", "gpu": gpu, "command": command, "log": str(log)}
        for replicate, (process, handle, output, log, command) in processes.items():
            code = process.wait()
            handle.close()
            receipt_path = output / "M4_CORRIDOR_PREFLIGHT.json"
            if code != 0 or not receipt_path.is_file():
                rows[replicate] = {"status": "RUNNER_ERROR", "reason": f"RETURN_CODE_{code}_RECEIPT_{'PRESENT' if receipt_path.is_file() else 'MISSING'}", "gpu": gpus[REPLICATES.index(replicate)], "command": command, "log": str(log), "return_code": code}
                continue
            try:
                receipt = self._receipt(receipt_path, key, replicate)
                rows[replicate] = {"status": receipt["status"], "reason": receipt.get("reason"), "gpu": gpus[REPLICATES.index(replicate)], "command": command, "log": str(log), "return_code": code, "receipt": str(receipt_path), "receipt_sha256": _sha(receipt_path), "trajectory_sha256": receipt.get("trajectory_sha256"), "probe_count": receipt.get("probe_count")}
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                rows[replicate] = {"status": "RUNNER_ERROR", "reason": f"{type(exc).__name__}:{exc}", "gpu": gpus[REPLICATES.index(replicate)], "command": command, "log": str(log), "return_code": code}
        for replicate in REPLICATES:
            rows.setdefault(replicate, {"status": "RUNNER_ERROR", "reason": "PROCESS_NOT_STARTED", "gpu": gpus[REPLICATES.index(replicate)]})
        pair_status, stable = _status_pair(rows)
        result = {
            "schema": "STAGE_V_M4_POST_HOLD_PAIR_RECONCILIATION_V1_1",
            "canonical_parent_key": key,
            "suite": parent["suite"],
            "selection_rank": parent["selection_rank"],
            "replicates": rows,
            "status_pair": pair_status,
            "stable_pass_pass": stable,
            "structural_failure": any(row["status"] == "RUNNER_ERROR" for row in rows.values()),
            "reconciled_utc": _utc(),
            "outcomes_read": False,
            "protected_counters": dict(COUNTERS),
        }
        path = self.root / "pair_reconciliation" / f"{_safe_key(key)}.json"
        _write(path, result)
        _sidecar(path)
        return result

    def _suite_worker(self, suite: str) -> None:
        for parent in self.queues[suite]:
            if self.stop.is_set() or self.stable[suite] >= self.targets[suite]:
                return
            gpus = self._acquire_pair()
            if gpus is None:
                return
            try:
                self._mark_attempt(parent, gpus)
                result = self._run_pair(parent, gpus)
            finally:
                self._release_pair(gpus)
            with self.lock:
                self.results.append(result)
                if result["stable_pass_pass"]:
                    self.stable[suite] += 1
                if result["structural_failure"]:
                    self.failures.append(f"STRUCTURAL_PAIR_FAILURE:{result['canonical_parent_key']}:{result['status_pair']}")
                    self.stop.set()
                self.launch["completed_pairs"] = len(self.results)
                self.launch["stable_by_suite"] = dict(self.stable)
                self.launch["results"] = list(self.results)
                _write(self.root / "POST_HOLD_V1_1_LAUNCH_MANIFEST.json", self.launch)
            if result["stable_pass_pass"] and self.stable[suite] >= self.targets[suite]:
                return
        if self.stable[suite] < self.targets[suite]:
            with self.lock:
                self.failures.append(f"POOL_EXHAUSTED:{suite}:{self.stable[suite]}/{self.targets[suite]}")
                self.stop.set()

    def run(self) -> int:
        plan = self.plan()
        if not str(plan["status"]).startswith("PASS_"):
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 3
        self.root.mkdir(parents=True, exist_ok=False)
        prelaunch = {**plan, "schema": "STAGE_V_M4_POST_HOLD_V1_1_PRELAUNCH_BINDING_V1", "status": "PASS_PRELAUNCH", "created_utc": _utc()}
        _write(self.root / "PRELAUNCH_BINDING.json", prelaunch)
        _sidecar(self.root / "PRELAUNCH_BINDING.json")
        self.runtime_attempts = {
            "schema": "STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_RUNTIME_V1_1",
            "status": "ACTIVE",
            "prelaunch_attempt_registry_sha256": self.outer["reserve_sources"]["attempt_registry"]["sha256"],
            "attempted_identity_count": 0,
            "attempted_identities": [],
            "outcomes_read": False,
            "protected_counters": dict(COUNTERS),
        }
        _write(self.root / "CORRIDOR_ATTEMPT_REGISTRY_RUNTIME.json", self.runtime_attempts)
        self.launch = {
            "schema": "STAGE_V_M4_POST_HOLD_V1_1_LAUNCH_V1",
            "status": "RUNNING",
            "started_utc": _utc(),
            "outer_protocol_sha256": _sha(self.outer_path),
            "inner_protocol_sha256": _sha(self.inner_path),
            "authorization_sha256": _sha(self.auth_path),
            "compatibility_audit_sha256": _sha(self.compat_path),
            "eligible_gpu_pool_at_prelaunch": [row["index"] for row in plan["eligible_gpu_rows"]],
            "maximum_concurrent_replicates": 6,
            "completed_pairs": 0,
            "stable_by_suite": dict(self.stable),
            "targets": self.targets,
            "results": [],
            "outcomes_read": False,
            "protected_counters": dict(COUNTERS),
        }
        _write(self.root / "POST_HOLD_V1_1_LAUNCH_MANIFEST.json", self.launch)
        with ThreadPoolExecutor(max_workers=len(SUITES), thread_name_prefix="post-hold-suite") as pool:
            futures = [pool.submit(self._suite_worker, suite) for suite in SUITES]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:  # fail closed after already-started pairs finish
                    with self.lock:
                        self.failures.append(f"DISPATCHER_EXCEPTION:{type(exc).__name__}:{exc}")
                        self.stop.set()
        targets_met = all(self.stable[suite] >= self.targets[suite] for suite in SUITES)
        status = "PASS_POST_HOLD_CORRIDOR_TARGETS_REACHED" if targets_met and not self.failures else ("HOLD_STRUCTURAL_RUNTIME_FAILURE" if any(item.startswith(("STRUCTURAL", "DISPATCHER")) for item in self.failures) else "HOLD_POST_HOLD_CORRIDOR_POOL_EXHAUSTED")
        self.runtime_attempts.update({"status": "SEALED", "sealed_utc": _utc()})
        _write(self.root / "CORRIDOR_ATTEMPT_REGISTRY_RUNTIME.json", self.runtime_attempts)
        terminal = {
            "schema": "STAGE_V_M4_POST_HOLD_V1_1_RUNTIME_RECONCILIATION_V1",
            "status": status,
            "terminal": True,
            "sealed": True,
            "immutable": True,
            "retry_forbidden": True,
            "consumable_for_composite_reconciliation": status == "PASS_POST_HOLD_CORRIDOR_TARGETS_REACHED",
            "stable_by_suite": dict(self.stable),
            "target_stable_by_suite": dict(self.targets),
            "attempted_identity_count": self.runtime_attempts["attempted_identity_count"],
            "pair_count": len(self.results),
            "pairs": list(self.results),
            "failures": list(self.failures),
            "outcomes_read": False,
            "intervention_executed": False,
            "protected_counters": dict(COUNTERS),
            "next_action": "INDEPENDENT_COMPOSITE_RECONCILIATION" if status.startswith("PASS_") else "SEAL_HANDOFF_STOP_NO_RERUN",
        }
        terminal_path = self.root / "STAGE_V_M4_POST_HOLD_V1_1_RUNTIME_RECONCILIATION.json"
        _write(terminal_path, terminal)
        _sidecar(terminal_path)
        self.launch.update({"status": status, "completed_utc": _utc(), "stable_by_suite": dict(self.stable), "failures": list(self.failures)})
        _write(self.root / "POST_HOLD_V1_1_LAUNCH_MANIFEST.json", self.launch)
        self._seal_root()
        print(json.dumps({"status": status, "stable_by_suite": self.stable, "attempted_identity_count": self.runtime_attempts["attempted_identity_count"], "root": str(self.root)}, sort_keys=True))
        return 0 if status.startswith("PASS_") else 2

    def _seal_root(self) -> None:
        sums = self.root / "SHA256SUMS"
        seal = self.root / "ROOT_SEAL.sha256"
        files = sorted(path for path in self.root.rglob("*") if path.is_file() and path not in {sums, seal})
        sums.write_text("".join(f"{_sha(path)}  {path.relative_to(self.root).as_posix()}\n" for path in files), encoding="utf-8")
        seal.write_text(f"{_sha(sums)}  SHA256SUMS\n", encoding="utf-8")
        for path in sorted(self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        self.root.chmod(stat.S_IMODE(self.root.stat().st_mode) & ~0o222)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--self-check"]:
        _self_check()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-protocol", type=Path, required=True)
    parser.add_argument("--inner-protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--compatibility-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--owner-basis", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    dispatcher = Dispatcher(args)
    if args.plan_only:
        plan = dispatcher.plan()
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if str(plan["status"]).startswith("PASS_") else 3
    return dispatcher.run()


if __name__ == "__main__":
    raise SystemExit(main())
