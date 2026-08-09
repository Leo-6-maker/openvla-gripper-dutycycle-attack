from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5 import audit_stage_v_r2_control_qualification_v2 as auditor
from scripts.detector_v5 import freeze_stage_q2_protocol as freezer
from scripts.detector_v5 import run_stage_v_r2_q2_control_qualification as q2
from scripts.detector_v5 import run_stage_v_r2_q2_supervisor as q2_supervisor
from scripts.monitoring import materialize_stage_v_r2_next_plan as materializer


SUITES = q2.EXPECTED_SUITES
SOURCE_COMMIT = "c" * 40
SOURCE_TREE = "t" * 40


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def candidate(suite: str, index: int) -> dict[str, object]:
    return {
        "canonical_parent_key": f"{suite}/task_{index:02d}/state_48",
        "suite": suite, "task_index": index, "state_index": 48,
        "legacy_g10_test_only": True, "source_artifact_read": False, "old_artifacts_reused": False,
    }


def result(key: str, *, clean_success: bool = True, terminal: str = "a", horizon: bool = True, identity: str = "identity") -> dict[str, object]:
    return {
        "schema": "STAGE_Q2_CLEAN_CONTROL_RESULT_V1", "status": "PASS" if clean_success else "TASK_FAILURE",
        "exit_code": 0, "process_exit_code": 0, "clean_success": clean_success,
        "snapshot_restore_valid": True, "task_identity_valid": True, "runtime_valid": True,
        "metrics_finite": True, "artifact_validation_pass": True, "old_artifacts_reused": False,
        "source_commit": SOURCE_COMMIT, "source_tree": SOURCE_TREE, "canonical_parent_key": key,
        "key_state_identity_sha256": identity, "terminal_state_sha256": terminal,
        "remaining_horizon_complete": horizon, "eval160_reads": 0, "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
    }


def test_q2_qualification_does_not_gate_on_terminal_hash_or_horizon() -> None:
    row = q2._ranked([candidate("libero_goal", 0)], q2.DEFAULT_SALT)[0]
    a = result(row["canonical_parent_key"], terminal="a", horizon=False)
    b = result(row["canonical_parent_key"], terminal="b", horizon=True)
    ok, classification, errors = q2.qualify_pair(row, a, b, True, True, SOURCE_COMMIT, SOURCE_TREE)
    assert (ok, classification, errors) == (True, "QUALIFIED", [])


def test_q2_clean_failure_is_not_infrastructure_retry() -> None:
    row = candidate("libero_goal", 0)
    a = result(row["canonical_parent_key"], clean_success=True)
    b = result(row["canonical_parent_key"], clean_success=False)
    ok, classification, errors = q2.qualify_pair(row, a, b, True, True, SOURCE_COMMIT, SOURCE_TREE)
    assert not ok
    assert classification == "CLEAN_REPEATABILITY_FAIL_A_SUCCESS_B_FAIL"
    assert "B_CLEAN_SUCCESS_FALSE" in errors


def test_q2_initial_state_identity_mismatch_is_engineering_invalid() -> None:
    row = q2._ranked([candidate("libero_goal", 0)], q2.DEFAULT_SALT)[0]
    a = result(row["canonical_parent_key"])
    b = result(row["canonical_parent_key"])
    b["key_state_identity_sha256"] = "different-identity"
    producer = q2.qualify_pair(row, a, b, True, True, SOURCE_COMMIT, SOURCE_TREE)
    independent = auditor._pair(row, {"A": a, "B": b}, {"A": True, "B": True})
    assert producer[0] is False and producer[1] == "ENGINEERING_INVALID"
    assert independent[0] is False and independent[1] == "ENGINEERING_INVALID"


def test_q2_missing_replicate_is_engineering_failure_not_producer_crash() -> None:
    row = q2._ranked([candidate("libero_goal", 0)], q2.DEFAULT_SALT)[0]
    key = row["canonical_parent_key"]
    record = q2._build_parent_record(
        row,
        {"A": {"result": result(key), "process_exit_code": 0, "engineering_valid": True, "attempts": []}, "B": None},
        SOURCE_COMMIT,
        SOURCE_TREE,
    )
    assert record["qualified"] is False
    assert record["classification"] == "ENGINEERING_INVALID"


def test_q2_protocol_freeze_binds_candidate_and_q1_forensics(tmp_path: Path) -> None:
    candidates = [candidate(suite, 0) for suite in SUITES]
    universe = {
        "schema": "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1", "candidate_count": len(candidates),
        "candidates": candidates, "candidates_per_suite": 1,
        "selection_frozen_before_new_rollouts": True,
        "gates": {"eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0},
    }
    universe_path = tmp_path / "universe.json"
    q1_matrix = tmp_path / "q1.json"
    q1_semantic = tmp_path / "q1_semantic.json"
    write_json(universe_path, universe)
    write_json(q1_matrix, {"schema": "Q1"})
    write_json(q1_semantic, {"schema": "Q1_SEMANTIC"})
    protocol = freezer.freeze(SimpleNamespace(
        output_dir=tmp_path / "protocol", candidate_universe=universe_path,
        q1_matrix=q1_matrix, q1_semantic_audit=q1_semantic,
        source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE,
        expected_candidate_sha256=hashlib.sha256(universe_path.read_bytes()).hexdigest(),
    ))
    assert protocol["status"] == "FROZEN"
    assert protocol["approved_gpus"] == [0, 1, 2, 3, 4, 6, 7]
    assert protocol["worker_count"] == 7
    assert protocol["gpu5_authorized"] is False
    assert (tmp_path / "protocol" / "STAGE_Q2_PROTOCOL.sha256").is_file()


def test_independent_audit_accepts_descriptive_hash_and_horizon_mismatch(tmp_path: Path) -> None:
    candidates = [candidate(suite, index) for suite in SUITES for index in range(20)]
    universe = {
        "schema": "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1", "candidate_count": len(candidates),
        "candidates": candidates, "candidates_per_suite": 20,
        "selection_frozen_before_new_rollouts": True,
        "gates": {"eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0},
    }
    universe_path = tmp_path / "universe.json"
    write_json(universe_path, universe)
    protocol_dir = tmp_path / "protocol"
    q1_matrix = tmp_path / "q1.json"
    q1_semantic = tmp_path / "q1_semantic.json"
    write_json(q1_matrix, {"schema": "Q1"})
    write_json(q1_semantic, {"schema": "Q1_SEMANTIC"})
    freezer.freeze(SimpleNamespace(
        output_dir=protocol_dir, candidate_universe=universe_path,
        q1_matrix=q1_matrix, q1_semantic_audit=q1_semantic,
        source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE,
        expected_candidate_sha256=hashlib.sha256(universe_path.read_bytes()).hexdigest(),
    ))
    root = tmp_path / "q2"
    root.mkdir()
    rows = []
    for raw in auditor._ranked(candidates):
        key = str(raw["canonical_parent_key"])
        clean_ok = int(raw["task_index"]) != 0
        base = root / "qualification" / str(raw["suite"]) / key.replace("/", "__")
        dirs = {}
        reps = {}
        for replicate, terminal, horizon in (("A", "hash-a", False), ("B", "hash-b", True)):
            output = base / replicate / "attempt_01"
            output.mkdir(parents=True)
            actual = result(key, clean_success=clean_ok, terminal=terminal, horizon=horizon)
            write_json(output / "CONTROL_RESULT.json", actual)
            dirs[replicate] = str(output)
            reps[replicate] = {**actual, "process_exit_code": 0}
        rows.append({
            "schema": "STAGE_Q2_CONTROL_QUALIFICATION_ROW_V1", **raw,
            "replicates": reps, "replicate_output_dirs": dirs,
            "replicate_attempts": {replicate: [{"attempt": 1, "output_dir": dirs[replicate]}] for replicate in ("A", "B")},
            "qualified": clean_ok, "classification": "QUALIFIED" if clean_ok else "CLEAN_REPEATABILITY_FAIL_BOTH_FAIL",
        })
    report = {
        "schema": "STAGE_Q2_CONTROL_QUALIFICATION_REPORT_V1", "status": "PASS",
        "protocol_sha256": hashlib.sha256((protocol_dir / "STAGE_Q2_PROTOCOL.json").read_bytes()).hexdigest(),
        "candidate_universe_sha256": hashlib.sha256(universe_path.read_bytes()).hexdigest(),
        "source_commit": SOURCE_COMMIT, "source_tree": SOURCE_TREE,
        "gpus": [0, 1, 2, 3, 4, 6, 7], "worker_count": 7,
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
    }
    report_path = root / "Q2_CONTROL_QUALIFICATION_REPORT.json"
    rows_path = root / "Q2_CONTROL_QUALIFICATION_ROWS.jsonl"
    write_json(report_path, report)
    rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    audited = auditor.audit(SimpleNamespace(
        output_dir=root, protocol=protocol_dir / "STAGE_Q2_PROTOCOL.json",
        candidate_universe=universe_path, report=report_path, rows=rows_path,
        source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE,
    ))
    assert audited["verdict"] == "PASS"
    assert audited["classifications"]["CLEAN_REPEATABILITY_FAIL_BOTH_FAIL"] == 4
    assert audited["terminal_state_sha256_gate_used"] is False
    assert audited["remaining_horizon_complete_gate_used"] is False
    assert (root / "Q2_PARENT_MANIFEST_A.json").is_file()
    assert (root / "STAGE_V_FORMAL_PARENT_MANIFEST_V1.json").is_file()


def test_independent_audit_rejects_initial_state_identity_mismatch(tmp_path: Path) -> None:
    candidates = [candidate(suite, index) for suite in SUITES for index in range(10)]
    universe = {
        "schema": "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1", "candidate_count": len(candidates),
        "candidates": candidates, "candidates_per_suite": 10,
        "selection_frozen_before_new_rollouts": True,
        "gates": {"eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0},
    }
    universe_path = tmp_path / "universe.json"
    protocol_dir = tmp_path / "protocol"
    q1_matrix = tmp_path / "q1.json"
    q1_semantic = tmp_path / "q1_semantic.json"
    write_json(universe_path, universe)
    write_json(q1_matrix, {"schema": "Q1"})
    write_json(q1_semantic, {"schema": "Q1_SEMANTIC"})
    freezer.freeze(SimpleNamespace(
        output_dir=protocol_dir, candidate_universe=universe_path,
        q1_matrix=q1_matrix, q1_semantic_audit=q1_semantic,
        source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE,
        expected_candidate_sha256=hashlib.sha256(universe_path.read_bytes()).hexdigest(),
    ))
    root = tmp_path / "q2"
    root.mkdir()
    rows = []
    for raw in auditor._ranked(candidates):
        key = str(raw["canonical_parent_key"])
        mismatch = key == "libero_goal/task_00/state_48"
        base = root / "qualification" / str(raw["suite"]) / key.replace("/", "__")
        dirs, reps = {}, {}
        for replicate in ("A", "B"):
            output = base / replicate / "attempt_01"
            output.mkdir(parents=True)
            actual = result(key, identity="other" if mismatch and replicate == "B" else "identity")
            write_json(output / "CONTROL_RESULT.json", actual)
            dirs[replicate] = str(output)
            reps[replicate] = {**actual, "process_exit_code": 0}
        rows.append({
            "schema": "STAGE_Q2_CONTROL_QUALIFICATION_ROW_V1", **raw,
            "replicates": reps, "replicate_output_dirs": dirs,
            "replicate_attempts": {replicate: [{"attempt": 1, "output_dir": dirs[replicate]}] for replicate in ("A", "B")},
            "qualified": not mismatch,
            "classification": "ENGINEERING_INVALID" if mismatch else "QUALIFIED",
        })
    report = {
        "schema": "STAGE_Q2_CONTROL_QUALIFICATION_REPORT_V1", "status": "PASS",
        "protocol_sha256": hashlib.sha256((protocol_dir / "STAGE_Q2_PROTOCOL.json").read_bytes()).hexdigest(),
        "candidate_universe_sha256": hashlib.sha256(universe_path.read_bytes()).hexdigest(),
        "source_commit": SOURCE_COMMIT, "source_tree": SOURCE_TREE,
        "gpus": [0, 1, 2, 3, 4, 6, 7], "worker_count": 7,
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
    }
    report_path = root / "Q2_CONTROL_QUALIFICATION_REPORT.json"
    rows_path = root / "Q2_CONTROL_QUALIFICATION_ROWS.jsonl"
    write_json(report_path, report)
    rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    audited = auditor.audit(SimpleNamespace(
        output_dir=root, protocol=protocol_dir / "STAGE_Q2_PROTOCOL.json",
        candidate_universe=universe_path, report=report_path, rows=rows_path,
        source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE,
    ))
    assert audited["verdict"] == "FAIL"
    assert audited["classifications"]["ENGINEERING_INVALID"] == 1
    assert any("AB_INITIAL_STATE_IDENTITY_MISMATCH" in error for error in audited["errors"])


def test_q2_producer_uses_fresh_queue_and_reaches_quota(tmp_path: Path, monkeypatch) -> None:
    candidates = [candidate(suite, 0) for suite in SUITES]
    universe = {
        "schema": "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1", "candidate_count": len(candidates),
        "candidates": candidates, "candidates_per_suite": 1,
        "selection_frozen_before_new_rollouts": True,
        "gates": {"eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0},
    }
    universe_path = tmp_path / "universe.json"
    write_json(universe_path, universe)
    protocol_dir = tmp_path / "protocol"
    q1_matrix = tmp_path / "q1.json"
    q1_semantic = tmp_path / "q1_semantic.json"
    write_json(q1_matrix, {"schema": "Q1"})
    write_json(q1_semantic, {"schema": "Q1_SEMANTIC"})
    freezer.freeze(SimpleNamespace(
        output_dir=protocol_dir, candidate_universe=universe_path,
        q1_matrix=q1_matrix, q1_semantic_audit=q1_semantic,
        source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE,
        expected_candidate_sha256=hashlib.sha256(universe_path.read_bytes()).hexdigest(),
    ))

    def fake_run_once(template: str, *, candidate_path: Path, output_dir: Path, replicate: str,
                      source_commit: str, source_tree: str, gpu: int):
        key = json.loads(candidate_path.read_text(encoding="utf-8"))["canonical_parent_key"]
        payload = result(key)
        write_json(output_dir / "CONTROL_RESULT.json", payload)
        return 0, payload

    monkeypatch.setattr(q2, "_run_once", fake_run_once)
    args = SimpleNamespace(
        protocol=protocol_dir / "STAGE_Q2_PROTOCOL.json", candidate_universe=universe_path,
        output_dir=tmp_path / "run", runner_command="clean", source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE, source_clean_root=str(tmp_path / "clean"), salt=q2.DEFAULT_SALT,
        gpus="0,1,2,3,4,6,7", initial_per_suite=1, batch_size=1, target_per_suite=1,
        max_infrastructure_retries=1,
    )
    report, rows = q2.qualify(args)
    assert report["status"] == "PASS"
    assert report["evaluated_rows"] == 4
    assert all(row["qualified"] is True for row in rows)


def test_q2_pass_materializes_c0_from_full_universe(tmp_path: Path) -> None:
    candidates = [candidate(suite, index) for suite in SUITES for index in range(12)]
    universe = {
        "schema": "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1", "candidate_count": len(candidates),
        "candidates": candidates, "candidates_per_suite": 12,
        "gates": {"eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0},
    }
    candidate_path = tmp_path / "candidate.json"
    write_json(candidate_path, universe)
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    write_json(qualification / "Q2_CONTROL_QUALIFICATION_REPORT.json", {
        "status": "PASS", "source_commit": SOURCE_COMMIT, "source_tree": SOURCE_TREE,
        "qualified_by_suite": {suite: 10 for suite in SUITES},
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
    })
    write_json(qualification / "Q2_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json", {"verdict": "PASS"})
    formal_rows = candidates[:10] + candidates[12:22] + candidates[24:34] + candidates[36:46]
    write_json(qualification / "Q2_PARENT_MANIFEST_A.json", {
        "schema": "STAGE_Q2_PARENT_MANIFEST_A_V1", "status": "PASS", "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE, "selected_count": 40, "selected_parents": formal_rows,
    })
    science = tmp_path / "science.json"
    write_json(science, {"status": "PASS"})
    plan, plan_path, diagnostic_path = materializer.build_c0_plan(
        repo_root=Path(__file__).resolve().parents[2], state_root=tmp_path / "state",
        qualification_root=qualification, candidate_manifest=candidate_path, science_provenance=science,
        source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE, python_executable="python", external_pid=0,
        allow_gpu5=True,
    )
    assert plan["stage"] == "C0"
    assert len(json.loads(diagnostic_path.read_text(encoding="utf-8"))["selected_parents"]) == 8
    assert plan_path.is_file()
    assert plan["resource_policy"]["gpu5_authorized"] is True
    assert "--allow-gpu5" in plan["command_template"]
    assert "--allow-gpu5" in plan["audit_command_template"]


def test_q2_supervisor_heartbeat_is_local_and_keeps_external_process_untouched(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"candidate_universe_count": 120})
    state = tmp_path / "state"
    run = tmp_path / "run"
    state.mkdir()
    run.mkdir()
    supervisor = q2_supervisor.Q2Supervisor(SimpleNamespace(
        state_root=state, run_root=run, repo_root=tmp_path, protocol=protocol,
        candidate_universe=tmp_path / "candidate.json", source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE, gpus=[0], external_pid=99999999, min_available_ram_gib=0,
        min_free_memory_mib=0, gpu_query_command="false", python_executable="python",
        producer_script=tmp_path / "producer.py", producer_args=[], auditor_script=tmp_path / "audit.py",
        lock_path=tmp_path / "lock", expected_candidate_sha256="", poll_seconds=1, audit_timeout=1,
    ))
    supervisor.last_resource = {"queue": {"active_workers": [], "progress": {}}, "resource_errors": [], "gpu_memory": [], "gpu_xid_status": "NOT_CHECKED"}
    supervisor._heartbeat(supervisor.last_resource)
    heartbeat = json.loads((state / "Q2_LOCAL_HEARTBEAT.json").read_text(encoding="utf-8"))
    assert heartbeat["control_plane_mode"] == "LOCAL_AUTONOMOUS"
    assert heartbeat["ssh_is_hard_stop"] is False
    assert heartbeat["external_root_process_terminated"] is False
    assert heartbeat["eval160_reads"] == 0


def test_q2_supervisor_accepts_stale_lock_audit_only(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip()
    candidate = tmp_path / "candidate.json"
    write_json(candidate, {"schema": "candidate"})
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {
        "schema": "STAGE_Q2_PROTOCOL_V1", "status": "FROZEN", "source_commit": commit, "source_tree": tree,
        "candidate_universe_sha256": q2_supervisor.sha256_file(candidate),
        "approved_gpus": [0, 1, 2, 3, 4, 6, 7], "worker_count": 7, "gpu5_authorized": False,
    })
    state = tmp_path / "state"
    state.mkdir()
    write_json(state / "STALE_LOCK_AUDIT.json", {"schema": "STAGE_V_STALE_LOCK_AUDIT_V1"})
    supervisor = q2_supervisor.Q2Supervisor(SimpleNamespace(
        state_root=state, run_root=tmp_path / "run", repo_root=repo, protocol=protocol,
        candidate_universe=candidate, source_commit=commit, source_tree=tree,
        expected_candidate_sha256=q2_supervisor.sha256_file(candidate),
        gpus=[0, 1, 2, 3, 4, 6, 7],
    ))
    supervisor._prepare()
    assert (state / "STALE_LOCK_AUDIT.json").is_file()
    assert (state / "SUPERVISOR_START.json").is_file()


def test_q2_supervisor_queue_reports_duplicate_gpu_assignments(tmp_path: Path) -> None:
    db = tmp_path / "queue.sqlite"
    connection = sqlite3.connect(db)
    connection.executescript("""
        CREATE TABLE run_meta (state TEXT, updated_at TEXT);
        CREATE TABLE tasks (cell_id TEXT, parent_id TEXT, suite TEXT, arm TEXT, state TEXT, attempt_count INTEGER);
        CREATE TABLE attempts (attempt_id TEXT, cell_id TEXT, pid INTEGER, gpu_id INTEGER, worker_id TEXT, heartbeat_at TEXT, output_dir TEXT);
        INSERT INTO run_meta VALUES ('ACTIVE', '2026-08-07T00:00:00+00:00');
        INSERT INTO tasks VALUES ('a', 'p1', 'libero_goal', 'A', 'RUNNING', 1);
        INSERT INTO tasks VALUES ('b', 'p2', 'libero_goal', 'B', 'RUNNING', 1);
        INSERT INTO attempts VALUES ('aa', 'a', 10, 0, 'w0', 'now', 'a');
        INSERT INTO attempts VALUES ('bb', 'b', 11, 0, 'w1', 'now', 'b');
    """)
    connection.commit()
    connection.close()
    snapshot = q2_supervisor._queue_snapshot(db)
    assert len(snapshot["active_workers"]) == 2
    assert [row["gpu_id"] for row in snapshot["active_workers"]] == [0, 0]
