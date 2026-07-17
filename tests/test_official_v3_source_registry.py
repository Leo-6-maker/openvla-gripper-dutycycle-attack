import csv
import json
from pathlib import Path

from gripper_attack.official_v3_contract import audit_artifact, json_sha, load_contract, sha256_file
from official_v3.audit_official_v3_incremental_snapshot import audit_snapshot
from official_v3.audit_official_v3_worker_strata import audit_worker_strata
from official_v3.build_official_v3_formal_registry import _read_stale_recovery_audit, build_registry, write_registry


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/OFFICIAL_V3_SOURCE_CONTRACT_V1.json"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _build_artifact(root: Path, *, key="libero_object/task_00/state_00", success=True, provenance="A_CURRENT_HEAD_CLEAN_START_VERIFIED", generation=1):
    contract = load_contract(CONTRACT_PATH)
    suite, task, state = key.split("/")[0], int(key.split("/")[1].split("_")[1]), int(key.split("/")[2].split("_")[1])
    model_sha = "1" * 64
    processor_sha = "2" * 64
    protocol_sha = "3" * 64
    worker = {
        "slot_id": "slot_gpu0", "gpu_id": 0, "pid": 123, "started_at": "2026-07-16T00:00:00Z",
        "collector_head": "4" * 40, "worktree_clean": True, "worker_script_sha256": "5" * 64,
        "adapter_sha256": "6" * 64, "protocol_sha256": protocol_sha, "config_sha256": "7" * 64,
        "queue_epoch_sha256": "8" * 64, "model_tree_sha256": model_sha, "processor_tree_sha256": processor_sha,
        "python_version": "3.10", "torch_version": "2", "transformers_version": "4",
        "supervisor_archive_commit": "9" * 40, "provenance_class": provenance,
        "first_canary_pass": True,
    }
    meta = {
        "schema": "OPENVLA_OFFICIAL_CLEAN_EPISODE_V3", "condition": "CLEAN", "runtime_valid": True,
        "suite": suite, "task_idx": task, "state_id": state, "canonical_parent_key": key,
        "split": "FIT_TRAIN" if state < 20 else "FIT_DEV", "official_horizon": contract["official_horizons"][suite],
        "num_steps_wait": 10, "success": success, "env_success": success,
        "official_execution_adapter": "OfficialOpenVLAActionAdapter.predict_action",
        "generation_passes_per_step": 1, "feature_names_25d": contract["feature_names_25d"],
        "policy_intent_feature_names_9d": contract["policy_intent_feature_names_9d"],
        "initial_state_sha256": "a" * 64, "model_tree_sha256": model_sha,
        "processor_tokenizer_sha256": processor_sha, "protocol_sha256": protocol_sha,
        "worker_start_git_head": worker["collector_head"], "worker_start_script_sha256": worker["worker_script_sha256"],
        "worker_start_adapter_sha256": worker["adapter_sha256"], "worker_start_protocol_sha256": protocol_sha,
        "worker_start_model_tree_sha256": model_sha, "worker_start_processor_tokenizer_sha256": processor_sha,
    }
    step_rows, policy_rows, sidecar_rows = [], [], []
    for step in range(2):
        action_tokens = [step] * 7
        step_rows.append({
            "step": step, "features_25d": [0.0] * 25, "clean_action_raw_7d": [0.0] * 7,
            "applied_action_7d": [0.0] * 7, "action_token_ids": action_tokens, "score_head_summary": [0.0] * 7,
            "generation_passes_per_step": generation, "single_generation_parity_pass": generation == 1,
            "score_adapter_parity_pass": generation == 1,
        })
        policy_rows.append({
            "step": step, "clean_policy_intent_9d": [0.0] * 9, "action_token_ids": action_tokens,
            "generation_passes_per_step": generation, "single_generation_parity_pass": generation == 1,
            "score_adapter_parity_pass": generation == 1,
        })
        sidecar_rows.append({"step": step, "robot0_eef_pos": [0.0, 0.0, 0.0], "robot0_gripper_qpos": [0.0, 0.0]})
    _write(root / "episode_metadata.json", meta)
    _write(root / "episode_summary.json", {"step_count": 2})
    _write(root / "runtime_audit.json", {"runtime_valid": True, "official_horizon": meta["official_horizon"], "generation_passes_per_step": 1})
    _write(root / "condition_config.json", {"condition": "CLEAN"})
    _write(root / "attack_config.json", {"attack_enabled": False})
    for name, rows in (("step_records.jsonl", step_rows), ("policy_intent_records.jsonl", policy_rows), ("privileged_teacher_sidecar.jsonl", sidecar_rows)):
        _write(root / name, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    worker_path = root / "worker_start_manifest.json"
    _write(worker_path, worker)
    _write(root / "worker_start_manifest.json.sha256", f"{sha256_file(worker_path)}  worker_start_manifest.json\n")
    _reseal(root)
    return root


def _reseal(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.json":
            rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    _write(root / "artifact_sha256.json", {"files": rows, "recursive_sha256": json_sha(rows)})


def test_success_and_task_failure_are_both_formal_candidates(tmp_path: Path):
    contract = load_contract(CONTRACT_PATH)
    success = audit_artifact(_build_artifact(tmp_path / "success", success=True), contract)
    failure = audit_artifact(_build_artifact(tmp_path / "failure", key="libero_goal/task_01/state_01", success=False), contract)
    assert success["status"] == "PASS_FORMAL_CANDIDATE"
    assert failure["status"] == "PASS_FORMAL_CANDIDATE"
    assert failure["task_success"] is False


def test_provenance_generation_and_checksum_fail_closed(tmp_path: Path):
    contract = load_contract(CONTRACT_PATH)
    quarantined = audit_artifact(_build_artifact(tmp_path / "dirty", provenance="D_DIRTY_START_QUARANTINE"), contract)
    assert quarantined["status"] == "HOLD_PROVENANCE"
    bad_generation = _build_artifact(tmp_path / "generation", generation=0)
    assert audit_artifact(bad_generation, contract)["status"] == "HOLD_GENERATION"
    tampered = _build_artifact(tmp_path / "tampered")
    (tampered / "step_records.jsonl").write_text("corrupt\n", encoding="utf-8")
    assert audit_artifact(tampered, contract)["status"] == "HOLD_CHECKSUM"


def test_old_head_requires_equivalence_then_becomes_eligible(tmp_path: Path):
    contract = load_contract(CONTRACT_PATH)
    artifact = _build_artifact(tmp_path / "old", provenance="B_PREVIOUS_HEAD_EQUIVALENT")
    assert audit_artifact(artifact, contract)["status"] == "HOLD_PROVENANCE"
    assert audit_artifact(artifact, contract, equivalence_status="PASS")["status"] == "PASS_FORMAL_CANDIDATE"


def test_25d_audit_does_not_require_policy_intent_stream(tmp_path: Path):
    contract = load_contract(CONTRACT_PATH)
    artifact = _build_artifact(tmp_path / "25d_only")
    (artifact / "policy_intent_records.jsonl").write_text("not-jsonl\n", encoding="utf-8")
    _reseal(artifact)
    report = audit_artifact(artifact, contract, mode="25d")
    assert report["status"] == "PASS_FORMAL_CANDIDATE"
    assert report["audit_mode"] == "25d"


def test_registry_keeps_task_failure_and_rejects_duplicate_identity(tmp_path: Path):
    contract = load_contract(CONTRACT_PATH)
    one = _build_artifact(tmp_path / "one", success=False)
    report = audit_artifact(one, contract)
    manifest = [{"canonical_parent_key": "libero_object/task_00/state_00", "suite": "libero_object", "task_idx": "0", "state_id": "0", "split": "FIT_TRAIN"}]
    ledger = [{"canonical_parent_key": manifest[0]["canonical_parent_key"], "status": "TASK_FAILURE"}]
    rows, summary = build_registry(manifest, ledger, {manifest[0]["canonical_parent_key"]: [report, report]}, expected_identity_count=1)
    assert rows[0]["task_success"] is False
    assert rows[0]["formal_selected"] is False
    assert rows[0]["selection_reason"] == "DUPLICATE_AUDIT_CANDIDATE"
    assert summary["formal_training_authorized"] is False


def test_incremental_diff_and_worker_strata_detect_duplicate_lease(tmp_path: Path):
    contract = load_contract(CONTRACT_PATH)
    artifact = _build_artifact(tmp_path / "artifact")
    key = "libero_object/task_00/state_00"
    manifest = [{"canonical_parent_key": key, "artifact_root": str(artifact), "suite": "libero_object", "task_idx": "0", "state_id": "0", "split": "FIT_TRAIN"}]
    ledger = [{"canonical_parent_key": key, "status": "RUNNING", "worker_id": "slot_gpu0"}]
    snap = audit_snapshot(manifest, ledger, tmp_path, contract)
    assert snap["raw_sealed_count"] == 0
    assert snap["running_or_leased_excluded"] == [key]
    registry = [{"canonical_parent_key": key, "worker_id": "slot_gpu0", "task_success": "true", "formal_eligible": "True"}]
    strata = audit_worker_strata(
        registry,
        [
            {"canonical_parent_key": key, "status": "RUNNING", "worker_id": "slot_gpu0"},
            {"canonical_parent_key": key, "status": "RUNNING", "worker_id": "slot_gpu1"},
        ],
        [{"slot_id": "slot_gpu0", "first_canary_pass": True}, {"slot_id": "slot_gpu1", "first_canary_pass": True}],
        [{"slot_id": "slot_gpu0", "gpu_id": 0}, {"slot_id": "slot_gpu1", "gpu_id": 1}],
    )
    assert strata["status"] == "HOLD"
    assert strata["duplicate_active_canonical_keys"] == [key]


def test_registry_output_is_non_overwriting(tmp_path: Path):
    output = tmp_path / "registry"
    write_registry([], {"formal_fit_ready": False}, output)
    try:
        write_registry([], {"formal_fit_ready": False}, output)
    except ValueError:
        pass
    else:
        raise AssertionError("registry writer overwrote a sealed root")


def test_stale_audit_uses_real_schema_and_allows_closed_recovery(tmp_path: Path):
    path = tmp_path / "stale.json"
    payload = {
        "schema": "OFFICIAL_V3_STALE_LEASE_RECOVERY_AUDIT_V1",
        "status": "RECOVERY_SAFE",
        "stale_keys": ["libero_object/task_00/state_00"],
        "unexpected_stale_keys": [], "missing_expected_stale_keys": [],
        "missing_recovery_records": [], "unexpected_recovery_records": [],
        "duplicate_formal_result_keys": [], "missing_formal_result_keys": [],
        "duplicate_active_canonical_keys": [], "fence_violations": [],
        "late_result_violations": [], "ledger_mutated": False,
        "formal_training_authorized": False, "formal_attack_authorized": False,
        "runner_binding": {
            "runner_head": "a" * 40, "runner_worktree_clean": True,
            "runner_script_sha256": "b" * 64, "config_sha256": "c" * 64,
        },
    }
    _write(path, payload)
    _write(path.with_name(path.name + ".sha256"), f"{sha256_file(path)}  {path.name}\n")
    count, digest = _read_stale_recovery_audit(path)
    assert count == 0 and digest == sha256_file(path)
