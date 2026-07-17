import csv
import json
from pathlib import Path

import pytest

from gripper_attack import b3_official_v3_s1 as s1
from gripper_attack.official_v3_contract import SUITES, canonical_key
from official_v3.build_official_v3_formal_registry import build_registry, write_registry


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/B3_OFFICIAL_V3_S1_PROTOCOL_V1.json"
CONTRACT_PATH = ROOT / "configs/OFFICIAL_V3_SOURCE_CONTRACT_V1.json"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fit_rows() -> list[dict[str, str]]:
    rows = []
    for suite in SUITES:
        for task in range(10):
            for state in range(20):
                rows.append({
                    "canonical_parent_key": canonical_key(suite, task, state),
                    "suite": suite,
                    "task_idx": str(task),
                    "state_id": str(state),
                    "split": "FIT_TRAIN",
                    "selected_artifact_root": f"/synthetic/{suite}/task_{task:02d}/state_{state:02d}",
                    "selected_artifact_recursive_sha256": "a" * 64,
                    "artifact_audit_path": "",
                    "artifact_audit_sha256": "b" * 64,
                    "formal_eligible": "true",
                    "formal_selected": "true",
                    "provenance_class": "A_CURRENT_HEAD_CLEAN_START_VERIFIED",
                })
    return rows


def _full_registry_rows(fit_rows=None) -> list[dict[str, str]]:
    rows = list(_fit_rows() if fit_rows is None else fit_rows)
    for suite in SUITES:
        for task in range(10):
            for state in range(20, 50):
                split = "FIT_DEV" if state < 24 else ("CAL" if state < 27 else ("CHECK" if state < 30 else "FINAL_EVAL_CANDIDATE"))
                rows.append({
                    "canonical_parent_key": canonical_key(suite, task, state),
                    "suite": suite,
                    "task_idx": str(task),
                    "state_id": str(state),
                    "split": split,
                    "selected_artifact_root": "",
                    "selected_artifact_recursive_sha256": "",
                    "artifact_audit_path": "",
                    "artifact_audit_sha256": "",
                    "formal_eligible": "false",
                    "formal_selected": "false",
                    "provenance_class": "",
                })
    return rows


def _write_registry(tmp_path: Path, rows=None, **summary_overrides):
    rows = _full_registry_rows(rows)
    audit_root = tmp_path / "audits"
    for row in rows:
        if row.get("split") == "FIT_TRAIN" and row.get("formal_selected") == "true":
            audit_path = audit_root / (row["canonical_parent_key"].replace("/", "_") + ".json")
            _write(audit_path, {"schema": "OFFICIAL_V3_ARTIFACT_AUDIT_V1", "canonical_parent_key": row["canonical_parent_key"]})
            row["artifact_audit_path"] = str(audit_path)
            row["artifact_audit_sha256"] = s1.sha256_file(audit_path)
    registry = tmp_path / "OFFICIAL_V3_FORMAL_REGISTRY_V1.csv"
    fields = list(rows[0]) if rows else [
        "canonical_parent_key", "suite", "task_idx", "state_id", "split",
        "selected_artifact_root", "selected_artifact_recursive_sha256",
        "artifact_audit_sha256", "formal_eligible", "formal_selected", "provenance_class",
    ]
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "identity_count": 2000,
        "unique_identity_count": 2000,
        "formal_fit_ready": True,
        "formal_selected_count": 800,
        "global_formal_selected_count": 800,
        "fit_formal_selected_count": 800,
        "fit_by_suite_formal_selected": {suite: 200 for suite in SUITES},
        "fit_by_task_formal_selected": {f"{suite}/task_{task}": 20 for suite in SUITES for task in range(10)},
        "full_artifact_audit_pass_count": 800,
        "unresolved_provenance_count": 0,
        "unfinished_remediation_count": 0,
        "stale_recovery_unresolved_count": 0,
        "duplicate_selection_count": 0,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "registry_sha256": s1.sha256_file(registry),
        "stale_recovery_summary_sha256": "c" * 64,
    }
    summary.update(summary_overrides)
    summary_path = tmp_path / "OFFICIAL_V3_FORMAL_REGISTRY_SUMMARY_V1.json"
    _write(summary_path, summary)
    return registry, summary_path


def _source_fixture(tmp_path: Path, *, key="libero_10/task_00/state_00") -> Path:
    source = tmp_path / "source"
    suite, task, state = key.split("/")[0], 0, 0
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    steps, policies, sidecars = [], [], []
    for step in range(20):
        tokens = [step] * 7
        action = [0.0] * 6 + [0.0]
        env_action = [0.0] * 6 + [1.0]
        eef = [step * 0.01, 0.0, 0.0]
        steps.append({
            "step": step,
            "features_25d": [float(step)] * 25,
            "clean_action_raw_7d": action,
            "applied_action_7d": env_action,
            "action_token_ids": tokens,
            "score_head_summary": [0.0] * 7,
            "generation_passes_per_step": 1,
            "single_generation_parity_pass": True,
            "score_adapter_parity_pass": True,
        })
        policies.append({
            "step": step,
            "clean_policy_intent_9d": [0.0] * 9,
            "action_token_ids": tokens,
            "generation_passes_per_step": 1,
            "single_generation_parity_pass": True,
            "score_adapter_parity_pass": True,
        })
        sidecars.append({
            "step": step,
            "robot0_eef_pos": eef,
            "robot0_gripper_qpos": [0.1, 0.1],
        })
    meta = {
        "schema": "OPENVLA_OFFICIAL_CLEAN_EPISODE_V3",
        "condition": "CLEAN",
        "runtime_valid": True,
        "suite": suite,
        "task_idx": task,
        "state_id": state,
        "canonical_parent_key": key,
        "split": "FIT_TRAIN",
        "official_horizon": 520,
        "num_steps_wait": 10,
        "success": False,
        "env_success": False,
        "official_execution_adapter": "OfficialOpenVLAActionAdapter.predict_action",
        "generation_passes_per_step": 1,
        "feature_names_25d": contract["feature_names_25d"],
        "policy_intent_feature_names_9d": contract["policy_intent_feature_names_9d"],
        "initial_state_sha256": "c" * 64,
        "model_tree_sha256": "d" * 64,
        "processor_tokenizer_sha256": "e" * 64,
        "protocol_sha256": "f" * 64,
    }
    for name, value in (
        ("episode_metadata.json", meta),
        ("episode_summary.json", {"step_count": 20}),
        ("runtime_audit.json", {"runtime_valid": True, "official_horizon": 520, "generation_passes_per_step": 1}),
        ("condition_config.json", {"condition": "CLEAN"}),
        ("attack_config.json", {"attack_enabled": False}),
    ):
        _write(source / name, value)
    for name, rows in (
        ("step_records.jsonl", steps),
        ("policy_intent_records.jsonl", policies),
        ("privileged_teacher_sidecar.jsonl", sidecars),
    ):
        _write(source / name, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return source


def _teacher_rows(count=12):
    rows = []
    for step in range(count):
        unknown = step >= count - 9
        rows.append({
            "step": step,
            "canonical_parent_key": "libero_10/task_00/state_00",
            "event_id": 0 if not unknown else -1,
            "event_ordinal": 0 if not unknown else -1,
            "valid": True,
            "event_evidence_valid": True,
            "grasp_support": None if unknown else True,
            "grasp_support_mask": not unknown,
            "retention_active": None if unknown else True,
            "retention_active_mask": not unknown,
            "retention_continuation_t10": None if unknown else False,
            "retention_unknown_mask": unknown,
            "release_imminent": None if unknown else False,
            "release_imminent_mask": not unknown,
            "retention_unknown_reason": "INSUFFICIENT_FUTURE_HORIZON" if unknown else None,
        })
    return rows


def test_exact_fit_registry_and_census_gate(tmp_path: Path):
    registry, summary = _write_registry(tmp_path)
    rows = s1.load_formal_fit_registry(registry, summary)
    census, census_summary = s1.build_fit_census(registry, summary)
    assert len(rows) == len(census) == 800
    assert census_summary["formal_training_authorized"] is False
    assert {row["suite"] for row in rows} == set(SUITES)


def test_census_bundle_is_directory_transaction(tmp_path: Path):
    registry, summary = _write_registry(tmp_path)
    rows, census_summary = s1.build_fit_census(registry, summary)
    output = tmp_path / "census"
    s1.write_census_bundle(output, rows, census_summary)
    assert (output / "OFFICIAL_V3_FIT_CENSUS_V1.csv").exists()
    assert (output / "OFFICIAL_V3_FIT_CENSUS_SUMMARY_V1.json.sha256").exists()
    assert not list(tmp_path.glob(".census.*.staging"))
    with pytest.raises(s1.V3S1ContractViolation):
        s1.write_census_bundle(output, rows, census_summary)


def test_builder_full_2000_registry_then_fit_census_filters_exact_800(tmp_path: Path):
    manifest_rows = []
    ledger_rows = []
    audit_reports = {}
    audit_root = tmp_path / "artifact_audits"
    for suite in SUITES:
        for task in range(10):
            for state in range(50):
                key = canonical_key(suite, task, state)
                split = "FIT_TRAIN" if state < 20 else ("FIT_DEV" if state < 24 else ("CAL" if state < 27 else ("CHECK" if state < 30 else "FINAL_EVAL_CANDIDATE")))
                manifest_rows.append({"canonical_parent_key": key, "suite": suite, "task_idx": str(task), "state_id": str(state), "split": split})
                ledger_rows.append({"canonical_parent_key": key, "status": "PASS", "task_success": "true"})
                if split == "FIT_TRAIN":
                    audit_path = audit_root / (key.replace("/", "_") + ".json")
                    _write(audit_path, {"schema": "OFFICIAL_V3_ARTIFACT_AUDIT_V1", "canonical_parent_key": key})
                    audit_reports[key] = [{
                        "schema": "OFFICIAL_V3_ARTIFACT_AUDIT_V1", "status": "PASS_FORMAL_CANDIDATE",
                        "canonical_parent_key": key, "artifact_root": f"/synthetic/{key}",
                        "artifact_recursive_sha256": "a" * 64, "provenance_class": "A_CURRENT_HEAD_CLEAN_START_VERIFIED",
                        "formal_eligible": True, "artifact_audit_path": str(audit_path),
                        "artifact_audit_sha256": s1.sha256_file(audit_path),
                    }]
    rows, summary = build_registry(
        manifest_rows, ledger_rows, audit_reports,
        expected_identity_count=2000,
        stale_recovery_unresolved_count=0,
        stale_recovery_summary_sha256="c" * 64,
    )
    assert summary["formal_fit_ready"] is True
    assert summary["fit_formal_selected_count"] == 800
    registry_root = tmp_path / "registry"
    write_registry(rows, summary, registry_root)
    fit_rows = s1.load_formal_fit_registry(
        registry_root / "OFFICIAL_V3_FORMAL_REGISTRY_V1.csv",
        registry_root / "OFFICIAL_V3_FORMAL_REGISTRY_SUMMARY_V1.json",
    )
    assert len(fit_rows) == 800
    assert all(row["split"] == "FIT_TRAIN" for row in fit_rows)


def test_fit_registry_799_is_hold(tmp_path: Path):
    registry, summary = _write_registry(tmp_path, _fit_rows()[:-1], formal_selected_count=799)
    with pytest.raises(s1.V3S1ContractViolation):
        s1.load_formal_fit_registry(registry, summary)


def test_teacher_unknown_tail_and_binary_masks_pass():
    report = s1.audit_teacher_episode(
        _teacher_rows(), [{"event_id": 0, "start_step": 0, "end_step": 2}], "libero_10/task_00/state_00"
    )
    assert report["status"] == "PASS"
    assert report["t10_positive_count"] == 0


def test_teacher_unknown_cannot_be_encoded_as_negative():
    rows = _teacher_rows()
    rows[-1]["retention_unknown_mask"] = True
    rows[-1]["retention_continuation_t10"] = False
    report = s1.audit_teacher_episode(
        rows, [{"event_id": 0, "start_step": 0, "end_step": 2}], "libero_10/task_00/state_00"
    )
    assert report["status"] == "HOLD"
    assert "STEP_11_retention_continuation_t10_UNKNOWN_NOT_NULL" in report["violations"]


def test_teacher_aggregate_requires_exact_identity_set():
    registry = _fit_rows()
    reports = [
        {
            "canonical_parent_key": row["canonical_parent_key"], "status": "PASS", "violations": [],
            "event_count": 2 if row["suite"] == "libero_10" else 1,
            "episodes_with_multiple_teacher_events": int(row["suite"] == "libero_10"),
            "later_event_known_t10_steps": int(row["suite"] == "libero_10"),
            "later_event_positive_t10_steps": int(row["suite"] == "libero_10"),
            "event_ordinal_stats": {"0": {"events": 1, "known_t10_steps": 1, "positive_t10_steps": 1}, **({"1": {"events": 1, "known_t10_steps": 1, "positive_t10_steps": 1}} if row["suite"] == "libero_10" else {})},
            "teacher_counts": {
                head: {"known": 1, "unknown": 0, "positive": int(head == "retention_continuation_t10"), "negative": int(head != "retention_continuation_t10"), "positive_episodes": int(head == "retention_continuation_t10"), "all_unknown_episodes": 0, "all_negative_episodes": int(head != "retention_continuation_t10")}
                for head in s1.HEADS
            },
        } for row in registry
    ]
    aggregate = s1.aggregate_teacher_audit(reports, registry)
    assert aggregate["status"] == "PASS"
    assert aggregate["actual_identity_count"] == 800
    assert aggregate["suite_episode_counts"] == {suite: 200 for suite in sorted(SUITES)}
    reports.pop()
    assert s1.aggregate_teacher_audit(reports, registry)["status"] == "HOLD"


def test_materialized_student_is_separate_from_teacher_and_policy(tmp_path: Path, monkeypatch):
    source = _source_fixture(tmp_path)
    key = "libero_10/task_00/state_00"
    row = _fit_rows()[-1] | {
        "canonical_parent_key": key,
        "suite": "libero_10",
        "task_idx": "0",
        "state_id": "0",
        "selected_artifact_root": str(source),
    }

    def fake_audit(root, contract, **kwargs):
        return {
            "status": "PASS_FORMAL_CANDIDATE",
            "formal_eligible": True,
            "canonical_parent_key": key,
            "artifact_recursive_sha256": "a" * 64,
        }

    monkeypatch.setattr(s1, "audit_artifact", fake_audit)
    output = tmp_path / "materialized"
    manifest = s1.materialize_episode(row, CONTRACT_PATH, PROTOCOL_PATH, output)
    student = [json.loads(line) for line in (output / "student_input_records.jsonl").read_text().splitlines()]
    assert manifest["student_teacher_physical_separation"] is True
    assert all(set(item) == {
        "schema", "source_schema", "feature_order_sha256", "suite", "task_idx",
        "state_id", "canonical_parent_key", "step", "features_25d", "valid",
    } for item in student)
    assert (output / "teacher_retention_records.jsonl").exists()
    assert not (output / "policy_intent_9d_records.jsonl").exists()


def test_primary_25d_materializer_does_not_require_or_open_9d_stream(tmp_path: Path, monkeypatch):
    source = _source_fixture(tmp_path)
    (source / "policy_intent_records.jsonl").unlink()
    key = "libero_10/task_00/state_00"
    row = _fit_rows()[-1] | {
        "canonical_parent_key": key,
        "suite": "libero_10",
        "task_idx": "0",
        "state_id": "0",
        "selected_artifact_root": str(source),
    }

    def fake_audit(root, contract, **kwargs):
        assert kwargs.get("mode") == "25d"
        return {
            "status": "PASS_FORMAL_CANDIDATE", "formal_eligible": True,
            "canonical_parent_key": key, "artifact_recursive_sha256": "a" * 64,
        }

    monkeypatch.setattr(s1, "audit_artifact", fake_audit)
    output = tmp_path / "materialized_25d_only"
    manifest = s1.materialize_episode(row, CONTRACT_PATH, PROTOCOL_PATH, output)
    assert manifest["policy_intent_9d_exported"] is False
    assert (output / "student_input_records.jsonl").exists()


def test_equivalent_previous_head_is_verified_with_passed_equivalence(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    row = _fit_rows()[0] | {
        "selected_artifact_root": str(source),
        "provenance_class": "B_PREVIOUS_HEAD_EQUIVALENT",
    }
    seen = {}

    def fake_audit(root, contract, **kwargs):
        seen.update(kwargs)
        return {
            "status": "PASS_FORMAL_CANDIDATE",
            "formal_eligible": True,
            "canonical_parent_key": row["canonical_parent_key"],
            "artifact_recursive_sha256": row["selected_artifact_recursive_sha256"],
        }

    monkeypatch.setattr(s1, "audit_artifact", fake_audit)
    assert s1._audit_source(row, {})["status"] == "PASS_FORMAL_CANDIDATE"
    assert seen == {"equivalence_status": "PASS", "mode": "25d"}


def test_materialize_fit_cleans_staging_on_dry_run_failure(tmp_path: Path, monkeypatch):
    registry, summary = _write_registry(tmp_path)
    output = tmp_path / "fit_materialized"

    def fail_once(*args, **kwargs):
        raise s1.V3S1ContractViolation("synthetic dry-run failure")

    monkeypatch.setattr(s1, "dry_run_episode", fail_once)
    with pytest.raises(s1.V3S1ContractViolation):
        s1.materialize_fit(registry, summary, CONTRACT_PATH, PROTOCOL_PATH, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".fit_materialized.*.staging"))


def test_empty_sealed_csv_has_frozen_header_and_sidecar_is_fail_closed(tmp_path: Path):
    output = tmp_path / "empty.csv"
    s1.write_sealed_csv(output, [], ["canonical_parent_key", "status"])
    assert output.read_text(encoding="utf-8").splitlines() == ["canonical_parent_key,status"]
    assert output.with_name("empty.csv.sha256").exists()
    other = tmp_path / "other.csv"
    other.with_name("other.csv.sha256").write_text("stale\n", encoding="utf-8")
    with pytest.raises(s1.V3S1ContractViolation):
        s1.write_sealed_csv(other, [], ["x"])
    assert not other.exists()
