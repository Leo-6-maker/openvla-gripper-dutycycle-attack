from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.detector_v5.audit_factorized_v2_production_inputs import (
    ProductionInputAuditError,
    audit_raw_action,
    audit_roots,
    verify_sealed_root,
)
from scripts.detector_v5.audit_factorized_calibration_design_feasibility_v2 import audit_v2
from scripts.detector_v5.materialize_factorized_v2_production_bundles import ProductionBundleError, materialize
from scripts.detector_v5.materialize_factorized_v2_production_bundles import _recursive_seal, sha256_file
from gripper_attack.b3_training_protocol import seal_directory


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifact_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_sha256.json":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        })
    recursive = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    _write(root / "artifact_sha256.json", {"files": rows, "recursive_sha256": recursive})


def test_candidate_close_uses_clean_raw_action_and_abstains_at_boundary(tmp_path: Path):
    root = tmp_path / "libero_object" / "task_00" / "state_00"
    _write(root / "episode_metadata.json", {"condition": "CLEAN", "attack_enabled": False})
    rows = []
    for step, raw in enumerate((0.0, 0.2, 0.5, 1.0)):
        rows.append({"canonical_parent_key": "libero_object/task_00/state_00", "step": step, "clean_action_raw_7d": [0, 0, 0, 0, 0, 0, raw]})
    _write(root / "step_records.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    result = audit_raw_action(root)
    assert result["semantic_certification"] == "DIRECT_CLEAN_OPENVLA_RAW_ACTION"
    assert result["counts"] == {"close": 2, "open": 1, "boundary": 1, "unknown": 0}
    assert result["max_close_streak"] == 2


def test_fallback_action_without_clean_certification_is_hold(tmp_path: Path):
    root = tmp_path / "libero_object" / "task_00" / "state_00"
    _write(root / "episode_metadata.json", {"condition": "UNKNOWN", "attack_enabled": False})
    _write(root / "step_records.jsonl", json.dumps({"canonical_parent_key": "libero_object/task_00/state_00", "step": 0, "action_raw": [0, 0, 0, 0, 0, 0, 0.0]}) + "\n")
    result = audit_raw_action(root)
    assert result["semantic_certification"] == "HOLD"
    assert result["counts"]["unknown"] == 1


def test_invalid_clean_raw_action_is_not_certified(tmp_path: Path):
    root = tmp_path / "libero_object" / "task_00" / "state_00"
    _write(root / "episode_metadata.json", {"condition": "CLEAN", "attack_enabled": False})
    _write(root / "step_records.jsonl", json.dumps({"canonical_parent_key": "libero_object/task_00/state_00", "step": 0, "clean_action_raw_7d": [0, 0, 0, 0, 0, 0, 2.0]}) + "\n")
    result = audit_raw_action(root)
    assert result["semantic_certification"] == "HOLD"
    assert result["invalid_raw_action"] is True


def test_root_seal_missing_fails_closed(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    result = verify_sealed_root(root)
    assert result["pass"] is False
    assert result["reason"] == "SEAL_FILES_MISSING"


def test_production_audit_never_promotes_unsealed_roots(tmp_path: Path):
    roots = {name: tmp_path / name for name in ("w32", "splits", "s1", "clean", "teacher")}
    for root in roots.values():
        root.mkdir()
    result = audit_roots(**{f"{name}_root": path for name, path in roots.items()})
    assert result["production_chain_ready"] is False
    assert result["model_inference"] is False
    assert "W32_ROOT_SEAL_HOLD" in result["blockers"]


def test_identity_audit_missing_plan_is_not_oof(tmp_path: Path):
    result = audit_v2(tmp_path / "missing-plan.json")
    assert result["verdict"] == "BLOCKED_ROOTS_NOT_MOUNTED"
    assert result["production_inference"] is False


def test_materializer_requires_exact_plan_and_never_creates_partial_output(tmp_path: Path):
    plan = tmp_path / "plan.json"
    _write(plan, {"schema": "FACTORIZED_V2_PRODUCTION_INPUT_PLAN_V1", "splits": []})
    output = tmp_path / "output"
    with pytest.raises(ProductionBundleError, match="EXACT_12_SPLIT_CLOSURE_REQUIRED"):
        materialize(plan, output)
    assert not output.exists()


def test_duplicate_json_key_is_rejected_by_production_audit(tmp_path: Path):
    root = tmp_path / "libero_object" / "task_00" / "state_00"
    _write(root / "episode_metadata.json", '{"condition":"CLEAN","condition":"CLEAN","attack_enabled":false}\n')
    _write(root / "step_records.jsonl", "")
    with pytest.raises(ProductionInputAuditError, match="DUPLICATE_JSON_KEY"):
        audit_raw_action(root)


def test_synthetic_production_chain_materializes_all_four_bundle_streams(tmp_path: Path):
    sha = "a" * 64
    commit = "b" * 40
    feature = "c" * 64
    jobs = []

    def seal_root(root: Path, recursive: bool = False):
        if recursive:
            _recursive_seal(root)
        else:
            seal_directory(root)

    for outer in range(4):
        for inner in range(3):
            split = f"o{outer}_i{inner}"
            identity = f"libero_object/task_{outer:02d}/state_{inner:02d}"
            train_identity = f"libero_object/task_{outer:02d}/state_{10 + inner:02d}"
            calibration_identity = f"libero_object/task_{outer:02d}/state_{20 + inner:02d}"
            policy_identity = f"libero_object/task_{outer:02d}/state_{30 + inner:02d}"
            base = tmp_path / split
            prediction = base / "prediction"
            policy_prediction = base / "policy_prediction"
            run = base / "run"
            s1 = base / "s1"
            clean = base / "clean"
            teacher = base / "teacher"
            for root in (prediction, policy_prediction, run, s1, clean, teacher):
                root.mkdir(parents=True)

            def prediction_row(key: str):
                return {
                    "canonical_parent_key": key,
                    "step_index": 0,
                    "mechanism_route": "single_object_pick_place",
                    "route_supported": True,
                    "grasp_prob": 0.8,
                    "manipulation_prob": 0.8,
                    "release_prob": 0.1,
                    "grasp_logit": 1.4,
                    "manipulation_logit": 1.4,
                    "release_logit": -1.4,
                    "grasp_target": True,
                    "manipulation_target": True,
                    "release_target": False,
                    "grasp_known_mask": True,
                    "manipulation_known_mask": True,
                    "release_known_mask": True,
                }

            calibration_prediction = base / "calibration_prediction"
            calibration_prediction.mkdir()
            _write(prediction / "heldout_step_predictions.jsonl", json.dumps(prediction_row(identity)) + "\n")
            _write(prediction / "prediction_manifest.json", {"formal_selection_eligible": False, "identities": [identity]})
            _write(calibration_prediction / "heldout_step_predictions.jsonl", json.dumps(prediction_row(calibration_identity)) + "\n")
            _write(calibration_prediction / "prediction_manifest.json", {"formal_selection_eligible": False, "identities": [calibration_identity]})
            _write(policy_prediction / "heldout_step_predictions.jsonl", json.dumps(prediction_row(policy_identity)) + "\n")
            _write(policy_prediction / "prediction_manifest.json", {"formal_selection_eligible": False, "identities": [policy_identity]})
            _write(run / "checkpoint.pt", "synthetic checkpoint")
            _write(run / "source_binding.json", {"source_commit": commit})

            for root, key in ((s1, identity), (clean, identity), (teacher, identity), (s1, calibration_identity), (clean, calibration_identity), (teacher, calibration_identity), (s1, policy_identity), (clean, policy_identity), (teacher, policy_identity)):
                episode = root.joinpath(*key.split("/"))
                episode.mkdir(parents=True, exist_ok=True)
                if root == s1:
                    _write(episode / "student_input_records.jsonl", json.dumps({"canonical_parent_key": key, "step": 0, "valid": True, "feature_order_sha256": feature}) + "\n")
                elif root == clean:
                    _write(episode / "episode_metadata.json", {"condition": "CLEAN", "attack_enabled": False})
                    _write(episode / "step_records.jsonl", json.dumps({"canonical_parent_key": key, "step": 0, "clean_action_raw_7d": [0, 0, 0, 0, 0, 0, 0.0]}) + "\n")
                else:
                    _write(episode / "factorized_teacher_v1.jsonl", json.dumps({"canonical_parent_key": key, "step": 0, "strict_k10_feasible": True, "strict_k10_known_mask": True, "event_id": 1, "event_role": "VALID"}) + "\n")
            for key in (identity, calibration_identity, policy_identity):
                _write_artifact_manifest(clean.joinpath(*key.split("/")))
            seal_root(prediction)
            seal_root(base / "calibration_prediction")
            seal_root(policy_prediction)
            seal_root(run)
            seal_root(s1, recursive=True)
            seal_root(clean, recursive=True)
            seal_root(teacher, recursive=True)
            for name, values in (("training", [train_identity]), ("heldout", [identity]), ("calibrator_fit", [calibration_identity]), ("policy_selection", [policy_identity])):
                _write(base / f"{name}.json", values)
            jobs.append({
                "split": split,
                "prediction_root": str(prediction),
                "run_root": str(run),
                "s1_root": str(s1),
                "clean_root": str(clean),
                "teacher_root": str(teacher),
                "training_identity_manifest_path": str(base / "training.json"),
                "heldout_identity_manifest_path": str(base / "heldout.json"),
                "calibrator_fit_manifest_path": str(base / "calibrator_fit.json"),
                "policy_selection_manifest_path": str(base / "policy_selection.json"),
                "calibration_prediction_root": str(calibration_prediction),
                "policy_prediction_root": str(policy_prediction),
                "checkpoint_sha256": sha256_file(run / "checkpoint.pt"),
                "source_commit": commit,
                "feature_order_sha256": feature,
                "predictor_source_sha256": sha,
            })

    plan = tmp_path / "plan.json"
    for job in jobs:
        job["checkpoint_root"] = job["run_root"]
        job["feature_root"] = job["s1_root"]
    _write(plan, {"schema": "FACTORIZED_V2_PRODUCTION_INPUT_PLAN_V1", "splits": jobs})
    identity_result = audit_v2(plan)
    assert identity_result["verdict"] == "GROUP_CROSS_FITTED_OOF_FEASIBLE"
    bad_plan = tmp_path / "bad-checkpoint-plan.json"
    bad_jobs = json.loads(json.dumps(jobs))
    bad_jobs[0]["checkpoint_sha256"] = "d" * 64
    _write(bad_plan, {"schema": "FACTORIZED_V2_PRODUCTION_INPUT_PLAN_V1", "splits": bad_jobs})
    with pytest.raises(ProductionBundleError, match="CHECKPOINT_SHA_MISMATCH"):
        materialize(bad_plan, tmp_path / "bad-bundle")
    assert not (tmp_path / "bad-bundle").exists()
    output = tmp_path / "bundle"
    result = materialize(plan, output)
    assert result["split_count"] == 12
    assert (output / "SHA256SUMS").is_file()
    assert all((output / directory).is_dir() for directory in ("runtime", "calibration", "policy_selection", "evaluation"))
    assert len(list((output / "runtime").iterdir())) == 12
    assert json.loads((output / "runtime" / "o0_i0" / "runtime_scheduler_inputs.jsonl").read_text())["candidate_close"] is True
    policy_row = json.loads((output / "policy_selection" / "o0_i0" / "policy_selection_runtime_records.jsonl").read_text())
    assert "strict_k10_feasible" not in policy_row
    assert "teacher_label_seal" not in policy_row
    label_row = json.loads((output / "policy_selection" / "o0_i0" / "policy_selection_evaluation_labels.jsonl").read_text())
    assert label_row["strict_k10_feasible"] is True
