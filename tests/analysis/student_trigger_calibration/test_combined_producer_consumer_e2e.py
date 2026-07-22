from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis/student_trigger_calibration"))
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import seal_directory
from gripper_attack.factorized_scheduler_adapter import FactorizedV2SchedulerAdapter
from scripts.detector_v5.materialize_factorized_v2_production_bundles import (
    _recursive_seal,
    materialize,
    sha256_file,
)

from run_factorized_l3_analysis import compute_l3_metrics, _group, _read_jsonl

SPLITS = [f"o{outer}_i{inner}" for outer in range(4) for inner in range(3)]
HEADS = ("grasp", "manipulation", "release")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_schema(value: dict, schema_path: Path) -> None:
    import jsonschema

    jsonschema.validate(value, json.loads(schema_path.read_text(encoding="utf-8")))


def _write_artifact_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_sha256.json":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha(path),
            "size": path.stat().st_size,
        })
    recursive = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write(root / "artifact_sha256.json", {"files": rows, "recursive_sha256": recursive})


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _prediction_row(identity: str, step: int, positive: bool) -> dict:
    logit = 2.0 if positive else -2.0
    return {
        "episode": identity,
        "step": step,
        "route": "single_object_pick_place",
        "route_supported": True,
        "grasp_logit": logit,
        "grasp_probability": _sigmoid(logit),
        "manipulation_logit": logit,
        "manipulation_probability": _sigmoid(logit),
        "release_logit": -2.0,
        "release_probability": _sigmoid(-2.0),
    }


def _teacher_row(identity: str, step: int, *, positive_episode: bool, short: bool = False) -> dict:
    alternating = step % 2 == 0
    return {
        "canonical_parent_key": identity,
        "step": step,
        "strict_k10_feasible": bool(positive_episode and step == 0 and not short),
        "strict_k10_known_mask": True,
        "grasp_established": alternating,
        "grasp_established_known_mask": True,
        "manipulation_active": alternating,
        "manipulation_active_known_mask": True,
        "release_or_instability": not alternating,
        "release_or_instability_known_mask": True,
        "event_id": 1,
        "event_role": "VALID" if alternating else "INVALID",
    }


def _build_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path], str, str, str, str]:
    commit = "b" * 40
    feature_sha = "c" * 64
    predictor_sha = "a" * 64
    scheduler_sha = sha256_file(ROOT / "src/gripper_attack/factorized_scheduler.py")
    structural_sha = sha256_file(ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json")
    jobs = []

    for split_index, split in enumerate(SPLITS):
        outer, inner = split[1], split[4]
        heldout = [f"libero_object/task_{outer}{inner}/state_00", f"libero_object/task_{outer}{inner}/state_01"]
        calibrator = [f"libero_object/task_{outer}{inner}/state_10", f"libero_object/task_{outer}{inner}/state_11"]
        policy = [f"libero_object/task_{outer}{inner}/state_20", f"libero_object/task_{outer}{inner}/state_21"]
        training = [f"libero_object/task_{outer}{inner}/state_30"]
        base = tmp_path / split
        prediction = base / "prediction"
        calibration_prediction = base / "calibration_prediction"
        policy_prediction = base / "policy_prediction"
        run = base / "run"
        s1 = base / "s1"
        clean = base / "clean"
        teacher = base / "teacher"
        for root in (prediction, calibration_prediction, policy_prediction, run, s1, clean, teacher):
            root.mkdir(parents=True)

        heldout_rows = []
        calibration_rows = []
        policy_rows = []
        all_identities = []
        for identity in heldout + calibrator + policy:
            all_identities.append(identity)
            positive = identity.endswith("state_00") or identity.endswith("state_10") or identity.endswith("state_20")
            count = 10 if identity in heldout + policy else 2
            target_rows = [_teacher_row(identity, step, positive_episode=positive) for step in range(count)]
            episode = teacher.joinpath(*identity.split("/"))
            episode.mkdir(parents=True, exist_ok=True)
            _write(episode / "factorized_teacher_v1.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in target_rows))

            student_episode = s1.joinpath(*identity.split("/"))
            student_episode.mkdir(parents=True, exist_ok=True)
            _write(student_episode / "student_input_records.jsonl", "".join(json.dumps({
                "canonical_parent_key": identity,
                "step": step,
                "valid": True,
                "feature_order_sha256": feature_sha,
            }, sort_keys=True) + "\n" for step in range(count)))

            clean_episode = clean.joinpath(*identity.split("/"))
            clean_episode.mkdir(parents=True, exist_ok=True)
            _write(clean_episode / "episode_metadata.json", {"condition": "CLEAN", "attack_enabled": False})
            _write(clean_episode / "step_records.jsonl", "".join(json.dumps({
                "canonical_parent_key": identity,
                "step": step,
                "clean_action_raw_7d": [0, 0, 0, 0, 0, 0, 0.0],
            }, sort_keys=True) + "\n" for step in range(count)))
            _write_artifact_manifest(clean_episode)

            rows = [_prediction_row(identity, step, positive) for step in range(count)]
            if identity in heldout:
                heldout_rows.extend(rows)
            elif identity in calibrator:
                calibration_rows.extend([_prediction_row(identity, step, step % 2 == 0) for step in range(count)])
            else:
                policy_rows.extend(rows)

        _write(prediction / "heldout_step_predictions.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in heldout_rows))
        _write(prediction / "prediction_manifest.json", {"formal_selection_eligible": False, "identities": heldout})
        _write(calibration_prediction / "heldout_step_predictions.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in calibration_rows))
        _write(calibration_prediction / "prediction_manifest.json", {"formal_selection_eligible": False, "identities": calibrator})
        _write(policy_prediction / "heldout_step_predictions.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in policy_rows))
        _write(policy_prediction / "prediction_manifest.json", {"formal_selection_eligible": False, "identities": policy})
        _write(run / "checkpoint.pt", f"synthetic checkpoint {split}\n")
        _write(run / "source_binding.json", {"source_commit": commit})
        for root in (prediction, calibration_prediction, policy_prediction, run):
            seal_directory(root)
        for root in (s1, clean, teacher):
            _recursive_seal(root)

        manifests = {
            "training": {"identities": training, "training_identities": training},
            "heldout": {"identities": heldout, "heldout_identities": heldout},
            "calibrator_fit": {"identities": calibrator, "fit_identities": calibrator},
            "policy_selection": {"identities": policy, "policy_selection_identities": policy},
        }
        for name, identities in manifests.items():
            _write(base / f"{name}.json", identities)

        jobs.append({
            "split": split,
            "prediction_root": str(prediction),
            "run_root": str(run),
            "s1_root": str(s1),
            "clean_root": str(clean),
            "teacher_root": str(teacher),
            "checkpoint_root": str(run),
            "feature_root": str(s1),
            "training_identity_manifest_path": str(base / "training.json"),
            "heldout_identity_manifest_path": str(base / "heldout.json"),
            "calibrator_fit_manifest_path": str(base / "calibrator_fit.json"),
            "policy_selection_manifest_path": str(base / "policy_selection.json"),
            "calibration_prediction_root": str(calibration_prediction),
            "policy_prediction_root": str(policy_prediction),
            "checkpoint_sha256": sha256_file(run / "checkpoint.pt"),
            "source_commit": commit,
            "feature_order_sha256": feature_sha,
            "predictor_source_sha256": predictor_sha,
            "scheduler_source_sha256": scheduler_sha,
            "structural_config_sha256": structural_sha,
        })

    plan = tmp_path / "production_plan.json"
    _write(plan, {
        "schema": "FACTORIZED_V2_PRODUCTION_INPUT_PLAN_V1",
        "splits": jobs,
        "formal_selection_eligible": False,
        "training_authorized": False,
        "attack_authorized": False,
    })
    bundle = tmp_path / "production_bundle"
    plan_value = json.loads(plan.read_text(encoding="utf-8"))
    _validate_schema(plan_value, ROOT / "schemas/factorized_v2_production_input_plan.schema.json")
    materialize(plan, bundle)
    for split in SPLITS:
        for row in _read_jsonl(bundle / "runtime" / split / "runtime_scheduler_inputs.jsonl"):
            _validate_schema(
                row,
                ROOT / "schemas/factorized_v2_runtime_scheduler_input.schema.json",
            )
        _validate_schema(
            json.loads((bundle / "calibration" / split / "manifest.json").read_text()),
            ROOT / "schemas/factorized_v2_offline_calibration_bundle.schema.json",
        )
        _validate_schema(
            json.loads((bundle / "policy_selection" / split / "manifest.json").read_text()),
            ROOT / "schemas/factorized_v2_offline_policy_selection_bundle.schema.json",
        )
        _validate_schema(
            json.loads((bundle / "evaluation" / split / "manifest.json").read_text()),
            ROOT / "schemas/factorized_v2_offline_evaluation_bundle.schema.json",
        )

    fit_manifest_root = tmp_path / "fit_manifests"
    checkpoint_manifest_root = tmp_path / "checkpoint_manifests"
    for job in jobs:
        split = job["split"]
        fit_dir = fit_manifest_root / split
        checkpoint_dir = checkpoint_manifest_root / split
        fit_dir.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)
        fit_ids = json.loads(Path(job["calibrator_fit_manifest_path"]).read_text())["fit_identities"]
        train_ids = json.loads(Path(job["training_identity_manifest_path"]).read_text())["training_identities"]
        _write(fit_dir / "manifest.json", {"schema": "SYNTHETIC_FIT_MANIFEST_V1", "fit_identities": fit_ids})
        _write(checkpoint_dir / "manifest.json", {
            "schema": "SYNTHETIC_CHECKPOINT_MANIFEST_V1",
            "checkpoint_sha256": job["checkpoint_sha256"],
            "training_identities": train_ids,
        })
    seal_directory(fit_manifest_root)
    seal_directory(checkpoint_manifest_root)
    return bundle, {"fit": fit_manifest_root, "checkpoint": checkpoint_manifest_root}, commit, feature_sha, scheduler_sha, structural_sha


def _run(script: Path, *args: object) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(script), *(str(arg) for arg in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{script.name}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result


def test_combined_producer_consumer_12_split_cpu_e2e(tmp_path: Path) -> None:
    bundle, roots, commit, feature_sha, scheduler_sha, structural_sha = _build_fixture(tmp_path)
    calibration_root = tmp_path / "calibration_contracts"
    calibration_root.mkdir()

    jobs = json.loads((tmp_path / "production_plan.json").read_text())["splits"]
    fitter = ROOT / "analysis/student_trigger_calibration/fit_factorized_calibrators.py"
    first = jobs[0]
    blocked = subprocess.run(
        [
            sys.executable,
            str(fitter),
            "--calibration-bundle-root", str(bundle / "calibration"),
            "--calibration-fit-manifest", first["calibrator_fit_manifest_path"],
            "--heldout-manifest", first["heldout_identity_manifest_path"],
            "--checkpoint-manifest", str(roots["checkpoint"] / first["split"] / "manifest.json"),
            "--output-root", str(tmp_path / "blocked-calibration"),
            "--split", first["split"],
            "--method", "RAW",
            "--checkpoint-sha256", first["checkpoint_sha256"],
            "--student-source-commit", "d" * 40,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode != 0
    assert "CALIBRATION_BUNDLE_SOURCE_MISMATCH" in blocked.stdout + blocked.stderr
    assert not (tmp_path / "blocked-calibration").exists()

    for job in jobs:
        split = job["split"]
        _run(
            fitter,
            "--calibration-bundle-root", bundle / "calibration",
            "--calibration-fit-manifest", job["calibrator_fit_manifest_path"],
            "--heldout-manifest", job["heldout_identity_manifest_path"],
            "--checkpoint-manifest", roots["checkpoint"] / split / "manifest.json",
            "--output-root", calibration_root / split,
            "--split", split,
            "--method", "RAW",
            "--checkpoint-sha256", job["checkpoint_sha256"],
            "--student-source-commit", commit,
        )

    threshold_root = tmp_path / "threshold_selection"
    selector = ROOT / "analysis/student_trigger_calibration/select_factorized_scheduler_thresholds.py"
    _run(
        selector,
        "--policy-selection-bundle-root", bundle / "policy_selection",
        "--calibration-contract-root", calibration_root,
        "--calibration-fit-manifest-root", roots["fit"],
        "--checkpoint-manifest-root", roots["checkpoint"],
        "--structure-config", ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json",
        "--output-root", threshold_root,
        "--grasp-grid", "0.5",
        "--manipulation-grid", "0.5",
        "--release-grid", "0.5",
        "--max-false-start", "1.0",
    )
    threshold_contract_path = threshold_root / "threshold_contract.json"
    threshold_contract = json.loads(threshold_contract_path.read_text())
    assert threshold_contract["expected_splits"] == SPLITS
    assert threshold_contract["formal_selection_eligible"] is True

    v3_root = tmp_path / "v3_contracts"
    producer = ROOT / "analysis/student_trigger_calibration/produce_factorized_calibration_threshold_contract.py"
    for job in jobs:
        split = job["split"]
        _run(
            producer,
            "--calibration-fit-contract", calibration_root / split / "calibration_contract.json",
            "--threshold-selection-contract", threshold_contract_path,
            "--scheduler-source-sha256", scheduler_sha,
            "--structural-config-sha256", structural_sha,
            "--feature-order-sha256", feature_sha,
            "--student-source-commit", commit,
            "--output-root", v3_root / split,
            "--split", split,
        )
        contract = json.loads((v3_root / split / "calibration_and_threshold_contract.json").read_text())
        assert contract["schema"] == "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V3"
        assert contract["l3_evaluation_eligible"] is True

    structure = json.loads((ROOT / "configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json").read_text())
    pooled = []
    for split in SPLITS:
        contract = json.loads((v3_root / split / "calibration_and_threshold_contract.json").read_text())
        adapter = FactorizedV2SchedulerAdapter(structure, contract, require_l3_eligible=True)
        runtime_rows = _read_jsonl(bundle / "runtime" / split / "runtime_scheduler_inputs.jsonl")
        evaluation_rows = _read_jsonl(bundle / "evaluation" / split / "evaluation_records.jsonl")
        runtime_episodes = _group(runtime_rows, "episode", "step")
        evaluation_episodes = _group(evaluation_rows, "episode", "step")
        scheduler_results = {}
        for episode, rows in runtime_episodes.items():
            result = adapter.run_episode(rows)
            scheduler_results[episode] = {
                "emitted": result["ever_emitted"],
                "emit_step": result["first_emit_step"] if result["first_emit_step"] is not None else -1,
            }
        metrics = compute_l3_metrics(evaluation_episodes, scheduler_results, "step")
        pooled.append(metrics)
    assert len(pooled) == 12
    assert all("negative_episode_false_start_rate" in metrics for metrics in pooled)
