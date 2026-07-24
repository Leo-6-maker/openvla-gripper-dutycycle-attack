import hashlib
import json
from pathlib import Path

import pytest

from tools.multisuite_detector.validate_c6_matrix_artifact_v1 import C6MatrixValidationError, validate

FREEZE = "6" * 64
REPLAY = "7" * 64
CONDITIONS = ["CLEAN", "TRUE_T10", "RAND_T10", "RANDOM_TIME", "EARLY_SHIFT", "ORACLE"]
BOUNDARIES = {"label_mutation": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outcomes(path: Path, conditions=CONDITIONS):
    lines = ["condition,count"]
    lines += [f"{name},1" for name in conditions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sums(root: Path):
    lines = []
    for p in sorted(x for x in root.iterdir() if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}):
        lines.append(f"{sha(p)}  {p.name}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha(sums)}  SHA256SUMS\n", encoding="utf-8")


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "c6"
    root.mkdir()
    write_json(root / "matrix_manifest.json", {"status": "PASS", "conditions": CONDITIONS, **BOUNDARIES})
    write_json(root / "detector_freeze_identity.json", {"freeze_manifest_sha256": FREEZE, **BOUNDARIES})
    write_json(root / "replay_identity.json", {"replay_manifest_sha256": REPLAY, **BOUNDARIES})
    write_json(root / "run_config.json", {"exact_prefix_shared": True, "clean_success_parent_denominator": True, **BOUNDARIES})
    write_json(root / "metrics_summary.json", {"status": "PASS", **BOUNDARIES})
    write_json(root / "gripper_bridge_report.json", {"status": "PASS", **BOUNDARIES})
    write_json(root / "command_duty_report.json", {"status": "PASS", **BOUNDARIES})
    write_json(root / "control_integrity_report.json", {"status": "PASS", **BOUNDARIES})
    write_outcomes(root / "outcomes_overall.csv")
    write_outcomes(root / "outcomes_by_suite.csv")
    write_outcomes(root / "outcomes_by_task.csv")
    write_sums(root)
    return root


def call(root: Path):
    return validate(root, FREEZE, REPLAY)


def test_c6_matrix_positive(tmp_path):
    report = call(make_root(tmp_path))
    assert report["status"] == "PASS"
    assert set(report["conditions"]) >= set(CONDITIONS)


@pytest.mark.parametrize("filename, mutation, message", [
    ("detector_freeze_identity.json", lambda o: {**o, "freeze_manifest_sha256": "0" * 64}, "freeze sha256"),
    ("replay_identity.json", lambda o: {**o, "replay_manifest_sha256": "0" * 64}, "replay sha256"),
    ("run_config.json", lambda o: {**o, "exact_prefix_shared": False}, "exact-prefix"),
    ("run_config.json", lambda o: {**o, "clean_success_parent_denominator": False}, "parent denominator"),
    ("metrics_summary.json", lambda o: {**o, "status": "FAIL"}, "status is not PASS"),
])
def test_c6_matrix_rejects_tamper(tmp_path, filename, mutation, message):
    root = make_root(tmp_path)
    path = root / filename
    write_json(path, mutation(json.loads(path.read_text())))
    write_sums(root)
    with pytest.raises(C6MatrixValidationError, match=message):
        call(root)


def test_c6_matrix_rejects_missing_condition(tmp_path):
    root = make_root(tmp_path)
    (root / "matrix_manifest.json").unlink()
    write_json(root / "matrix_manifest.json", {"status": "PASS", **BOUNDARIES})
    write_outcomes(root / "outcomes_overall.csv", CONDITIONS[:-1])
    write_sums(root)
    with pytest.raises(C6MatrixValidationError, match="missing required conditions"):
        call(root)


def test_c6_matrix_rejects_sha_tamper(tmp_path):
    root = make_root(tmp_path)
    (root / "metrics_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(C6MatrixValidationError, match="digest mismatch"):
        call(root)


def test_c6_matrix_rejects_missing_boundaries(tmp_path):
    root = make_root(tmp_path)
    for name in ["matrix_manifest.json", "detector_freeze_identity.json", "replay_identity.json", "run_config.json", "metrics_summary.json", "gripper_bridge_report.json", "command_duty_report.json", "control_integrity_report.json"]:
        obj = json.loads((root / name).read_text())
        for key in list(BOUNDARIES):
            obj.pop(key, None)
        write_json(root / name, obj)
    write_sums(root)
    with pytest.raises(C6MatrixValidationError, match="missing boundary markers"):
        call(root)
