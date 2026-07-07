import hashlib
import json
from pathlib import Path

import pytest

from tools.multisuite_detector.validate_c4_bundle_audit_v1 import C4BundleAuditError, validate_audit


CKPT = "5747a9c967b5b08f0e4b8fc8ba0cbf47c13533ffb5e347c38470e84efe17d79b"
DATASET = "f7808c4ef2a74887689804758c131a19a7fecbbc0e5400bcc3322d08c796010a"
SPLIT = "df23607b3791e414d0e07900508c095bda6a190e8f6500502b056f0988e02673"
STATE = "e4fafbb01e70418ec04b7dc19294b1f6b9c0b52ecc0d8aaa5b56997c3ba53691"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, header, rows):
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sha_files(root: Path):
    entries = []
    for path in sorted(p for p in root.iterdir() if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}):
        entries.append(f"{sha256(path)}  {path.name}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(entries) + "\n", encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha256(sums)}  SHA256SUMS\n", encoding="utf-8")


def make_audit_root(tmp_path: Path) -> Path:
    root = tmp_path / "audit"
    root.mkdir()
    write_json(root / "bundle_identity.json", {"schema_version": "bundle_identity_v1"})
    write_json(root / "checkpoint_identity.json", {"checkpoint_sha256": CKPT})
    write_json(root / "dataset_identity.json", {
        "dataset_csv_sha256": DATASET,
        "split_csv_sha256": SPLIT,
        "state_index_sha256": STATE,
    })
    write_json(root / "threshold_identity.json", {"threshold": 0.95})
    common_non_actions = {
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "attack": "NOT_PERFORMED",
    }
    write_json(root / "metrics_overall.json", {"status": "PASS", **common_non_actions, "f1": 0.93})
    write_json(root / "safety_false_trigger_report.json", {"status": "PASS", **common_non_actions, "false_trigger_rate": 0.01})
    write_json(root / "emission_rate_report.json", {"status": "PASS", **common_non_actions, "emission_rate": 0.15})
    write_json(root / "bundle_load_report.json", {"status": "PASS", **common_non_actions})
    write_csv(root / "metrics_by_suite.csv", ["suite", "f1"], [{"suite": "object", "f1": 0.9}])
    write_csv(root / "metrics_by_task.csv", ["task", "f1"], [{"task": "object/task0", "f1": 0.9}])
    write_csv(root / "metrics_by_population.csv", ["population", "f1"], [{"population": "DETECTOR_ELIGIBLE", "f1": 0.9}])
    write_sha_files(root)
    return root


def test_c4_bundle_audit_validator_positive(tmp_path):
    root = make_audit_root(tmp_path)
    report = validate_audit(
        audit_root=root,
        expected_checkpoint_sha256=CKPT,
        expected_dataset_csv_sha256=DATASET,
        expected_split_csv_sha256=SPLIT,
        expected_state_index_sha256=STATE,
        expected_threshold=0.95,
        output_json=tmp_path / "report.json",
    )
    assert report["status"] == "PASS"
    assert report["metrics_by_suite_rows"] == 1
    assert (tmp_path / "report.json").is_file()


@pytest.mark.parametrize("filename,mutation,message", [
    ("checkpoint_identity.json", lambda obj: {**obj, "checkpoint_sha256": "0" * 64}, "checkpoint sha256"),
    ("dataset_identity.json", lambda obj: {**obj, "dataset_csv_sha256": "0" * 64}, "dataset_csv_sha256"),
    ("dataset_identity.json", lambda obj: {**obj, "split_csv_sha256": "0" * 64}, "split_csv_sha256"),
    ("dataset_identity.json", lambda obj: {**obj, "state_index_sha256": "0" * 64}, "state_index_sha256"),
    ("threshold_identity.json", lambda obj: {**obj, "threshold": 0.90}, "threshold mismatch"),
    ("bundle_load_report.json", lambda obj: {**obj, "status": "FAIL"}, "non-pass status"),
])
def test_c4_bundle_audit_rejects_identity_tamper(tmp_path, filename, mutation, message):
    root = make_audit_root(tmp_path)
    path = root / filename
    write_json(path, mutation(json.loads(path.read_text())))
    write_sha_files(root)
    with pytest.raises(C4BundleAuditError, match=message):
        validate_audit(
            audit_root=root,
            expected_checkpoint_sha256=CKPT,
            expected_dataset_csv_sha256=DATASET,
            expected_split_csv_sha256=SPLIT,
            expected_state_index_sha256=STATE,
            expected_threshold=0.95,
        )


def test_c4_bundle_audit_rejects_sha256sum_tamper(tmp_path):
    root = make_audit_root(tmp_path)
    (root / "metrics_overall.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(C4BundleAuditError, match="digest mismatch"):
        validate_audit(
            audit_root=root,
            expected_checkpoint_sha256=CKPT,
            expected_dataset_csv_sha256=DATASET,
            expected_split_csv_sha256=SPLIT,
            expected_state_index_sha256=STATE,
            expected_threshold=0.95,
        )


def test_c4_bundle_audit_rejects_missing_non_action_marker(tmp_path):
    root = make_audit_root(tmp_path)
    write_json(root / "metrics_overall.json", {"status": "PASS"})
    write_json(root / "safety_false_trigger_report.json", {"status": "PASS"})
    write_json(root / "emission_rate_report.json", {"status": "PASS"})
    write_json(root / "bundle_load_report.json", {"status": "PASS"})
    write_sha_files(root)
    with pytest.raises(C4BundleAuditError, match="missing non-action markers"):
        validate_audit(
            audit_root=root,
            expected_checkpoint_sha256=CKPT,
            expected_dataset_csv_sha256=DATASET,
            expected_split_csv_sha256=SPLIT,
            expected_state_index_sha256=STATE,
            expected_threshold=0.95,
        )
