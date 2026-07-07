import hashlib
import json
from pathlib import Path

import pytest

from tools.multisuite_detector.validate_c4_detector_freeze_v1 import C4DetectorFreezeError, validate_freeze

CKPT = "5747a9c967b5b08f0e4b8fc8ba0cbf47c13533ffb5e347c38470e84efe17d79b"
DATASET = "f7808c4ef2a74887689804758c131a19a7fecbbc0e5400bcc3322d08c796010a"
SPLIT = "df23607b3791e414d0e07900508c095bda6a190e8f6500502b056f0988e02673"
STATE = "e4fafbb01e70418ec04b7dc19294b1f6b9c0b52ecc0d8aaa5b56997c3ba53691"

NON_ACTIONS = {
    "OpenVLA": "NOT_PERFORMED",
    "LIBERO": "NOT_PERFORMED",
    "rollout": "NOT_PERFORMED",
    "attack": "NOT_PERFORMED",
    "exact_prefix_replay": "NOT_PERFORMED",
    "victim_inference": "NOT_PERFORMED",
}


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


def make_freeze_root(tmp_path: Path, *, split_type="object_task_heldout_with_val_v1") -> Path:
    root = tmp_path / "freeze"
    root.mkdir()
    write_json(root / "freeze_manifest.json", {"status": "FROZEN", "split_type": split_type, **NON_ACTIONS})
    write_json(root / "bundle_identity.json", {"bundle_id": "detector_bundle_v1", **NON_ACTIONS})
    write_json(root / "checkpoint_identity.json", {"checkpoint_sha256": CKPT, **NON_ACTIONS})
    write_json(root / "dataset_identity.json", {
        "dataset_csv_sha256": DATASET,
        "split_csv_sha256": SPLIT,
        "state_index_sha256": STATE,
        **NON_ACTIONS,
    })
    write_json(root / "split_identity.json", {"split_type": split_type, "split_csv_sha256": SPLIT, **NON_ACTIONS})
    write_json(root / "normalization_identity.json", {"normalization_source": "train_only", **NON_ACTIONS})
    write_json(root / "threshold_identity.json", {"threshold": 0.95, "threshold_source": "validation", **NON_ACTIONS})
    write_json(root / "metrics_summary.json", {"status": "PASS", "f1": 0.91, **NON_ACTIONS})
    write_json(root / "bundle_load_report.json", {"status": "PASS", **NON_ACTIONS})
    write_csv(root / "metrics_by_suite.csv", ["suite", "f1"], [{"suite": "Object", "f1": 0.9}])
    write_csv(root / "metrics_by_task.csv", ["task", "f1"], [{"task": "task_a", "f1": 0.9}])
    write_sha_files(root)
    return root


def call_validate(root: Path, **kwargs):
    return validate_freeze(
        freeze_root=root,
        expected_checkpoint_sha256=CKPT,
        expected_dataset_csv_sha256=DATASET,
        expected_split_csv_sha256=SPLIT,
        expected_state_index_sha256=STATE,
        expected_threshold=0.95,
        **kwargs,
    )


def test_detector_freeze_validator_positive_scientific_split(tmp_path):
    root = make_freeze_root(tmp_path)
    report = call_validate(root, output_json=tmp_path / "report.json")
    assert report["status"] == "PASS"
    assert report["split_types"] == ["object_task_heldout_with_val_v1"]
    assert report["threshold_source"] == "validation"
    assert (tmp_path / "report.json").is_file()


def test_detector_freeze_allows_parent_random_candidate_only_with_flag(tmp_path):
    root = make_freeze_root(tmp_path, split_type="parent_random_split_v1")
    with pytest.raises(C4DetectorFreezeError, match="parent-random detector"):
        call_validate(root)
    report = call_validate(root, allow_parent_random_candidate=True)
    assert report["status"] == "PASS"
    assert report["allow_parent_random_candidate"] is True


@pytest.mark.parametrize("filename,mutation,message", [
    ("checkpoint_identity.json", lambda obj: {**obj, "checkpoint_sha256": "0" * 64}, "checkpoint sha256"),
    ("dataset_identity.json", lambda obj: {**obj, "dataset_csv_sha256": "0" * 64}, "dataset_csv_sha256"),
    ("dataset_identity.json", lambda obj: {**obj, "split_csv_sha256": "0" * 64}, "split_csv_sha256"),
    ("dataset_identity.json", lambda obj: {**obj, "state_index_sha256": "0" * 64}, "state_index_sha256"),
    ("threshold_identity.json", lambda obj: {**obj, "threshold": 0.90}, "threshold mismatch"),
    ("threshold_identity.json", lambda obj: {**obj, "threshold_source": "test"}, "validation-selected"),
    ("normalization_identity.json", lambda obj: {**obj, "normalization_source": "all_data"}, "train_only"),
    ("bundle_load_report.json", lambda obj: {**obj, "status": "FAIL"}, "status is not PASS"),
])
def test_detector_freeze_rejects_identity_tamper(tmp_path, filename, mutation, message):
    root = make_freeze_root(tmp_path)
    path = root / filename
    write_json(path, mutation(json.loads(path.read_text())))
    write_sha_files(root)
    with pytest.raises(C4DetectorFreezeError, match=message):
        call_validate(root)


def test_detector_freeze_rejects_sha_tamper(tmp_path):
    root = make_freeze_root(tmp_path)
    (root / "metrics_summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(C4DetectorFreezeError, match="digest mismatch"):
        call_validate(root)


def test_detector_freeze_rejects_missing_non_action_markers(tmp_path):
    root = make_freeze_root(tmp_path)
    for filename in [
        "freeze_manifest.json",
        "bundle_identity.json",
        "checkpoint_identity.json",
        "dataset_identity.json",
        "split_identity.json",
        "normalization_identity.json",
        "threshold_identity.json",
        "metrics_summary.json",
        "bundle_load_report.json",
    ]:
        obj = json.loads((root / filename).read_text())
        for key in list(NON_ACTIONS):
            obj.pop(key, None)
        write_json(root / filename, obj)
    write_sha_files(root)
    with pytest.raises(C4DetectorFreezeError, match="missing non-action markers"):
        call_validate(root)


def test_detector_freeze_rejects_unsupported_split(tmp_path):
    root = make_freeze_root(tmp_path, split_type="random_bad_split")
    with pytest.raises(C4DetectorFreezeError, match="unsupported split"):
        call_validate(root)
