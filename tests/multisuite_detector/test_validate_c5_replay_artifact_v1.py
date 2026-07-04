import hashlib
import json
from pathlib import Path

import pytest

from tools.multisuite_detector.validate_c5_replay_artifact_v1 import C5ReplayValidationError, validate

FREEZE = "1" * 64
CKPT = "2" * 64
DATASET = "3" * 64
SPLIT = "4" * 64
STATE = "5" * 64
NON_ACTIONS = {"simulator": "NOT_PERFORMED", "policy_run": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path):
    path.write_text("name,value\nObject,1\n", encoding="utf-8")


def write_sums(root: Path):
    lines = []
    for p in sorted(x for x in root.iterdir() if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}):
        lines.append(f"{sha(p)}  {p.name}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha(sums)}  SHA256SUMS\n", encoding="utf-8")


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "c5"
    root.mkdir()
    write_json(root / "replay_manifest.json", {"status": "PASS", **NON_ACTIONS})
    write_json(root / "detector_freeze_identity.json", {"freeze_manifest_sha256": FREEZE, "checkpoint_sha256": CKPT, **NON_ACTIONS})
    write_json(root / "dataset_identity.json", {"dataset_csv_sha256": DATASET, "split_csv_sha256": SPLIT, "state_index_sha256": STATE, **NON_ACTIONS})
    write_json(root / "threshold_identity.json", {"threshold": 0.95, "threshold_source": "validation", **NON_ACTIONS})
    write_json(root / "replay_config.json", {"exact_prefix": True, "detector_only": True, **NON_ACTIONS})
    write_json(root / "metrics_overall.json", {"status": "PASS", **NON_ACTIONS})
    write_json(root / "timing_error_report.json", {"status": "PASS", **NON_ACTIONS})
    write_json(root / "emission_rate_report.json", {"status": "PASS", **NON_ACTIONS})
    write_json(root / "safety_false_trigger_report.json", {"status": "PASS", **NON_ACTIONS})
    write_csv(root / "metrics_by_suite.csv")
    write_csv(root / "metrics_by_task.csv")
    write_sums(root)
    return root


def call(root: Path):
    return validate(root, FREEZE, CKPT, DATASET, SPLIT, STATE, 0.95)


def test_c5_replay_artifact_positive(tmp_path):
    report = call(make_root(tmp_path))
    assert report["status"] == "PASS"
    assert report["metrics_by_suite_rows"] == 1


@pytest.mark.parametrize("filename, mutation, message", [
    ("detector_freeze_identity.json", lambda o: {**o, "checkpoint_sha256": "0" * 64}, "checkpoint sha256"),
    ("dataset_identity.json", lambda o: {**o, "dataset_csv_sha256": "0" * 64}, "dataset_csv_sha256"),
    ("threshold_identity.json", lambda o: {**o, "threshold": 0.9}, "threshold mismatch"),
    ("threshold_identity.json", lambda o: {**o, "threshold_source": "test"}, "validation-selected"),
    ("replay_config.json", lambda o: {**o, "exact_prefix": False}, "exact-prefix"),
])
def test_c5_replay_artifact_rejects_tamper(tmp_path, filename, mutation, message):
    root = make_root(tmp_path)
    path = root / filename
    write_json(path, mutation(json.loads(path.read_text())))
    write_sums(root)
    with pytest.raises(C5ReplayValidationError, match=message):
        call(root)


def test_c5_replay_artifact_rejects_sha_tamper(tmp_path):
    root = make_root(tmp_path)
    (root / "metrics_overall.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(C5ReplayValidationError, match="digest mismatch"):
        call(root)


def test_c5_replay_artifact_rejects_missing_non_action(tmp_path):
    root = make_root(tmp_path)
    for name in ["replay_manifest.json", "detector_freeze_identity.json", "dataset_identity.json", "threshold_identity.json", "replay_config.json", "metrics_overall.json", "timing_error_report.json", "emission_rate_report.json", "safety_false_trigger_report.json"]:
        obj = json.loads((root / name).read_text())
        obj.clear()
        obj["status"] = "PASS"
        if name == "replay_config.json":
            obj.update({"exact_prefix": True, "detector_only": True})
        write_json(root / name, obj)
    write_sums(root)
    with pytest.raises(C5ReplayValidationError, match="missing non-action markers"):
        call(root)
