import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.audit_cross_suite_clean_300_postrun import audit_episode, duplicate_rows  # noqa: E402
from scripts.stageb.build_teacher_label_eval_table import build_coverage  # noqa: E402
from scripts.stageb.build_paper_table_schemas import TABLES  # noqa: E402


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _make_episode(root: Path, name: str, *, attack=False, source_commit="63793972743f667c6a6bcc12e9700f322f261147", invalid=0):
    ep = root / name
    _write_json(ep / "episode_manifest.json", {
        "condition": "CLEAN",
        "attack_enabled": attack,
        "vis_enabled": False,
        "rand_enabled": False,
        "suite": "libero_spatial",
        "task_idx": 0,
        "state_id": 0,
        "eval_seed": 0,
        "source_commit": source_commit,
        "gpu_snapshot": {"cuda_visible_devices": "2,6"},
    })
    _write_json(ep / "episode_summary.json", {
        "condition": "CLEAN",
        "vis_or_rand_run": False,
        "task_success": True,
        "n_steps": 10,
        "invalid_feature_steps": invalid,
        "mlp_triggered": False,
        "mlp_emit_step": -1,
        "checkpoint_sha256": "ckpt",
        "dataset_sha256": "data",
        "privileged_valid": False,
        "teacher_abstain": True,
    })
    _write_csv(ep / "step_telemetry.csv", [{"step": 0, "condition": "CLEAN"}])
    _write_csv(ep / "detector_telemetry.csv", [{"step": 0, "emit_step": -1, "emitted": False}])
    _write_csv(ep / "frame_index.csv", [{"step": 0, "raw_agentview_sha256": "abc"}])
    _write_json(ep / "privileged_sidecar.json", {"privileged_valid": False, "teacher_abstain": True})
    _write_json(ep / "sim_state_manifest.json", {"steps": 1})
    (ep / "sim_state_stream.npz").write_bytes(b"placeholder")
    _write_json(ep / "artifact_sha256.json", {"files": [{"path": "episode_summary.json"}], "recursive_sha256": "seal"})
    return ep


def test_postrun_audit_accepts_complete_clean_episode(tmp_path):
    ep = _make_episode(tmp_path, "episode")
    row = audit_episode(ep)
    assert row["status"] == "COMPLETE_VALID"
    assert row["clean_only_contract"] is True
    assert row["gpu_pair"] == "2,6"


def test_postrun_audit_rejects_attack_polluted_episode(tmp_path):
    ep = _make_episode(tmp_path, "episode", attack=True)
    row = audit_episode(ep)
    assert row["status"] == "INVALID_OR_INCOMPLETE"
    assert row["clean_only_contract"] is False


def test_postrun_audit_marks_invalid_features_scientific_invalid(tmp_path):
    ep = _make_episode(tmp_path, "episode", invalid=1)
    row = audit_episode(ep)
    assert row["status"] == "SCIENTIFIC_INVALID"


def test_duplicate_rows_reports_canonical_key_conflict(tmp_path):
    rows = [
        audit_episode(_make_episode(tmp_path, "a")),
        audit_episode(_make_episode(tmp_path, "b")),
    ]
    dupes = duplicate_rows(rows)
    assert len(dupes) == 1
    assert dupes[0]["duplicate_count"] == 2


def test_teacher_coverage_does_not_infer_labels_from_abstain():
    rows = [
        {"suite": "libero_spatial", "condition": "CLEAN", "status": "COMPLETE_VALID", "privileged_valid": "False", "teacher_abstain": "True"},
        {"suite": "libero_spatial", "condition": "CLEAN", "status": "COMPLETE_VALID", "privileged_valid": "False", "teacher_abstain": "True"},
    ]
    coverage = build_coverage(rows)
    assert coverage[0]["privileged_valid_count"] == 0
    assert coverage[0]["teacher_abstain_count"] == 2
    assert coverage[0]["usable_for_teacher_timing_eval"] is False
    assert coverage[0]["usable_for_clean_sr_only"] is True


def test_paper_table_schemas_are_empty_claim_boundary_templates():
    assert "table1_end_to_end_attack_results_schema.csv" in TABLES
    assert "condition" in TABLES["table1_end_to_end_attack_results_schema.csv"]
    assert "eligible_denominator" in TABLES["table2_detector_localization_transfer_schema.csv"]
