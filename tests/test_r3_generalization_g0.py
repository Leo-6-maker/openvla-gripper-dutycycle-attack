from pathlib import Path

import pytest

from scripts.detector_v5 import audit_r3_generalization_g0 as g0
from scripts.detector_v5.audit_r3_generalization_g0 import _effective_step_value, _event_metrics, event_label


def _label(value):
    return {"value": value}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["TRUE", "UNKNOWN"], "TRUE"),
        (["FALSE", "UNKNOWN"], "UNKNOWN"),
        (["FALSE", "FALSE"], "FALSE"),
        (["UNKNOWN"], "UNKNOWN"),
        ([], "UNKNOWN"),
    ],
)
def test_g0_event_label_is_tri_valued_or(values, expected):
    assert event_label([_label(value) for value in values]) == expected


def test_g0_event_label_does_not_treat_unknown_as_false():
    assert event_label([_label("FALSE"), _label("UNKNOWN")]) != "FALSE"


def test_g0_masked_true_is_unknown_before_event_aggregation():
    assert _effective_step_value({"value": "TRUE", "valid_mask": False, "mask": False, "right_censored": False, "reason": "MISSING_EVIDENCE"}) == "UNKNOWN"


def test_g0_geometry_not_applicable_is_not_unknown_bucket():
    assert _effective_step_value({"value": "UNKNOWN", "valid_mask": False, "mask": False, "right_censored": False, "reason": "GEOMETRY_NOT_APPLICABLE"}) == "NOT_APPLICABLE"


def test_g0_known_true_remains_observed_if_other_step_is_censored():
    known_true = {"value": "TRUE", "valid_mask": True, "mask": True, "right_censored": False}
    censored_false = {"value": "FALSE", "valid_mask": True, "mask": True, "right_censored": True}
    assert event_label([known_true, {"value": "UNKNOWN"}]) == "TRUE"
    assert _effective_step_value(censored_false) == "UNKNOWN"


def test_g0_teacher_true_interval_is_deduplicated_across_candidate_spans():
    def label(value, *, censored=False):
        return {"value": value, "valid_mask": True, "mask": True, "right_censored": censored}

    rows = [
        {"episode_id": "e", "candidate_close": True, "labels": {"h": label("TRUE")}},
        {"episode_id": "e", "candidate_close": False, "labels": {"h": label("TRUE")}},
        {"episode_id": "e", "candidate_close": True, "labels": {"h": label("TRUE")}},
    ]
    stats = _event_metrics(rows, "h", {"suite": "s", "task_id": 0})
    assert stats["candidate_event_count"] == 2
    assert stats["teacher_true_intervals"] == 1
    assert stats["teacher_true_intervals_touched_by_candidate"] == 1


def test_g0_output_path_rejects_relative_and_forbidden(tmp_path):
    allowed = tmp_path / "phase"
    allowed.mkdir()
    with pytest.raises(ValueError):
        g0._validate_output_path(Path("relative"), allowed)
    with pytest.raises(ValueError):
        g0._validate_output_path(allowed / "attack" / "root", allowed)


def test_g0_output_path_rejects_symlink(tmp_path):
    allowed = tmp_path / "phase"
    allowed.mkdir()
    target = allowed / "real"
    target.mkdir()
    link = allowed / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        g0._validate_output_path(link, allowed)


def test_g0_run_publishes_sealed_root_and_rejects_overwrite(monkeypatch, tmp_path):
    phase = tmp_path / "phase"
    teacher_root = phase / "teacher"
    teacher_root.mkdir(parents=True)
    binding = {
        "t4_seal_sha256sums_sha256": "a" * 64,
        "teacher_root": str(teacher_root),
        "teacher_root_sha256sums_sha256": "b" * 64,
        "teacher_manifest_sha256": "c" * 64,
        "teacher_records_sha256": "d" * 64,
        "coverage_root": str(phase / "coverage"),
        "coverage_root_sha256sums_sha256": "e" * 64,
        "feature_binding_sha256": "f" * 64,
        "feature_order_sha256": "0" * 64,
        "t0a_manifest": {"episode_bindings": {str(i): {} for i in range(670)}},
    }
    empty = {head: g0._empty_head() for head in g0.HEADS}
    monkeypatch.setattr(g0, "_load_records", lambda _: ([{}] * 670, binding))
    monkeypatch.setattr(g0, "_identity_metadata", lambda _: {str(i): {} for i in range(670)})
    monkeypatch.setattr(g0, "_audit_rows", lambda *_: (empty, {}, 196483))
    output = phase / "g0"
    report = g0.run(Path("/unused/t4"), output)
    assert report["consumable"] is False
    assert (output / "G0_LABEL_BASELINE_AUDIT.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    with pytest.raises(FileExistsError):
        g0.run(Path("/unused/t4"), output)
