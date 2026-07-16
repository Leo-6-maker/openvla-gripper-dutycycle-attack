import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_b3_retention_materializer import _build_source_artifact  # noqa: E402

from detector import materialize_b3_causal_25d_episode as causal_materializer  # noqa: E402
from detector.audit_b3_causal_25d_materialization import audit  # noqa: E402


ROOT = Path(__file__).parents[1]
SOURCE_PROTOCOL = ROOT / "configs" / "B3_RETENTION_PROTOCOL_V1.json"
FEATURE_CONFIG = ROOT / "configs" / "B3_CAUSAL_25D_MULTIEVENT_V1.json"
MATERIALIZATION_CONFIG = ROOT / "configs" / "B3_CAUSAL_25D_S1_MATERIALIZATION_V1.json"


def test_causal_fit_materializer_keeps_student_25d_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "episode"
    output = tmp_path / "materialized"
    _build_source_artifact(source)

    original_load_jsonl = causal_materializer.load_jsonl

    def reject_policy_stream(path: Path):
        if path.name == "policy_intent_records.jsonl":
            raise AssertionError("causal 25D materializer opened the 9D policy stream")
        return original_load_jsonl(path)

    monkeypatch.setattr(causal_materializer, "load_jsonl", reject_policy_stream)
    manifest = causal_materializer.materialize(
        source, output, SOURCE_PROTOCOL, FEATURE_CONFIG, MATERIALIZATION_CONFIG
    )
    result = audit(output)

    assert manifest["schema"] == "B3_CAUSAL_25D_S1_MATERIALIZED_EPISODE_V1"
    assert manifest["student_policy_intent_read"] is False
    assert result["status"] == "PASS"
    student = [
        __import__("json").loads(line)
        for line in (output / "student_input_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert student
    assert all("clean_policy_intent_9d" not in row for row in student)
    assert all("event_id" not in row for row in student)
    assert len(student[0]["features_25d"]) == 25


def test_causal_fit_materializer_rejects_non_fit_state(tmp_path: Path):
    source = tmp_path / "episode"
    _build_source_artifact(source)
    metadata_path = source / "episode_metadata.json"
    metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
    metadata["state_id"] = 20
    metadata_path.write_text(__import__("json").dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="state_id<=19"):
        causal_materializer.validate_materialization_inputs(
            source, SOURCE_PROTOCOL, FEATURE_CONFIG, MATERIALIZATION_CONFIG
        )
