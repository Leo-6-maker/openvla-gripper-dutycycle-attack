from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.detector_v5.validate_factorized_v2_handoff import validate


def test_handoff_uses_full_non_self_referential_commit_binding(tmp_path: Path):
    value = {
        "schema": "DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V2",
        "status": "READY_FOR_DEEPSEEK_STATIC_INTEGRATION",
        "code_snapshot_commit": "a" * 40,
        "metadata_parent_commit": "b" * 40,
    }
    from scripts.detector_v5.validate_factorized_v2_handoff import canonical_handoff_sha
    value["handoff_blob_sha256"] = canonical_handoff_sha(value)
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert validate(path)["status"] == "PASS"
    value["full_head"] = "c" * 40
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="SELF_REFERENTIAL"):
        validate(path)
