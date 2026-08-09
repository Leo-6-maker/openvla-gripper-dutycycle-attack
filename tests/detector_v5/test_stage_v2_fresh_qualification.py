from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from scripts.detector_v5.run_stage_v_r2_fresh_qualification_atomic import _load_manifest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((ROOT / "configs/stage_v_r2_fresh_qualification_protocol_v2.json").read_text())


def test_fresh_qualification_protocol_is_parent_atomic_and_partial_fleet() -> None:
    policy = PROTOCOL["resource_policy"]
    atomicity = PROTOCOL["parent_atomicity"]
    assert policy["minimum_free_memory_mib"] == 20480
    assert policy["excluded_gpus"] == []
    assert policy["partial_fleet_allowed"] is True
    assert PROTOCOL["runtime_environment"] == {"OPENVLA_ATTN_IMPLEMENTATION": "eager"}
    assert PROTOCOL["runtime_adapter"] == "OPENVLA_UPSTREAM_ATTENTION_OVERRIDE_V1"
    assert atomicity["same_physical_gpu_required"] is True
    assert atomicity["parallelize_across_parents_only"] is True


def test_candidate_manifest_loader_preserves_frozen_rank_order(tmp_path: Path) -> None:
    salt = PROTOCOL["salt"]
    rows = []
    for suite in PROTOCOL["suites"]:
        for index in range(10):
            key = f"{suite}/task_{index:02d}/state_47"
            rows.append({
                "canonical_parent_key": key, "suite": suite, "task_index": index, "state_index": 47,
                "qualification_rank_sha256": hashlib.sha256(f"{salt}::{key}".encode()).hexdigest(),
                "old_artifacts_reused": False, "source_artifact_read": False,
            })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "STAGE_V_R2_QUALIFICATION_CANDIDATE_MANIFEST_V1", "status": "FROZEN", "selected_count": len(rows), "selected_parents": rows}), encoding="utf-8")
    _, loaded = _load_manifest(path, salt)
    assert [row["canonical_parent_key"] for row in loaded] == [row["canonical_parent_key"] for row in rows]


def test_clean_wrapper_binds_eager_attention(monkeypatch) -> None:
    from scripts.detector_v5 import run_stage_v_clean_replay_frozen as clean

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return kwargs

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForVision2Seq=FakeAutoModel))
    monkeypatch.setenv("OPENVLA_ATTN_IMPLEMENTATION", "eager")
    assert clean._bind_runtime_attention() == "OPENVLA_UPSTREAM_ATTENTION_OVERRIDE_V1"
    assert FakeAutoModel.from_pretrained("model", attn_implementation="flash_attention_2")["attn_implementation"] == "eager"
