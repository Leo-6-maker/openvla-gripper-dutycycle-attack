from __future__ import annotations

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "detector_v5" / "audit_clean2000_raw_assets.py"
    spec = importlib.util.spec_from_file_location("clean2000_raw_asset_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["canonical_parent_key", "suite", "task_idx", "state_id", "split", "task_success"])
        for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
            for task in range(10):
                for state in range(50):
                    split = "FIT_TRAIN" if state < 20 else "PROTECTED"
                    writer.writerow([f"{suite}/task_{task:02d}/state_{state:02d}", suite, task, state, split, "True"])


def test_raw_asset_audit_separates_official_policy_from_unbound_rgb(tmp_path: Path):
    module = _module()
    registry = tmp_path / "registry.csv"
    _registry(registry)
    official = tmp_path / "official_v3_clean"
    official_ep = official / "libero_object" / "task_00" / "state_00"
    official_ep.mkdir(parents=True)
    (official_ep / "episode_metadata.json").write_text(json.dumps({"schema": "OFFICIAL_V3", "task_language": "pick"}), encoding="utf-8")
    (official_ep / "step_records.jsonl").write_text(json.dumps({"step": 0, "clean_action_token_top_logits": [1.0]}) + "\n", encoding="utf-8")
    (official_ep / "policy_intent_records.jsonl").write_text(json.dumps({"step": 0, "clean_policy_intent_9d": [0.0]}) + "\n", encoding="utf-8")
    (official_ep / "privileged_teacher_sidecar.jsonl").write_text(json.dumps({"step": 0, "contact": 1}) + "\n", encoding="utf-8")
    protected = official / "libero_object" / "task_00" / "state_20"
    protected.mkdir(parents=True)
    (protected / "episode_metadata.json").write_text("not-json", encoding="utf-8")

    parallel = tmp_path / "c2f_raw"
    parallel_ep = parallel / "shards" / "libero_object" / "worker_00" / "episodes" / "libero_object" / "libero_object" / "task_00" / "state_000" / "clean" / "attempt_01"
    (parallel_ep / "rgb").mkdir(parents=True)
    (parallel_ep / "rgb" / "frame_000000.png").write_bytes(b"png")
    (parallel_ep / "episode_metadata.json").write_text(json.dumps({"schema": "C2F", "task_language": "pick", "collector_commit": "old"}), encoding="utf-8")
    (parallel_ep / "step_records.jsonl").write_text(json.dumps({"step": 0, "rgb_path": "rgb/frame_000000.png", "teacher_hazard": 0}) + "\n", encoding="utf-8")

    report, rows = module.audit(Namespace(
        registry_csv=str(registry),
        official_s1_root=None,
        asset_root=[f"official_v3_clean={official}", f"c2f_raw={parallel}"],
        max_semantic_state=19,
    ))

    assert report["status"] == "PASS_METADATA_ONLY"
    assert report["protected_split_semantic_reads"] == []
    official_row = next(row for row in rows if row["root_label"] == "official_v3_clean" and row["canonical_parent_key"] == "libero_object/task_00/state_00")
    parallel_row = next(row for row in rows if row["root_label"] == "c2f_raw" and row["canonical_parent_key"] == "libero_object/task_00/state_00")
    assert official_row["logits_class"] == "LOGITS_DIRECT"
    assert official_row["privileged_class"] == "PRIVILEGED_READY"
    assert official_row["rgb_class"] == "RGB_MISSING"
    assert parallel_row["rgb_class"] == "RGB_DIRECT"
    assert parallel_row["rgb_alignment_status"] == "PASS"
    assert parallel_row["source_compatibility"] == "UNBOUND_PARALLEL_SOURCE"
    assert parallel_row["teacher_or_privileged_fields_detected"] is True
