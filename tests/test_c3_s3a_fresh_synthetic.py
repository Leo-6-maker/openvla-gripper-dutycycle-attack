import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n5" / "phase3_student"))
import run_c3_s3a_fresh_synthetic as fresh  # noqa: E402


def test_gate_split_separates_c3_s3a_and_d0():
    config = json.loads((ROOT / "configs" / "C3_S3A_D0_GATE_SPLIT_V1.json").read_text(encoding="utf-8"))
    assert config["gate_split"]["C3-S3A"]["clean2000_payload_read"] is False
    assert config["gate_split"]["D0"]["status"] == "HOLD"


def test_synthetic_relation_plan_is_exactly_11_31_2():
    plan = fresh.build_relation_plan()
    assert len(plan) == 44
    counts = {category: sum(row["category"] == category for row in plan) for category in fresh.EXPECTED}
    assert counts == {"STATIC": 11, "DYNAMIC": 31, "ARTICULATED": 2}
    assert min(row["step_count"] for row in plan if row["category"] == "STATIC") >= 10
    assert min(row["step_count"] for row in plan if row["category"] != "STATIC") >= 100


def test_canonical_payload_excludes_run_id():
    plan = fresh.build_relation_plan()
    assert len({row["relation_id"] for row in plan}) == 44
