from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.build_cross_suite_smoke_manifest import (  # noqa: E402
    build_smoke_manifest,
    build_task_inventory,
    load_protocol,
)


def test_task_inventory_contains_expected_cross_suite_rows():
    rows = build_task_inventory()
    assert len(rows) == 30
    suites = {row["suite"] for row in rows}
    assert {"libero_spatial", "libero_goal", "libero_10"} <= suites
    spatial0 = next(row for row in rows if row["suite"] == "libero_spatial" and row["task_idx"] == "0")
    assert spatial0["mechanism_type"] == "single_object_pick_place"
    assert spatial0["eligible_for_gripper_duty"] == "true"
    assert spatial0["primary_object"] == "black_bowl"
    assert spatial0["primary_target"] == "plate"
    assert spatial0["resolver_status"] == "RESOLVED"


def test_articulated_and_push_tasks_are_not_positive_denominator():
    rows = build_task_inventory()
    drawer = next(row for row in rows if row["suite"] == "libero_goal" and row["task_idx"] == "0")
    push = next(row for row in rows if row["suite"] == "libero_goal" and row["task_idx"] == "5")
    assert drawer["mechanism_type"] == "articulated_object"
    assert drawer["eligible_for_gripper_duty"] == "false"
    assert drawer["resolver_status"] == "ABSTAIN_UNSUPPORTED"
    assert push["mechanism_type"] == "planar_or_push"
    assert push["eligible_for_gripper_duty"] == "false"
    assert push["resolver_status"] == "ABSTAIN_UNSUPPORTED"


def test_smoke_manifest_is_preregistered_18_clean_rollouts():
    protocol = load_protocol(REPO_ROOT / "configs" / "sc5_cross_suite_protocol_v1.yaml")
    rows = build_smoke_manifest(protocol)
    assert len(rows) == 18
    assert {row["condition"] for row in rows} == {"CLEAN"}
    assert all(row["attack_allowed_in_phase1"] == "false" for row in rows)
    assert all("primary_object" in row for row in rows)
    assert all("resolver_status" in row for row in rows)
