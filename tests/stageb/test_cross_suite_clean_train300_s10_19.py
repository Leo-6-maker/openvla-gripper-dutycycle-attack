import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.run_cross_suite_clean_train300_s10_19 import (  # noqa: E402
    POLICIES,
    build_jobs,
    split_role,
    validate_master_manifest,
)


def test_train300_manifest_has_exact_300_unique_keys(tmp_path):
    jobs = build_jobs(tmp_path)
    validation = validate_master_manifest(jobs)
    assert validation["planned_count"] == 300
    assert validation["unique_planned_count"] == 300
    assert validation["duplicate_keys"] == []
    assert validation["overlap_with_clean300_state0_9"] == 0
    assert validation["missing_expected_keys"] == []
    assert validation["by_suite"] == {
        "libero_spatial": 100,
        "libero_goal": 100,
        "libero_10": 100,
    }


def test_train_validation_split_is_state_frozen():
    assert split_role(10) == "train_pool"
    assert split_role(17) == "train_pool"
    assert split_role(18) == "validation_pool"
    assert split_role(19) == "validation_pool"
    for bad in [0, 9, 20]:
        try:
            split_role(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("state outside 10-19 should be rejected")


def test_gpu_ownership_mapping_is_frozen():
    assert POLICIES["libero_spatial"]["cuda_visible_devices"] == "1,3"
    assert POLICIES["libero_spatial"]["render_gpu"] == 1
    assert POLICIES["libero_goal"]["cuda_visible_devices"] == "2,6"
    assert POLICIES["libero_goal"]["render_gpu"] == 6
    assert POLICIES["libero_10"]["cuda_visible_devices"] == "5,4"
    assert POLICIES["libero_10"]["render_gpu"] == 5
    for spec in POLICIES.values():
        assert not str(spec["cuda_visible_devices"]).startswith(("3", "4", "0", "7"))


def test_jobs_are_clean_only_and_state10_19(tmp_path):
    jobs = build_jobs(tmp_path)
    assert {j.condition for j in jobs} == {"CLEAN"}
    assert min(j.state_id for j in jobs) == 10
    assert max(j.state_id for j in jobs) == 19
    assert {j.eval_seed for j in jobs} == {0}
    assert {j.split_role for j in jobs} == {"train_pool", "validation_pool"}
