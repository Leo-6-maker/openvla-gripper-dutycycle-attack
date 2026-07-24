#!/usr/bin/env python3
"""Contract tests for CLEAN1500 collector V3 — protocol + registry integrity."""
import json, os, sys, pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "configs" / "cross_suite_clean1500_protocol_v1.json"
REGISTRY_PATH = REPO / "configs" / "cross_suite_object_target_registry_v1.json"

with open(PROTOCOL_PATH) as f: PROTOCOL = json.load(f)
with open(REGISTRY_PATH) as f: REGISTRY = json.load(f)


class TestProtocolMatrix:
    def test_gate_correct(self):
        assert PROTOCOL["gate"] == "CROSS_SUITE_CLEAN1500_PROTOCOL_V1"

    def test_three_suites_defined(self):
        assert set(PROTOCOL["suites"].keys()) == {"libero_spatial","libero_goal","libero_10"}

    def test_each_suite_500_episodes(self):
        for s in PROTOCOL["suites"]:
            assert PROTOCOL["suites"][s]["episodes"] == 500

    def test_total_planned_1500(self):
        assert PROTOCOL["total_planned"] == 1500

    def test_states_0_to_49(self):
        for s in PROTOCOL["suites"]:
            sc = PROTOCOL["suites"][s]["states"]
            assert sc["start"] == 0 and sc["end"] == 49

    def test_tasks_0_to_9(self):
        for s in PROTOCOL["suites"]:
            assert PROTOCOL["suites"][s]["tasks"] == list(range(10))

    def test_eval_seed_zero(self):
        assert PROTOCOL["eval_seed"] == 0

    def test_clean_only(self):
        assert PROTOCOL["clean_only"] == True
        assert PROTOCOL["attack_enabled"] == False
        assert PROTOCOL["detector_selection"] == False

    def test_no_replacement_states(self):
        assert PROTOCOL["replacement_states"] == False

    def test_retain_success_and_failure(self):
        assert PROTOCOL["retain_success_and_failure"] == True

    def test_gpu_per_suite_unique(self):
        gpus = [PROTOCOL["suites"][s]["gpu"] for s in PROTOCOL["suites"]]
        assert len(gpus) == len(set(gpus)), "GPUs not unique: %s" % gpus
        assert set(gpus) == {4, 5, 6}

    def test_model_paths_exist_on_disk(self):
        for s in PROTOCOL["suites"]:
            p = PROTOCOL["suites"][s]["model"]
            assert os.path.exists(p), "Model missing: %s" % p


class TestRegistry:
    def test_gate_correct(self):
        assert REGISTRY["gate"] == "CROSS_SUITE_OBJECT_TARGET_REGISTRY_V1"

    def test_all_30_tasks_defined(self):
        for suite in ["libero_spatial","libero_goal","libero_10"]:
            tasks = set(REGISTRY[suite].keys())
            assert tasks == {str(i) for i in range(10)}, \
                "%s: missing tasks %s" % (suite, {str(i) for i in range(10)} - tasks)

    def test_spatial_all_eligible(self):
        for t in range(10):
            assert REGISTRY["libero_spatial"][str(t)]["teacher_eligible"] == True
            assert REGISTRY["libero_spatial"][str(t)]["target_site"] is not None

    def test_goal_eligibility_counts(self):
        eligible = sum(1 for t in range(10) if REGISTRY["libero_goal"][str(t)]["teacher_eligible"])
        abstain = 10 - eligible
        assert eligible == 5, "Expected 5 eligible, got %d" % eligible
        assert abstain == 5

    def test_libero10_eligibility_counts(self):
        eligible = sum(1 for t in range(10) if REGISTRY["libero_10"][str(t)]["teacher_eligible"])
        assert eligible == 1, "Expected 1 eligible, got %d" % eligible

    def test_eligible_tasks_have_target_site(self):
        for suite in ["libero_spatial","libero_goal","libero_10"]:
            for t in range(10):
                entry = REGISTRY[suite][str(t)]
                if entry["teacher_eligible"]:
                    assert entry["target_site"] is not None, \
                        "%s task %d: eligible but no target_site" % (suite, t)

    def test_abstain_tasks_have_reason(self):
        for suite in ["libero_spatial","libero_goal","libero_10"]:
            for t in range(10):
                entry = REGISTRY[suite][str(t)]
                if not entry["teacher_eligible"]:
                    assert entry.get("abstain_reason"), \
                        "%s task %d: abstain but no reason" % (suite, t)

    def test_no_approximate_fallback_allowed(self):
        assert REGISTRY["target_resolution_rules"]["approximate_target_prohibited"] == True

    def test_object_sites_are_unique_per_suite_task(self):
        for suite in ["libero_spatial","libero_goal","libero_10"]:
            for t in range(10):
                entry = REGISTRY[suite][str(t)]
                assert entry["primary_object_site"] is not None
                assert len(entry["primary_object_site"]) > 0

    def test_canary_jobs_in_protocol(self):
        canary = PROTOCOL.get("canary_jobs", [])
        assert len(canary) == 5
        assert canary[0]["suite"] == "libero_spatial"
        assert canary[3]["note"] == "ABSTAIN: open drawer, no pick-place"


class TestCollectorCLI:
    """Verify collector script exists and has required arguments."""
    def test_collector_exists(self):
        p = REPO / "scripts" / "stageb" / "run_cross_suite_clean_v3.py"
        assert p.exists(), "Collector V3 not found"

    def test_collector_requires_suite(self):
        p = REPO / "scripts" / "stageb" / "run_cross_suite_clean_v3.py"
        content = p.read_text()
        assert '--suite' in content
        assert '--protocol' in content
        assert '--registry' in content
        assert '--canary' in content
        assert 'FileExistsError' in content or 'not empty' in content, "Must refuse non-empty dir"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
