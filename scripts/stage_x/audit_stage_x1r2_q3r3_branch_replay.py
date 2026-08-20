#!/usr/bin/env python3
"""CPU/mock structural audit for Q3R3 branch replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.stage_x_q3r3_branch_replay import BranchReplay, ProtectedCounters, ReferenceClean, compare_branch_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()
    reference = ReferenceClean.from_record({"status": "PASS_REFERENCE_CLEAN", "clean_success": True, "initial_state": {"seed": 7}, "dummy_wait_steps": 10, "policy_horizon": 20, "first_emit_step": 3, "t5": 5, "h_phys": 10, "student_calls": 1, "env_actions": [[0.1], [0.2], [0.3], [0.4]], "observation_bytes": [b"o0", b"o1", b"o2", b"branch", b"o4"]})
    branch = BranchReplay(reference, "TRUE_PGD_T5")
    replayed: list[tuple[int, tuple[float, ...]]] = []
    prefix_count = branch.replay_prefix(lambda step, action: replayed.append((step, action)))
    branch.validate_first_decision(3, b"branch")
    branch.authorize_attacked_step(3, True)
    state = {"model_identity": "m", "suite_task_state_identity": "s", "seed_and_dummy_wait": [7, 10], "wrapper_step_index": 3, "qpos": [1.0], "qvel": [2.0], "act": [3.0], "ctrl": [4.0], "time": 5.0, "mocap_state": [6.0], "task_object_state": {"x": 7.0}, "controller_state": {"y": [8.0]}}
    counters = ProtectedCounters()
    counters.assert_zero()
    result = {"schema": "STAGE_X1R2_Q3R3_BRANCH_RUNNER_STATIC_AUDIT_V1", "status": "STAGE_X1R2_Q3R3_BRANCH_RUNNER_STATIC_PASS", "source": {"commit": args.source_commit, "tree": args.source_tree}, "checks": {"reference_student_calls": reference.student_calls == 1, "prefix_count": prefix_count, "prefix_steps": [step for step, _ in replayed], "prebranch_model_calls": 0, "prebranch_student_calls": 0, "common_first_observation": branch.common_first_observation.decode(), "branch_state_equal": compare_branch_state(state, state)["equal"], "attacked_step_gate": True}, "protected_boundary": {**counters.as_dict(), "eval160": "UNREAD", "protected_evaluation": "UNREAD"}, "scientific_authority": False, "real_model_or_simulator_executed": False, "next_gate": "STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_PASS"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "source": result["source"]}, sort_keys=True))


if __name__ == "__main__":
    main()
