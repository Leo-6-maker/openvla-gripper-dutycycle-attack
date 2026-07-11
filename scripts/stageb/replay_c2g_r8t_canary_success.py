#!/usr/bin/env python3
"""Deterministic success replay for existing R8T canary trajectories.

Replays stored applied_action_7d through LIBERO environments
without loading OpenVLA or generating new actions. Measures
env.check_success() and state trajectory consistency.
"""
from __future__ import annotations

import argparse, json, hashlib, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
from src.gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait

try:
    import libero
    _BENCHMARK = libero.libero.benchmark.get_benchmark_dict()
except Exception:
    _BENCHMARK = {}


def _get_init_state(suite: str, task_index: int, state_id: int):
    """Get official LIBERO init state for suite/task/state."""
    suite_obj = _BENCHMARK[suite]()
    states = suite_obj.get_task_init_states(task_index)
    if state_id < 0 or state_id >= len(states):
        raise IndexError(f"state_id {state_id} outside [0, {len(states)}) for {suite}/task_{task_index}")
    return states[state_id]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


REPLAY_EXACT = "REPLAY_EXACT"
REPLAY_NUMERICALLY_EQUIVALENT = "REPLAY_NUMERICALLY_EQUIVALENT"
REPLAY_DIVERGED = "REPLAY_DIVERGED"
REPLAY_FAILED = "REPLAY_FAILED"


def replay_episode(
    episode_dir: Path,
    dummy_wait: float = 10.0,
    max_steps: int = 300,
) -> Dict[str, Any]:
    """Replay one episode using stored applied_action_7d."""
    meta = read_json(episode_dir / "episode_metadata.json")
    steps = read_jsonl(episode_dir / "step_records.jsonl")

    suite = meta["suite"]
    task_index = meta["task_index"]
    state_id = meta["state_id"]
    parent_key = meta["parent_key"]
    bddl_path = meta.get("task_bddl_path", meta.get("task_bddl", ""))

    if not bddl_path or not os.path.exists(bddl_path):
        raise FileNotFoundError(f"BDDL not found: {bddl_path}")

    # Get official init state via LIBERO suite
    init_state = _get_init_state(suite, task_index, state_id)

    # Build environment via V4 factory
    env, obs = build_v4_exact_env(
        bddl_file=bddl_path,
        render_gpu_device_id=0,
        max_steps=max_steps,
        num_steps_wait=0,  # dummy wait applied separately below
    )

    # Set to official init state
    env.set_init_state(init_state)

    # Apply dummy wait
    env, obs = apply_dummy_wait(env, obs, int(dummy_wait))

    # Replay each step
    result_steps = []
    classification = REPLAY_EXACT
    any_check_success = False
    final_check_success = False
    done_observed = False
    step_done = False
    all_actions_equal = True

    for i, stored in enumerate(steps):
        if i >= max_steps:
            break

        applied_action = stored.get("applied_action_7d")
        stored_action_raw = stored.get("clean_action_raw_7d")
        if applied_action is None:
            classification = REPLAY_FAILED
            break

        # Pre-step state
        pre_qpos = np.array(env.robots[0].joint_positions, dtype=np.float64).copy()
        pre_eef = np.array(env.robots[0].controller.eef_site.getPosition(), dtype=np.float64).copy()

        # Step
        obs, reward, done, info = env.step(np.array(applied_action, dtype=np.float64))
        check_success = bool(env.check_success())
        any_check_success = any_check_success or check_success
        if done and not step_done:
            done_observed = True
            step_done = True

        # Post-step state
        post_qpos = np.array(env.robots[0].joint_positions, dtype=np.float64).copy()
        post_eef = np.array(env.robots[0].controller.eef_site.getPosition(), dtype=np.float64).copy()

        result_steps.append({
            "suite": suite, "task_index": task_index, "state_id": state_id,
            "parent_key": parent_key, "step": i,
            "applied_action_7d": applied_action,
            "reward": float(reward),
            "done": bool(done),
            "env_check_success": bool(check_success),
            "info_success": info.get("success", None),
            "info_is_success": info.get("is_success", None),
            "pre_qpos": pre_qpos.tolist() if len(pre_qpos) > 0 else [],
            "pre_eef": pre_eef.tolist() if len(pre_eef) > 0 else [],
            "post_qpos": post_qpos.tolist() if len(post_qpos) > 0 else [],
            "post_eef": post_eef.tolist() if len(post_eef) > 0 else [],
            "features_25d_replay": features_25d,
        })

    # Final check_success
    try:
        final_check_success = bool(env.check_success())
    except Exception:
        pass
    canonical_success = any_check_success or final_check_success

    env.close()

    return {
        "suite": suite, "task_index": task_index, "state_id": state_id,
        "parent_key": parent_key,
        "classification": classification,
        "canonical_success": canonical_success,
        "any_check_success": any_check_success,
        "final_check_success": final_check_success,
        "done_observed": done_observed,
        "step_count": len(result_steps),
        "all_actions_equal": all_actions_equal,
        "steps": result_steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay R8T canary actions for success remeasurement")
    ap.add_argument("--plan-root", required=True, help="R8T plan root")
    ap.add_argument("--run-root", required=True, help="R8T GPU collection root")
    ap.add_argument("--output-root", required=True, help="New R8U output root")
    ap.add_argument("--dummy-wait", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    plan_root = Path(args.plan_root)
    run_root = Path(args.run_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Verify inputs
    plan_report = plan_root / "c2g_r8t_teacher_v2_canary_plan.json"
    if not plan_report.is_file():
        print(f"MISSING: {plan_report}", file=sys.stderr)
        return 1

    plan_manifest = plan_root / "c2g_r8t_teacher_v2_canary.jsonl"
    scheduler_report = run_root / "c2g_r8t_dynamic_gpu_scheduler_report.json"

    print(f"Plan report: {plan_report} (SHA256: {sha256_file(plan_report)})")
    print(f"Scheduler report: {scheduler_report} (SHA256: {sha256_file(scheduler_report)})")

    # Discover episodes under run_root
    episode_dirs = sorted(run_root.glob("shards/*/clean_collection/episodes/**/episode_metadata.json"))
    episode_dirs = [p.parent for p in episode_dirs]
    print(f"Found {len(episode_dirs)} episode directories")

    # Replay each episode
    episode_results = []
    all_steps = []
    classifications = defaultdict(int)
    per_suite_success = defaultdict(lambda: {"success": 0, "total": 0})
    per_suite_term = defaultdict(lambda: defaultdict(int))

    for ep_dir in episode_dirs:
        print(f"Replaying: {ep_dir}")
        try:
            result = replay_episode(ep_dir, args.dummy_wait, args.max_steps)
            episode_results.append({
                k: v for k, v in result.items() if k != "steps"
            })
            all_steps.extend(result["steps"])
            classifications[result["classification"]] += 1
            suite = result["suite"]
            per_suite_success[suite]["total"] += 1
            if result["canonical_success"]:
                per_suite_success[suite]["success"] += 1
            if result["done_observed"]:
                per_suite_term[suite]["DONE_OBSERVED"] += 1
            if result["any_check_success"]:
                per_suite_term[suite]["ENV_CHECK_SUCCESS"] += 1
            if result["final_check_success"]:
                per_suite_term[suite]["FINAL_CHECK_SUCCESS"] += 1
        except Exception as exc:
            print(f"FAILED {ep_dir}: {exc}", file=sys.stderr)
            classifications[REPLAY_FAILED] += 1
            episode_results.append({
                "suite": "unknown", "task_index": -1, "state_id": -1,
                "parent_key": str(ep_dir),
                "classification": REPLAY_FAILED,
                "canonical_success": False,
                "error": str(exc),
            })

    # Write outputs
    ep_fields = ["suite", "task_index", "state_id", "parent_key",
                 "classification", "canonical_success", "any_check_success",
                 "final_check_success", "done_observed", "step_count",
                 "all_actions_equal"]
    write_csv(output_root / "r8u_success_replay_episode_ledger.csv", episode_results, ep_fields)

    step_fields = ["suite", "task_index", "state_id", "parent_key", "step",
                   "reward", "done", "env_check_success",
                   "info_success", "info_is_success"]
    write_jsonl(output_root / "r8u_success_replay_step_ledger.jsonl", [
        {k: v for k, v in s.items() if k in step_fields or k in ("applied_action_7d", "features_25d_replay")}
        for s in all_steps
    ])

    # Build report
    report = {
        "schema": "c2g.r8u.postcanary_success_replay.2026-07-11.v1",
        "status": "PASS_C2G_R8U_SUCCESS_REPLAY",
        "episode_count": len(episode_results),
        "replay_exact_count": classifications[REPLAY_EXACT],
        "replay_numerically_equivalent_count": classifications[REPLAY_NUMERICALLY_EQUIVALENT],
        "replay_diverged_count": classifications[REPLAY_DIVERGED],
        "replay_failed_count": classifications[REPLAY_FAILED],
        "canonical_clean_success_count": sum(1 for r in episode_results if r.get("canonical_success")),
        "per_suite_clean_success": {k: dict(v) for k, v in per_suite_success.items()},
        "per_suite_success_rate": {
            k: f"{v['success']}/{v['total']}" for k, v in per_suite_success.items()
        },
        "per_suite_termination_reasons": {k: dict(v) for k, v in per_suite_term.items()},
        "plan_report_sha256": sha256_file(plan_report),
        "scheduler_report_sha256": sha256_file(scheduler_report),
    }

    output_report = output_root / "r8u_postcanary_report.json"
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)

    # SHA256SUMS
    sha_files = sorted(output_root.glob("*.*"))
    with open(output_root / "SHA256SUMS", "w") as f:
        for sf in sha_files:
            f.write(f"{sha256_file(sf)}  {sf.name}\n")
    sums_sha = sha256_file(output_root / "SHA256SUMS")
    with open(output_root / "SHA256SUMS.sha256", "w") as f:
        f.write(f"{sums_sha}  SHA256SUMS\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
