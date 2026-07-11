#!/usr/bin/env python3
"""Deterministic success replay for R8T canary — fail-closed, hash-bound.

Replays stored applied_action_7d through LIBERO environments
without loading OpenVLA. Measures env.check_success() and
rebuilds 25D features for state trajectory consistency.

Requires all input SHA hashes — never prints hashes without asserting them.
"""
from __future__ import annotations

import argparse, json, hashlib, os, sys, time, traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np

REPLAY_EXACT = "REPLAY_EXACT"
REPLAY_NUMERICALLY_EQUIVALENT = "REPLAY_NUMERICALLY_EQUIVALENT"
REPLAY_DIVERGED = "REPLAY_DIVERGED"
REPLAY_FAILED = "REPLAY_FAILED"

REPORT_PASS = "PASS_C2G_R8U_SUCCESS_REPLAY_INTEGRITY"
REPORT_HOLD_DIVERGED = "HOLD_C2G_R8U_REPLAY_DIVERGED"
REPORT_HOLD_FAILED = "HOLD_C2G_R8U_REPLAY_FAILED"
REPORT_HOLD_INPUT = "HOLD_C2G_R8U_INPUT_INTEGRITY"


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


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual[:16]}... != {expected[:16]}...")


def _validate_step_records(steps: List[dict], n_steps: int, parent_key: str) -> None:
    if len(steps) != n_steps:
        raise ValueError(f"{parent_key}: step_records count {len(steps)} != metadata n_steps {n_steps}")
    if n_steps < 16:
        raise ValueError(f"{parent_key}: n_steps={n_steps} < 16 minimum")
    for i, row in enumerate(steps):
        if int(row.get("step", -1)) != i:
            raise ValueError(f"{parent_key}: discontinuous steps at index {i}")


def _validate_action(row: dict, key: str, parent_key: str) -> np.ndarray:
    action = np.asarray(row[key], dtype=np.float32).reshape(-1)
    if action.shape != (7,):
        raise ValueError(f"{parent_key}: {key} shape {action.shape}, expected (7,)")
    if not np.isfinite(action).all():
        raise ValueError(f"{parent_key}: {key} has non-finite values")
    return action


def _replay_episode(
    episode_dir: Path,
    render_gpu: int = 0,
) -> Dict[str, Any]:
    """Replay one episode. Raises on any integrity failure."""
    from libero.libero import benchmark as _bench

    meta = read_json(episode_dir / "episode_metadata.json")
    steps = read_jsonl(episode_dir / "step_records.jsonl")

    suite = meta["suite"]
    task_index = int(meta["task_index"])
    state_id = int(meta["state_id"])
    parent_key = meta["parent_key"]
    bddl_path = meta["bddl_path"]
    bddl_sha = meta["bddl_sha256"]
    init_sha = meta["official_init_state_sha256"]
    meta_max_steps = int(meta["max_steps"])
    metadata_dummy_wait = int(meta["dummy_wait"])
    n_steps = int(meta["n_steps"])

    # Verify stored hashes
    if sha256_file(Path(bddl_path)) != bddl_sha:
        raise ValueError(f"{parent_key}: BDDL hash mismatch")

    _validate_step_records(steps, n_steps, parent_key)

    if len(steps) > meta_max_steps:
        raise ValueError(f"{parent_key}: {len(steps)} steps > metadata max_steps {meta_max_steps}")

    # Build environment
    from src.gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
    from src.gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2

    suite_obj = _bench.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    init_states = suite_obj.get_task_init_states(task_index)
    if state_id < 0 or state_id >= len(init_states):
        raise IndexError(f"state_id {state_id} outside [0, {len(init_states)})")
    init_state = init_states[state_id]

    env = None
    try:
        env, obs = build_v4_exact_env(
            bddl_file=str(bddl_path),
            render_gpu_device_id=render_gpu,
            max_steps=meta_max_steps,
            num_steps_wait=metadata_dummy_wait,
        )
        obs = env.set_init_state(init_state)
        env, obs = apply_dummy_wait(env, obs, metadata_dummy_wait)

        eef_site_id = env.sim.model.site_name2id("gripper0_grip_site")

        streamer = SC5StreamingFeatureAdapterV2()
        previous_eef: Optional[np.ndarray] = None
        classification = REPLAY_EXACT
        any_check_success = False
        done_observed_at: Optional[int] = None
        result_steps: List[dict] = []
        all_numeric = True
        any_not_exact = False

        for i in range(n_steps):
            stored = steps[i]

            raw_action = _validate_action(stored, "clean_action_raw_7d", parent_key)
            applied_action = _validate_action(stored, "applied_action_7d", parent_key)

            # Pre-step proprio
            from src.gripper_attack.libero_v4_env_factory import physical_gripper_state
            gs = physical_gripper_state(env, obs)
            qpos_arr = np.asarray(gs.get("qpos", []), dtype=np.float32).reshape(-1)
            qpos_sum = float(qpos_arr[:2].sum()) if qpos_arr.size >= 2 else 0.0
            opening = float(np.abs(qpos_arr[:2]).sum()) if qpos_arr.size >= 2 else 0.0
            eef = np.asarray(env.sim.data.site_xpos[eef_site_id], dtype=np.float32).copy()
            velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else (eef - previous_eef)
            previous_eef = eef.copy()

            # Step environment
            obs_after, reward, done, info = env.step(applied_action)
            check_success = bool(env.check_success())
            any_check_success = any_check_success or check_success
            if done and done_observed_at is None:
                done_observed_at = i

            # Rebuild 25D
            stream_result = streamer.update(
                step_id=i,
                raw_gripper=float(raw_action[-1]),
                env_gripper=float(applied_action[-1]),
                gripper_qpos=qpos_sum,
                gripper_opening_proxy=opening,
                eef_x=float(eef[0]),
                eef_y=float(eef[1]),
                eef_z=float(eef[2]),
                eef_vx=float(velocity[0]),
                eef_vy=float(velocity[1]),
                eef_vz=float(velocity[2]),
                action_dx=float(applied_action[0]),
                action_dy=float(applied_action[1]),
                action_dz=float(applied_action[2]),
                action_gripper=float(raw_action[-1]),
            )

            original_25d = np.asarray(stored.get("features_25d", []), dtype=np.float32)
            replayed_25d_arr = np.asarray(list(stream_result["features"].values()), dtype=np.float32)

            exact = bool(np.array_equal(original_25d, replayed_25d_arr))
            numeric = bool(np.allclose(original_25d, replayed_25d_arr, rtol=1e-5, atol=1e-5, equal_nan=False))
            max_abs = float(np.max(np.abs(original_25d.astype(np.float64) - replayed_25d_arr.astype(np.float64))))
            l2_err = float(np.sqrt(np.sum((original_25d.astype(np.float64) - replayed_25d_arr.astype(np.float64)) ** 2)))

            if not numeric:
                all_numeric = False
            if not exact:
                any_not_exact = True

            result_steps.append({
                "suite": suite, "task_index": task_index, "state_id": state_id,
                "parent_key": parent_key, "step": i,
                "reward": float(reward), "done": bool(done),
                "env_check_success": bool(check_success),
                "info_success": info.get("success"),
                "info_task_success": info.get("task_success"),
                "info_is_success": info.get("is_success"),
                "feature_exact_equal": exact,
                "feature_numeric_equal": numeric,
                "feature_max_abs_error": max_abs,
                "feature_l2_error": l2_err,
                "original_features_25d": original_25d.tolist(),
                "replayed_features_25d": replayed_25d_arr.tolist(),
            })

            obs = obs_after

        # Final check
        try:
            final_check_success = bool(env.check_success())
        except Exception:
            final_check_success = False
        canonical_success = any_check_success or final_check_success

        # Classification
        if done_observed_at is not None and done_observed_at < n_steps - 1:
            classification = REPLAY_DIVERGED
        elif not all_numeric:
            classification = REPLAY_DIVERGED
        elif any_not_exact:
            classification = REPLAY_NUMERICALLY_EQUIVALENT

        reward_list = [s["reward"] for s in result_steps]
        return {
            "suite": suite, "task_index": task_index, "state_id": state_id,
            "parent_key": parent_key, "classification": classification,
            "canonical_success": canonical_success,
            "any_check_success": any_check_success,
            "final_check_success": final_check_success,
            "done_observed_at": done_observed_at,
            "termination_alignment": done_observed_at == n_steps - 1 if done_observed_at is not None else None,
            "step_count": n_steps,
            "clean_success_first_step": result_steps.index(next((s for s in result_steps if s["env_check_success"]), result_steps[-1])) if any_check_success else None,
            "reward_sum": float(sum(reward_list)),
            "reward_max": float(max(reward_list)) if reward_list else 0.0,
            "reward_nonzero_step_count": sum(1 for r in reward_list if r != 0.0),
            "steps": result_steps,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="R8U deterministic success replay")
    ap.add_argument("--plan-root", required=True)
    ap.add_argument("--expected-plan-report-sha256", required=True)
    ap.add_argument("--expected-plan-manifest-sha256", required=True)
    ap.add_argument("--expected-shard-index-sha256", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--expected-scheduler-report-sha256", required=True)
    ap.add_argument("--old-audit-root", required=True)
    ap.add_argument("--expected-old-audit-report-sha256", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--render-gpus", default="0")
    args = ap.parse_args()

    plan_root = Path(args.plan_root)
    run_root = Path(args.run_root)
    old_audit_root = Path(args.old_audit_root)
    output_root = Path(args.output_root)

    if output_root.exists():
        raise FileExistsError(str(output_root))
    output_root.mkdir(parents=True)

    # ── Hash-bound input verification ──
    plan_report = plan_root / "c2g_r8t_teacher_v2_canary_plan.json"
    plan_manifest = plan_root / "c2g_r8t_teacher_v2_canary.jsonl"
    shard_index = plan_root / "c2g_r8t_teacher_v2_canary_shards.jsonl"
    scheduler_report = run_root / "c2g_r8t_dynamic_gpu_scheduler_report.json"
    old_audit_report = old_audit_root / "c2g_r8t_teacher_v2_canary_audit.json"

    assert_hash(plan_report, args.expected_plan_report_sha256, "plan report")
    assert_hash(plan_manifest, args.expected_plan_manifest_sha256, "plan manifest")
    assert_hash(shard_index, args.expected_shard_index_sha256, "shard index")
    assert_hash(scheduler_report, args.expected_scheduler_report_sha256, "scheduler report")
    assert_hash(old_audit_report, args.expected_old_audit_report_sha256, "old audit report")

    # ── Validate plan and scheduler status ──
    plan = read_json(plan_report)
    if plan.get("status") != "PASS_C2G_R8T_TEACHER_V2_CANARY_PLAN":
        raise ValueError("plan status not PASS")
    if plan.get("episode_count") != 24:
        raise ValueError(f"plan episode count {plan.get('episode_count')} != 24")

    sched = read_json(scheduler_report)
    if sched.get("status") != "PASS_C2G_R8T_DYNAMIC_GPU_CANARY":
        raise ValueError("scheduler status not PASS")
    if sched.get("completed_shard_count") != 4:
        raise ValueError("scheduler completed != 4")
    if sched.get("failed_shard_count") != 0:
        raise ValueError("scheduler had failed shards")
    if sched.get("pending_shard_ids"):
        raise ValueError("scheduler has pending shards")

    # ── Discover and validate episodes ──
    episode_dirs = sorted(run_root.glob("shards/*/clean_collection/episodes/**/episode_metadata.json"))
    episode_dirs = [p.parent for p in episode_dirs]

    if len(episode_dirs) != 24:
        raise ValueError(f"found {len(episode_dirs)} episode dirs, expected 24")

    plan_ids = set()
    for row in read_jsonl(plan_manifest):
        plan_ids.add((row["suite"], int(row["task_index"]), int(row["state_id"]), row["parent_key"]))

    meta_ids = set()
    for ed in episode_dirs:
        meta = read_json(ed / "episode_metadata.json")
        key = (meta["suite"], int(meta["task_index"]), int(meta["state_id"]), meta["parent_key"])
        if key in meta_ids:
            raise ValueError(f"duplicate identity: {key}")
        if meta["cohort"] != "DETECTOR_TRAIN":
            raise ValueError(f"non-train cohort: {key}")
        if meta["split"] != "train":
            raise ValueError(f"non-train split: {key}")
        meta_ids.add(key)

    if meta_ids != plan_ids:
        missing = plan_ids - meta_ids
        outside = meta_ids - plan_ids
        raise ValueError(f"identity mismatch: missing={len(missing)} outside={len(outside)}")

    render_gpus = [int(g) for g in args.render_gpus.split(",")]
    gpu_idx = 0

    episode_results = []
    all_steps = []
    classifications: Dict[str, int] = defaultdict(int)
    per_suite_success: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    for ed in episode_dirs:
        gpu = render_gpus[gpu_idx % len(render_gpus)]
        gpu_idx += 1
        print(f"Replaying: {ed.parent_key if hasattr(ed, 'parent_key') else ed}")
        try:
            result = _replay_episode(ed, render_gpu=gpu)
            ep_summary = {k: v for k, v in result.items() if k != "steps"}
            episode_results.append(ep_summary)
            all_steps.extend(result["steps"])
            classifications[result["classification"]] += 1
            suite = result["suite"]
            per_suite_success[suite]["total"] += 1
            if result["canonical_success"]:
                per_suite_success[suite]["success"] += 1
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"FAILED {ed}: {exc}", file=sys.stderr)
            print(tb, file=sys.stderr)
            episode_results.append({
                "suite": "unknown", "task_index": -1, "state_id": -1,
                "parent_key": str(ed), "classification": REPLAY_FAILED,
                "canonical_success": False, "error": str(exc),
            })
            classifications[REPLAY_FAILED] += 1

    # ── Determine final status ──
    if len(episode_results) != 24:
        report_status = REPORT_HOLD_INPUT
    elif classifications[REPLAY_FAILED] > 0:
        report_status = REPORT_HOLD_FAILED
    elif classifications[REPLAY_DIVERGED] > 0:
        report_status = REPORT_HOLD_DIVERGED
    else:
        report_status = REPORT_PASS

    # ── Write outputs ──
    ep_fields = ["suite", "task_index", "state_id", "parent_key",
                 "classification", "canonical_success", "any_check_success",
                 "final_check_success", "done_observed_at", "termination_alignment",
                 "step_count", "clean_success_first_step",
                 "reward_sum", "reward_max", "reward_nonzero_step_count"]
    write_csv(output_root / "r8u_success_replay_episode_ledger.csv", episode_results, ep_fields)

    step_fields = ["suite", "task_index", "state_id", "parent_key", "step",
                   "reward", "done", "env_check_success",
                   "info_success", "info_task_success", "info_is_success",
                   "feature_exact_equal", "feature_numeric_equal",
                   "feature_max_abs_error", "feature_l2_error"]
    write_jsonl(output_root / "r8u_success_replay_step_ledger.jsonl",
                [{k: v for k, v in s.items() if k in step_fields} for s in all_steps])

    report = {
        "schema": "c2g.r8u.postcanary_success_replay.2026-07-11.v1",
        "status": report_status,
        "episode_count": len(episode_results),
        "replay_exact_count": classifications[REPLAY_EXACT],
        "replay_numerically_equivalent_count": classifications[REPLAY_NUMERICALLY_EQUIVALENT],
        "replay_diverged_count": classifications[REPLAY_DIVERGED],
        "replay_failed_count": classifications[REPLAY_FAILED],
        "canonical_clean_success_count": sum(1 for r in episode_results if r.get("canonical_success")),
        "per_suite_clean_success": {k: dict(v) for k, v in per_suite_success.items()},
        "per_suite_success_rate": {k: f"{v['success']}/{v['total']}" for k, v in per_suite_success.items()},
        "plan_report_sha256": sha256_file(plan_report),
    }
    with open(output_root / "r8u_postcanary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)

    sha_files = sorted(output_root.glob("*.*"))
    with open(output_root / "SHA256SUMS", "w") as f:
        for sf in sha_files:
            f.write(f"{sha256_file(sf)}  {sf.name}\n")
    sums_sha = sha256_file(output_root / "SHA256SUMS")
    with open(output_root / "SHA256SUMS.sha256", "w") as f:
        f.write(f"{sums_sha}  SHA256SUMS\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report_status == REPORT_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
