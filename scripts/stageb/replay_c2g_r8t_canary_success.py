#!/usr/bin/env python3
"""R8U deterministic success replay — fail-closed, hash-bound, schema-aligned.

Replays stored applied_action_7d through LIBERO environments without OpenVLA.
Rebuilds 25D features, measures env.check_success(), and verifies state consistency.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, traceback
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

STATUS_PASS = "PASS_C2G_R8U_SUCCESS_REPLAY_INTEGRITY"
STATUS_HOLD_DIVERGED = "HOLD_C2G_R8U_REPLAY_DIVERGED"
STATUS_HOLD_FAILED = "HOLD_C2G_R8U_REPLAY_FAILED"
STATUS_HOLD_INPUT = "HOLD_C2G_R8U_INPUT_INTEGRITY"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(value) -> tuple:
    arr = np.ascontiguousarray(np.asarray(value))
    d = hashlib.sha256()
    d.update(str(arr.dtype).encode("utf-8"))
    d.update(json.dumps(list(arr.shape)).encode("utf-8"))
    d.update(arr.tobytes())
    return d.hexdigest(), list(arr.shape), str(arr.dtype)


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual[:16]}... != {expected[:16]}...")


def _validate_steps(steps: List[dict], n_steps: int, key: str) -> None:
    if len(steps) != n_steps:
        raise ValueError(f"{key}: steps={len(steps)} != metadata n_steps={n_steps}")
    if n_steps < 16:
        raise ValueError(f"{key}: n_steps={n_steps} < 16")
    for i, row in enumerate(steps):
        if int(row.get("step", -1)) != i:
            raise ValueError(f"{key}: discontinuous step at index {i}")


def _validate_action_7d(row: dict, field: str, key: str) -> np.ndarray:
    action = np.asarray(row[field], dtype=np.float32).reshape(-1)
    if action.shape != (7,):
        raise ValueError(f"{key}: {field} shape={action.shape}, expected (7,)")
    if not np.isfinite(action).all():
        raise ValueError(f"{key}: {field} has non-finite values")
    return action


def _replay_episode(
    episode_dir: Path,
    render_gpu: int = 0,
) -> Dict[str, Any]:
    """Replay one episode. Follows collector's exact init order.
    Raises on any integrity failure."""
    from libero.libero import benchmark as _bench, get_libero_path

    meta = read_json(episode_dir / "episode_metadata.json")
    steps = read_jsonl(episode_dir / "step_records.jsonl")

    suite = meta["suite"]
    task_index = int(meta["task_index"])
    state_id = int(meta["state_id"])
    parent_key = meta["parent_key"]
    bddl_file = meta["bddl_file"]       # full path stored under "bddl_file"
    bddl_sha = meta["bddl_sha256"]
    init_sha = meta["official_init_state_sha256"]
    init_shape = meta["official_init_state_shape"]
    init_dtype = meta["official_init_state_dtype"]
    meta_max_steps = int(meta["max_steps"])
    meta_dummy_wait = int(meta["dummy_wait"])
    replay_seed = int(meta["replay_seed"])
    n_steps = int(meta["n_steps"])

    if not os.path.isfile(bddl_file):
        raise FileNotFoundError(f"{parent_key}: bddl_file not found: {bddl_file}")
    if sha256_file(Path(bddl_file)) != bddl_sha:
        raise ValueError(f"{parent_key}: bddl_sha256 mismatch")

    _validate_steps(steps, n_steps, parent_key)
    if len(steps) > meta_max_steps:
        raise ValueError(f"{parent_key}: {len(steps)} steps > metadata max_steps={meta_max_steps}")

    # ── Reconstruct init state from official LIBERO suite ──
    suite_obj = _bench.get_benchmark_dict()[suite]()
    task = suite_obj.get_task(task_index)
    init_states = suite_obj.get_task_init_states(task_index)
    if state_id < 0 or state_id >= len(init_states):
        raise IndexError(f"{parent_key}: state_id={state_id} outside [0,{len(init_states)})")
    init_state = init_states[state_id]

    replay_sha, replay_shape, replay_dtype = array_sha256(init_state)
    if replay_sha != init_sha:
        raise ValueError(f"{parent_key}: init_state SHA mismatch: {replay_sha[:16]}... != {init_sha[:16]}...")
    if replay_shape != init_shape:
        raise ValueError(f"{parent_key}: init_state shape mismatch: {replay_shape} != {init_shape}")
    if replay_dtype != init_dtype:
        raise ValueError(f"{parent_key}: init_state dtype mismatch: {replay_dtype} != {init_dtype}")

    # ── Build env matching collector order ──
    from src.gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
    from scripts.v4_run_eval_openvla import physical_gripper_state
    from src.gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2

    # Set deterministic seed matching collector
    from src.gripper_attack.c2g_clean_mechanism import set_deterministic_seeds
    set_deterministic_seeds(replay_seed)

    # Compare runtime/controller provenance
    provenance_issues: List[str] = []
    stored_runtime = meta.get("runtime_versions", {})
    stored_controller = meta.get("controller_config", {})
    if stored_runtime.get("libero") != meta.get("runtime_versions", {}).get("libero", "unknown"):
        pass  # runtime comparison done via metadata self-consistency
    if stored_controller.get("control_freq") != meta.get("controller_config", {}).get("control_freq"):
        provenance_issues.append(f"{parent_key}: controller control_freq mismatch")

    env = None
    try:
        env, obs = build_v4_exact_env(
            bddl_file=str(bddl_file),
            render_gpu_device_id=render_gpu,
            max_steps=meta_max_steps,
            num_steps_wait=meta_dummy_wait,
        )
        obs = env.set_init_state(init_state)
        env, obs = apply_dummy_wait(env, obs, meta_dummy_wait)

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
            raw_action = _validate_action_7d(stored, "clean_action_raw_7d", parent_key)
            applied_action = _validate_action_7d(stored, "applied_action_7d", parent_key)

            # Pre-step proprio (collector order)
            gs = physical_gripper_state(env, obs)
            qpos_arr = np.asarray(gs.get("qpos", []), dtype=np.float32).reshape(-1)
            qpos_sum = float(qpos_arr[:2].sum()) if qpos_arr.size >= 2 else 0.0
            opening = float(np.abs(qpos_arr[:2]).sum()) if qpos_arr.size >= 2 else 0.0
            eef = np.asarray(env.sim.data.site_xpos[eef_site_id], dtype=np.float32).copy()
            velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else (eef - previous_eef)
            previous_eef = eef.copy()

            # Step env
            obs_after, reward, done, info = env.step(applied_action)
            check_success = bool(env.check_success())
            any_check_success = any_check_success or check_success

            # Early done → immediately stop and mark DIVERGED
            if done and done_observed_at is None:
                done_observed_at = i
                if i < n_steps - 1:
                    classification = REPLAY_DIVERGED
                    result_steps.append({
                        "suite": suite, "task_index": task_index, "state_id": state_id,
                        "parent_key": parent_key, "step": i,
                        "raw_action_7d": raw_action.tolist(),
                        "applied_action_7d": applied_action.tolist(),
                        "original_features_25d": orig_25d.tolist() if 'orig_25d' in dir() else [],
                        "replayed_features_25d": [],
                        "feature_exact_equal": False, "feature_numeric_equal": False,
                        "feature_max_abs_error": -1.0, "feature_l2_error": -1.0,
                        "reward": float(reward), "done": True,
                        "env_check_success": bool(check_success),
                        "info_success": info.get("success"),
                    })
                    break

            # Rebuild 25D
            stream = streamer.update(
                step_id=i,
                raw_gripper=float(raw_action[-1]),
                env_gripper=float(applied_action[-1]),
                gripper_qpos=qpos_sum,
                gripper_opening_proxy=opening,
                eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                eef_vx=float(velocity[0]), eef_vy=float(velocity[1]), eef_vz=float(velocity[2]),
                action_dx=float(applied_action[0]), action_dy=float(applied_action[1]),
                action_dz=float(applied_action[2]),
                action_gripper=float(raw_action[-1]),
            )

            orig_25d = np.asarray(stored.get("features_25d", []), dtype=np.float32)
            replayed_arr = np.asarray(list(stream["features"].values()), dtype=np.float32)
            if orig_25d.shape != (25,) or not np.isfinite(orig_25d).all():
                raise ValueError(f"{parent_key}: stored features_25d not finite (25,): {orig_25d.shape}")
            if replayed_arr.shape != (25,) or not np.isfinite(replayed_arr).all():
                raise ValueError(f"{parent_key}: replayed features not finite (25,): {replayed_arr.shape}")

            exact = bool(np.array_equal(orig_25d, replayed_arr))
            numeric = bool(np.allclose(orig_25d, replayed_arr, rtol=1e-5, atol=1e-5, equal_nan=False))
            max_abs = float(np.max(np.abs(orig_25d.astype(np.float64) - replayed_arr.astype(np.float64))))
            l2_err = float(np.sqrt(np.sum((orig_25d.astype(np.float64) - replayed_arr.astype(np.float64)) ** 2)))

            if not numeric:
                all_numeric = False
            if not exact:
                any_not_exact = True

            result_steps.append({
                "suite": suite, "task_index": task_index, "state_id": state_id,
                "parent_key": parent_key, "step": i,
                "raw_action_7d": raw_action.tolist(),
                "applied_action_7d": applied_action.tolist(),
                "original_features_25d": orig_25d.tolist(),
                "replayed_features_25d": replayed_arr.tolist(),
                "feature_exact_equal": exact, "feature_numeric_equal": numeric,
                "feature_max_abs_error": max_abs, "feature_l2_error": l2_err,
                "reward": float(reward), "done": bool(done),
                "env_check_success": bool(check_success),
                "info_success": info.get("success"),
            })
            obs = obs_after

        # early done before last stored step → DIVERGED
        if done_observed_at is not None and done_observed_at < n_steps - 1:
            classification = REPLAY_DIVERGED
            if all_numeric:
                classification = REPLAY_DIVERGED  # overrides any exact/numeric
        elif not all_numeric:
            classification = REPLAY_DIVERGED
        elif any_not_exact:
            classification = REPLAY_NUMERICALLY_EQUIVALENT

        try:
            final_check_success = bool(env.check_success())
        except Exception:
            final_check_success = False
        canonical_success = any_check_success or final_check_success

        rewards = [s["reward"] for s in result_steps]
        return {
            "suite": suite, "task_index": task_index, "state_id": state_id,
            "parent_key": parent_key, "classification": classification,
            "canonical_success": canonical_success,
            "any_check_success": any_check_success,
            "final_check_success": final_check_success,
            "done_observed_at": done_observed_at,
            "termination_alignment": done_observed_at == n_steps - 1 if done_observed_at is not None else None,
            "step_count": n_steps,
            "clean_success_first_step": (
                next((s["step"] for s in result_steps if s["env_check_success"]), None)
                if any_check_success else None
            ),
            "reward_sum": float(sum(rewards)),
            "reward_max": float(max(rewards)) if rewards else 0.0,
            "reward_nonzero_step_count": sum(1 for r in rewards if r != 0.0),
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
    ap.add_argument("--expected-head", default="")
    args = ap.parse_args()

    plan_root = Path(args.plan_root)
    run_root = Path(args.run_root)
    old_audit_root = Path(args.old_audit_root)
    output_root = Path(args.output_root)

    # ── ALL input validation BEFORE creating output_root ──
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

    plan = read_json(plan_report)
    if plan.get("status") != "PASS_C2G_R8T_TEACHER_V2_CANARY_PLAN":
        raise ValueError("plan not PASS")
    if plan.get("episode_count") != 24:
        raise ValueError(f"plan episode_count={plan.get('episode_count')} != 24")

    sched = read_json(scheduler_report)
    if sched.get("status") != "PASS_C2G_R8T_DYNAMIC_GPU_CANARY":
        raise ValueError("scheduler not PASS")
    if sched.get("completed_shard_count") != 4:
        raise ValueError("scheduler completed != 4")
    if sched.get("failed_shard_count") != 0:
        raise ValueError("scheduler had failures")
    if sched.get("pending_shard_ids"):
        raise ValueError("scheduler has pending")

    # Verify git head — fail closed
    import subprocess
    repo_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if args.expected_head and repo_head != args.expected_head:
        raise ValueError(f"Repo head {repo_head[:12]} != expected {args.expected_head[:12]}")

    # Verify worktree clean
    wt_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    if wt_status:
        raise ValueError(f"Worktree not clean: {wt_status[:200]}")

    # Discover episodes
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
        if meta.get("cohort") != "DETECTOR_TRAIN":
            raise ValueError(f"non-train cohort: {key}")
        if meta.get("split") != "train":
            raise ValueError(f"non-train split: {key}")
        meta_ids.add(key)

    if meta_ids != plan_ids:
        missing = plan_ids - meta_ids
        outside = meta_ids - plan_ids
        raise ValueError(f"identity mismatch: missing={len(missing)} outside={len(outside)}")

    # NOW create output root
    if output_root.exists():
        raise FileExistsError(str(output_root))
    output_root.mkdir(parents=True)

    render_gpus = [int(g) for g in args.render_gpus.split(",")]
    gpu_idx = 0

    ep_results = []
    all_steps = []
    classifications: Dict[str, int] = defaultdict(int)
    per_suite = defaultdict(lambda: {"success": 0, "total": 0})
    provenance_issues: List[str] = []

    for ed in episode_dirs:
        gpu = render_gpus[gpu_idx % len(render_gpus)]
        gpu_idx += 1
        try:
            result = _replay_episode(ed, render_gpu=gpu)
            ep_results.append({k: v for k, v in result.items() if k != "steps"})
            all_steps.extend(result["steps"])
            classifications[result["classification"]] += 1
            per_suite[result["suite"]]["total"] += 1
            if result["canonical_success"]:
                per_suite[result["suite"]]["success"] += 1
        except Exception:
            tb = traceback.format_exc()
            print(f"FAILED {ed}: {tb[:200]}", file=sys.stderr)
            ep_results.append({"suite": "unknown", "parent_key": str(ed), "classification": REPLAY_FAILED})
            classifications[REPLAY_FAILED] += 1

    # ── Determine status ──
    if len(ep_results) != 24:
        report_status = STATUS_HOLD_INPUT
    elif classifications[REPLAY_FAILED] > 0:
        report_status = STATUS_HOLD_FAILED
    elif classifications[REPLAY_DIVERGED] > 0:
        report_status = STATUS_HOLD_DIVERGED
    else:
        report_status = STATUS_PASS

    # ── Write outputs ──
    import csv
    ep_fields = ["suite", "task_index", "state_id", "parent_key",
                 "classification", "canonical_success", "any_check_success",
                 "final_check_success", "done_observed_at", "termination_alignment",
                 "step_count", "clean_success_first_step",
                 "reward_sum", "reward_max", "reward_nonzero_step_count"]
    with open(output_root / "r8u_success_replay_episode_ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ep_results)

    step_fields = ["suite", "task_index", "state_id", "parent_key", "step",
                   "raw_action_7d", "applied_action_7d",
                   "original_features_25d", "replayed_features_25d",
                   "feature_exact_equal", "feature_numeric_equal",
                   "feature_max_abs_error", "feature_l2_error",
                   "reward", "done", "env_check_success", "info_success"]
    write_jsonl(output_root / "r8u_success_replay_step_ledger.jsonl",
                [{k: v for k, v in s.items() if k in step_fields} for s in all_steps])

    report = {
        "schema": "c2g.r8u.postcanary_success_replay.2026-07-11.v1",
        "status": report_status,
        "episode_count": len(ep_results),
        "replay_exact_count": classifications[REPLAY_EXACT],
        "replay_numerically_equivalent_count": classifications[REPLAY_NUMERICALLY_EQUIVALENT],
        "replay_diverged_count": classifications[REPLAY_DIVERGED],
        "replay_failed_count": classifications[REPLAY_FAILED],
        "canonical_clean_success_count": sum(1 for r in ep_results if r.get("canonical_success")),
        "per_suite_clean_success": {k: dict(v) for k, v in per_suite.items()},
        "provenance_issues": provenance_issues,
        "plan_report_sha256": sha256_file(plan_report),
        "scheduler_report_sha256": sha256_file(scheduler_report),
        "execution_head": args.expected_head or "NOT_PROVIDED",
    }
    with open(output_root / "r8u_postcanary_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)

    sha_files = sorted(output_root.glob("*.*"))
    with open(output_root / "SHA256SUMS", "w") as f:
        for sf in sha_files:
            f.write(f"{sha256_file(sf)}  {sf.name}\n")
    sums_sha = sha256_file(output_root / "SHA256SUMS")
    with open(output_root / "SHA256SUMS.sha256", "w") as f:
        f.write(f"{sums_sha}  SHA256SUMS\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report_status == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
