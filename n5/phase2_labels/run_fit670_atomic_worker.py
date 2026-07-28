"""[DeepSeek] FIT670 Atomic Worker — Gate F670-G.

Per-GPU collector: loads one shard from the shard plan, collects each episode
with identity-level atomic publish. Independent process per GPU.

Requires:
  - fit_collection_core.py (shared collection primitives)
  - FIT670 transition receipt (verified before model load)
  - FIT670 shard plan (identity assignment)

Usage:
  python n5/phase2_labels/run_fit670_atomic_worker.py \
    --shard-id 0 \
    --gpu 0 \
    --model-path /path/to/model \
    --official-worker /path/to/official_clean_worker.py \
    --transition-receipt /path/to/transition_root \
    --identity-allowlist /path/to/FIT670_IDENTITY_ALLOWLIST.json \
    --shard-plan /path/to/FIT670_GPU_SHARD_PLAN.json \
    --registry-root /path/to/registry/per_task \
    --alias-ledger /path/to/ALIAS_LEDGER.json \
    --upstream-root /path/to/upstream \
    --output-root /path/to/d670_output \
    --seed 20260717
"""
import argparse, copy, importlib, json, math, os, pickle, shutil
import random, subprocess, sys, time, uuid
from pathlib import Path

import numpy as np

# Import shared core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_collection_core import (
    HORIZONS, FOUR_SUITES, FORBIDDEN_PATH_TOKENS, CollectionHold,
    sha256_file, sha256_bytes, sha256_image, sha256_numpy, git_value, reject_path,
    mat_to_quat, jsonable,
    _verify_source_stability, collect_entity, verify_entity_identity,
    collect_contact_pairs, compute_gripper_width, compute_eef_velocity,
    get_geom_extents, capture_gpu_identity,
    _validate_episode_shapes, seal_root,
    load_resolutions, capture_model_geometry_snapshot,
    make_episode_staging, compute_episode_target, publish_episode, stage_cleanup,
)


def capture_one_fit670_episode(module, suite, task_idx, state_id, collection_seed,
                               registry_dir, canonical_state, task, adapter,
                               output_root, save_student_rgb=True):
    """Collect a single episode with FIT670 telemetry upgrades.

    Returns (episode_data, published_target_path).
    """
    from experiments.robot.libero.libero_utils import get_libero_image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    reg_path = Path(registry_dir) / f"{suite}_task_{task_idx:02d}.json"
    registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
    legacy = registry_data.get("legacy", registry_data)
    is_articulated = legacy.get("task_disposition") == "ARTICULATED_UNSUPPORTED"
    resolutions, relations = load_resolutions(str(reg_path), allow_articulated=True)

    bddl_root = Path(get_libero_path("bddl_files")).resolve()
    task_bddl = (bddl_root / task.problem_folder / task.bddl_file).resolve()
    task_bddl_sha = sha256_file(task_bddl)

    ep_id = f"{suite}/task_{task_idx:02d}/state_{state_id:02d}"
    target = compute_episode_target(output_root, suite, task_idx, state_id)

    # Check if already published (resume support)
    if target.exists():
        existing_seal = target / "SHA256SUMS.sha256"
        if existing_seal.is_file():
            print(f"    SKIP (already sealed): {ep_id}")
            return None, target
        raise CollectionHold(f"target exists but not sealed: {target}")

    staging = make_episode_staging(ep_id, output_root)
    published = False
    try:
        module.set_official_seed(collection_seed)
        env = OffScreenRenderEnv(bddl_file_name=str(task_bddl), camera_heights=256, camera_widths=256)
        try:
            env.seed(collection_seed)
            env.reset()
            obs = env.set_init_state(copy.deepcopy(canonical_state))
            for _ in range(int(module.NUM_STEPS_WAIT)):
                obs = env.step([0, 0, 0, 0, 0, 0, -1])[0]

            model = env.sim.model
            for (etype, eid), res in resolutions.items():
                expected_name = res.get("alias_to", res.get("name", "?"))
                verify_entity_identity(model, etype, eid, expected_name)

            # ── Model geometry snapshot (once per episode) ──
            geometry_snapshot = capture_model_geometry_snapshot(
                model,
                registry_resolutions=resolutions,
                bddl_sha=task_bddl_sha,
                c1_registry_binding=str(reg_path),
            )
            (staging / "SNAPSHOT_model_geometry.json").write_text(
                json.dumps(geometry_snapshot, indent=2, sort_keys=True), encoding="utf-8")

            # Create steps directory for RGB PNGs
            steps_dir = staging / "steps"
            if save_student_rgb:
                steps_dir.mkdir()

            rows = []
            privileged = []
            generation_counts = []
            prev_obs = None
            from fit_collection_core import compute_eef_velocity, get_geom_extents

            for step_num in range(HORIZONS[suite]):
                # ── forward-before-capture protocol ──
                qpos_pre = env.sim.data.qpos.copy()
                qvel_pre = env.sim.data.qvel.copy()
                act_pre = env.sim.data.act.copy() if (hasattr(env.sim.data, 'act') and
                            env.sim.data.act is not None) else None
                time_pre = float(env.sim.data.time)

                if not all(math.isfinite(float(x)) for x in qpos_pre):
                    raise CollectionHold(f"non-finite qpos at step {step_num}")
                if not all(math.isfinite(float(x)) for x in qvel_pre):
                    raise CollectionHold(f"non-finite qvel at step {step_num}")

                env.sim.forward()
                _verify_source_stability(qpos_pre, qvel_pre, act_pre, time_pre,
                                         env.sim.data, step_num, "capture_forward")
                model = env.sim.model; data = env.sim.data
                sim_state = env.sim.get_state()
                entities = [collect_entity(model, data, res) for res in resolutions.values()]
                contact_pairs = collect_contact_pairs(model, data,
                                                      registry_resolutions=resolutions)

                # Add geom extents to entity records
                for ent, (etype, eid) in zip(entities, resolutions.keys()):
                    if etype == "geom":
                        ent["geom_extents"] = get_geom_extents(model, eid)

                # EEF velocity (finite difference)
                eef_velocity = compute_eef_velocity(obs, prev_obs)
                gripper_qpos = jsonable(obs.get("robot0_gripper_qpos", []))
                gripper_vel = None
                if prev_obs is not None and isinstance(gripper_qpos, list) and len(gripper_qpos) >= 1:
                    prev_gripper = jsonable(prev_obs.get("robot0_gripper_qpos", []))
                    if isinstance(prev_gripper, list) and len(prev_gripper) >= 1:
                        gripper_vel = float(gripper_qpos[0]) - float(prev_gripper[0])

                prev_obs = obs

                # Save student RGB frame
                rgb_image = get_libero_image(obs, 224)
                frame_sha = None
                if save_student_rgb:
                    from PIL import Image as PILImage
                    pil_img = PILImage.fromarray(np.asarray(rgb_image)).convert("RGB")
                    png_path = steps_dir / f"step_{step_num:04d}.png"
                    pil_img.save(str(png_path), format="PNG")
                    frame_sha = sha256_file(png_path)

                privileged.append({
                    "step": step_num, "suite": suite, "task_idx": task_idx,
                    "state_id": state_id,
                    "horizon": HORIZONS[suite],
                    "sim_state": {
                        "time": float(data.time),
                        "qpos": sim_state.qpos.tolist(),
                        "qvel": sim_state.qvel.tolist(),
                        "act": getattr(sim_state, "act", None).tolist() if getattr(sim_state, "act", None) is not None else None,
                    },
                    "robot0_eef_pos": jsonable(obs.get("robot0_eef_pos", [])),
                    "robot0_eef_quat": jsonable(obs.get("robot0_eef_quat", [])),
                    "robot0_eef_vel": eef_velocity,
                    "robot0_gripper_qpos": gripper_qpos,
                    "robot0_gripper_vel": gripper_vel,
                    "gripper_width": compute_gripper_width(obs),
                    "object_state": jsonable(obs.get("object-state", [])),
                    "entities": entities,
                    "contact_pairs": contact_pairs,
                    "contact_count": int(data.ncon),
                    "forward_before_capture": True,
                    "protocol_amendment": "PROTOCOL_AMENDMENT_V6_FIT670_ATOMIC",
                })

                image = get_libero_image(obs, 224)
                clean_action, generation, score_meta = adapter.predict_action_with_scores(
                    image, str(task.language))
                count = score_meta.get("generation_passes_per_step")
                if isinstance(count, bool) or not isinstance(count, int) or count != 1:
                    raise CollectionHold(f"generation pass count: {count}")
                generation_counts.append(count)
                score_action = [float(x) for x in jsonable(score_meta["score_action"])]
                raw_action = [float(x) for x in jsonable(clean_action)]
                if len(raw_action) != 7 or len(score_action) != 7:
                    raise CollectionHold(f"action shape failed at step {step_num}")
                if max(abs(a - b) for a, b in zip(raw_action, score_action)) > 1e-6:
                    raise CollectionHold(f"action parity failed at step {step_num}")
                executed = [float(x) for x in jsonable(adapter.postprocess(clean_action))]
                if len(executed) != 7:
                    raise CollectionHold(f"executed action shape failed at step {step_num}")
                for action_label, action_arr in [("raw", raw_action), ("score", score_action), ("executed", executed)]:
                    if not all(math.isfinite(x) for x in action_arr):
                        raise CollectionHold(f"non-finite {action_label}_action at step {step_num}: {action_arr}")

                row = {
                    "step": step_num, "suite": suite, "task_idx": task_idx,
                    "state_id": state_id,
                    "action_raw_7d": raw_action, "score_action_7d": score_action,
                    "action_env_7d": executed, "generation_passes_per_step": count,
                    "single_generation_parity_pass": True,
                    "action_mutation_by_detector": False,
                }
                if frame_sha:
                    row["frame_rgb_png_sha256"] = frame_sha
                rows.append(row)

                obs, _reward, done, _info = env.step(executed)
                if done:
                    break
        finally:
            env.close()

        if not generation_counts or any(x != 1 for x in generation_counts):
            raise CollectionHold("generation closure failed")

        episode = {
            "episode_id": ep_id,
            "suite": suite, "task_id": task_idx, "state_id": state_id,
            "collection_seed": collection_seed,
            "pilot_identity_bound": True,
            "task_bddl_sha256": task_bddl_sha,
            "registry_task_sha256": sha256_file(str(reg_path)),
            "step_count": len(rows), "official_horizon": HORIZONS[suite],
            "generation_passes_per_step": generation_counts,
            "task_language": str(task.language),
            "steps": rows, "telemetry": privileged,
            "relations": relations,
            "source_mode": "FIT670_ATOMIC_COLLECTION",
            "forward_before_capture": True,
            "protocol_amendment": "PROTOCOL_AMENDMENT_V6_FIT670_ATOMIC",
            "geometry_status": "NOT_APPLICABLE" if (is_articulated and not resolutions) else "OK",
            "model_inference": True, "attack_enabled": False,
            "detector_loaded": False, "teacher_labels_generated": False,
            "student_rgb_saved": save_student_rgb,
        }

        _validate_episode_shapes(episode)
        (staging / "episode.json").write_text(
            json.dumps(episode, indent=2, sort_keys=True), encoding="utf-8")

        # Atomic publish
        publish_episode(staging, target)
        published = True
        return episode, target

    finally:
        if not published:
            stage_cleanup(staging)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU number")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path, required=True)
    parser.add_argument("--identity-allowlist", type=Path, required=True)
    parser.add_argument("--shard-plan", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--no-student-rgb", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    # Path safety audit
    for path in [args.model_path, args.official_worker, args.transition_receipt,
                 args.identity_allowlist, args.shard_plan, args.registry_root,
                 args.alias_ledger, args.upstream_root]:
        reject_path(path)

    out_root = Path(args.output_root).resolve()
    label = f"gpu_{args.gpu}"
    worker_root = out_root / label
    if worker_root.exists():
        raise SystemExit(f"worker output exists: {worker_root}")

    # ── Load shard identities ──
    shard_plan = json.loads(Path(args.shard_plan).read_text(encoding="utf-8"))
    shard = shard_plan["shards"][args.shard_id]
    identities = shard["identities"]
    print(f"Worker shard {args.shard_id} (GPU {args.gpu}): {len(identities)} identities")
    print(f"  cost: {shard['total_cost']}")
    print(f"  suites: {shard['suite_counts']}")

    # ── Verify transition receipt ──
    from fit_transition import verify_transition
    verify_transition(
        args.transition_receipt,
        execution_source_commit=None,
        script_sha=None,
        model_path=args.model_path,
        official_worker_path=args.official_worker,
        pilot_manifest_path=args.identity_allowlist,
        registry_root=args.registry_root,
        alias_ledger_path=args.alias_ledger,
        upstream_root=args.upstream_root,
        libero_root=Path(args.upstream_root) / ".." / "libero",
        output_root=str(out_root),
        gpu=0,
        physical_gpu=args.gpu,
        repo_root=None,
        nd_diagnostic_mode=True,  # FIT670 uses generalized verification
        expected_identity_count=670,
    )
    print("Transition receipt: VERIFIED")

    # ── Preflight-only ──
    if args.preflight_only:
        print("\nPREFLIGHT_ONLY: All validations passed. No model loaded.")
        return

    # ── Load model ──
    print(f"\n[*] Loading model on GPU {args.gpu}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    worker_path = Path(args.official_worker).resolve()
    saved_argv = sys.argv[:]
    dummy = "0" * 64
    sys.argv = [
        str(worker_path), "--suite", "libero_10", "--gpu", str(args.gpu),
        "--worker-id", f"fit670_shard_{args.shard_id}",
        "--model-path", str(args.model_path),
        "--manifest", str(args.identity_allowlist),
        "--output-root", str(out_root),
        "--upstream-root", str(args.upstream_root),
        "--worker-start-manifest-dir", str(out_root),
        "--prelease-gate-dir", str(out_root),
        "--queue-epoch-id", "FIT670",
        "--queue-manifest-sha256", dummy,
        "--canonical-manifest-sha256", dummy,
        "--runtime-config-sha256", dummy,
        "--protocol-config", str(args.identity_allowlist),
        "--processor-path", str(args.model_path),
        "--supervisor-pid", "0",
        "--supervisor-config-sha256", dummy,
        "--relay-archive-commit", "fit670",
        "--provenance-path", str(args.identity_allowlist),
        "--seed", str(args.seed),
    ]
    try:
        spec = importlib.util.spec_from_file_location("official_clean_worker", str(worker_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["official_clean_worker"] = module
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv

    module.set_official_seed(args.seed)
    model, processor, device, unnorm_key = module.load_policy()
    adapter = module.OfficialOpenVLAActionAdapter(
        model, processor, device, unnorm_key, center_crop=True,
        base_vla_name=str(args.model_path))

    print(f"  Model loaded: GPU {args.gpu}")

    # ── LIBERO setup ──
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_cache = {}

    # ── GPU info (record once) ──
    import torch
    gpu_identity = capture_gpu_identity(args.gpu)
    gpu_info = {
        "physical_gpu": args.gpu,
        "logical_gpu": 0,
        "gpu_uuid": gpu_identity.get("gpu_uuid", "UNAVAILABLE"),
        "pci_bus_id": gpu_identity.get("pci_bus_id", "UNAVAILABLE"),
        "gpu_name": gpu_identity.get("gpu_name", "N/A"),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "sdpa_available": hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
    }
    (worker_root / "GPU_IDENTITY.json").write_text(
        json.dumps(gpu_info, indent=2, sort_keys=True), encoding="utf-8")

    # ── Collect episodes ──
    staging = out_root.parent / f".{label}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)

    episodes_dir = out_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    t_start = time.time()

    for idx, ident in enumerate(identities):
        suite = ident["suite"]
        task_id = ident["task_id"]
        state_id = ident["state_id"]
        ep_id = ident["episode_id"]
        seed = ident["collection_seed"]
        declared_sha = ident["initial_state_sha256"]

        print(f"\n  [{idx+1}/{len(identities)}] {ep_id}...", end=" ", flush=True)
        t_ep = time.perf_counter()

        try:
            # Load init state from LIBERO
            if suite not in suite_cache:
                suite_cache[suite] = benchmark_dict[suite]()
            suite_obj = suite_cache[suite]
            task = suite_obj.get_task(task_id)
            states = suite_obj.get_task_init_states(task_id)
            if state_id >= len(states):
                raise CollectionHold(f"state_id {state_id} >= {len(states)}")
            canonical_state = copy.deepcopy(states[state_id])

            # Verify initial-state SHA
            init_sha = sha256_bytes(pickle.dumps(canonical_state, protocol=4))
            if init_sha != declared_sha:
                raise CollectionHold(
                    f"initial_state_sha mismatch: computed={init_sha[:16]} "
                    f"declared={declared_sha[:16]}")

            episode_data, target_path = capture_one_fit670_episode(
                module, suite, task_id, state_id, seed,
                str(args.registry_root), canonical_state, task, adapter,
                episodes_dir,
                save_student_rgb=not args.no_student_rgb,
            )

            if episode_data is None:
                # Already sealed (resume)
                results.append({
                    "episode_id": ep_id, "status": "SKIPPED",
                    "reason": "already_sealed",
                })
                print("SKIP (sealed)")
                continue

            elapsed = time.perf_counter() - t_ep
            results.append({
                "episode_id": ep_id, "status": "OK",
                "steps": episode_data["step_count"],
                "target": str(target_path),
            })
            print(f"steps={episode_data['step_count']} elapsed={elapsed:.0f}s OK")

        except CollectionHold as e:
            elapsed = time.perf_counter() - t_ep
            failures.append({"episode_id": ep_id, "error": str(e), "elapsed": elapsed})
            print(f"HOLD: {e}")
        except Exception as e:
            elapsed = time.perf_counter() - t_ep
            failures.append({"episode_id": ep_id, "error": f"{type(e).__name__}: {e}", "elapsed": elapsed})
            print(f"ERROR: {type(e).__name__}: {e}")

    # ── Seal worker output ──
    total_elapsed = time.time() - t_start
    worker_manifest = {
        "gate": "FIT670_ATOMIC_WORKER",
        "shard_id": args.shard_id,
        "gpu": args.gpu,
        "n_assigned": len(identities),
        "n_success": sum(1 for r in results if r["status"] == "OK"),
        "n_skipped": sum(1 for r in results if r["status"] == "SKIPPED"),
        "n_fail": len(failures),
        "total_steps": sum(r.get("steps", 0) for r in results if r["status"] == "OK"),
        "elapsed_s": round(total_elapsed, 1),
        "results": results,
        "failures": failures,
        "gpu_info": gpu_info,
    }
    (staging / "WORKER_MANIFEST.json").write_text(
        json.dumps(worker_manifest, indent=2, sort_keys=True), encoding="utf-8")

    seal_root(staging)
    staging.rename(worker_root)

    print(f"\nWorker shard {args.shard_id} complete:")
    print(f"  success: {worker_manifest['n_success']}")
    print(f"  skipped: {worker_manifest['n_skipped']}")
    print(f"  fail: {worker_manifest['n_fail']}")
    print(f"  total_steps: {worker_manifest['total_steps']}")
    print(f"  elapsed: {total_elapsed:.0f}s")
    print(f"  sealed: {worker_root}")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    {f['episode_id']}: {f['error'][:120]}")


if __name__ == "__main__":
    main()
