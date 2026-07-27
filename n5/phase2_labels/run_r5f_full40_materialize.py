"""[DeepSeek] R5-F: Corrected FIT Full40 Materialization.

Runs the corrected forward-before-capture collector across all 40 task
identities. Produces a sealed evidence root with relation-bound entity
poses. Designed for A/B independent materialization with identical digests.

Protocol: PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE
Collector fix: R5-C1 (sim.forward() before entity capture)
Resolver fix: R5-D (black_book alias-before-geom)

Usage (server):
  python n5/phase2_labels/run_r5f_full40_materialize.py \
    --model-path /path/to/openvla-checkpoint \
    --upstream-root /path/to/openvla-upstream \
    --official-worker /path/to/official_clean_worker.py \
    --pilot-manifest /path/to/GREC_FALLBACK_FULL40_MANIFEST.json \
    --registry-root /path/to/c1_v2_registry/run_A/per_task \
    --alias-ledger /path/to/c1_v2_registry/ALIAS_LEDGER.json \
    --output-root /path/to/output \
    --run-label A \
    --gpu 0
"""
import argparse, copy, hashlib, importlib, json, math, os, pickle
import platform, random, socket, subprocess, sys, time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
ALL_TASKS = [(suite, tid) for suite in HORIZONS for tid in range(10)]


class CollectionHold(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_value(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def mat_to_quat(m):
    values = [float(x) for x in m]
    a00, a01, a02, a10, a11, a12, a20, a21, a22 = values
    trace = a00 + a11 + a22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (0.25 * s, (a21 - a12) / s, (a02 - a20) / s, (a10 - a01) / s)
    elif a00 > a11 and a00 > a22:
        s = math.sqrt(1 + a00 - a11 - a22) * 2
        q = ((a21 - a12) / s, 0.25 * s, (a01 + a10) / s, (a02 + a20) / s)
    elif a11 > a22:
        s = math.sqrt(1 + a11 - a00 - a22) * 2
        q = ((a02 - a20) / s, (a01 + a10) / s, 0.25 * s, (a12 + a21) / s)
    else:
        s = math.sqrt(1 + a22 - a00 - a11) * 2
        q = ((a10 - a01) / s, (a02 + a20) / s, (a12 + a21) / s, 0.25 * s)
    norm = math.sqrt(sum(x * x for x in q))
    return [x / norm for x in q]


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def _verify_source_stability(qpos_before, qvel_before, act_before, time_before, data, step, label):
    qpos_after = data.qpos.copy()
    qvel_after = data.qvel.copy()
    act_after = data.act.copy() if hasattr(data, 'act') and data.act is not None else None
    time_after = float(data.time)
    pos_drift = float(np.max(np.abs(qpos_before - qpos_after)))
    vel_drift = float(np.max(np.abs(qvel_before - qvel_after)))
    time_drift = abs(float(time_before) - time_after)
    act_drift = 0.0
    if act_before is not None and act_after is not None:
        act_drift = float(np.max(np.abs(act_before - act_after)))
    if pos_drift > 0 or vel_drift > 0 or time_drift > 0 or act_drift > 0:
        raise CollectionHold(
            f"source state mutated by {label} at step {step}: "
            f"qpos_drift={pos_drift:.2e} qvel_drift={vel_drift:.2e} "
            f"time_drift={time_drift:.2e} act_drift={act_drift:.2e}")
    return True


def collect_entity(model, data, resolution):
    kind = str(resolution.get("entity_type") or "")
    entity_id = int(resolution.get("entity_id", -1))
    if kind == "body":
        if entity_id < 0 or entity_id >= int(model.nbody):
            raise CollectionHold(f"body id out of range: {entity_id}")
        actual_name = str(model.body(entity_id).name or "")
        pose = {"position": data.body_xpos[entity_id].tolist(),
                "quaternion": [float(x) for x in data.body_xquat[entity_id]]}
        parent = int(model.body_parentid[entity_id])
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": parent, "world_pose": pose}
    if kind == "site":
        if entity_id < 0 or entity_id >= int(model.nsite):
            raise CollectionHold(f"site id out of range: {entity_id}")
        actual_name = str(model.site(entity_id).name or "")
        body_id = int(model.site_bodyid[entity_id])
        pose = {"position": data.site_xpos[entity_id].tolist(),
                "quaternion": mat_to_quat(data.site_xmat[entity_id])}
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": body_id, "world_pose": pose}
    if kind == "geom":
        if entity_id < 0 or entity_id >= int(model.ngeom):
            raise CollectionHold(f"geom id out of range: {entity_id}")
        actual_name = str(model.geom(entity_id).name or "")
        body_id = int(model.geom_bodyid[entity_id])
        pose = {"position": data.geom_xpos[entity_id].tolist(),
                "quaternion": mat_to_quat(data.geom_xmat[entity_id])}
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": body_id, "world_pose": pose}
    raise CollectionHold(f"unsupported entity kind: {kind}")


def load_resolutions(registry_path):
    """Load relation-bound entity resolutions from C1-V2 per-task registry."""
    with open(registry_path) as f:
        data = json.load(f)
    legacy = data.get("legacy", data)
    relations = legacy.get("relations", [])
    unique = {}
    for rel in relations:
        for side in ("object_resolution", "target_resolution"):
            res = rel[side]
            if res.get("resolution") in ("EXACT_BODY", "EXACT_SITE", "EXACT_GEOM",
                                          "APPROVED_STRUCTURAL_ALIAS"):
                unique[(res["entity_type"], res["entity_id"])] = res
    return unique, relations


def capture_one_episode(module, args, suite, task_idx, state_id, registry_dir,
                        state, task, suite_seed, adapter):
    """Collect a single episode using the corrected forward-before-capture protocol."""
    from experiments.robot.libero.libero_utils import get_libero_image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    reg_path = Path(registry_dir) / f"{suite}_task_{task_idx:02d}.json"
    if not reg_path.is_file():
        raise CollectionHold(f"registry missing: {reg_path}")
    resolutions, relations = load_resolutions(str(reg_path))
    if not resolutions:
        raise CollectionHold(f"no relation-bound entities for {suite}/task_{task_idx:02d}")

    bddl_root = Path(get_libero_path("bddl_files")).resolve()
    task_bddl = (bddl_root / task.problem_folder / task.bddl_file).resolve()

    module.set_official_seed(suite_seed)
    env = OffScreenRenderEnv(bddl_file_name=str(task_bddl), camera_heights=256, camera_widths=256)
    env.seed(suite_seed)
    env.reset()
    obs = env.set_init_state(copy.deepcopy(state))
    for _ in range(int(module.NUM_STEPS_WAIT)):
        obs = env.step([0, 0, 0, 0, 0, 0, -1])[0]

    rows = []
    privileged = []
    generation_counts = []
    try:
        for step in range(HORIZONS[suite]):
            # R5-C1: forward-before-capture protocol
            qpos_pre = env.sim.data.qpos.copy()
            qvel_pre = env.sim.data.qvel.copy()
            act_pre = env.sim.data.act.copy() if hasattr(env.sim.data, 'act') and env.sim.data.act is not None else None
            time_pre = float(env.sim.data.time)
            env.sim.forward()
            _verify_source_stability(qpos_pre, qvel_pre, act_pre, time_pre, env.sim.data, step, "capture_forward")
            model = env.sim.model; data = env.sim.data
            sim_state = env.sim.get_state()
            entities = [collect_entity(model, data, res) for res in resolutions.values()]

            privileged.append({
                "step": step, "suite": suite, "task_idx": task_idx, "state_id": state_id,
                "sim_state": {"time": float(data.time), "qpos": sim_state.qpos.tolist(),
                              "qvel": sim_state.qvel.tolist(),
                              "act": getattr(sim_state, "act", None).tolist() if getattr(sim_state, "act", None) is not None else None},
                "robot0_eef_pos": jsonable(obs.get("robot0_eef_pos", [])),
                "robot0_eef_quat": jsonable(obs.get("robot0_eef_quat", [])),
                "robot0_gripper_qpos": jsonable(obs.get("robot0_gripper_qpos", [])),
                "object_state": jsonable(obs.get("object-state", [])),
                "entities": entities,
                "forward_before_capture": True,
                "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
                "contact_count": int(data.ncon),
            })

            image = get_libero_image(obs, 224)
            clean_action, generation, score_meta = adapter.predict_action_with_scores(image, str(task.language))
            count = score_meta.get("generation_passes_per_step")
            if isinstance(count, bool) or not isinstance(count, int) or count != 1:
                raise CollectionHold(f"generation pass count: {count}")
            generation_counts.append(count)
            score_action = [float(x) for x in jsonable(score_meta["score_action"])]
            raw_action = [float(x) for x in jsonable(clean_action)]
            if len(raw_action) != 7 or len(score_action) != 7:
                raise CollectionHold(f"action shape failed at step {step}")
            if max(abs(a - b) for a, b in zip(raw_action, score_action)) > 1e-6:
                raise CollectionHold(f"action parity failed at step {step}")
            executed = [float(x) for x in jsonable(adapter.postprocess(clean_action))]
            if len(executed) != 7:
                raise CollectionHold(f"executed action shape failed at step {step}")
            rows.append({
                "step": step, "suite": suite, "task_idx": task_idx, "state_id": state_id,
                "action_raw_7d": raw_action, "score_action_7d": score_action,
                "action_env_7d": executed, "generation_passes_per_step": count,
                "single_generation_parity_pass": True, "action_mutation_by_detector": False,
            })
            obs, _reward, done, _info = env.step(executed)
            if done:
                break
    finally:
        env.close()

    if not generation_counts or any(x != 1 for x in generation_counts):
        raise CollectionHold("generation closure failed")

    task_bddl_sha = sha256_file(task_bddl)
    return {
        "episode_id": f"{suite}/task_{task_idx:02d}/state_{state_id}",
        "suite": suite, "task_id": task_idx, "state_id": state_id,
        "collection_seed": suite_seed,
        "task_bddl_sha256": task_bddl_sha,
        "registry_task_sha256": sha256_file(str(reg_path)),
        "step_count": len(rows), "official_horizon": HORIZONS[suite],
        "generation_passes_per_step": generation_counts,
        "steps": rows, "telemetry": privileged,
        "relations": relations,
        "source_mode": "NEW_FIT_ONLY_CORRECTED_COLLECTOR",
        "forward_before_capture": True,
        "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
        "original_payload_target_pose_available": False,
        "model_inference": True, "attack_enabled": False,
        "detector_loaded": False, "teacher_labels_generated": False,
    }


def seal_root(staging):
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    sums = "".join(f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}\n" for p in payload)
    (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "file_count": str(len(payload))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", required=True, choices=["A", "B"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--worker-id", default="full40")
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve() / f"run_{args.run_label}"
    if out_root.exists():
        raise SystemExit(f"output exists: {out_root}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Load module
    worker = Path(args.official_worker).resolve()
    if not worker.is_file():
        raise SystemExit(f"worker missing: {worker}")
    argv = [
        str(worker), "--suite", "libero_10", "--gpu", str(args.gpu),
        "--worker-id", args.worker_id, "--model-path", str(args.model_path),
        "--manifest", str(args.pilot_manifest), "--output-root", str(out_root.parent),
        "--upstream-root", str(args.upstream_root),
        "--worker-start-manifest-dir", str(out_root.parent),
        "--prelease-gate-dir", str(out_root.parent),
        "--queue-epoch-id", "GREC_FULL40_CORRECTED",
        "--queue-manifest-sha256", "0" * 64,
        "--canonical-manifest-sha256", "0" * 64,
        "--runtime-config-sha256", "0" * 64,
        "--protocol-config", str(args.pilot_manifest),
        "--processor-path", str(args.model_path),
        "--supervisor-pid", "0", "--supervisor-config-sha256", "0" * 64,
        "--relay-archive-commit", "GREC_FULL40_CORRECTED",
        "--provenance-path", str(args.pilot_manifest),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        spec = importlib.util.spec_from_file_location("official_clean_worker", str(worker))
        if spec is None or spec.loader is None:
            raise SystemExit("cannot load worker")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv

    module.set_official_seed(args.seed)
    model, processor, device, unnorm_key = module.load_policy()
    adapter = module.OfficialOpenVLAActionAdapter(
        model, processor, device, unnorm_key, center_crop=True,
        base_vla_name=str(args.model_path))

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero import benchmark

    staging = out_root.parent / f".{out_root.name}.staging.{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "episodes").mkdir()

    print("=" * 70)
    print(f"[DeepSeek] R5-F: Corrected FIT Full40 Materialization — Run {args.run_label}")
    print(f"  model={args.model_path}  gpu={args.gpu}  seed={args.seed}")
    print("=" * 70)

    suite_dict = benchmark.get_benchmark_dict()
    collections = []
    failures = []
    total_start = time.time()

    for suite, task_idx in ALL_TASKS:
        task_key = f"{suite}/task_{task_idx:02d}"
        print(f"\n  {task_key}...", end=" ", flush=True)
        try:
            suite_obj = suite_dict[suite]()
            task = suite_obj.get_task(task_idx)
            states = suite_obj.get_task_init_states(task_idx)
            state_id = 0  # Use state 0 for each task
            state = copy.deepcopy(states[state_id])

            episode = capture_one_episode(
                module, args, suite, task_idx, state_id,
                str(args.registry_root), state, task, args.seed, adapter)

            ep_dir = staging / "episodes" / f"{suite}_task_{task_idx:02d}"
            ep_dir.mkdir()
            (ep_dir / "episode.json").write_text(
                json.dumps(episode, indent=2, sort_keys=True, default=str), encoding="utf-8")

            ep_sha = sha256_file(ep_dir / "episode.json")
            collections.append({
                "task_key": task_key, "episode_id": episode["episode_id"],
                "steps": episode["step_count"],
                "entities": len(episode["telemetry"][0]["entities"]) if episode["telemetry"] else 0,
                "sha256": ep_sha,
            })
            print(f"steps={episode['step_count']} entities={len(episode['telemetry'][0]['entities']) if episode['telemetry'] else 0} OK")
        except Exception as e:
            print(f"HOLD: {e}")
            failures.append({"task_key": task_key, "error": str(e)})

    elapsed = time.time() - total_start

    # Manifest
    manifest = {
        "gate": "R5-F_CORRECTED_FULL40_MATERIALIZATION",
        "schema": "G_REC_CORRECTED_FULL40_V1",
        "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
        "run_label": args.run_label,
        "status": "COMPLETE" if not failures else "PARTIAL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": git_value(Path(__file__).resolve().parent.parent.parent, "rev-parse", "HEAD"),
        "source_tree": git_value(Path(__file__).resolve().parent.parent.parent, "rev-parse", "HEAD^{tree}"),
        "model_path": str(args.model_path.resolve()),
        "upstream_root": str(args.upstream_root.resolve()),
        "upstream_commit": git_value(args.upstream_root, "rev-parse", "HEAD"),
        "registry_root": str(args.registry_root),
        "alias_ledger_sha256": sha256_file(args.alias_ledger),
        "n_tasks_attempted": len(ALL_TASKS),
        "n_tasks_collected": len(collections),
        "n_tasks_failed": len(failures),
        "total_steps": sum(c["steps"] for c in collections),
        "elapsed_s": elapsed,
        "collections": collections,
        "failures": failures,
        "forward_before_capture": True,
        "no_detector": True, "attack_enabled": False,
        "teacher_labels_generated": False,
        "protected_payload_read": False,
        "consumer_eligible": False,
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (staging / "SEAL_RECEIPT.json").write_text(json.dumps({
        "schema": "V23_G_REC_CORRECTED_FULL40_SEAL_V1",
        "status": "SEALED_AFTER_PAYLOAD",
        "run_label": args.run_label,
    }, indent=2, sort_keys=True), encoding="utf-8")

    seal = seal_root(staging)
    staging.rename(out_root)

    print(f"\n{'=' * 70}")
    print(f"Run {args.run_label}: {len(collections)}/{len(ALL_TASKS)} episodes collected")
    print(f"  Failures: {len(failures)}")
    print(f"  Total steps: {sum(c['steps'] for c in collections)}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"  Sealed: {out_root}")
    print(f"  SHA256SUMS: {seal['sha256sums_sha256']}")

    if failures:
        for f in failures:
            print(f"  FAIL: {f['task_key']}: {f['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
