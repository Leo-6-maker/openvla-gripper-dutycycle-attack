"""Collect a FIT-only clean telemetry canary with direct MuJoCo poses.

This deliberately runs the official clean OpenVLA action path, but never
loads a detector, changes an action, emits an attack, or creates Teacher
labels.  The output is a replacement telemetry canary, not a reconstruction of
the historical episode payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import pickle
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping


HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
FORBIDDEN = {"cal", "check", "g10", "t2r", "attack"}


class CollectionHold(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def git_value(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def reject_path(path: Path) -> None:
    parts = {part.lower() for part in path.resolve().parts}
    if parts & FORBIDDEN:
        raise CollectionHold(f"forbidden path: {path}")


def qnorm(q: Any) -> list[float]:
    values = [float(x) for x in q]
    norm = math.sqrt(sum(x * x for x in values))
    if len(values) != 4 or not math.isfinite(norm) or norm <= 0:
        raise CollectionHold("invalid quaternion")
    return [x / norm for x in values]


def mat_to_quat(m: Any) -> list[float]:
    values = [float(x) for x in m]
    if len(values) != 9 or not all(math.isfinite(x) for x in values):
        raise CollectionHold("invalid rotation matrix")
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
    return qnorm(q)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def load_official_worker(path: Path, args: argparse.Namespace, pilot_path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise CollectionHold(f"official worker missing: {path}")
    argv = [
        str(path), "--suite", args.suite, "--gpu", str(args.gpu), "--worker-id", args.worker_id,
        "--model-path", str(args.model_path), "--manifest", str(pilot_path),
        "--output-root", str(args.output.parent), "--upstream-root", str(args.upstream_root),
        "--worker-start-manifest-dir", str(args.output.parent), "--prelease-gate-dir", str(args.output.parent),
        "--queue-epoch-id", "GREC_FALLBACK_CANARY", "--queue-manifest-sha256", "0" * 64,
        "--canonical-manifest-sha256", "0" * 64, "--runtime-config-sha256", "0" * 64,
        "--protocol-config", str(pilot_path), "--processor-path", str(args.model_path),
        "--supervisor-pid", "0", "--supervisor-config-sha256", "0" * 64,
        "--relay-archive-commit", "GREC_FALLBACK_CANARY", "--provenance-path", str(pilot_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        spec = importlib.util.spec_from_file_location("official_clean_worker_for_grec_canary", path)
        if spec is None or spec.loader is None:
            raise CollectionHold("cannot load official worker")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


def verify_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(record["source_episode_root"])).resolve()
    reject_path(root)
    files = {str(x["name"]): x for x in record.get("source_files", [])}
    meta_path = root / "episode_metadata.json"
    spec = files.get("episode_metadata.json")
    if spec is None or not meta_path.is_file() or meta_path.is_symlink() or sha256_file(meta_path) != spec.get("sha256"):
        raise CollectionHold(f"pilot metadata binding failed: {record['episode_id']}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"metadata": meta, "metadata_sha256": sha256_file(meta_path), "root": str(root)}


def capture_contacts(env: Any) -> dict[str, Any]:
    pairs = []
    data = env.sim.data
    model = env.sim.model
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        left = model.geom_id2name(int(contact.geom1))
        right = model.geom_id2name(int(contact.geom2))
        if left and right:
            pairs.append([str(left), str(right)])
    return {"contact_count": len(pairs), "mujoco_contact_pairs": pairs, "contact_capture_valid": True}


def collect_entity(model: Any, data: Any, resolution: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(resolution.get("entity_type") or "")
    entity_id = int(resolution.get("entity_id", -1))
    if kind == "body":
        if entity_id < 0 or entity_id >= int(model.nbody):
            raise CollectionHold(f"body id out of range: {entity_id}")
        actual_name = str(model.body(entity_id).name or "")
        pose = {"position": data.body_xpos[entity_id].tolist(), "quaternion": qnorm(data.body_xquat[entity_id])}
        parent = int(model.body_parentid[entity_id])
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name, "parent_body_id": parent, "world_pose": pose}
    if kind == "site":
        if entity_id < 0 or entity_id >= int(model.nsite):
            raise CollectionHold(f"site id out of range: {entity_id}")
        actual_name = str(model.site(entity_id).name or "")
        body_id = int(model.site_bodyid[entity_id])
        pose = {"position": data.site_xpos[entity_id].tolist(), "quaternion": mat_to_quat(data.site_xmat[entity_id])}
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name, "parent_body_id": body_id, "world_pose": pose}
    if kind == "geom":
        if entity_id < 0 or entity_id >= int(model.ngeom):
            raise CollectionHold(f"geom id out of range: {entity_id}")
        actual_name = str(model.geom(entity_id).name or "")
        body_id = int(model.geom_bodyid[entity_id])
        pose = {"position": data.geom_xpos[entity_id].tolist(), "quaternion": mat_to_quat(data.geom_xmat[entity_id])}
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name, "parent_body_id": body_id, "world_pose": pose}
    raise CollectionHold(f"unsupported entity kind: {kind}")


def registry_task(registry_root: Path, suite: str, task_id: int) -> tuple[dict[str, Any], Path]:
    path = registry_root / "run_A" / "per_task" / f"{suite}_task_{task_id:02d}.json"
    if not path.is_file() or path.is_symlink():
        raise CollectionHold(f"registry task missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_key") != f"{suite}/task_{task_id:02d}" or payload.get("legacy", {}).get("status") != "OK":
        raise CollectionHold(f"registry task binding failed: {path}")
    return payload["legacy"], path


def capture_episode(module: ModuleType, args: argparse.Namespace, record: Mapping[str, Any], registry_root: Path, state: Any, task: Any, suite_seed: int, adapter: Any) -> dict[str, Any]:
    from experiments.robot.libero.libero_utils import get_libero_image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = str(record["suite"]); task_id = int(record["task_id"]); state_id = int(record["state_id"])
    legacy, registry_path = registry_task(registry_root, suite, task_id)
    bddl_root = Path(get_libero_path("bddl_files")).resolve()
    task_bddl = (bddl_root / task.problem_folder / task.bddl_file).resolve()
    if sha256_file(task_bddl) != legacy.get("bddl_sha256"):
        raise CollectionHold(f"BDDL SHA mismatch: {record['episode_id']}")
    module.set_official_seed(suite_seed)
    env = OffScreenRenderEnv(bddl_file_name=str(task_bddl), camera_heights=256, camera_widths=256)
    env.seed(suite_seed)
    env.reset()
    obs = env.set_init_state(copy.deepcopy(state))
    for _ in range(int(module.NUM_STEPS_WAIT)):
        obs = env.step([0, 0, 0, 0, 0, 0, -1])[0]

    resolved_relations = list(legacy.get("relations", []))
    unique_resolutions: dict[tuple[str, int], Mapping[str, Any]] = {}
    for relation in resolved_relations:
        for side in ("object_resolution", "target_resolution"):
            resolution = relation[side]
            unique_resolutions[(str(resolution.get("entity_type")), int(resolution.get("entity_id", -1)))] = resolution
    rows = []
    privileged = []
    generation_counts = []
    try:
        for step in range(HORIZONS[suite]):
            model = env.sim.model; data = env.sim.data
            entities = [collect_entity(model, data, resolution) for resolution in unique_resolutions.values()]
            sim_state = env.sim.get_state()
            privileged.append({
                "step": step, "suite": suite, "task_idx": task_id, "state_id": state_id,
                "sim_state": {"time": float(data.time), "qpos": sim_state.qpos.tolist(), "qvel": sim_state.qvel.tolist(), "act": getattr(sim_state, "act", None).tolist() if getattr(sim_state, "act", None) is not None else None},
                "robot0_eef_pos": jsonable(obs.get("robot0_eef_pos", [])),
                "robot0_eef_quat": jsonable(obs.get("robot0_eef_quat", [])),
                "robot0_gripper_qpos": jsonable(obs.get("robot0_gripper_qpos", [])),
                "object_state": jsonable(obs.get("object-state", [])),
                "entities": entities,
                **capture_contacts(env),
            })
            image = get_libero_image(obs, 224)
            clean_action, generation, score_meta = adapter.predict_action_with_scores(image, str(task.language))
            count = score_meta.get("generation_passes_per_step")
            if isinstance(count, bool) or not isinstance(count, int) or count != 1:
                raise CollectionHold(f"generation pass count: {count}")
            generation_counts.append(count)
            score_action = [float(x) for x in jsonable(score_meta["score_action"])]
            raw_action = [float(x) for x in jsonable(clean_action)]
            if len(raw_action) != 7 or len(score_action) != 7 or max(abs(a - b) for a, b in zip(raw_action, score_action)) > 1e-6:
                raise CollectionHold(f"official action parity failed at step {step}")
            executed = [float(x) for x in jsonable(adapter.postprocess(clean_action))]
            if len(executed) != 7:
                raise CollectionHold(f"executed action shape failed at step {step}")
            rows.append({
                "step": step, "suite": suite, "task_idx": task_id, "state_id": state_id,
                "action_raw_7d": raw_action, "score_action_7d": score_action, "action_env_7d": executed,
                "generation_passes_per_step": count, "single_generation_parity_pass": True,
                "action_mutation_by_detector": False,
            })
            obs, _reward, done, _info = env.step(executed)
            if done:
                break
    finally:
        env.close()
    if not generation_counts or any(x != 1 for x in generation_counts):
        raise CollectionHold("generation closure failed")
    return {
        "episode_id": record["episode_id"], "suite": suite, "task_id": task_id, "state_id": state_id,
        "collection_seed": suite_seed, "source_parent_identity": record["episode_id"],
        "task_bddl_sha256": sha256_file(task_bddl), "registry_task_sha256": sha256_file(registry_path),
        "step_count": len(rows), "official_horizon": HORIZONS[suite],
        "generation_passes_per_step": generation_counts, "steps": rows, "telemetry": privileged,
        "relations": resolved_relations, "source_mode": "NEW_FIT_ONLY_CLEAN_RUNTIME_TELEMETRY",
        "original_payload_target_pose_available": False, "model_inference": True,
        "attack_enabled": False, "detector_loaded": False, "teacher_labels_generated": False,
    }


def seal_root(staging: Path) -> dict[str, str]:
    payload = sorted(path for path in staging.rglob("*") if path.is_file())
    sums = "".join(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in payload)
    (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "file_count": str(len(payload))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.model_path, args.upstream_root, args.official_worker, args.pilot_manifest, args.registry_root, args.alias_ledger):
        reject_path(path)
    if args.output.exists() or args.output.is_symlink():
        raise CollectionHold(f"output exists: {args.output}")
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    if pilot.get("protected_payload_read") is not False or pilot.get("no_attack") is not True:
        raise CollectionHold("pilot manifest boundary failed")
    records = sorted((row for row in pilot.get("records", []) if row.get("suite") == args.suite), key=lambda row: row["episode_id"])
    if not records:
        raise CollectionHold(f"suite missing from pilot: {args.suite}")
    record = records[0]
    source_meta = verify_source_record(record)
    declared_model = source_meta["metadata"].get("checkpoint_path_verified") or source_meta["metadata"].get("checkpoint_path_declared")
    if not isinstance(declared_model, str) or Path(declared_model).resolve() != args.model_path.resolve():
        raise CollectionHold(f"checkpoint path is not the source-bound path: declared={declared_model} actual={args.model_path.resolve()}")
    if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
        raise CollectionHold("model checkpoint missing")
    alias = json.loads(args.alias_ledger.read_text(encoding="utf-8"))
    if not isinstance(alias.get("entries"), list):
        raise CollectionHold("alias ledger missing")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    random.seed(args.seed); os.environ["PYTHONHASHSEED"] = str(args.seed)
    module = load_official_worker(args.official_worker, args, args.pilot_manifest)
    module.set_official_seed(args.seed)
    model, processor, device, unnorm_key = module.load_policy()
    adapter = module.OfficialOpenVLAActionAdapter(model, processor, device, unnorm_key, center_crop=True, base_vla_name=str(args.model_path))
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero import benchmark
    suite_instance = benchmark.get_benchmark_dict()[args.suite]()
    task_id = int(record["task_id"]); state_id = int(record["state_id"])
    task = suite_instance.get_task(task_id)
    state = suite_instance.get_task_init_states(task_id)[state_id]
    if sha256_bytes(pickle.dumps(state, protocol=4)) != source_meta["metadata"].get("initial_state_sha256"):
        raise CollectionHold("initial state binding failed")
    collection = capture_episode(module, args, record, args.registry_root, state, task, args.seed, adapter)
    final_parent = args.output.parent
    final_parent.mkdir(parents=True, exist_ok=True)
    staging = final_parent / f".{args.output.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise CollectionHold(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    manifest = {
        "schema": "V23_G_REC_DATA_FALLBACK_CANARY_V1",
        "status": "DERIVED_FIT_ONLY_CLEAN_TELEMETRY",
        "source_parent_identity": record["episode_id"],
        "collection_identity_is_original_payload": False,
        "pilot_manifest_sha256": sha256_file(args.pilot_manifest),
        "registry_task_sha256": collection["registry_task_sha256"],
        "alias_ledger_sha256": sha256_file(args.alias_ledger),
        "official_worker_sha256": sha256_file(args.official_worker),
        "collector_source_commit": git_value(args.official_worker.parent.parent, "rev-parse", "HEAD"),
        "collector_source_tree": git_value(args.official_worker.parent.parent, "rev-parse", "HEAD^{tree}"),
        "upstream_root": str(args.upstream_root.resolve()),
        "upstream_commit": git_value(args.upstream_root, "rev-parse", "HEAD"),
        "libero_root": str(Path(get_libero_path("bddl_files")).resolve().parents[2]),
        "model_path": str(args.model_path.resolve()),
        "model_tree_sha256": module.checkpoint_tree_fingerprint(args.model_path)[0],
        "processor_sha256": sha256_file(args.model_path / "preprocessor_config.json"),
        "environment": {"python": sys.executable, "python_version": platform.python_version(), "torch": module.torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "hostname": socket.gethostname()},
        "no_detector": True, "attack_enabled": False, "teacher_labels_generated": False,
        "protected_payload_read": False, "source_mode": "NEW_FIT_ONLY_CLEAN_RUNTIME_TELEMETRY",
        "original_payload_target_pose_available": False,
    }
    (staging / "FALLBACK_CANARY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "episode.json").write_text(json.dumps(collection, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (staging / "SEAL_RECEIPT.json").write_text(json.dumps({"schema": "V23_G_REC_DATA_FALLBACK_CANARY_SEAL_V1", "status": "SEALED_AFTER_PAYLOAD"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal = seal_root(staging)
    if args.output.exists():
        raise CollectionHold(f"output appeared during collection: {args.output}")
    os.rename(staging, args.output)
    return {"status": "PASS_FIT_ONLY_CANARY", "output": str(args.output), "episode_id": record["episode_id"], "step_count": collection["step_count"], "seal": seal}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=sorted(HORIZONS))
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "error": str(exc), "attack_enabled": False, "detector_loaded": False}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
