#!/usr/bin/env python3
"""Resident official-action-path CLEAN worker for one LIBERO suite."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

try:
    import fcntl
except ImportError:  # pragma: no cover - the resident runner is Linux-only
    fcntl = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"])
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--upstream-root", required=True, type=Path)
    return ap.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "flash_attention_2")
sys.path.insert(0, str(args.upstream_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from gripper_attack.official_libero_protocol import OFFICIAL_HORIZONS, NUM_STEPS_WAIT
from gripper_attack.official_detector_features import (
    CANONICAL_25D_FEATURES,
    CLEAN_POLICY_FEATURE_NAMES,
    SC5StreamingFeatureAdapterV2,
)
from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def state_sha(state: object) -> str:
    return sha256_bytes(pickle.dumps(state, protocol=4))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def disk_free_gb() -> float:
    return float(shutil.disk_usage(args.output_root).free / (1024 ** 3))


def update_global_ledger(cell_id: str, status: str, *, result_status: str = "", attempt_increment: bool = False) -> None:
    """Atomically update one real canonical cell in the frozen global ledger."""
    ledger = args.output_root / "manifests" / "OFFICIAL_GLOBAL_CELL_LEDGER_V1.csv"
    if not ledger.is_file():
        raise SystemExit(f"GLOBAL_LEDGER_MISSING {ledger}")
    lock_path = ledger.with_suffix(ledger.suffix + ".lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with ledger.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        matches = [row for row in rows if row["cell_id"] == cell_id]
        if len(matches) != 1:
            raise SystemExit(f"GLOBAL_LEDGER_CELL_ID_FAIL {cell_id} matches={len(matches)}")
        row = matches[0]
        if status in {"LEASED", "RUNNING"} and row["status"] in {"LEASED", "RUNNING"}:
            if row["worker_id"] and row["worker_id"] != args.worker_id:
                raise SystemExit(f"GLOBAL_LEDGER_CONCURRENT_LEASE {cell_id} owner={row['worker_id']}")
        now = str(time.time())
        row.update({
            "status": status,
            "worker_id": args.worker_id if status in {"LEASED", "RUNNING"} else row.get("worker_id", args.worker_id),
            "gpu_id": str(args.gpu) if status in {"LEASED", "RUNNING"} else row.get("gpu_id", ""),
            "pid": str(os.getpid()) if status in {"LEASED", "RUNNING"} else row.get("pid", ""),
            "lease_timestamp": now if status == "LEASED" else row.get("lease_timestamp", ""),
            "last_heartbeat": now,
            "result_status": result_status,
        })
        if attempt_increment:
            row["attempt_count"] = str(int(row.get("attempt_count") or 0) + 1)
        fields = list(rows[0].keys())
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=ledger.parent, delete=False) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            temp_name = tmp.name
        os.replace(temp_name, ledger)
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def seal(path: Path) -> str:
    rows = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or item.name == "artifact_sha256.json":
            continue
        rows.append({"path": item.relative_to(path).as_posix(), "size": item.stat().st_size, "sha256": sha256_file(item)})
    payload = {"files": rows, "recursive_sha256": json_sha(rows)}
    write_json(path / "artifact_sha256.json", payload)
    return payload["recursive_sha256"]


def artifact_valid(path: Path) -> bool:
    manifest = path / "artifact_sha256.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        rows = payload["files"]
        if payload["recursive_sha256"] != json_sha(rows):
            return False
        return all((path / row["path"]).is_file() and sha256_file(path / row["path"]) == row["sha256"] for row in rows)
    except Exception:
        return False


def retryable_runtime_error(exc: Exception) -> bool:
    """Allow only transient/runtime classes one automatic retry."""
    text = f"{type(exc).__name__}:{exc}".lower()
    markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "filesystem",
        "input/output error",
        "i/o error",
        "environment reset",
        "egl",
        "mujoco",
        "model load",
        "checkpoint",
    )
    return isinstance(exc, (OSError, RuntimeError, TimeoutError)) and any(marker in text for marker in markers)


def load_rows() -> list[dict[str, str]]:
    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f) if row["suite"] == args.suite]
    if len(rows) != 500:
        raise SystemExit(f"OFFICIAL_MANIFEST_SUITE_FAIL {args.suite}: {len(rows)}")
    return rows


def load_artifact_provenance() -> dict[str, str]:
    path = args.output_root / "provenance" / "UPSTREAM_PROVENANCE.json"
    if not path.is_file():
        raise SystemExit(f"PROVENANCE_MISSING {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = payload["checkpoints"][args.suite]
    source_root = Path(__file__).resolve().parents[1] / "src" / "gripper_attack"
    source_hash = json_sha({
        "official_libero_protocol.py": sha256_file(source_root / "official_libero_protocol.py"),
        "official_openvla_adapter.py": sha256_file(source_root / "official_openvla_adapter.py"),
        "official_detector_features.py": sha256_file(source_root / "official_detector_features.py"),
        "official_clean_worker.py": sha256_file(Path(__file__).resolve()),
    })
    return {
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "checkpoint_config_sha256": checkpoint["config_sha256"],
        "dataset_statistics_sha256": checkpoint["dataset_statistics_sha256"],
        "processor_preprocessor_sha256": payload["processor_files"]["preprocessor_config_sha256"],
        "processor_tokenizer_sha256": payload["processor_files"]["tokenizer_config_sha256"],
        "official_adapter_sha256": source_hash,
        "official_protocol_id": payload["protocol_id"],
    }


def load_policy():
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(args.model_path),
        load_in_8bit=False,
        load_in_4bit=False,
    )
    model = get_model(cfg)
    processor = get_processor(cfg)
    model.eval()
    device = next(model.parameters()).device
    key = args.suite
    stats = getattr(model, "norm_stats", {})
    if key not in stats and f"{key}_no_noops" in stats:
        key = f"{key}_no_noops"
    if key not in stats:
        raise RuntimeError(f"missing unnorm key {args.suite}")
    return model, processor, device, key


def make_env(task):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.robot.libero.libero_utils import get_libero_dummy_action

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    return env, get_libero_dummy_action("openvla")


def mujoco_contact_pairs(env) -> list[list[str]]:
    """Return compact privileged contact evidence for offline teacher labeling."""
    pairs: list[list[str]] = []
    try:
        data = env.sim.data
        model = env.sim.model
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            first = model.geom_id2name(int(contact.geom1))
            second = model.geom_id2name(int(contact.geom2))
            if first and second:
                pairs.append([str(first), str(second)])
    except Exception:
        return []
    return pairs


def run_episode(adapter, task, initial_state, row: dict[str, str], out: Path, model_path: Path, artifact_provenance: dict[str, str]) -> dict[str, object]:
    from experiments.robot.libero.libero_utils import get_libero_image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = args.suite
    task_idx = int(row["task_idx"])
    state_id = int(row["state_id"])
    horizon = OFFICIAL_HORIZONS[suite]
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    obs = env.set_init_state(copy.deepcopy(initial_state))
    dummy = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(NUM_STEPS_WAIT):
        obs, _reward, _done, _info = env.step(dummy)

    step_rows: list[dict[str, object]] = []
    policy_intent_rows: list[dict[str, object]] = []
    privileged_rows: list[dict[str, object]] = []
    feature_stream = SC5StreamingFeatureAdapterV2()
    try:
        eef_site = env.sim.model.site_name2id("gripper0_grip_site")
    except Exception:
        eef_site = None
    previous_eef: np.ndarray | None = None
    success = False
    try:
        for step in range(horizon):
            image = get_libero_image(obs, 224)
            gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float32).reshape(-1)
            qpos_sum = float(gripper_qpos[:2].sum()) if gripper_qpos.size >= 2 else 0.0
            opening_proxy = float(np.abs(gripper_qpos[:2]).sum()) if gripper_qpos.size >= 2 else 0.0
            if eef_site is not None:
                eef = np.asarray(env.sim.data.site_xpos[eef_site], dtype=np.float32).reshape(3).copy()
            else:
                eef = np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float32).reshape(-1)
                if eef.size != 3:
                    raise RuntimeError("missing robot0_eef_pos/gripper0_grip_site for 25D features")
                eef = eef.copy()
            velocity = np.zeros(3, dtype=np.float32) if previous_eef is None else eef - previous_eef
            previous_eef = eef.copy()
            privileged_rows.append({
                "step": step,
                "suite": suite,
                "task_idx": task_idx,
                "state_id": state_id,
                "task_language": str(task.language),
                "robot0_eef_pos": np.asarray(obs.get("robot0_eef_pos", [])).tolist(),
                "robot0_eef_quat": np.asarray(obs.get("robot0_eef_quat", [])).tolist(),
                "robot0_gripper_qpos": np.asarray(obs.get("robot0_gripper_qpos", [])).tolist(),
                "object_state": np.asarray(obs.get("object-state", [])).tolist(),
                "mujoco_contact_pairs": mujoco_contact_pairs(env),
            })
            clean_action, _official_meta = adapter.predict_action(image, str(task.language))
            score_action, generation, score_meta = adapter.score_action(image, str(task.language))
            action_error = float(np.max(np.abs(clean_action - score_action)))
            env_action = adapter.postprocess(clean_action)
            tokens = generation.sequences[0, -7:].detach().cpu().tolist()
            score_summary = []
            for scores in getattr(generation, "scores", [])[:7]:
                probs = torch.softmax(scores[0].float(), dim=-1)
                top_prob, top_id = torch.max(probs, dim=-1)
                score_summary.append({"top_token": int(top_id), "top_probability": float(top_prob)})
            policy, policy_top_ids, policy_top_logits = adapter.detector_policy_features(generation)
            stream = feature_stream.update(
                step_id=step,
                raw_gripper=float(clean_action[-1]),
                env_gripper=float(env_action[-1]),
                gripper_qpos=qpos_sum,
                gripper_opening_proxy=opening_proxy,
                eef_x=float(eef[0]),
                eef_y=float(eef[1]),
                eef_z=float(eef[2]),
                eef_vx=float(velocity[0]),
                eef_vy=float(velocity[1]),
                eef_vz=float(velocity[2]),
                action_dx=float(env_action[0]),
                action_dy=float(env_action[1]),
                action_dz=float(env_action[2]),
                action_gripper=float(clean_action[-1]),
            )
            if not stream.get("valid") or stream.get("features") is None:
                raise RuntimeError(f"invalid official 25D feature stream at step {step}: {stream.get('error', '')}")
            features_25d = [float(stream["features"][name]) for name in CANONICAL_25D_FEATURES]
            if len(features_25d) != 25 or not np.isfinite(np.asarray(features_25d, dtype=np.float32)).all():
                raise RuntimeError(f"invalid official 25D feature vector at step {step}")
            policy_named = {name: policy[index] for index, name in enumerate(CLEAN_POLICY_FEATURE_NAMES)}
            step_rows.append({
                "step": step,
                "suite": suite,
                "task_idx": task_idx,
                "state_id": state_id,
                "condition": "CLEAN",
                "official_execution": True,
                "features_25d": features_25d,
                "clean_policy_intent_9d": policy,
                **policy_named,
                "action_raw": [float(x) for x in np.asarray(clean_action)],
                "action_env": [float(x) for x in np.asarray(env_action)],
                "clean_action_raw_7d": [float(x) for x in np.asarray(clean_action)],
                "applied_action_7d": [float(x) for x in np.asarray(env_action)],
                "action_token_ids": [int(x) for x in tokens],
                "clean_action_token_top_ids": policy_top_ids,
                "clean_action_token_top_logits": policy_top_logits,
                "score_adapter_action_max_abs_error": action_error,
                "score_adapter_parity_pass": bool(action_error <= 1e-6),
                "score_head_summary": score_summary,
                "prompt": score_meta["prompt"],
            })
            policy_intent_rows.append({
                "step": step,
                "action_token_ids": [int(x) for x in tokens],
                "clean_policy_intent_9d": policy,
                **policy_named,
                "clean_action_token_top_ids": policy_top_ids,
                "clean_action_token_top_logits": policy_top_logits,
                "score_head_summary": score_summary,
                "score_adapter_parity_pass": bool(action_error <= 1e-6),
            })
            obs, _reward, done, _info = env.step(env_action.tolist())
            if done:
                success = True
                break
        if not success and hasattr(env, "check_success"):
            success = bool(env.check_success())
    finally:
        env.close()

    meta = {
        "schema": "OPENVLA_OFFICIAL_CLEAN_EPISODE_V2",
        "protocol_id": "OPENVLA_LIBERO_OFFICIAL_V1",
        "condition": "CLEAN",
        "suite": suite,
        "task_idx": task_idx,
        "task_name": str(task.name),
        "task_language": str(task.language),
        "state_id": state_id,
        "canonical_parent_key": row["canonical_parent_key"],
        "split": row["split"],
        "initial_state_sha256": row["initial_state_sha256"],
        "official_horizon": horizon,
        "max_steps": horizon,
        "num_steps_wait": NUM_STEPS_WAIT,
        "runtime_valid": True,
        "env_success": success,
        "success": success,
        "model_path": str(model_path),
        "unnorm_key": adapter.unnorm_key,
        "official_execution_adapter": "OfficialOpenVLAActionAdapter.predict_action",
        "score_adapter": "OfficialOpenVLAScoreAdapter.generate_same_inputs",
        "policy_intent_records": "policy_intent_records.jsonl",
        "privileged_teacher_sidecar": "privileged_teacher_sidecar.jsonl",
        "feature_names_25d": list(CANONICAL_25D_FEATURES),
        "policy_intent_feature_names_9d": list(CLEAN_POLICY_FEATURE_NAMES),
        "student_allowed_modalities": ["features_25d", "clean_policy_intent_9d", "task_language"],
        "student_forbidden_modalities": ["object_state", "mujoco_contact_pairs", "attack_outcome"],
        "detector_retraining_input_ready": True,
        "teacher_labels_materialized": False,
        "teacher_label_source": "privileged_teacher_sidecar.jsonl",
        "attack_enabled": False,
        "video_saved": False,
        **artifact_provenance,
    }
    write_json(out / "episode_metadata.json", meta)
    write_json(out / "episode_summary.json", {"success": success, "steps": len(step_rows), "clean": True})
    write_json(out / "runtime_audit.json", {"runtime_valid": True, "exception": "", "official_horizon": horizon})
    write_json(out / "condition_config.json", {"condition": "CLEAN", "protocol_id": "OPENVLA_LIBERO_OFFICIAL_V1"})
    write_json(out / "attack_config.json", {"attack_enabled": False, "condition": "CLEAN"})
    (out / "step_records.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in step_rows), encoding="utf-8")
    (out / "policy_intent_records.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in policy_intent_rows), encoding="utf-8")
    (out / "privileged_teacher_sidecar.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in privileged_rows), encoding="utf-8")
    seal(out)
    return {"status": "PASS" if success else "TASK_FAILURE", **meta, "steps": len(step_rows), "artifact_root": str(out)}


def main() -> int:
    rows = load_rows()
    model, processor, device, unnorm_key = load_policy()
    adapter = OfficialOpenVLAActionAdapter(
        model, processor, device, unnorm_key, center_crop=True,
        base_vla_name=str(args.model_path),
    )
    artifact_provenance = load_artifact_provenance()
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    suite_instance = suites[args.suite]()
    status_path = args.output_root / "worker_status" / f"{args.worker_id}.json"
    results = []
    stop_stage = ""
    for index, row in enumerate(rows):
        free_gb = disk_free_gb()
        if free_gb < 30:
            write_json(status_path, {
                "worker_id": args.worker_id, "gpu": args.gpu, "suite": args.suite,
                "stage": "DISK_HARD_STOP", "disk_free_gb": free_gb,
                "completed": index, "total": len(rows), "last_heartbeat": time.time(),
            })
            return 2
        if free_gb < 40:
            stop_stage = "DISK_SOFT_STOP"
            write_json(status_path, {
                "worker_id": args.worker_id, "gpu": args.gpu, "suite": args.suite,
                "stage": "DISK_SOFT_STOP", "disk_free_gb": free_gb,
                "completed": index, "total": len(rows), "last_heartbeat": time.time(),
            })
            break
        out = args.output_root / "clean" / args.suite / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        cell_id = f"CLEAN|{row['canonical_parent_key']}"
        if artifact_valid(out):
            result = {"status": "SKIP_CHECKSUM_PASS", **row, "artifact_root": str(out)}
            update_global_ledger(cell_id, "PASS", result_status=result["status"])
        else:
            task = suite_instance.get_task(int(row["task_idx"]))
            states = suite_instance.get_task_init_states(int(row["task_idx"]))
            if state_sha(states[int(row["state_id"])]) != row["initial_state_sha256"]:
                update_global_ledger(cell_id, "PROTOCOL_HOLD", result_status="INITIAL_STATE_HASH_FAIL")
                raise SystemExit(f"INITIAL_STATE_HASH_FAIL {row['canonical_parent_key']}")
            update_global_ledger(cell_id, "LEASED", attempt_increment=True)
            update_global_ledger(cell_id, "RUNNING")
            retry_count = 0
            while True:
                try:
                    result = run_episode(adapter, task, states[int(row["state_id"])], row, out, args.model_path, artifact_provenance)
                    if retry_count:
                        result["runtime_retry_count"] = retry_count
                        write_json(out / "runtime_retries.json", {"retry_count": retry_count})
                        seal(out)
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}:{str(exc)[:500]}"
                    if retry_count == 0 and retryable_runtime_error(exc):
                        retry_count = 1
                        out.mkdir(parents=True, exist_ok=True)
                        write_json(out / "runtime_retries.json", {
                            "retry_count": retry_count,
                            "reason": error,
                            "policy": "same formal CLEAN attempt; one runtime retry",
                        })
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    result = {"status": "RUNTIME_INVALID", **row, "error": error, "runtime_retry_count": retry_count}
                    write_json(out / "episode_metadata.json", result)
                    write_json(out / "runtime_audit.json", {"runtime_valid": False, "exception": error, "runtime_retry_count": retry_count})
                    seal(out)
                    break
            result_ledger_status = {
                "PASS": "PASS",
                "TASK_FAILURE": "TASK_FAILURE",
                "RUNTIME_INVALID": "RUNTIME_HOLD",
            }.get(str(result.get("status")), "PROTOCOL_HOLD")
            update_global_ledger(cell_id, result_ledger_status, result_status=str(result.get("status")))
        results.append(result)
        write_json(status_path, {
            "worker_id": args.worker_id,
            "gpu": args.gpu,
            "suite": args.suite,
            "completed": index + 1,
            "total": len(rows),
            "pass": sum(r.get("status") in {"PASS", "SKIP_CHECKSUM_PASS"} for r in results),
            "task_failure": sum(r.get("status") == "TASK_FAILURE" for r in results),
            "runtime_invalid": sum(r.get("status") == "RUNTIME_INVALID" for r in results),
            "last_parent": row["canonical_parent_key"],
            "disk_free_gb": disk_free_gb(),
            "last_heartbeat": time.time(),
        })

    ledger = args.output_root / "ledgers" / f"OFFICIAL_CLEAN_LEDGER_{args.suite}.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in results for key in row})
    with ledger.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    write_json(status_path, {
        "worker_id": args.worker_id,
        "gpu": args.gpu,
        "suite": args.suite,
        "stage": stop_stage or "CLEAN_500_DONE",
        "completed": len(results),
        "total": len(rows),
        "disk_free_gb": disk_free_gb(),
        "last_heartbeat": time.time(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
