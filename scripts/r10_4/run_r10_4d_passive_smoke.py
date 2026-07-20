#!/usr/bin/env python3
"""Execute the one authorized R10.4D passive smoke.

The script has no command-OPEN, VIS, RAND, attack, or action-override path.
A machine-built authorization receipt bound to the final Git HEAD, model tree,
detector checkpoint, bundle seal, protocol, and one frozen parent is mandatory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R10.4D single-episode passive smoke")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--detector-bundle", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--authorization-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--render-gpu", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def checkpoint_tree_fingerprint(path: Path) -> tuple[str, int, int]:
    rows = []
    total_bytes = 0
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if not item.is_file():
            continue
        size = item.stat().st_size
        total_bytes += size
        rows.append({
            "path": item.relative_to(path).as_posix(),
            "size": size,
            "sha256": sha256_file(item),
        })
    if not rows:
        raise SystemExit("MODEL_TREE_EMPTY")
    return canonical_json_sha(rows), len(rows), total_bytes


def verify_receipt_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise SystemExit("AUTH_RECEIPT_OR_SIDECAR_MISSING")
    rows = sidecar.read_text(encoding="utf-8").splitlines()
    tokens = rows[0].split() if rows else []
    if len(tokens) != 2 or tokens[1] != path.name:
        raise SystemExit("AUTH_RECEIPT_SIDECAR_FORMAT_FAIL")
    actual = sha256_file(path)
    if tokens[0] != actual:
        raise SystemExit("AUTH_RECEIPT_SIDECAR_DIGEST_FAIL")
    return actual


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_clean(root: Path) -> bool:
    return not subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def pickle4_sha(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def seal_root(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        })
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="utf-8")
    sums_sha = sha256_file(sums)
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"file_count": len(rows), "sha256sums_sha256": sums_sha, "files": rows}


def selected_parent(manifest: dict[str, Any]) -> tuple[str, str | None]:
    selected = manifest.get("selected_parent")
    if isinstance(selected, dict):
        identity = selected.get("identity") or selected.get("canonical_parent_key")
        initial_sha = selected.get("initial_state_sha256")
    else:
        identity = selected
        initial_sha = manifest.get("initial_state_sha256")
    if not isinstance(identity, str):
        raise RuntimeError("PARENT_MANIFEST_SELECTED_PARENT_MISSING")
    return identity, initial_sha if isinstance(initial_sha, str) else None


def resolve_parent(identity: str) -> dict[str, Any]:
    from libero.libero import benchmark, get_libero_path

    parts = identity.split("/")
    if len(parts) != 3:
        raise RuntimeError(f"PARENT_IDENTITY_FORMAT:{identity}")
    suite, task_name, state_name = parts
    task_index = int(task_name.split("_")[1])
    state_index = int(state_name.split("_")[1])
    constructors = benchmark.get_benchmark_dict()
    if suite not in constructors:
        raise RuntimeError(f"PARENT_SUITE_MISSING:{suite}")
    suite_instance = constructors[suite]()
    task = suite_instance.get_task(task_index)
    states = suite_instance.get_task_init_states(task_index)
    initial_state = states[state_index]
    bddl_path = Path(get_libero_path("bddl_files")) / str(task.problem_folder) / str(task.bddl_file)
    if not bddl_path.is_file():
        raise RuntimeError(f"PARENT_BDDL_MISSING:{bddl_path}")
    return {
        "identity": identity,
        "suite": suite,
        "task_index": task_index,
        "state_index": state_index,
        "task_name": str(task.name),
        "task_language": str(task.language),
        "initial_state": initial_state,
        "initial_state_sha256": pickle4_sha(initial_state),
        "bddl_path": str(bddl_path),
        "bddl_sha256": sha256_file(bddl_path),
    }


def load_openvla(model_path: Path, suite: str):
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import get_model

    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(model_path),
        load_in_8bit=False,
        load_in_4bit=False,
    )
    model = get_model(cfg)
    processor = get_processor(cfg)
    model.eval()
    device = next(model.parameters()).device
    unnorm_key = suite
    stats = getattr(model, "norm_stats", {})
    if unnorm_key not in stats and f"{suite}_no_noops" in stats:
        unnorm_key = f"{suite}_no_noops"
    if unnorm_key not in stats:
        raise RuntimeError(f"OPENVLA_UNNORM_KEY_MISSING:{suite}")
    return model, processor, device, unnorm_key


def privileged_observer(env: Any, observation: Any, step: int) -> dict[str, Any]:
    contacts: list[list[str]] = []
    try:
        data = env.sim.data
        model = env.sim.model
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            first = model.geom_id2name(int(contact.geom1))
            second = model.geom_id2name(int(contact.geom2))
            if first and second:
                contacts.append([str(first), str(second)])
    except Exception:
        contacts = []
    return {
        "step": step,
        "object_state": list(map(float, observation.get("object-state", []))),
        "robot0_gripper_qpos": list(map(float, observation.get("robot0_gripper_qpos", []))),
        "robot0_eef_pos": list(map(float, observation.get("robot0_eef_pos", []))),
        "mujoco_contact_pairs": contacts,
        "teacher_labels_materialized": False,
    }


def main() -> int:
    args = parse_args()
    if args.gpu != 0:
        raise SystemExit("R10_4D_GPU_MUST_BE_0")
    render_gpu = args.gpu if args.render_gpu is None else args.render_gpu
    if render_gpu != 0:
        raise SystemExit("R10_4D_RENDER_GPU_MUST_BE_0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "flash_attention_2")
    sys.path.insert(0, str(args.upstream_root.resolve()))

    # Imports below this line may import torch, but no model/device allocation is
    # allowed until the receipt and all immutable bindings pass.
    import torch
    from experiments.robot.libero.libero_utils import get_libero_image
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env
    from gripper_attack.official_libero_protocol import OFFICIAL_HORIZONS
    from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
    from gripper_attack.r10_4d_passive import (
        SUPPORTED_PARENT,
        load_detector_bundle,
        run_passive_episode,
        validate_authorization_receipt,
    )

    repo_root = Path(__file__).resolve().parents[2]
    head = git_head(repo_root)
    if not git_clean(repo_root):
        raise SystemExit("R10_4D_WORKTREE_DIRTY")
    if args.output_root.exists():
        raise SystemExit(f"R10_4D_OUTPUT_EXISTS:{args.output_root}")
    if not args.model_path.is_dir():
        raise SystemExit(f"MODEL_PATH_MISSING:{args.model_path}")
    if not args.protocol.is_file() or not args.parent_manifest.is_file():
        raise SystemExit("R10_4D_PROTOCOL_OR_PARENT_MISSING")

    receipt_sha = verify_receipt_sidecar(args.authorization_receipt)
    receipt = json.loads(args.authorization_receipt.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    parent_manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    identity, manifest_initial_sha = selected_parent(parent_manifest)
    if identity != SUPPORTED_PARENT:
        raise SystemExit(f"R10_4D_PARENT_FAIL:{identity}")
    if protocol.get("schema") != "R10_4D_SINGLE_EPISODE_PASSIVE_SMOKE_PROTOCOL_V1":
        raise SystemExit("R10_4D_PROTOCOL_SCHEMA_FAIL")
    if protocol.get("selected_parent") != identity or int(protocol.get("episodes_authorized", 0)) != 1:
        raise SystemExit("R10_4D_PROTOCOL_PARENT_OR_COUNT_FAIL")
    if int(protocol.get("gpu", -1)) != 0 or int(protocol.get("render_gpu", -1)) != 0:
        raise SystemExit("R10_4D_PROTOCOL_GPU_FAIL")
    if receipt.get("parent_manifest_sha256") != sha256_file(args.parent_manifest):
        raise SystemExit("R10_4D_RECEIPT_PARENT_MANIFEST_SHA_FAIL")
    if receipt.get("protocol_sha256") != sha256_file(args.protocol):
        raise SystemExit("R10_4D_RECEIPT_PROTOCOL_SHA_FAIL")
    if not isinstance(receipt.get("authorization_comment_id"), int) or int(receipt["authorization_comment_id"]) <= 0:
        raise SystemExit("R10_4D_AUTH_COMMENT_ID_FAIL")
    for key in (
        "second_episode_authorized",
        "parent_substitution_authorized",
        "threshold_or_fsm_change_authorized",
        "output_overwrite_authorized",
    ):
        if receipt.get(key) is not False:
            raise SystemExit(f"R10_4D_RECEIPT_SCOPE_FAIL:{key}")

    parent = resolve_parent(identity)
    if manifest_initial_sha is not None and manifest_initial_sha != parent["initial_state_sha256"]:
        raise SystemExit("R10_4D_PARENT_INITIAL_STATE_SHA_FAIL")

    model_tree_sha, model_file_count, model_bytes = checkpoint_tree_fingerprint(args.model_path)
    expected_checkpoint_sha = str(receipt.get("detector_checkpoint_sha256", ""))
    expected_bundle_sha = str(receipt.get("bundle_sha256s_sha256", ""))
    validate_authorization_receipt(
        receipt,
        expected_head=head,
        expected_parent=identity,
        expected_checkpoint_sha256=expected_checkpoint_sha,
        expected_bundle_sha256s=expected_bundle_sha,
        expected_model_tree_sha256=model_tree_sha,
    )

    # Detector bundle verification and strict load are CPU-only.  The 7B model
    # is loaded only after the receipt and every source binding above pass.
    detector, detector_meta = load_detector_bundle(
        args.detector_bundle,
        device=torch.device("cpu"),
        expected_checkpoint_sha256=expected_checkpoint_sha,
        expected_bundle_sha256s=expected_bundle_sha,
    )

    model, processor, model_device, unnorm_key = load_openvla(args.model_path, parent["suite"])
    openvla = OfficialOpenVLAActionAdapter(
        model,
        processor,
        model_device,
        unnorm_key,
        center_crop=True,
        base_vla_name=str(args.model_path),
    )

    max_steps = int(OFFICIAL_HORIZONS[parent["suite"]])
    env, _observation = build_v4_exact_env(
        parent["bddl_path"],
        render_gpu_device_id=render_gpu,
        max_steps=max_steps,
        num_steps_wait=10,
    )

    staging = args.output_root.parent / f".{args.output_root.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise SystemExit(f"R10_4D_STAGING_EXISTS:{staging}")
    staging.mkdir(parents=True)
    try:
        result = run_passive_episode(
            env=env,
            initial_state=copy.deepcopy(parent["initial_state"]),
            task_language=parent["task_language"],
            identity=identity,
            openvla_adapter=openvla,
            detector=detector,
            image_getter=lambda observation: get_libero_image(observation, 224),
            max_steps=max_steps,
            privileged_observer=privileged_observer,
        )
        write_jsonl(staging / "step_records.jsonl", result.pop("step_records"))
        write_jsonl(staging / "detector_records.jsonl", result.pop("detector_records"))
        write_jsonl(staging / "privileged_teacher_sidecar.jsonl", result.pop("privileged_records"))
        metadata = {
            "schema": "R10_4D_SINGLE_EPISODE_PASSIVE_METADATA_V1",
            "source_commit": head,
            "parent": {key: value for key, value in parent.items() if key != "initial_state"},
            "parent_manifest_sha256": sha256_file(args.parent_manifest),
            "protocol_sha256": sha256_file(args.protocol),
            "authorization_receipt_sha256": receipt_sha,
            "authorization_comment_id": receipt["authorization_comment_id"],
            "detector": detector_meta,
            "model_path": str(args.model_path.resolve()),
            "model_tree_sha256": model_tree_sha,
            "model_file_count": model_file_count,
            "model_bytes": model_bytes,
            "unnorm_key": unnorm_key,
            "gpu": args.gpu,
            "render_gpu": render_gpu,
            "openvla_model_loaded": True,
            "detector_executed": True,
            "action_mutation": False,
            "attack_enabled": False,
            "command_open_enabled": False,
            "visual_attack_enabled": False,
            "random_attack_enabled": False,
            "privileged_runtime_input": False,
        }
        write_json(staging / "episode_metadata.json", metadata)
        write_json(staging / "episode_summary.json", result)
        write_json(staging / "runtime_audit.json", {
            "runtime_valid": result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"},
            "status": result["status"],
            "violations": result["violations"],
            "action_mutation": result["action_mutation"],
            "privileged_runtime_input": result["privileged_runtime_input"],
        })
        seal = seal_root(staging)
        write_json(staging / "ROOT_SEAL_RECEIPT.json", seal)
        seal = seal_root(staging)
        os.replace(staging, args.output_root)
        print(json.dumps({
            "status": result["status"],
            "output_root": str(args.output_root),
            "sha256sums_sha256": seal["sha256sums_sha256"],
            "n_steps": result["n_steps"],
            "emit_count": result["emit_count"],
            "task_success": result["task_success"],
        }, indent=2, sort_keys=True))
        return 0 if result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"} else 1
    except Exception as exc:
        write_json(staging / "RUNTIME_FAILURE.json", {
            "schema": "R10_4D_RUNTIME_FAILURE_V1",
            "exception_type": type(exc).__name__,
            "exception": str(exc)[:2000],
            "source_commit": head,
            "parent": identity,
        })
        seal_root(staging)
        os.replace(staging, args.output_root)
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
