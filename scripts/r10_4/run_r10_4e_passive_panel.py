#!/usr/bin/env python3
"""R10.4E E-R3a passive canary runner.

Authorized scope in this revision:
- reuse the existing sealed task_00/state_20 evidence;
- execute task_01/state_20 exactly once;
- stop unconditionally after task_01;
- never mutate the clean OpenVLA action;
- never enable command-OPEN, VIS, RAND, or training.

All immutable authorization checks complete before torch, LIBERO, or OpenVLA
imports. Every attempt is durably ledgered and every promoted root is immutable.
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

PHASE = "E_R3A_TASK01_CANARY"
TASK00 = "libero_10/task_00/state_20"
TASK01 = "libero_10/task_01/state_20"
EXPECTED_MANIFEST = [
    {"identity": TASK00, "reuse": True},
    {"identity": TASK01, "reuse": False},
]
PASS_STATUSES = {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R10.4E E-R3a task01 passive canary")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--detector-bundle", required=True, type=Path)
    parser.add_argument("--panel-receipt", required=True, type=Path)
    parser.add_argument("--panel-protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--task00-root", required=True, type=Path)
    parser.add_argument("--authorization-comment-id", required=True, type=int)
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


def pickle4_sha(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


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


def checkpoint_tree_fingerprint(path: Path) -> tuple[str, int, int]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if not item.is_file():
            continue
        size = item.stat().st_size
        total_bytes += size
        rows.append(
            {
                "path": item.relative_to(path).as_posix(),
                "size": size,
                "sha256": sha256_file(item),
            }
        )
    if not rows:
        raise SystemExit("MODEL_TREE_EMPTY")
    return canonical_json_sha(rows), len(rows), total_bytes


def verify_receipt_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise SystemExit("PANEL_RECEIPT_OR_SIDECAR_MISSING")
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    tokens = lines[0].split() if lines else []
    if len(tokens) != 2 or tokens[1] != path.name:
        raise SystemExit("PANEL_RECEIPT_SIDECAR_FORMAT_FAIL")
    actual = sha256_file(path)
    if tokens[0] != actual:
        raise SystemExit("PANEL_RECEIPT_SIDECAR_DIGEST_FAIL")
    return actual


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def seal_root(root: Path) -> dict[str, Any]:
    """Seal all current files; no caller may write inside root afterwards."""
    if (root / "SHA256SUMS").exists() or (root / "SHA256SUMS.sha256").exists():
        raise RuntimeError(f"ROOT_ALREADY_SEALED:{root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )
    sums_sha = sha256_file(sums)
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"file_count": len(rows), "sha256sums_sha256": sums_sha, "files": rows}


def write_ledger_revision(
    panel_root: Path,
    attempts: list[dict[str, Any]],
    revision: int,
    previous_sha: str | None,
) -> str:
    payload = {
        "schema": "R10_4E_PANEL_LEDGER_REVISION_V1",
        "revision": revision,
        "previous_ledger_sha256": previous_sha,
        "attempts": list(attempts),
        "n_attempts": len(attempts),
    }
    temporary = panel_root / f".ledger_rev{revision:04d}.{uuid.uuid4().hex}.tmp"
    target = panel_root / f"panel_ledger_rev{revision:04d}.json"
    if target.exists():
        raise RuntimeError(f"LEDGER_REVISION_EXISTS:{target}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return sha256_file(target)


def validate_receipt(
    receipt: dict[str, Any],
    *,
    head: str,
    expected_comment_id: int,
    protocol_sha: str,
    task00_root: Path,
    task00_seal_sha: str,
    task00_summary_sha: str,
    model_tree_sha: str,
    model_file_count: int,
    model_bytes: int,
    detector_checkpoint_sha: str,
    detector_bundle_sha: str,
    feature_order_sha: str,
    gpu: int,
    render_gpu: int,
) -> None:
    exact = {
        "schema": "R10_4E_TEN_TASK_PASSIVE_PANEL_RECEIPT_V1",
        "scope": "R10_4E_E_R3A_TASK01_CANARY",
        "phase": PHASE,
        "source_commit": head,
        "authorization_comment_id": expected_comment_id,
        "episodes_authorized": 2,
        "fresh_executions_authorized": 1,
        "reuse_authorized": 1,
        "task_manifest": EXPECTED_MANIFEST,
        "task_manifest_sha256": canonical_json_sha(EXPECTED_MANIFEST),
        "task00_root": str(task00_root.resolve()),
        "task00_root_sha256s": task00_seal_sha,
        "task00_summary_sha256": task00_summary_sha,
        "protocol_sha256": protocol_sha,
        "detector_checkpoint_sha256": detector_checkpoint_sha,
        "bundle_sha256s_sha256": detector_bundle_sha,
        "model_tree_sha256": model_tree_sha,
        "model_file_count": model_file_count,
        "model_bytes": model_bytes,
        "feature_order_sha256": feature_order_sha,
        "gpu": gpu,
        "render_gpu": render_gpu,
        "passive_only": True,
        "model_load_authorized": True,
        "detector_execution_authorized": True,
        "action_mutation_authorized": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "command_open_authorized": False,
        "visual_attack_authorized": False,
        "random_attack_authorized": False,
        "retry_authorized": False,
        "parent_substitution_authorized": False,
        "threshold_or_fsm_change_authorized": False,
        "output_overwrite_authorized": False,
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise SystemExit(f"RECEIPT_FIELD_FAIL:{key}")


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

    config = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(model_path),
        load_in_8bit=False,
        load_in_4bit=False,
    )
    model = get_model(config)
    processor = get_processor(config)
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
    if args.authorization_comment_id <= 0:
        raise SystemExit("AUTHORIZATION_COMMENT_ID_INVALID")
    if args.gpu != 0:
        raise SystemExit("R10_4E_GPU_MUST_BE_0")
    render_gpu = args.gpu if args.render_gpu is None else args.render_gpu
    if render_gpu != 0:
        raise SystemExit("R10_4E_RENDER_GPU_MUST_BE_0")

    repo_root = Path(__file__).resolve().parents[2]
    head = git_head(repo_root)
    if not git_clean(repo_root):
        raise SystemExit("R10_4E_WORKTREE_DIRTY")
    if args.output_root.exists():
        raise SystemExit(f"R10_4E_OUTPUT_EXISTS:{args.output_root}")
    for path, label in (
        (args.model_path, "MODEL_PATH"),
        (args.detector_bundle, "DETECTOR_BUNDLE"),
        (args.task00_root, "TASK00_ROOT"),
        (args.upstream_root, "UPSTREAM_ROOT"),
    ):
        if not path.is_dir():
            raise SystemExit(f"{label}_MISSING:{path}")
    if not args.panel_protocol.is_file():
        raise SystemExit("PANEL_PROTOCOL_MISSING")

    # Phase 0: immutable checks only. No torch/LIBERO/OpenVLA imports above this point.
    receipt_sha = verify_receipt_sidecar(args.panel_receipt)
    receipt = json.loads(args.panel_receipt.read_text(encoding="utf-8"))
    protocol = json.loads(args.panel_protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "R10_4E_TEN_TASK_PASSIVE_PANEL_PROTOCOL_V1":
        raise SystemExit("PROTOCOL_SCHEMA_FAIL")
    for key in (
        "command_open_authorized",
        "visual_attack_authorized",
        "random_attack_authorized",
        "formal_training_authorized",
        "formal_attack_authorized",
    ):
        if protocol.get(key) is not False:
            raise SystemExit(f"PROTOCOL_FORBIDDEN_FAIL:{key}")

    from gripper_attack.r10_4_runtime import FEATURE_ORDER_SHA256, verify_checksum_manifest

    task00_seal = verify_checksum_manifest(args.task00_root)
    task00_summary_path = args.task00_root / "episode_summary.json"
    if not task00_summary_path.is_file():
        raise SystemExit("TASK00_SUMMARY_MISSING")
    task00_summary = json.loads(task00_summary_path.read_text(encoding="utf-8"))
    if task00_summary.get("identity") != TASK00:
        raise SystemExit("TASK00_IDENTITY_FAIL")
    if task00_summary.get("status") not in PASS_STATUSES:
        raise SystemExit("TASK00_STATUS_FAIL")

    model_tree_sha, model_file_count, model_bytes = checkpoint_tree_fingerprint(args.model_path)
    detector_bundle_seal = verify_checksum_manifest(args.detector_bundle)
    checkpoint_path = args.detector_bundle / "full_fit_deploy.pt"
    if not checkpoint_path.is_file():
        raise SystemExit("DETECTOR_CHECKPOINT_MISSING")
    detector_checkpoint_sha = sha256_file(checkpoint_path)

    validate_receipt(
        receipt,
        head=head,
        expected_comment_id=args.authorization_comment_id,
        protocol_sha=sha256_file(args.panel_protocol),
        task00_root=args.task00_root,
        task00_seal_sha=task00_seal["sha256sums_sha256"],
        task00_summary_sha=sha256_file(task00_summary_path),
        model_tree_sha=model_tree_sha,
        model_file_count=model_file_count,
        model_bytes=model_bytes,
        detector_checkpoint_sha=detector_checkpoint_sha,
        detector_bundle_sha=detector_bundle_seal["sha256sums_sha256"],
        feature_order_sha=FEATURE_ORDER_SHA256,
        gpu=args.gpu,
        render_gpu=render_gpu,
    )

    # Phase 1: authorization passed. Heavy imports and allocations may begin.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "flash_attention_2")
    sys.path.insert(0, str(args.upstream_root.resolve()))

    import torch
    from experiments.robot.libero.libero_utils import get_libero_image
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env
    from gripper_attack.official_libero_protocol import OFFICIAL_HORIZONS
    from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
    from gripper_attack.r10_4d_passive import load_detector_bundle, run_passive_episode

    detector, detector_meta = load_detector_bundle(
        args.detector_bundle,
        device=torch.device("cpu"),
        expected_checkpoint_sha256=detector_checkpoint_sha,
        expected_bundle_sha256s=detector_bundle_seal["sha256sums_sha256"],
    )
    max_steps = int(OFFICIAL_HORIZONS["libero_10"])
    model, processor, model_device, unnorm_key = load_openvla(args.model_path, "libero_10")
    openvla = OfficialOpenVLAActionAdapter(
        model,
        processor,
        model_device,
        unnorm_key,
        center_crop=True,
        base_vla_name=str(args.model_path),
    )
    image_getter = lambda observation: get_libero_image(observation, 224)

    panel_root = args.output_root
    panel_root.mkdir(parents=True)

    reuse_dir = panel_root / TASK00.replace("/", "_")
    reuse_dir.mkdir()
    write_json(
        reuse_dir / "REUSE_BINDING.json",
        {
            "schema": "R10_4E_TASK00_REUSE_BINDING_V1",
            "identity": TASK00,
            "external_root": str(args.task00_root.resolve()),
            "external_sha256sums_sha256": task00_seal["sha256sums_sha256"],
            "external_summary_sha256": sha256_file(task00_summary_path),
            "original_status": task00_summary["status"],
            "n_steps": task00_summary.get("n_steps", 0),
            "emit_count": task00_summary.get("emit_count", 0),
        },
    )
    seal_root(reuse_dir)

    attempts: list[dict[str, Any]] = [
        {
            "identity": TASK00,
            "status": task00_summary["status"],
            "ledger_status": "REUSE_VERIFIED",
            "reuse": True,
            "external_root": str(args.task00_root.resolve()),
            "sha256sums_sha256": task00_seal["sha256sums_sha256"],
        }
    ]
    revision = 0
    previous_sha = write_ledger_revision(panel_root, attempts, revision, None)

    identity = TASK01
    revision += 1
    attempts.append({"identity": identity, "status": "PREPARING", "reuse": False})
    previous_sha = write_ledger_revision(panel_root, attempts, revision, previous_sha)

    staging = panel_root / f".{identity.replace('/', '_')}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    revision += 1
    attempts[-1]["status"] = "RUNNING"
    previous_sha = write_ledger_revision(panel_root, attempts, revision, previous_sha)

    env = None
    promoted_dir: Path | None = None
    runtime_valid = False
    panel_ok = False
    try:
        parent = resolve_parent(identity)
        env, _ = build_v4_exact_env(
            parent["bddl_path"],
            render_gpu_device_id=render_gpu,
            max_steps=max_steps,
            num_steps_wait=10,
        )
        result = run_passive_episode(
            env=env,
            initial_state=copy.deepcopy(parent["initial_state"]),
            task_language=parent["task_language"],
            identity=identity,
            openvla_adapter=openvla,
            detector=detector,
            image_getter=image_getter,
            max_steps=max_steps,
            authorized_parents=frozenset({TASK01}),
            privileged_observer=privileged_observer,
        )

        write_jsonl(staging / "step_records.jsonl", result.pop("step_records"))
        write_jsonl(staging / "detector_records.jsonl", result.pop("detector_records"))
        write_jsonl(staging / "privileged_teacher_sidecar.jsonl", result.pop("privileged_records"))
        write_json(
            staging / "episode_metadata.json",
            {
                "schema": "R10_4E_SINGLE_EPISODE_PASSIVE_METADATA_V1",
                "identity": identity,
                "source_commit": head,
                "parent": {key: value for key, value in parent.items() if key != "initial_state"},
                "panel_receipt_sha256": receipt_sha,
                "authorization_comment_id": args.authorization_comment_id,
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
            },
        )
        write_json(staging / "episode_summary.json", result)
        runtime_valid = result["status"] in PASS_STATUSES
        write_json(
            staging / "runtime_audit.json",
            {
                "runtime_valid": runtime_valid,
                "status": result["status"],
                "termination_reason": result.get("termination_reason", ""),
                "is_hard_failure": result["status"] == "FAIL_TERMINATION",
                "violations": result["violations"],
                "action_mutation": result["action_mutation"],
                "privileged_runtime_input": result["privileged_runtime_input"],
            },
        )
        write_json(
            staging / "ROOT_SEAL_RECEIPT.json",
            {
                "schema": "R10_4E_ROOT_SEAL_RECEIPT_V1",
                "identity": identity,
                "source_commit": head,
                "panel_receipt_sha256": receipt_sha,
            },
        )
        seal = seal_root(staging)

        episode_dir = panel_root / identity.replace("/", "_")
        if episode_dir.exists():
            raise RuntimeError(f"EPISODE_OUTPUT_EXISTS:{episode_dir}")
        os.replace(staging, episode_dir)
        promoted_dir = episode_dir

        revision += 1
        attempts[-1] = {
            "identity": identity,
            "status": result["status"],
            "ledger_status": "SEALED_PASS" if runtime_valid else "SEALED_FAIL",
            "n_steps": result["n_steps"],
            "emit_count": result["emit_count"],
            "termination_reason": result.get("termination_reason", ""),
            "task_success": result["task_success"],
            "violations": result["violations"],
            "sha256sums_sha256": seal["sha256sums_sha256"],
            "reuse": False,
        }
        previous_sha = write_ledger_revision(panel_root, attempts, revision, previous_sha)
        panel_ok = runtime_valid
    except Exception as exc:
        # Never delete or overwrite a promoted root.
        if promoted_dir is None:
            if not staging.exists():
                staging.mkdir(parents=True, exist_ok=False)
            write_json(
                staging / "RUNTIME_FAILURE.json",
                {
                    "schema": "R10_4E_RUNTIME_FAILURE_V1",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc)[:2000],
                    "source_commit": head,
                    "identity": identity,
                },
            )
            if not (staging / "SHA256SUMS").exists():
                failure_seal = seal_root(staging)
            else:
                failure_seal = {"sha256sums_sha256": sha256_file(staging / "SHA256SUMS")}
            failure_dir = panel_root / identity.replace("/", "_")
            if failure_dir.exists():
                raise RuntimeError(f"FAILURE_OUTPUT_EXISTS_NO_OVERWRITE:{failure_dir}") from exc
            os.replace(staging, failure_dir)
            promoted_dir = failure_dir
            seal_sha = failure_seal["sha256sums_sha256"]
        else:
            seal_sha = sha256_file(promoted_dir / "SHA256SUMS")
            write_json(
                panel_root / "PANEL_FAILURE_AFTER_PROMOTION.json",
                {
                    "schema": "R10_4E_PANEL_FAILURE_AFTER_PROMOTION_V1",
                    "identity": identity,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc)[:2000],
                    "episode_root_preserved": str(promoted_dir),
                },
            )

        revision += 1
        attempts[-1] = {
            "identity": identity,
            "status": "FAIL_EXCEPTION",
            "ledger_status": "SEALED_FAIL",
            "exception_type": type(exc).__name__,
            "exception": str(exc)[:500],
            "sha256sums_sha256": seal_sha,
            "reuse": False,
        }
        try:
            previous_sha = write_ledger_revision(panel_root, attempts, revision, previous_sha)
        except Exception:
            pass
        runtime_valid = False
        panel_ok = False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    write_json(
        panel_root / "panel_ledger.json",
        {
            "schema": "R10_4E_PANEL_LEDGER_V1",
            "revision": revision,
            "previous_ledger_sha256": previous_sha,
            "attempts": attempts,
            "n_attempts": len(attempts),
            "all_runtime_valid": runtime_valid,
            "panel_ok": panel_ok,
        },
    )
    write_json(
        panel_root / "panel_summary.json",
        {
            "panel": "R10_4E",
            "phase": PHASE,
            "source_commit": head,
            "panel_receipt_sha256": receipt_sha,
            "authorization_comment_id": args.authorization_comment_id,
            "n_tasks_attempted": len(attempts),
            "n_reuse": 1,
            "n_fresh": 1,
            "all_runtime_valid": runtime_valid,
            "panel_ok": panel_ok,
            "per_task": [
                {
                    "identity": entry["identity"],
                    "status": entry["status"],
                    "reuse": entry.get("reuse", False),
                }
                for entry in attempts
            ],
        },
    )
    seal_root(panel_root)
    print(f"E-R3a complete: task01 runtime_valid={runtime_valid} panel_ok={panel_ok}")
    return 0 if panel_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
