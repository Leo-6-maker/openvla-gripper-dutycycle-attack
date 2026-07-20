#!/usr/bin/env python3
"""R10.4E passive panel runner — ten tasks, one resident model load.

Task_00 is REUSE-ONLY (seal verification, no re-run).
Task_01-09 are FIRST_SEALED_EXECUTION (one run each, no retries).
Any hard failure stops the panel immediately.

The script has no command-OPEN, VIS, RAND, attack, or action-override path.
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R10.4E ten-task passive panel")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--detector-bundle", required=True, type=Path)
    parser.add_argument("--panel-receipt", required=True, type=Path)
    parser.add_argument("--panel-protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--task00-root", required=True, type=Path,
                        help="Existing sealed R10.4D task_00 root (reuse-only)")
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--render-gpu", type=int, default=None)
    return parser.parse_args()


# ── Hashing utilities ────────────────────────────────────────────────────────

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
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def git_clean(root: Path) -> bool:
    return not subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def checkpoint_tree_fingerprint(path: Path) -> tuple[str, int, int]:
    rows = []
    total_bytes = 0
    for item in sorted(path.rglob("*"), key=lambda v: v.relative_to(path).as_posix()):
        if not item.is_file():
            continue
        size = item.stat().st_size
        total_bytes += size
        rows.append({"path": item.relative_to(path).as_posix(), "size": size, "sha256": sha256_file(item)})
    if not rows:
        raise SystemExit("MODEL_TREE_EMPTY")
    return canonical_json_sha(rows), len(rows), total_bytes


def verify_receipt_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise SystemExit("PANEL_RECEIPT_OR_SIDECAR_MISSING")
    rows = sidecar.read_text(encoding="utf-8").splitlines()
    tokens = rows[0].split() if rows else []
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
    sums.write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in rows), encoding="utf-8")
    sums_sha = sha256_file(sums)
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"file_count": len(rows), "sha256sums_sha256": sums_sha, "files": rows}


# ── Parent resolution ────────────────────────────────────────────────────────

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
    si = constructors[suite]()
    task = si.get_task(task_index)
    states = si.get_task_init_states(task_index)
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


# ── OpenVLA loading ──────────────────────────────────────────────────────────

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


# ── Privileged observer ──────────────────────────────────────────────────────

def privileged_observer(env: Any, observation: Any, step: int) -> dict[str, Any]:
    contacts: list[list[str]] = []
    try:
        data = env.sim.data
        model = env.sim.model
        for idx in range(int(data.ncon)):
            contact = data.contact[idx]
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


# ── Task_00 reuse ────────────────────────────────────────────────────────────

def verify_and_extract_task00(task00_root: Path) -> dict[str, Any]:
    """Verify R10.4D task_00 seal and extract telemetry. Does NOT re-run."""
    if not task00_root.is_dir():
        raise SystemExit(f"TASK00_ROOT_MISSING:{task00_root}")

    # Verify SHA256SUMS
    from gripper_attack.r10_4_runtime import verify_checksum_manifest
    seal = verify_checksum_manifest(task00_root)
    summary = json.loads((task00_root / "episode_summary.json").read_text(encoding="utf-8"))

    identity = summary.get("identity", "")
    if not identity.endswith("/task_00/state_20"):
        raise SystemExit(f"TASK00_IDENTITY_FAIL:{identity}")

    return {
        "identity": identity,
        "reuse": True,
        "original_status": summary.get("status", "?"),
        "n_steps": summary.get("n_steps", 0),
        "emit_count": summary.get("emit_count", 0),
        "task_success": summary.get("task_success", None),
        "violations": summary.get("violations", []),
        "root_seal": seal,
    }


# ── Main panel ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    if args.gpu != 0:
        raise SystemExit("R10_4E_GPU_MUST_BE_0")
    render_gpu = args.gpu if args.render_gpu is None else args.render_gpu
    if render_gpu != 0:
        raise SystemExit("R10_4E_RENDER_GPU_MUST_BE_0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OPENVLA_ATTN_IMPLEMENTATION", "flash_attention_2")
    sys.path.insert(0, str(args.upstream_root.resolve()))

    # Imports below this line may import torch, but no model/device allocation
    # is allowed until the receipt and all immutable bindings pass.
    import torch
    from experiments.robot.libero.libero_utils import get_libero_image
    from gripper_attack.libero_v4_env_factory import build_v4_exact_env
    from gripper_attack.official_libero_protocol import OFFICIAL_HORIZONS
    from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
    from gripper_attack.r10_4d_passive import (
        load_detector_bundle,
        run_passive_episode,
    )

    repo_root = Path(__file__).resolve().parents[2]
    head = git_head(repo_root)
    if not git_clean(repo_root):
        raise SystemExit("R10_4E_WORKTREE_DIRTY")
    if args.output_root.exists():
        raise SystemExit(f"R10_4E_OUTPUT_EXISTS:{args.output_root}")
    if not args.model_path.is_dir():
        raise SystemExit(f"MODEL_PATH_MISSING:{args.model_path}")
    if not args.panel_receipt.is_file() or not args.panel_protocol.is_file():
        raise SystemExit("R10_4E_RECEIPT_OR_PROTOCOL_MISSING")
    if not args.task00_root.is_dir():
        raise SystemExit(f"TASK00_ROOT_MISSING:{args.task00_root}")

    # Verify receipt
    receipt_sha = verify_receipt_sidecar(args.panel_receipt)
    receipt = json.loads(args.panel_receipt.read_text(encoding="utf-8"))
    protocol = json.loads(args.panel_protocol.read_text(encoding="utf-8"))

    # Validate receipt fields
    if receipt.get("schema") != "R10_4E_TEN_TASK_PASSIVE_PANEL_RECEIPT_V1":
        raise SystemExit("R10_4E_RECEIPT_SCHEMA_FAIL")
    if receipt.get("source_commit") != head:
        raise SystemExit("R10_4E_RECEIPT_HEAD_FAIL")
    for key in (
        "passive_only", "model_load_authorized", "detector_execution_authorized",
    ):
        if receipt.get(key) is not True:
            raise SystemExit(f"R10_4E_RECEIPT_SCOPE_FAIL:{key}")
    for key in (
        "action_mutation_authorized", "formal_training_authorized", "formal_attack_authorized",
        "command_open_authorized", "visual_attack_authorized", "random_attack_authorized",
        "retry_authorized", "parent_substitution_authorized",
        "threshold_or_fsm_change_authorized", "output_overwrite_authorized",
    ):
        if receipt.get(key) is not False:
            raise SystemExit(f"R10_4E_RECEIPT_FORBIDDEN_FAIL:{key}")
    if not isinstance(receipt.get("authorization_comment_id"), int) or int(receipt["authorization_comment_id"]) <= 0:
        raise SystemExit("R10_4E_AUTH_COMMENT_ID_FAIL")

    task_manifest = receipt.get("task_manifest", [])
    if len(task_manifest) != 10:
        raise SystemExit("R10_4E_TASK_MANIFEST_COUNT_FAIL")

    # Verify task00 reuse
    task00 = verify_and_extract_task00(args.task00_root)
    if task00["original_status"] not in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}:
        raise SystemExit(f"TASK00_STATUS_FAIL:{task00['original_status']}")
    print(f"[1/10] {task00['identity']} REUSE status={task00['original_status']} steps={task00['n_steps']}")

    # Model tree fingerprint
    model_tree_sha, model_file_count, model_bytes = checkpoint_tree_fingerprint(args.model_path)
    if model_tree_sha != receipt.get("model_tree_sha256"):
        raise SystemExit("R10_4E_MODEL_TREE_SHA_FAIL")

    # Detector bundle verification (CPU only)
    expected_ckpt_sha = str(receipt.get("detector_checkpoint_sha256", ""))
    expected_bundle_sha = str(receipt.get("bundle_sha256s_sha256", ""))
    detector, detector_meta = load_detector_bundle(
        args.detector_bundle,
        device=torch.device("cpu"),
        expected_checkpoint_sha256=expected_ckpt_sha,
        expected_bundle_sha256s=expected_bundle_sha,
    )

    # Load OpenVLA once
    suite = "libero_10"
    max_steps = int(OFFICIAL_HORIZONS[suite])
    model, processor, model_device, unnorm_key = load_openvla(args.model_path, suite)
    openvla = OfficialOpenVLAActionAdapter(
        model, processor, model_device, unnorm_key,
        center_crop=True, base_vla_name=str(args.model_path),
    )
    img_getter = lambda obs: get_libero_image(obs, 224)

    # Authorized parents from receipt (task_00 is reuse, 01-09 are fresh)
    authorized = frozenset(entry["identity"] for entry in task_manifest)

    panel_root = args.output_root
    panel_root.mkdir(parents=True)

    attempt_ledger: list[dict[str, Any]] = [task00]
    all_runtime_valid = True
    panel_ok = True

    for idx, entry in enumerate(task_manifest):
        identity = entry["identity"]
        task_num = idx + 1

        if idx == 0:
            continue  # task_00 already handled as reuse

        print(f"[{task_num}/10] {identity} RUNNING...", flush=True)

        parent = resolve_parent(identity)

        # Build env
        env, _observation = build_v4_exact_env(
            parent["bddl_path"],
            render_gpu_device_id=render_gpu,
            max_steps=max_steps,
            num_steps_wait=10,
        )

        # Staging directory
        staging = panel_root / f".{identity.replace('/', '_')}.staging-{uuid.uuid4().hex}"
        if staging.exists():
            raise SystemExit(f"R10_4E_STAGING_EXISTS:{staging}")
        staging.mkdir(parents=True)

        try:
            result = run_passive_episode(
                env=env,
                initial_state=copy.deepcopy(parent["initial_state"]),
                task_language=parent["task_language"],
                identity=identity,
                openvla_adapter=openvla,
                detector=detector,
                image_getter=img_getter,
                max_steps=max_steps,
                authorized_parents=authorized,
                privileged_observer=privileged_observer,
            )

            write_jsonl(staging / "step_records.jsonl", result.pop("step_records"))
            write_jsonl(staging / "detector_records.jsonl", result.pop("detector_records"))
            write_jsonl(staging / "privileged_teacher_sidecar.jsonl", result.pop("privileged_records"))
            metadata = {
                "schema": "R10_4E_SINGLE_EPISODE_PASSIVE_METADATA_V1",
                "source_commit": head,
                "parent": {k: v for k, v in parent.items() if k != "initial_state"},
                "panel_receipt_sha256": receipt_sha,
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
                "runtime_valid": result["status"] in {
                    "PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED",
                },
                "status": result["status"],
                "termination_reason": result.get("termination_reason", ""),
                "is_hard_failure": result["status"] == "FAIL_TERMINATION",
                "violations": result["violations"],
                "action_mutation": result["action_mutation"],
                "privileged_runtime_input": result["privileged_runtime_input"],
            })
            seal = seal_root(staging)
            write_json(staging / "ROOT_SEAL_RECEIPT.json", seal)

            # Atomic promotion
            ep_dir = panel_root / identity.replace("/", "_")
            os.replace(staging, ep_dir)

            ledger_entry = {
                "identity": identity,
                "status": result["status"],
                "n_steps": result["n_steps"],
                "emit_count": result["emit_count"],
                "termination_reason": result.get("termination_reason", ""),
                "task_success": result["task_success"],
                "violations": result["violations"],
                "sha256sums_sha256": seal["sha256sums_sha256"],
                "reuse": False,
            }
            attempt_ledger.append(ledger_entry)

            print(f"  {result['status']} steps={result['n_steps']} emits={result['emit_count']} "
                  f"term={result.get('termination_reason','?')}")

            is_runtime_valid = result["status"] in {
                "PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED",
            }
            if not is_runtime_valid:
                all_runtime_valid = False
                if result["status"] == "FAIL_TERMINATION":
                    print(f"  HARD FAILURE — stopping panel")
                    panel_ok = False
                    break
        except Exception as exc:
            # Failure: seal staging, record, stop
            write_json(staging / "RUNTIME_FAILURE.json", {
                "schema": "R10_4E_RUNTIME_FAILURE_V1",
                "exception_type": type(exc).__name__,
                "exception": str(exc)[:2000],
                "source_commit": head,
                "parent": identity,
            })
            seal_root(staging)
            ep_dir = panel_root / identity.replace("/", "_")
            os.replace(staging, ep_dir)

            ledger_entry = {
                "identity": identity,
                "status": "FAIL_EXCEPTION",
                "exception_type": type(exc).__name__,
                "exception": str(exc)[:500],
                "reuse": False,
            }
            attempt_ledger.append(ledger_entry)
            print(f"  FAIL_EXCEPTION: {type(exc).__name__}: {str(exc)[:200]}")
            all_runtime_valid = False
            panel_ok = False
            break
        finally:
            try:
                env.close()
            except Exception:
                pass

    # Write panel-level ledger and summary
    write_json(panel_root / "panel_ledger.json", {
        "schema": "R10_4E_PANEL_LEDGER_V1",
        "source_commit": head,
        "panel_receipt_sha256": receipt_sha,
        "authorization_comment_id": receipt["authorization_comment_id"],
        "attempts": attempt_ledger,
        "n_attempts": len(attempt_ledger),
        "all_runtime_valid": all_runtime_valid,
        "panel_ok": panel_ok,
    })

    summary = {
        "panel": "R10_4E",
        "source_commit": head,
        "n_tasks_attempted": len(attempt_ledger),
        "n_reuse": sum(1 for e in attempt_ledger if e.get("reuse")),
        "n_fresh": sum(1 for e in attempt_ledger if not e.get("reuse")),
        "all_runtime_valid": all_runtime_valid,
        "panel_ok": panel_ok,
        "per_task": [
            {"identity": e["identity"], "status": e["status"], "reuse": e.get("reuse", False)}
            for e in attempt_ledger
        ],
    }
    write_json(panel_root / "panel_summary.json", summary)

    print(f"\nPanel: {len(attempt_ledger)} tasks, all_runtime_valid={all_runtime_valid}, panel_ok={panel_ok}")
    return 0 if panel_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
