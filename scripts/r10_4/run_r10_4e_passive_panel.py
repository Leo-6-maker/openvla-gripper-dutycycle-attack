#!/usr/bin/env python3
"""R10.4E passive panel runner — phase-gated, receipt-first, durable ledger.

Phase E-R3a: task_00 external reuse + task_01 single fresh execution.
Any non-PASS_RUNTIME_* status or exception stops the panel immediately.

The script has no command-OPEN, VIS, RAND, attack, or action-override path.
P0-2: All torch/LIBERO/OpenVLA imports gated behind receipt validation.
P0-5: SHA256SUMS is the LAST file written (covers all content including seal receipt).
P0-6: Durable append-only ledger with atomic revision writes.
P0-7: Env construction failures captured in sealed staging.
P0-8: Any non-PASS_RUNTIME_* status halts the panel.
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
    parser = argparse.ArgumentParser(description="R10.4E phase-gated passive panel")
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


# ── Hashing / git / filesystem ───────────────────────────────────────────────

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
    """Write SHA256SUMS as the LAST file, then SHA256SUMS.sha256 as the very last.
    P0-5: No file is written after SHA256SUMS, so every file is covered.
    """
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


def write_ledger_revision(panel_root: Path, ledger: list[dict[str, Any]], revision: int,
                          previous_sha: str | None) -> str:
    """Atomically write a ledger revision via temp file + os.replace."""
    payload = {
        "schema": "R10_4E_PANEL_LEDGER_V1",
        "revision": revision,
        "previous_ledger_sha256": previous_sha,
        "attempts": list(ledger),
        "n_attempts": len(ledger),
    }
    tmp = panel_root / f".ledger_rev{revision}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    target = panel_root / f"panel_ledger_rev{revision:04d}.json"
    os.replace(str(tmp), str(target))
    return sha256_file(target)


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


# ── OpenVLA loading (gated behind receipt validation) ────────────────────────

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
    from gripper_attack.r10_4_runtime import verify_checksum_manifest

    if not task00_root.is_dir():
        raise SystemExit(f"TASK00_ROOT_MISSING:{task00_root}")
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


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT VALIDATION — must complete before any torch/LIBERO/OpenVLA import
# ═══════════════════════════════════════════════════════════════════════════════

def validate_panel_receipt(
    receipt: dict[str, Any],
    *,
    head: str,
    protocol_sha: str,
    task00_root: Path,
    task00_seal: dict[str, Any],
    model_tree_sha: str,
    detector_checkpoint_sha: str,
    detector_bundle_sha: str,
    gpu: int,
    render_gpu: int,
) -> tuple[frozenset[str], frozenset[str], str]:
    """Validate every receipt field. Returns (reuse_ids, fresh_ids, phase)."""
    # Schema
    if receipt.get("schema") != "R10_4E_TEN_TASK_PASSIVE_PANEL_RECEIPT_V1":
        raise SystemExit("RECEIPT_SCHEMA_FAIL")
    if receipt.get("source_commit") != head:
        raise SystemExit(f"RECEIPT_HEAD_FAIL:{receipt.get('source_commit','')[:16]}...")
    if receipt.get("gpu") != gpu or receipt.get("render_gpu") != render_gpu:
        raise SystemExit("RECEIPT_GPU_FAIL")

    # Authorization booleans
    for key in ("passive_only", "model_load_authorized", "detector_execution_authorized"):
        if receipt.get(key) is not True:
            raise SystemExit(f"RECEIPT_SCOPE_FAIL:{key}")
    for key in ("action_mutation_authorized", "formal_training_authorized", "formal_attack_authorized",
                "command_open_authorized", "visual_attack_authorized", "random_attack_authorized",
                "retry_authorized", "parent_substitution_authorized",
                "threshold_or_fsm_change_authorized", "output_overwrite_authorized"):
        if receipt.get(key) is not False:
            raise SystemExit(f"RECEIPT_FORBIDDEN_FAIL:{key}")
    comment_id = receipt.get("authorization_comment_id")
    if not isinstance(comment_id, int) or int(comment_id) <= 0:
        raise SystemExit("RECEIPT_AUTH_COMMENT_ID_FAIL")

    # Phase
    phase = str(receipt.get("phase", ""))
    if not phase:
        raise SystemExit("RECEIPT_PHASE_MISSING")

    # Task manifest — exact match required
    manifest = receipt.get("task_manifest", [])
    if not isinstance(manifest, list) or len(manifest) == 0:
        raise SystemExit("RECEIPT_MANIFEST_EMPTY")
    expected_manifest_sha = canonical_json_sha(manifest)
    if receipt.get("task_manifest_sha256") != expected_manifest_sha:
        raise SystemExit("RECEIPT_MANIFEST_SHA_FAIL")
    reuse_ids = frozenset(e["identity"] for e in manifest if e.get("reuse"))
    fresh_ids = frozenset(e["identity"] for e in manifest if not e.get("reuse"))
    if len(reuse_ids) != receipt.get("reuse_authorized", 0):
        raise SystemExit("RECEIPT_REUSE_COUNT_FAIL")
    if len(fresh_ids) != receipt.get("fresh_executions_authorized", 0):
        raise SystemExit("RECEIPT_FRESH_COUNT_FAIL")
    if (len(reuse_ids) + len(fresh_ids)) != receipt.get("episodes_authorized", 0):
        raise SystemExit("RECEIPT_TOTAL_COUNT_FAIL")

    # Protocol
    if receipt.get("protocol_sha256") != protocol_sha:
        raise SystemExit("RECEIPT_PROTOCOL_SHA_FAIL")

    # Task00 reuse
    if receipt.get("task00_root_sha256s") != task00_seal["sha256sums_sha256"]:
        raise SystemExit("RECEIPT_TASK00_SEAL_FAIL")

    # Detector
    if receipt.get("detector_checkpoint_sha256") != detector_checkpoint_sha:
        raise SystemExit("RECEIPT_CHECKPOINT_SHA_FAIL")
    if receipt.get("bundle_sha256s_sha256") != detector_bundle_sha:
        raise SystemExit("RECEIPT_BUNDLE_SHA_FAIL")

    # Model tree
    if receipt.get("model_tree_sha256") != model_tree_sha:
        raise SystemExit("RECEIPT_MODEL_TREE_SHA_FAIL")

    # Feature contract
    from gripper_attack.r10_4_runtime import FEATURE_ORDER_SHA256
    if receipt.get("feature_order_sha256") != FEATURE_ORDER_SHA256:
        raise SystemExit("RECEIPT_FEATURE_ORDER_FAIL")

    return reuse_ids, fresh_ids, phase


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — P0-2: All heavyweight imports gated behind receipt validation
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()
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
    if not args.model_path.is_dir():
        raise SystemExit(f"MODEL_PATH_MISSING:{args.model_path}")
    if not args.panel_receipt.is_file() or not args.panel_protocol.is_file():
        raise SystemExit("R10_4E_RECEIPT_OR_PROTOCOL_MISSING")
    if not args.task00_root.is_dir():
        raise SystemExit(f"TASK00_ROOT_MISSING:{args.task00_root}")

    # ── Phase 0: Read-only validation (no torch, no LIBERO, no OpenVLA) ──
    receipt_sha = verify_receipt_sidecar(args.panel_receipt)
    receipt = json.loads(args.panel_receipt.read_text(encoding="utf-8"))
    protocol = json.loads(args.panel_protocol.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(args.panel_protocol)

    if protocol.get("schema") != "R10_4E_TEN_TASK_PASSIVE_PANEL_PROTOCOL_V1":
        raise SystemExit("PROTOCOL_SCHEMA_FAIL")
    for key in ("command_open_authorized", "visual_attack_authorized", "random_attack_authorized",
                "formal_training_authorized", "formal_attack_authorized"):
        if protocol.get(key) is not False:
            raise SystemExit(f"PROTOCOL_FORBIDDEN_FAIL:{key}")

    # Task00 seal (read-only, no LIBERO)
    from gripper_attack.r10_4_runtime import verify_checksum_manifest
    task00 = verify_and_extract_task00(args.task00_root)
    task00_seal = verify_checksum_manifest(args.task00_root)
    if task00["original_status"] not in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}:
        raise SystemExit(f"TASK00_STATUS_FAIL:{task00['original_status']}")

    # Model tree fingerprint (read-only)
    model_tree_sha, model_file_count, model_bytes = checkpoint_tree_fingerprint(args.model_path)

    # Phase 0: Read-only detector verification (no torch import)
    detector_checkpoint_sha = sha256_file(args.detector_bundle / "full_fit_deploy.pt")
    detector_bundle_seal = verify_checksum_manifest(args.detector_bundle)
    detector_bundle_sha = detector_bundle_seal["sha256sums_sha256"]

    # ── Validate receipt against all bindings ──
    reuse_ids, fresh_ids, phase = validate_panel_receipt(
        receipt,
        head=head,
        protocol_sha=protocol_sha,
        task00_root=args.task00_root,
        task00_seal=task00_seal,
        model_tree_sha=model_tree_sha,
        detector_checkpoint_sha=detector_checkpoint_sha,
        detector_bundle_sha=detector_bundle_sha,
        gpu=args.gpu,
        render_gpu=render_gpu,
    )

    print(f"Receipt: {receipt_sha[:16]}... phase={phase} reuse={len(reuse_ids)} fresh={len(fresh_ids)}")
    for idx, entry in enumerate(receipt["task_manifest"]):
        tag = "REUSE" if entry.get("reuse") else "FRESH"
        print(f"  [{idx+1}/10] {entry['identity']} {tag}")

    # Check task_00 is in reuse set
    if "libero_10/task_00/state_20" not in reuse_ids:
        raise SystemExit("TASK00_NOT_IN_REUSE_SET")

    # ── Phase 1: Now authorized — import torch/LIBERO/OpenVLA ──
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

    # Detector strict load (CPU, now authorized)
    detector, detector_meta = load_detector_bundle(
        args.detector_bundle,
        device=torch.device("cpu"),
        expected_checkpoint_sha256=detector_checkpoint_sha,
        expected_bundle_sha256s=detector_bundle_sha,
    )

    # OpenVLA load once
    suite = "libero_10"
    max_steps = int(OFFICIAL_HORIZONS[suite])
    model, processor, model_device, unnorm_key = load_openvla(args.model_path, suite)
    openvla = OfficialOpenVLAActionAdapter(
        model, processor, model_device, unnorm_key,
        center_crop=True, base_vla_name=str(args.model_path),
    )
    img_getter = lambda obs: get_libero_image(obs, 224)

    # Setup panel root
    panel_root = args.output_root
    panel_root.mkdir(parents=True)

    # Task00 reuse binding
    reuse_dir = panel_root / "libero_10_task_00_state_20"
    reuse_dir.mkdir()
    write_json(reuse_dir / "REUSE_BINDING.json", {
        "schema": "R10_4E_TASK00_REUSE_BINDING_V1",
        "identity": "libero_10/task_00/state_20",
        "external_root": str(args.task00_root.resolve()),
        "external_sha256sums_sha256": task00_seal["sha256sums_sha256"],
        "original_status": task00["original_status"],
        "n_steps": task00["n_steps"],
        "emit_count": task00["emit_count"],
    })
    seal_root(reuse_dir)

    # Durable ledger — P0-6
    attempt_ledger: list[dict[str, Any]] = [task00]
    previous_ledger_sha = None

    def _commit_ledger(revision: int):
        nonlocal previous_ledger_sha
        previous_ledger_sha = write_ledger_revision(panel_root, attempt_ledger, revision, previous_ledger_sha)

    _commit_ledger(0)  # revision 0 = task00 reuse verified
    print(f"[1/10] {task00['identity']} REUSE status={task00['original_status']} steps={task00['n_steps']}")

    # ── Phase 2: Execute fresh episodes ──
    all_runtime_valid = True
    panel_ok = True
    auth_fresh = fresh_ids  # only identities authorized in receipt
    revision = 0

    for idx, entry in enumerate(receipt["task_manifest"]):
        identity = entry["identity"]
        is_reuse = entry.get("reuse", False)
        if is_reuse:
            continue

        task_num = idx + 1
        print(f"[{task_num}/10] {identity} RUNNING...", flush=True)

        # P0-6: PREPARING ledger revision
        revision += 1
        attempt_ledger.append({"identity": identity, "status": "PREPARING", "reuse": False})
        _commit_ledger(revision)

        # P0-7: Create staging BEFORE parent resolution / env construction
        staging = panel_root / f".{identity.replace('/', '_')}.staging-{uuid.uuid4().hex}"
        if staging.exists():
            raise SystemExit(f"R10_4E_STAGING_EXISTS:{staging}")
        staging.mkdir(parents=True)

        # P0-6: RUNNING ledger revision
        revision += 1
        attempt_ledger[-1]["status"] = "RUNNING"
        _commit_ledger(revision)

        env = None
        try:
            parent = resolve_parent(identity)
            env, _obs = build_v4_exact_env(
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
                image_getter=img_getter,
                max_steps=max_steps,
                authorized_parents=reuse_ids | auth_fresh,
                privileged_observer=privileged_observer,
            )

            # P0-5: Write all content files, then seal as LAST operation
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
            is_runtime_valid = result["status"] in {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}
            write_json(staging / "runtime_audit.json", {
                "runtime_valid": is_runtime_valid,
                "status": result["status"],
                "termination_reason": result.get("termination_reason", ""),
                "is_hard_failure": result["status"] == "FAIL_TERMINATION",
                "violations": result["violations"],
                "action_mutation": result["action_mutation"],
                "privileged_runtime_input": result["privileged_runtime_input"],
            })
            write_json(staging / "ROOT_SEAL_RECEIPT.json", {
                "schema": "R10_4E_ROOT_SEAL_RECEIPT_V1",
                "identity": identity,
                "source_commit": head,
                "panel_receipt_sha256": receipt_sha,
            })
            # P0-5: SHA256SUMS is the LAST content file written
            seal = seal_root(staging)

            # Atomic promotion
            ep_dir = panel_root / identity.replace("/", "_")
            os.replace(staging, ep_dir)

            # P0-6: SEALED ledger revision
            revision += 1
            ledger_status = "SEALED_PASS" if is_runtime_valid else "SEALED_FAIL"
            attempt_ledger[-1] = {
                "identity": identity,
                "status": result["status"],
                "ledger_status": ledger_status,
                "n_steps": result["n_steps"],
                "emit_count": result["emit_count"],
                "termination_reason": result.get("termination_reason", ""),
                "task_success": result["task_success"],
                "violations": result["violations"],
                "sha256sums_sha256": seal["sha256sums_sha256"],
                "reuse": False,
            }
            _commit_ledger(revision)

            print(f"  {result['status']} steps={result['n_steps']} emits={result['emit_count']} "
                  f"term={result.get('termination_reason','?')}")

            # P0-8: Any non-PASS_RUNTIME status halts the panel
            if not is_runtime_valid:
                all_runtime_valid = False
                panel_ok = False
                print(f"  NON-PASS STATUS — stopping panel")
                break

        except Exception as exc:
            # P0-7: Seal failure root, then stop
            write_json(staging / "RUNTIME_FAILURE.json", {
                "schema": "R10_4E_RUNTIME_FAILURE_V1",
                "exception_type": type(exc).__name__,
                "exception": str(exc)[:2000],
                "source_commit": head,
                "identity": identity,
            })
            seal_root(staging)
            ep_dir = panel_root / identity.replace("/", "_")
            if ep_dir.exists():
                import shutil
                shutil.rmtree(str(ep_dir))
            os.replace(staging, ep_dir)

            revision += 1
            attempt_ledger[-1] = {
                "identity": identity,
                "status": "FAIL_EXCEPTION",
                "ledger_status": "SEALED_FAIL",
                "exception_type": type(exc).__name__,
                "exception": str(exc)[:500],
                "reuse": False,
            }
            _commit_ledger(revision)
            print(f"  FAIL_EXCEPTION: {type(exc).__name__}: {str(exc)[:200]}")
            all_runtime_valid = False
            panel_ok = False
            break
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

    # ── Panel-level summary ──
    write_json(panel_root / "panel_ledger.json", {
        "schema": "R10_4E_PANEL_LEDGER_V1",
        "revision": revision,
        "previous_ledger_sha256": previous_ledger_sha,
        "attempts": attempt_ledger,
        "n_attempts": len(attempt_ledger),
        "all_runtime_valid": all_runtime_valid,
        "panel_ok": panel_ok,
    })
    write_json(panel_root / "panel_summary.json", {
        "panel": "R10_4E",
        "phase": phase,
        "source_commit": head,
        "panel_receipt_sha256": receipt_sha,
        "n_tasks_attempted": len(attempt_ledger),
        "n_reuse": sum(1 for e in attempt_ledger if e.get("reuse")),
        "n_fresh": sum(1 for e in attempt_ledger if not e.get("reuse")),
        "all_runtime_valid": all_runtime_valid,
        "panel_ok": panel_ok,
        "per_task": [
            {"identity": e["identity"], "status": e["status"], "reuse": e.get("reuse", False)}
            for e in attempt_ledger
        ],
    })
    # Panel aggregate seal
    seal_root(panel_root)

    print(f"\nPanel: {len(attempt_ledger)} tasks, all_runtime_valid={all_runtime_valid}, panel_ok={panel_ok}")
    return 0 if panel_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
