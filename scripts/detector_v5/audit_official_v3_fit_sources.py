#!/usr/bin/env python3
"""Bind Official V3 FIT policy telemetry and audit privileged physics fields.

Read-only against source roots.  It emits two non-overwrite, recursively
sealed roots: a Student policy-intent root and a Teacher-only privileged audit.
No protected split, replay, model inference, or attack is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


POLICY_FIELDS = (
    "step", "clean_policy_intent_9d", "clean_open_probability_mass",
    "clean_close_probability_mass", "clean_open_minus_close_log_mass",
    "clean_action_token_entropy_normalized", "clean_top1_probability",
    "clean_top1_is_open", "clean_top1_is_close",
    "clean_best_open_rank_normalized", "clean_best_close_rank_normalized",
    "action_token_ids", "clean_action_token_top_ids",
    "clean_action_token_top_logits", "score_head_summary",
    "generation_passes_per_step", "single_generation_parity_pass",
    "score_adapter_parity_pass",
)
PRIVILEGED_FIELDS = (
    "step", "suite", "task_idx", "state_id", "object_state",
    "mujoco_contact_pairs", "contact_count", "contact_capture_valid",
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    "eef_feature_pos", "eef_alias_valid",
)
META_FIELDS = (
    "schema", "official_protocol_id", "collector_git_head", "collector_script_sha256",
    "official_adapter_sha256", "checkpoint_tree_sha256", "processor_tokenizer_sha256",
    "prompt", "task_language", "model_path", "protocol_id", "split", "condition",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object row: {path}")
            rows.append(value)
    return rows


def verify_sealed_root(root: Path) -> str:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"unsealed root: {root}")
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError(f"SHA256SUMS sidecar mismatch: {root}")
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, sep, name = line.partition("  ")
        path = Path(name)
        if not sep or len(digest) != 64 or path.is_absolute() or ".." in path.parts or name in listed:
            raise ValueError(f"invalid checksum row: {name}")
        target = root / path
        if not target.is_file() or sha256_file(target) != digest.lower():
            raise ValueError(f"checksum mismatch: {name}")
        listed[path.as_posix()] = digest.lower()
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected = set(listed) | {"SHA256SUMS", "SHA256SUMS.sha256"}
    if actual != expected:
        raise ValueError(f"sealed file-set mismatch: {root}")
    return sha256_file(sums)


def verify_artifact(root: Path) -> str:
    manifest_path = root / "artifact_sha256.json"
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"bad artifact checksum manifest: {root}")
    listed = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"bad artifact checksum row: {root}")
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.name == "artifact_sha256.json":
            raise ValueError(f"unsafe artifact checksum path: {rel}")
        target = root / rel
        if not target.is_file() or target.stat().st_size != item.get("size") or sha256_file(target) != item.get("sha256"):
            raise ValueError(f"artifact checksum mismatch: {root}/{rel}")
        listed.append({"path": rel.as_posix(), "size": item["size"], "sha256": item["sha256"]})
    actual = {p.relative_to(root).as_posix() for p in root.iterdir() if p.is_file()}
    if actual != {item["path"] for item in listed} | {"artifact_sha256.json"}:
        raise ValueError(f"artifact file-set mismatch: {root}")
    if manifest.get("recursive_sha256") != json_sha(listed):
        raise ValueError(f"artifact recursive digest mismatch: {root}")
    return str(manifest["recursive_sha256"])


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_list(value: Any, length: int | None = None) -> bool:
    return isinstance(value, list) and (length is None or len(value) == length) and all(finite(x) for x in value)


def episode_root(source_root: Path, row: dict[str, Any]) -> Path:
    return source_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"


def load_fit(registry_csv: Path) -> list[dict[str, Any]]:
    with registry_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fit = [row for row in rows if int(row["state_id"]) < 20]
    if len(rows) != 2000 or len(fit) != 800:
        raise ValueError(f"registry closure failed: global={len(rows)} fit={len(fit)}")
    keys = {row["canonical_parent_key"] for row in fit}
    if len(keys) != 800:
        raise ValueError("duplicate FIT identities")
    for row in fit:
        key = row["canonical_parent_key"]
        expected = f"{row['suite']}/task_{int(row['task_idx']):02d}/state_{int(row['state_id']):02d}"
        if key != expected or row.get("split") != "FIT_TRAIN":
            raise ValueError(f"invalid FIT registry row: {key}")
    return sorted(fit, key=lambda row: row["canonical_parent_key"])


def _same_identity(row: dict[str, Any], key: str) -> bool:
    return row.get("canonical_parent_key", key) == key and row.get("suite") == key.split("/")[0] and int(row.get("task_idx", -1)) == int(key.split("/")[1].split("_")[1]) and int(row.get("state_id", -1)) == int(key.split("/")[2].split("_")[1])


def audit_episode(source_root: Path, registry_row: dict[str, Any], policy_out: list[dict[str, Any]], privileged_out: list[dict[str, Any]], policy_fields: Counter[str], privileged_fields: Counter[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    key = registry_row["canonical_parent_key"]
    root = episode_root(source_root, registry_row)
    artifact_sha = verify_artifact(root)
    metadata = read_json(root / "episode_metadata.json")
    runtime = read_json(root / "runtime_audit.json")
    steps = read_jsonl(root / "step_records.jsonl")
    policy = read_jsonl(root / "policy_intent_records.jsonl")
    privileged = read_jsonl(root / "privileged_teacher_sidecar.jsonl")
    if metadata.get("canonical_parent_key") != key or metadata.get("condition") != "CLEAN" or metadata.get("split") != "FIT":
        raise ValueError(f"metadata binding failed: {key}")
    if len(steps) != len(policy) or len(steps) != len(privileged):
        raise ValueError(f"step count mismatch: {key}")
    for index, (step, intent, priv) in enumerate(zip(steps, policy, privileged)):
        if any(int(x.get("step", -1)) != index for x in (step, intent, priv)):
            raise ValueError(f"step index mismatch: {key}:{index}")
        if not _same_identity(step, key) or not _same_identity(priv, key):
            raise ValueError(f"identity field mismatch: {key}:{index}")
        if intent.get("action_token_ids") != step.get("action_token_ids"):
            raise ValueError(f"policy/action token mismatch: {key}:{index}")
        score = intent.get("score_head_summary")
        token_ids = intent.get("action_token_ids")
        if not isinstance(score, list) or len(score) != len(token_ids) or any(item.get("top_token") != token for item, token in zip(score, token_ids)):
            raise ValueError(f"score adapter token mismatch: {key}:{index}")
        for field in POLICY_FIELDS:
            policy_fields[field] += 1
        for field in PRIVILEGED_FIELDS:
            privileged_fields[field] += field in priv
        if not finite_list(intent.get("clean_policy_intent_9d"), 9):
            raise ValueError(f"9D policy feature invalid: {key}:{index}")
        for field in ("clean_open_probability_mass", "clean_close_probability_mass", "clean_top1_probability", "clean_top1_is_open", "clean_top1_is_close"):
            if not finite(intent.get(field)) or not -1e-6 <= float(intent[field]) <= 1 + 1e-6:
                raise ValueError(f"policy range invalid: {key}:{index}:{field}")
        for field in ("clean_open_minus_close_log_mass", "clean_action_token_entropy_normalized", "clean_best_open_rank_normalized", "clean_best_close_rank_normalized"):
            if not finite(intent.get(field)):
                raise ValueError(f"policy numeric invalid: {key}:{index}:{field}")
        if intent.get("generation_passes_per_step") != 1 or intent.get("single_generation_parity_pass") is not True or intent.get("score_adapter_parity_pass") is not True:
            raise ValueError(f"policy generation/parity invalid: {key}:{index}")
        if not finite_list(priv.get("object_state")) or not finite_list(priv.get("robot0_eef_pos"), 3) or not finite_list(priv.get("robot0_gripper_qpos"), 2) or not finite_list(priv.get("eef_feature_pos"), 3):
            raise ValueError(f"privileged physics vector invalid: {key}:{index}")
        if priv.get("contact_capture_valid") is not True or priv.get("eef_alias_valid") is not True:
            raise ValueError(f"privileged validity invalid: {key}:{index}")
        clean = {"canonical_parent_key": key, "task_language": metadata.get("task_language", "")}
        clean.update({field: intent[field] for field in POLICY_FIELDS})
        clean["valid_intent"] = True
        policy_out.append(clean)
    policy_row = {
        "canonical_parent_key": key, "source_artifact_recursive_sha256": artifact_sha,
        "step_count": len(steps), "policy_step_count": len(policy), "privileged_step_count": len(privileged),
        "policy_valid_step_count": len(policy), "runtime_valid": runtime.get("runtime_valid") is True,
        "generation_parity_pass": all(item.get("generation_passes_per_step") == 1 for item in policy),
        "score_adapter_parity_pass": all(item.get("score_adapter_parity_pass") is True for item in policy),
        "single_generation_parity_pass": all(item.get("single_generation_parity_pass") is True for item in policy),
        **{field: metadata.get(field, "") for field in META_FIELDS},
    }
    privileged_row = {
        "canonical_parent_key": key, "source_artifact_recursive_sha256": artifact_sha,
        "step_count": len(privileged), "required_fields_present": all(all(field in item for field in PRIVILEGED_FIELDS) for item in privileged),
        "contact_capture_valid_count": sum(item.get("contact_capture_valid") is True for item in privileged),
        "eef_alias_valid_count": sum(item.get("eef_alias_valid") is True for item in privileged),
        "object_state_dimensions": sorted({len(item["object_state"]) for item in privileged}),
        "contact_pair_rows": sum(isinstance(item.get("mujoco_contact_pairs"), list) for item in privileged),
        "teacher_only": True,
    }
    return policy_row, privileged_row


def _write_recursive_seal(root: Path) -> str:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payloads = sorted((p for p in root.rglob("*") if p.is_file() and p.name not in excluded), key=lambda p: p.relative_to(root).as_posix())
    atomic_text(root / "SHA256SUMS", "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in payloads))
    atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")
    return sha256_file(root / "SHA256SUMS")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_roots = [args.policy_output_root.resolve(), args.privileged_output_root.resolve()]
    if any(path.exists() for path in output_roots):
        raise FileExistsError("refusing to overwrite output root")
    protocol = read_json(args.protocol.resolve())
    if protocol.get("schema") != "DETECTOR_V5_OFFICIAL_SOURCE_BINDING_PROTOCOL_V1":
        raise ValueError("unexpected source binding protocol")
    if tuple(protocol.get("student_policy_features", ())) != (
        "clean_open_probability_mass", "clean_close_probability_mass", "clean_open_minus_close_log_mass",
        "clean_action_token_entropy_normalized", "clean_top1_probability", "clean_top1_is_open",
        "clean_top1_is_close", "clean_best_open_rank_normalized", "clean_best_close_rank_normalized",
    ) or protocol.get("generation_contract") != {
        "generation_passes_per_step": 1, "single_generation_parity_pass": True, "score_adapter_parity_pass": True
    }:
        raise ValueError("source binding protocol feature/generation contract mismatch")
    if protocol.get("privileged_schema_scope") != "task_conditional_object_state_flattening" or protocol.get("object_state_dimensions_must_be_constant_within_task") is not True:
        raise ValueError("privileged schema contract mismatch")
    registry_root_sha = verify_sealed_root(args.registry_root.resolve())
    s1_root_sha = verify_sealed_root(args.s1_root.resolve())
    raw_audit_sha = verify_sealed_root(args.raw_audit_root.resolve())
    if args.registry_csv.resolve().parent != args.registry_root.resolve():
        raise ValueError("registry CSV must be inside the sealed registry root")
    rows = load_fit(args.registry_csv.resolve())
    policy_out: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    privileged_rows: list[dict[str, Any]] = []
    policy_fields: Counter[str] = Counter()
    privileged_fields: Counter[str] = Counter()
    for row in rows:
        p_row, t_row = audit_episode(args.source_root.resolve(), row, policy_out, privileged_rows, policy_fields, privileged_fields)
        policy_rows.append(p_row)
        privileged_rows.append(t_row)
    source_index_sha = json_sha([{"canonical_parent_key": row["canonical_parent_key"], "source_artifact_recursive_sha256": row["source_artifact_recursive_sha256"]} for row in policy_rows])
    common = {
        "registry_csv_sha256": sha256_file(args.registry_csv.resolve()),
        "source_binding_protocol_sha256": sha256_file(args.protocol.resolve()),
        "registry_root_sha256s_sha256": registry_root_sha,
        "s1_root_sha256s_sha256": s1_root_sha,
        "raw_asset_audit_root_sha256s_sha256": raw_audit_sha,
        "source_artifact_index_sha256": source_index_sha,
        "fit_identity_count": len(rows), "step_count_total": sum(row["step_count"] for row in policy_rows),
        "formal_training_authorized": False, "formal_attack_authorized": False,
    }
    policy_manifest = {
        "schema": "OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1", "status": "PASS",
        **common, "policy_feature_order": [
            "clean_open_probability_mass", "clean_close_probability_mass", "clean_open_minus_close_log_mass",
            "clean_action_token_entropy_normalized", "clean_top1_probability", "clean_top1_is_open",
            "clean_top1_is_close", "clean_best_open_rank_normalized", "clean_best_close_rank_normalized",
        ], "policy_step_count": len(policy_out), "valid_intent_step_count": len(policy_out),
        "student_forbidden_modalities": ["object_state", "mujoco_contact_pairs", "contact_count", "worker_id", "collector_git_head", "model_path", "attack_outcome"],
        "policy_field_coverage": dict(sorted(policy_fields.items())),
    }
    dimensions_by_task: dict[str, list[int]] = {}
    for row in privileged_rows:
        task = "/".join(row["canonical_parent_key"].split("/")[:2])
        dimensions_by_task.setdefault(task, [])
        dimensions_by_task[task] = sorted(set(dimensions_by_task[task]) | set(row["object_state_dimensions"]))
    schema_within_task = all(len(values) == 1 for values in dimensions_by_task.values()) and len(dimensions_by_task) == 40
    privileged_manifest = {
        "schema": "OFFICIAL_V3_PRIVILEGED_PHYSICS_TEACHER_AUDIT_V1",
        "status": "PASS_TASK_CONDITIONAL_SCHEMA" if schema_within_task else "HOLD_SCHEMA_VARIATION",
        **common, "teacher_only": True, "student_consumption_allowed": False,
        "privileged_step_count": sum(row["step_count"] for row in privileged_rows),
        "required_field_coverage": dict(sorted(privileged_fields.items())),
        "object_state_dimensions": sorted({dim for row in privileged_rows for dim in row["object_state_dimensions"]}),
        "object_state_dimensions_by_task": dimensions_by_task,
        "schema_scope": "task_conditional_object_state_flattening",
        "schema_constant_within_task": schema_within_task,
    }
    staging_policy = output_roots[0].with_name(f".{output_roots[0].name}.{uuid.uuid4().hex}.staging")
    staging_priv = output_roots[1].with_name(f".{output_roots[1].name}.{uuid.uuid4().hex}.staging")
    try:
        staging_policy.mkdir(parents=True); staging_priv.mkdir(parents=True)
        atomic_text(staging_policy / "OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1.json", json.dumps(policy_manifest, indent=2, sort_keys=True) + "\n")
        _write_csv(staging_policy / "identity_rows.csv", policy_rows, list(policy_rows[0]))
        _write_csv(staging_policy / "field_coverage.csv", [{"field": k, "count": v} for k, v in sorted(policy_fields.items())], ["field", "count"])
        atomic_text(staging_policy / "policy_intent_records.jsonl", "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in policy_out))
        atomic_text(staging_priv / "OFFICIAL_V3_PRIVILEGED_PHYSICS_TEACHER_AUDIT_V1.json", json.dumps(privileged_manifest, indent=2, sort_keys=True) + "\n")
        _write_csv(staging_priv / "identity_rows.csv", privileged_rows, list(privileged_rows[0]))
        _write_csv(staging_priv / "field_coverage.csv", [{"field": k, "count": v} for k, v in sorted(privileged_fields.items())], ["field", "count"])
        _write_recursive_seal(staging_policy); _write_recursive_seal(staging_priv)
        os.replace(staging_policy, output_roots[0]); os.replace(staging_priv, output_roots[1])
    except Exception:
        shutil.rmtree(staging_policy, ignore_errors=True); shutil.rmtree(staging_priv, ignore_errors=True)
        raise
    policy_manifest["output_sha256s_sha256"] = sha256_file(output_roots[0] / "SHA256SUMS")
    privileged_manifest["output_sha256s_sha256"] = sha256_file(output_roots[1] / "SHA256SUMS")
    return {"policy": policy_manifest, "privileged": privileged_manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--raw-audit-root", type=Path, required=True)
    parser.add_argument("--policy-output-root", type=Path, required=True)
    parser.add_argument("--privileged-output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
