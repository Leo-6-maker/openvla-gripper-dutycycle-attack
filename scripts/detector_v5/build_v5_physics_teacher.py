#!/usr/bin/env python3
"""Build the FIT-only Physics Teacher V2 clean-criticality derivative."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from gripper_attack import action_contract as _action_contract
from gripper_attack.v5_physics import derive_episode_rows, parse_bddl_task_role


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_sealed_root(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"sealed root missing checksum files: {root}")
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError(f"checksum sidecar mismatch: {root}")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not name or name in listed or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ValueError(f"invalid checksum row: {name}")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"unsafe sealed path: {name}")
        target = root / name
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"checksum mismatch: {root}/{name}")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != listed | {"SHA256SUMS", "SHA256SUMS.sha256"}:
        raise ValueError(f"sealed file-set mismatch: {root}")
    return {"sha256sums_sha256": sha256_file(sums), "file_count": len(listed)}


def _write_seal(root: Path) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name not in excluded),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    _atomic_text(root / "SHA256SUMS", "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files))
    _atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fit_rows(registry_csv: Path) -> list[dict[str, Any]]:
    with registry_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fit = [row for row in rows if row.get("split") == "FIT_TRAIN"]
    if len(fit) != 800 or len({row.get("canonical_parent_key") for row in fit}) != 800:
        raise ValueError("registry must contain exactly 800 unique FIT rows")
    for row in fit:
        expected = f"{row['suite']}/task_{int(row['task_idx']):02d}/state_{int(row['state_id']):02d}"
        if row.get("canonical_parent_key") != expected or int(row["state_id"]) not in range(20):
            raise ValueError(f"invalid FIT identity: {row.get('canonical_parent_key')}")
        if str(row.get("formal_selected", "")).lower() != "true":
            raise ValueError(f"FIT row not formal-selected: {row['canonical_parent_key']}")
        if not Path(row["selected_artifact_root"]).is_dir():
            raise ValueError(f"artifact root missing: {row['canonical_parent_key']}")
    return sorted(fit, key=lambda row: row["canonical_parent_key"])


def _parse_objects(text: str) -> list[str]:
    match = re.search(r"\(:objects\s*(.*?)\n\s*\)\s*\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError("BDDL objects section missing")
    names: list[str] = []
    for line in match.group(1).splitlines():
        item = re.fullmatch(r"([A-Za-z0-9_ ]+)\s+-\s+[A-Za-z0-9_]+", line.strip())
        if item:
            names.extend(item.group(1).split())
    if not names:
        raise ValueError("BDDL objects section empty")
    return names


def _task_specs() -> dict[tuple[str, int], dict[str, Any]]:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark

    specs: dict[tuple[str, int], dict[str, Any]] = {}
    for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        benchmark = get_benchmark(suite)(0)
        for task_idx in range(10):
            task = benchmark.get_task(task_idx)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            text = bddl.read_text(encoding="utf-8")
            specs[(suite, task_idx)] = {
                "task_name": task.name,
                "task_language": task.language,
                "bddl_path": str(bddl),
                "bddl_sha256": sha256_file(bddl),
                "text": text,
                "object_names": _parse_objects(text),
            }
    if len(specs) != 40:
        raise ValueError(f"expected 40 task specs, got {len(specs)}")
    return specs


def _load_object_slices(decoder_root: Path) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    with (decoder_root / "object_slices.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["suite"], int(row["task_idx"]))
        result.setdefault(key, {})[row["object_name"]] = {
            "pos": json.loads(row["pos"]),
            "quat": json.loads(row["quat"]),
            "to_eef_pos": json.loads(row["to_eef_pos"]),
            "to_eef_quat": json.loads(row["to_eef_quat"]),
            "offset_start": int(row["offset_start"]),
            "offset_end_exclusive": int(row["offset_end_exclusive"]),
        }
    return result


def _verify_source_artifact(root: Path, expected_recursive_sha256: str) -> None:
    metadata = _load_json(root / "episode_metadata.json")
    runtime = _load_json(root / "runtime_audit.json")
    if metadata.get("condition") != "CLEAN" or metadata.get("attack_enabled") is not False:
        raise ValueError(f"invalid source condition: {root}")
    if runtime.get("runtime_valid") is not True:
        raise ValueError(f"source runtime invalid: {root}")
    sha_path = root / "artifact_sha256.json"
    if not sha_path.is_file():
        raise ValueError(f"artifact checksum missing: {root}")
    value = _load_json(sha_path)
    entries = value.get("files") if isinstance(value, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"unsupported artifact checksum schema: {root}")
    listed = set()
    for item in entries:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"invalid artifact checksum row: {root}")
        name = str(item["path"])
        target = root / name
        if name in listed or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"invalid artifact checksum path: {root}/{name}")
        if not target.is_file() or sha256_file(target) != str(item["sha256"]):
            raise ValueError(f"artifact checksum mismatch: {root}/{name}")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "artifact_sha256.json"}
    if actual != listed:
        raise ValueError(f"artifact checksum file-set mismatch: {root}")
    if str(value.get("recursive_sha256")) != str(expected_recursive_sha256):
        raise ValueError(f"artifact recursive SHA mismatch: {root}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol = _load_json(args.protocol.resolve())
    if protocol.get("schema") not in {
        "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V1",
        "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21",
        "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL",
    }:
        raise ValueError("unexpected Physics Teacher protocol schema")
    v21 = protocol["schema"] in (
        "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21",
        "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL",
    )
    v21c = protocol["schema"] == "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL"
    registry_seal = verify_sealed_root(args.registry_root.resolve())
    decoder_seal = verify_sealed_root(args.decoder_root.resolve())
    physics_audit_seal = verify_sealed_root(args.physics_audit_root.resolve())
    decoder_summary = _load_json(args.decoder_root / "summary.json")
    if decoder_summary.get("status") != "PASS_TASK_CONDITIONAL_DECODER":
        raise ValueError("Physics Teacher requires a passing task decoder")
    rows = _fit_rows(args.registry_csv.resolve())
    specs = _task_specs()
    slices = _load_object_slices(args.decoder_root.resolve())
    roles: dict[tuple[str, int], Any] = {}
    for key, spec in specs.items():
        roles[key] = parse_bddl_task_role(spec["text"], suite=key[0], task_idx=key[1], object_names=spec["object_names"])
    role_counts = Counter(role.status for role in roles.values())
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        labels_root = staging / "labels"
        labels_root.mkdir()
        window_rows: list[dict[str, Any]] = []
        step_count = 0
        known_steps = 0
        tier_counts: Counter[str] = Counter()
        phase_counts: Counter[str] = Counter()
        identity_count = 0
        for row in rows:
            identity = row["canonical_parent_key"]
            suite = row["suite"]
            task_idx = int(row["task_idx"])
            root = Path(row["selected_artifact_root"])
            _verify_source_artifact(root, row["selected_artifact_recursive_sha256"])
            step_rows = _load_jsonl(root / "step_records.jsonl")
            sidecar_rows = _load_jsonl(root / "privileged_teacher_sidecar.jsonl")
            metadata = _load_json(root / "episode_metadata.json")
            if metadata.get("canonical_parent_key") != identity or len(step_rows) != len(sidecar_rows):
                raise ValueError(f"source identity/length mismatch: {identity}")
            role = roles[(suite, task_idx)]
            derived, windows = derive_episode_rows(step_rows, sidecar_rows, role, slices[(suite, task_idx)], protocol)
            for item in derived:
                item["canonical_parent_key"] = identity
                item["state_id"] = int(row["state_id"])
                item["source_artifact_recursive_sha256"] = row["selected_artifact_recursive_sha256"]
                item["physics_protocol_schema"] = protocol["schema"]
            target = labels_root / suite / f"task_{task_idx:02d}" / f"state_{int(row['state_id']):02d}"
            target.mkdir(parents=True)
            label_name = "physics_teacher_v21c.jsonl" if v21c else ("physics_teacher_v21.jsonl" if v21 else "physics_teacher_v2.jsonl")
            _atomic_text(target / label_name, "".join(json.dumps(item, sort_keys=True) + "\n" for item in derived))
            for window in windows:
                window_rows.append({"canonical_parent_key": identity, **window})
            identity_count += 1
            step_count += len(derived)
            known_steps += sum(bool(item["known_mask"]) for item in derived)
            tier_counts.update(str(item["utility_tier"]) for item in derived if item["utility_tier"] is not None)
            phase_counts.update(item["phase_name"] for item in derived)
        roles_csv = staging / "task_roles.csv"
        with roles_csv.open("w", newline="", encoding="utf-8") as handle:
            fields = ["suite", "task_idx", "manipulated_objects", "target_names", "support_names", "goal_predicates", "status", "reason"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for key, role in sorted(roles.items()):
                writer.writerow({
                    "suite": key[0], "task_idx": key[1],
                    "manipulated_objects": json.dumps(role.manipulated_objects),
                    "target_names": json.dumps(role.target_names),
                    "support_names": json.dumps(role.support_names),
                    "goal_predicates": json.dumps(role.goal_predicates),
                    "status": role.status, "reason": role.reason,
                })
        with (staging / "window_index.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(window_rows[0]) if window_rows else ["canonical_parent_key", "window_id", "start_step", "end_step", "step_count", "phase_name", "utility_tier", "known", "candidate_close", "rankable"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(window_rows)
        if v21c:
            manifest_schema = "DETECTOR_V5_PHYSICS_TEACHER_V21C_MANIFEST"
            teacher_version = "V2.1C"
            manifest_filename = "physics_teacher_v21c_manifest.json"
        elif v21:
            manifest_schema = "DETECTOR_V5_PHYSICS_TEACHER_V21_MANIFEST"
            teacher_version = "V2.1"
            manifest_filename = "physics_teacher_v2_manifest.json"
        else:
            manifest_schema = "DETECTOR_V5_PHYSICS_TEACHER_V2_MANIFEST"
            teacher_version = "V2"
            manifest_filename = "physics_teacher_v2_manifest.json"
        manifest = {
            "schema": manifest_schema,
            "teacher_version": teacher_version,
            "protocol_schema": protocol["schema"],
            "protocol_sha256": sha256_file(args.protocol.resolve()),
            "registry_csv_sha256": sha256_file(args.registry_csv.resolve()),
            "registry_root_sha256s_sha256": registry_seal["sha256sums_sha256"],
            "decoder_summary_sha256": sha256_file(args.decoder_root / "summary.json"),
            "decoder_root_sha256sums_sha256": decoder_seal["sha256sums_sha256"],
            "physics_audit_summary_sha256": sha256_file(args.physics_audit_root / "OFFICIAL_V3_PRIVILEGED_PHYSICS_TEACHER_AUDIT_V1.json"),
            "physics_audit_root_sha256sums_sha256": physics_audit_seal["sha256sums_sha256"],
            "identity_count": identity_count,
            "step_count": step_count,
            "known_step_count": known_steps,
            "window_count": len(window_rows),
            "task_count": 40,
            "task_role_status_counts": dict(sorted(role_counts.items())),
            "utility_tier_step_counts": dict(sorted(tier_counts.items())),
            "phase_step_counts": dict(sorted(phase_counts.items())),
            "teacher_is_clean_only_physics_proxy": True,
            "counterfactual_attack_label": False,
            "student_future_leakage": False,
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        if v21c:
            action_contract_path = Path(_action_contract.__file__).resolve()
            action_contract_sha256 = sha256_file(action_contract_path)
            v5_physics_path = Path(__import__("gripper_attack.v5_physics").__file__).resolve()
            v5_physics_sha256 = sha256_file(v5_physics_path)
            repo_root = action_contract_path.parent.parent
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
                ).strip()
            except Exception:
                git_commit = "UNKNOWN"
            manifest["action_contract_schema"] = _action_contract.ACTION_CONTRACT_SCHEMA
            manifest["action_contract_sha256"] = action_contract_sha256
            manifest["action_contract_source"] = str(action_contract_path)
            manifest["v5_physics_source"] = str(v5_physics_path)
            manifest["v5_physics_sha256"] = v5_physics_sha256
            manifest["source_git_commit"] = git_commit
            action_contract_doc = {
                "schema": _action_contract.ACTION_CONTRACT_SCHEMA,
                "file_sha256": action_contract_sha256,
                "source_path": str(action_contract_path),
                "raw_close_region": _action_contract.ACTION_CONTRACT["raw_close_region"],
                "raw_close_operator": _action_contract.ACTION_CONTRACT["raw_close_operator"],
                "boundary_policy": _action_contract.ACTION_CONTRACT["boundary_policy"],
                "postprocess": _action_contract.ACTION_CONTRACT["postprocess"],
            }
            _atomic_text(staging / "action_contract.json", json.dumps(action_contract_doc, indent=2, sort_keys=True) + "\n")
        _atomic_text(staging / manifest_filename, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _atomic_text(staging / "protocol.json", json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        report = {
            "schema": ("DETECTOR_V5_PHYSICS_TEACHER_V21C_AUDIT_V1" if v21c else
                        "DETECTOR_V5_PHYSICS_TEACHER_V21_AUDIT_V1" if v21 else
                        "DETECTOR_V5_PHYSICS_TEACHER_V2_AUDIT_V1"),
            "status": "PASS_WITH_EXPLICIT_NON_GRASP_TASKS" if role_counts.get("ABSTAIN_DECODER_HOLD", 0) == 0 else "ABSTAIN_DECODER_HOLD",
            "manifest": manifest,
            "role_holds": [key for key, role in sorted(roles.items()) if role.status == "ABSTAIN_DECODER_HOLD"],
            "formal_training_authorized": False,
            "formal_attack_authorized": False,
        }
        _atomic_text(staging / "audit_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
        _write_seal(staging)
        os.replace(staging, output)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--decoder-root", type=Path, required=True)
    parser.add_argument("--physics-audit-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
