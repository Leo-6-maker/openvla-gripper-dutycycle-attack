#!/usr/bin/env python3
"""Audit exact consecutive clean image snapshots for X1; no PGD or env steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
    "env_steps_with_perturbed_action": 0,
}
REQUIRED_OFFSETS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(worktree: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(worktree), *args], text=True).strip()


def array_descriptor(manifest: dict[str, Any], field: str) -> dict[str, Any] | None:
    for descriptor in manifest.get("arrays", []):
        if isinstance(descriptor, dict) and descriptor.get("field") == field:
            return descriptor
    return None


def exact_frame(manifest_path: Path, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    payload = manifest.get("payload")
    binding = manifest.get("binding")
    if not isinstance(payload, dict) or not isinstance(binding, dict):
        return False, ["payload_or_binding_missing"]
    if manifest.get("status") not in (None, "PASS", "PASS_CAUSAL_PROBE_SNAPSHOT", "SEALED_PROSPECTIVE_SNAPSHOT"):
        reasons.append("snapshot_status_not_pass")
    if not isinstance(binding.get("parent_key"), str) or not isinstance(binding.get("probe_id"), str) or not isinstance(binding.get("step"), int):
        reasons.append("binding_incomplete")
    current = array_descriptor(manifest, "payload.canonical_policy_rgb_224")
    if current is None:
        reasons.append("canonical_policy_rgb_224_missing")
    else:
        binary = manifest_path.parent / str(current.get("binary_path", ""))
        if not binary.is_file():
            reasons.append("canonical_policy_rgb_224_binary_missing")
        elif sha256_file(binary) != current.get("raw_sha256"):
            reasons.append("canonical_policy_rgb_224_sha_mismatch")
    if not isinstance(payload.get("clean_reference_action_window"), list):
        reasons.append("clean_reference_action_window_missing")
    return not reasons, reasons


def source_records(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for source in protocol["population"]["sources"]:
        stage = str(source["stage"])
        root = Path(source["path"])
        paths = sorted(root.rglob("CAUSAL_PROBE_SNAPSHOT_V2.json"))
        if len(paths) != int(source["snapshot_count"]):
            raise ValueError(f"snapshot count mismatch {stage}: {len(paths)}")
        for path in paths:
            manifest = load_json(path)
            binding = manifest.get("binding") or {}
            valid, reasons = exact_frame(path, manifest)
            parent = binding.get("parent_key")
            probe_id = binding.get("probe_id")
            step = binding.get("step")
            if not isinstance(parent, str) or not isinstance(probe_id, str) or not isinstance(step, int):
                raise ValueError(f"incomplete snapshot identity: {path}")
            records.append({
                "stage": stage,
                "suite": parent.split("/", 1)[0],
                "canonical_parent_key": parent,
                "probe_id": probe_id,
                "absolute_step": int(step),
                "snapshot_path": str(path),
                "snapshot_manifest_sha256": sha256_file(path),
                "exact_current_frame": valid,
                "invalid_reasons": reasons,
                "clean_action_window_count": len(manifest.get("payload", {}).get("clean_reference_action_window", [])) if isinstance(manifest.get("payload"), dict) else 0,
            })
            files.append({"stage": stage, "path": str(path), "sha256": sha256_file(path), "exact_current_frame": valid})
    return records, files


def sequence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_parent: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    duplicate_steps: list[dict[str, Any]] = []
    for record in records:
        key = (record["stage"], record["canonical_parent_key"])
        step = int(record["absolute_step"])
        if step in by_parent[key]:
            duplicate_steps.append({"stage": key[0], "canonical_parent_key": key[1], "absolute_step": step})
        else:
            by_parent[key][step] = record
    windows: Counter[str] = Counter()
    max_lengths: list[dict[str, Any]] = []
    eligible_starts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (stage, parent), steps in sorted(by_parent.items()):
        ordered = sorted(steps)
        longest = 0
        current = 0
        previous = None
        for step in ordered:
            if previous is not None and step == previous + 1 and steps[step]["exact_current_frame"] and steps[previous]["exact_current_frame"]:
                current += 1
            else:
                current = 1 if steps[step]["exact_current_frame"] else 0
            longest = max(longest, current)
            previous = step
        max_lengths.append({"stage": stage, "canonical_parent_key": parent, "max_consecutive_exact_frames": longest})
        for start in ordered:
            if not steps[start]["exact_current_frame"]:
                continue
            length = 1
            while length < len(REQUIRED_OFFSETS) and start + length in steps and steps[start + length]["exact_current_frame"]:
                length += 1
            windows[str(length)] += 1
            for requested in (3, 5, 10):
                if length >= requested:
                    eligible_starts[str(requested)].append({"stage": stage, "canonical_parent_key": parent, "start_step": start, "length": length})
    return {
        "parent_count": len(by_parent),
        "duplicate_absolute_steps": duplicate_steps,
        "window_start_count_by_available_length": dict(sorted(windows.items(), key=lambda item: int(item[0]))),
        "eligible_start_count_by_required_length": {str(length): len(eligible_starts[str(length)]) for length in (3, 5, 10)},
        "eligible_starts": {key: value for key, value in sorted(eligible_starts.items())},
        "max_consecutive_exact_frames": max_lengths,
    }


def seal(root: Path, summary: dict[str, Any]) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    (root / "ROOT_SEAL.json").write_text(json.dumps({
        "schema": "STAGE_X_X1_SEQUENCE_INPUT_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": sha256_file(root / "STAGE_X_X1_SEQUENCE_INPUT_AUDIT.json"),
        "sha256sums_sha256": sums_sha,
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    if protocol.get("schema") != "STAGE_X_X1_SEQUENTIAL_PGD_PROTOCOL_V1":
        raise ValueError("wrong X1 protocol")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    records, files = source_records(protocol)
    sequence = sequence_summary(records)
    valid_current = sum(bool(record["exact_current_frame"]) for record in records)
    status = "X1_CLEAN_SEQUENCE_INPUT_READY" if sequence["eligible_start_count_by_required_length"]["3"] else "STAGE_X_X1_CLEAN_SEQUENCE_UNAVAILABLE"
    summary = {
        "schema": "STAGE_X_X1_SEQUENCE_INPUT_AUDIT_V1",
        "status": status,
        "x1_pgd_executed": False,
        "x1_authorized": False,
        "source_commit": git_value(args.worktree, "rev-parse", "HEAD"),
        "source_tree": git_value(args.worktree, "rev-parse", "HEAD^{tree}"),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "record_count": len(records),
        "valid_current_frame_count": valid_current,
        "invalid_current_frame_count": len(records) - valid_current,
        "input_files": files,
        "sequence": sequence,
        "clean_reference_action_window_is_frame_sequence": False,
        "repeated_frame_padding": False,
        "attacked_frames": False,
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (root / "STAGE_X_X1_SEQUENCE_INPUT_AUDIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "X1_SEQUENCE_ROWS.jsonl").write_text("".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    (root / "PROVENANCE.json").write_text(json.dumps({
        "schema": "STAGE_X_X1_SEQUENCE_INPUT_PROVENANCE_V1",
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "source_script_sha256": summary["source_script_sha256"],
        "protocol_path": str(args.protocol),
        "protocol_sha256": summary["protocol_sha256"],
        "physical_intervention": False,
        "env_steps_with_perturbed_action": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal(root, summary)
    print(json.dumps({"status": status, "record_count": len(records), "valid_current_frame_count": valid_current, "output_root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
