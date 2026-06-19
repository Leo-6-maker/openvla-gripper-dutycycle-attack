#!/usr/bin/env python3
"""Post-run audit for the 300-episode cross-suite CLEAN census.

This is an offline, read-only reducer. It does not load OpenVLA, does not use
GPU, does not run LIBERO, and does not mutate the output roots it inspects.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SUITES = ("libero_spatial", "libero_goal", "libero_10")
EXPECTED_SOURCE_COMMIT = "63793972743f667c6a6bcc12e9700f322f261147"


def read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def iter_episode_dirs(roots: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for summary in root.rglob("episode_summary.json"):
            dirs.append(summary.parent)
    return sorted(set(dirs))


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def first_csv_values(path: Path, columns: list[str]) -> dict[str, str]:
    if not path.exists():
        return {col: "" for col in columns}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        try:
            row = next(reader)
        except StopIteration:
            row = {}
    return {col: str(row.get(col, "")) for col in columns}


def audit_episode(ep: Path) -> dict[str, Any]:
    manifest = read_json(ep / "episode_manifest.json")
    summary = read_json(ep / "episode_summary.json")
    sidecar = read_json(ep / "privileged_sidecar.json")
    artifact = read_json(ep / "artifact_sha256.json")
    detector_rows = csv_row_count(ep / "detector_telemetry.csv")
    step_rows = csv_row_count(ep / "step_telemetry.csv")
    first_values = first_csv_values(ep / "detector_telemetry.csv", ["emit_step", "emitted"])

    condition = str(manifest.get("condition", summary.get("condition", "")))
    source_commit = str(manifest.get("source_commit", summary.get("source_commit", "")))
    clean_only = (
        condition == "CLEAN"
        and manifest.get("attack_enabled") is False
        and manifest.get("vis_enabled") is False
        and manifest.get("rand_enabled") is False
        and summary.get("vis_or_rand_run") is False
    )
    required = [
        "episode_manifest.json",
        "episode_summary.json",
        "step_telemetry.csv",
        "detector_telemetry.csv",
        "frame_index.csv",
        "privileged_sidecar.json",
        "sim_state_stream.npz",
        "sim_state_manifest.json",
        "artifact_sha256.json",
    ]
    missing = [name for name in required if not (ep / name).exists()]
    artifact_file_count = len(artifact.get("files", [])) if isinstance(artifact.get("files"), list) else 0
    status = "COMPLETE_VALID" if clean_only and not missing and source_commit == EXPECTED_SOURCE_COMMIT else "INVALID_OR_INCOMPLETE"
    if summary.get("task_success") not in {True, False}:
        status = "INVALID_OR_INCOMPLETE"
    try:
        invalid_feature_steps = int(summary.get("invalid_feature_steps"))
    except Exception:
        invalid_feature_steps = -1
    if invalid_feature_steps != 0:
        status = "SCIENTIFIC_INVALID"
    return {
        "episode_path": str(ep),
        "root_path": str(next((p for p in ep.parents if p.name.startswith("cross_suite_clean_300_20260619")), ep.parent)),
        "suite": manifest.get("suite", summary.get("suite", "")),
        "task_idx": manifest.get("task_idx", summary.get("task_idx", "")),
        "state_id": manifest.get("state_id", summary.get("state_id", "")),
        "eval_seed": manifest.get("eval_seed", summary.get("eval_seed", "")),
        "condition": condition,
        "source_commit": source_commit,
        "gpu_pair": manifest.get("gpu_snapshot", {}).get("cuda_visible_devices", ""),
        "model_path": manifest.get("model_path", ""),
        "unnorm_key": manifest.get("unnorm_key", ""),
        "detector_checkpoint_sha256": summary.get("checkpoint_sha256", manifest.get("detector_checkpoint_sha256", "")),
        "detector_dataset_sha256": summary.get("dataset_sha256", manifest.get("detector_dataset_sha256", "")),
        "task_success": summary.get("task_success", ""),
        "n_steps": summary.get("n_steps", ""),
        "invalid_feature_steps": summary.get("invalid_feature_steps", ""),
        "mlp_triggered": summary.get("mlp_triggered", first_values.get("emitted", "")),
        "mlp_emit_step": summary.get("mlp_emit_step", first_values.get("emit_step", "")),
        "detector_rows": detector_rows,
        "step_rows": step_rows,
        "privileged_valid": sidecar.get("privileged_valid", summary.get("privileged_valid", "")),
        "teacher_abstain": sidecar.get("teacher_abstain", summary.get("teacher_abstain", "")),
        "artifact_file_count": artifact_file_count,
        "artifact_recursive_sha256": artifact.get("recursive_sha256", ""),
        "missing_required_artifacts": "|".join(missing),
        "clean_only_contract": clean_only,
        "status": status,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("suite", "")), str(row.get("gpu_pair", "")))].append(row)
    out: list[dict[str, Any]] = []
    for (suite, gpu_pair), rs in sorted(groups.items()):
        valid = [r for r in rs if r.get("status") == "COMPLETE_VALID"]
        states = {(str(r.get("task_idx")), str(r.get("state_id"))) for r in valid}
        out.append({
            "suite": suite,
            "gpu_pair": gpu_pair,
            "episode_count": len(rs),
            "valid_complete_count": len(valid),
            "scientific_invalid_count": sum(r.get("status") == "SCIENTIFIC_INVALID" for r in rs),
            "invalid_or_incomplete_count": sum(r.get("status") == "INVALID_OR_INCOMPLETE" for r in rs),
            "unique_task_states": len(states),
            "clean_success_count": sum(r.get("task_success") is True for r in valid),
            "clean_failure_count": sum(r.get("task_success") is False for r in valid),
            "detector_emit_count": sum(str(r.get("mlp_triggered")).lower() == "true" for r in valid),
            "privileged_valid_count": sum(str(r.get("privileged_valid")).lower() == "true" for r in valid),
        })
    return out


def duplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("suite", "")),
            str(row.get("task_idx", "")),
            str(row.get("state_id", "")),
            str(row.get("eval_seed", "")),
            str(row.get("condition", "")),
        )
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, rs in sorted(groups.items()):
        if len(rs) <= 1:
            continue
        statuses = Counter(str(r.get("status")) for r in rs)
        successes = sorted({str(r.get("task_success")) for r in rs})
        out.append({
            "suite": key[0],
            "task_idx": key[1],
            "state_id": key[2],
            "eval_seed": key[3],
            "condition": key[4],
            "duplicate_count": len(rs),
            "status_counts": json.dumps(statuses, sort_keys=True),
            "task_success_values": "|".join(successes),
            "paths": " | ".join(str(r.get("episode_path")) for r in rs),
        })
    return out


def write_report(path: Path, roots: list[Path], ledger: list[dict[str, Any]], summary: list[dict[str, Any]], dupes: list[dict[str, Any]]) -> None:
    valid = [r for r in ledger if r.get("status") == "COMPLETE_VALID"]
    lines = [
        "# Cross-Suite CLEAN 300 Postrun Audit",
        "",
        "This report is generated from completed CLEAN-only artifacts. It performs no GPU work, no rollout, and no attack execution.",
        "",
        "## Inputs",
        "",
        *[f"- `{root}`" for root in roots],
        "",
        "## Headline",
        "",
        f"- Episodes discovered: {len(ledger)}",
        f"- Valid complete CLEAN episodes: {len(valid)}",
        f"- Duplicate canonical keys: {len(dupes)}",
        f"- Required source commit: `{EXPECTED_SOURCE_COMMIT}`",
        "",
        "## Suite/GPU Summary",
        "",
        "| Suite | GPU pair | Episodes | Valid | Invalid | Unique task-states | Clean success | Detector emit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['suite']} | {row['gpu_pair']} | {row['episode_count']} | {row['valid_complete_count']} | "
            f"{row['scientific_invalid_count'] + row['invalid_or_incomplete_count']} | {row['unique_task_states']} | "
            f"{row['clean_success_count']} | {row['detector_emit_count']} |"
        )
    lines += [
        "",
        "## Claim Boundaries",
        "",
        "- This audit supports CLEAN corpus accounting only.",
        "- It does not validate Teacher timing, VIS, RAND, attack success, or closed-loop robustness.",
        "- Clean task failures remain in the denominator and are not replaced.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True, help="Completed or in-progress output roots to scan.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-gpu", action="store_true", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.roots]
    out = Path(args.output_dir)
    episodes = iter_episode_dirs(roots)
    ledger = [audit_episode(ep) for ep in episodes]
    summary = aggregate(ledger)
    dupes = duplicate_rows(ledger)
    write_csv(out / "tables" / "cross_suite_clean_300_master_ledger.csv", ledger)
    write_csv(out / "tables" / "cross_suite_clean_300_summary.csv", summary)
    write_csv(out / "tables" / "cross_suite_clean_300_duplicate_conflicts.csv", dupes)
    write_json(out / "reports" / "cross_suite_clean_300_postrun_audit.json", {
        "roots": [str(r) for r in roots],
        "episode_count": len(ledger),
        "valid_complete_count": sum(r.get("status") == "COMPLETE_VALID" for r in ledger),
        "duplicate_count": len(dupes),
    })
    write_report(out / "reports" / "CROSS_SUITE_CLEAN_300_POSTRUN_AUDIT.md", roots, ledger, summary, dupes)
    print(json.dumps({"result": "POSTRUN_AUDIT_DONE", "episodes": len(ledger), "valid": sum(r.get("status") == "COMPLETE_VALID" for r in ledger)}, sort_keys=True))


if __name__ == "__main__":
    main()
