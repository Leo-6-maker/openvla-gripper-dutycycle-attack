#!/usr/bin/env python3
"""Post-run audit for the 300-episode cross-suite CLEAN census.

This is an offline, read-only reducer. It does not load OpenVLA, does not use
GPU, does not run LIBERO, and does not mutate the output roots it inspects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


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


def npz_members(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return "|".join(sorted(zf.namelist()))
    except Exception as exc:
        return f"UNREADABLE:{type(exc).__name__}:{exc}"


def mp4_frame_count(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        import imageio.v2 as imageio  # type: ignore

        reader = imageio.get_reader(path)
        try:
            return str(reader.count_frames())
        finally:
            reader.close()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}:{exc}"


def deep_integrity(ep: Path, artifact: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    files = artifact.get("files", [])
    checked = 0
    mismatches: list[str] = []
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("path", ""))
            expected = str(item.get("sha256", ""))
            if not rel or rel == "artifact_sha256.json":
                continue
            path = ep / rel
            if not path.exists():
                mismatches.append(rel + ":missing")
                continue
            actual = sha256_file(path)
            checked += 1
            if expected and actual != expected:
                mismatches.append(rel)
    try:
        n_steps = int(summary.get("n_steps"))
    except Exception:
        n_steps = -1
    sim_manifest = read_json(ep / "sim_state_manifest.json")
    try:
        sim_steps = int(sim_manifest.get("steps"))
    except Exception:
        sim_steps = -1
    step_rows = csv_row_count(ep / "step_telemetry.csv")
    detector_rows = csv_row_count(ep / "detector_telemetry.csv")
    frame_rows = csv_row_count(ep / "frame_index.csv")
    return {
        "deep_integrity_enabled": True,
        "deep_sha_files_checked": checked,
        "deep_sha_mismatch_count": len(mismatches),
        "deep_sha_mismatches": "|".join(mismatches),
        "deep_step_rows_match_summary": step_rows == n_steps,
        "deep_detector_rows_match_summary": detector_rows == n_steps,
        "deep_frame_rows_match_summary": frame_rows == n_steps,
        "deep_sim_steps_match_summary": sim_steps == n_steps,
        "deep_agentview_npz_members": npz_members(ep / "agentview_frames_uint8.npz"),
        "deep_sim_state_npz_members": npz_members(ep / "sim_state_stream.npz"),
        "deep_raw_video_frame_count": mp4_frame_count(ep / "rollout_raw.mp4"),
        "deep_overlay_video_frame_count": mp4_frame_count(ep / "rollout_overlay.mp4"),
    }


def audit_episode(ep: Path, *, deep: bool = False) -> dict[str, Any]:
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
    row = {
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
    if deep:
        row.update(deep_integrity(ep, artifact, summary))
    else:
        row["deep_integrity_enabled"] = False
    return row


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


def canonical_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("suite", "")),
        str(row.get("task_idx", "")),
        str(row.get("state_id", "")),
        str(row.get("eval_seed", "")),
        str(row.get("condition", "CLEAN") or "CLEAN"),
    )


def key_text(key: tuple[str, str, str, str, str]) -> str:
    return "|".join(key)


def load_planned_rows(paths: list[Path]) -> list[dict[str, Any]]:
    planned: list[dict[str, Any]] = []
    for path in paths:
        for row in read_csv_rows(path):
            enriched = dict(row)
            enriched["queue_manifest_path"] = str(path)
            enriched["condition"] = row.get("condition") or "CLEAN"
            planned.append(enriched)
    return planned


def load_queue_status_rows(paths: list[Path]) -> dict[str, dict[str, str]]:
    status: dict[str, dict[str, str]] = {}
    for manifest in paths:
        for row in read_csv_rows(manifest.parent / "queue_status.csv"):
            for key in [row.get("job_id", ""), row.get("output_dir", ""), key_text(canonical_key(row))]:
                if key:
                    status[key] = row
    return status


def reconcile_planned(
    planned: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    queue_status: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    planned_by_key: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in planned:
        planned_by_key[canonical_key(row)].append(row)
    ledger_by_key: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        ledger_by_key[canonical_key(row)].append(row)

    rows: list[dict[str, Any]] = []
    for key, planned_rows in sorted(planned_by_key.items(), key=lambda item: key_text(item[0])):
        p0 = planned_rows[0]
        found = ledger_by_key.get(key, [])
        status_row = queue_status.get(str(p0.get("job_id", ""))) or queue_status.get(str(p0.get("output_dir", ""))) or queue_status.get(key_text(key)) or {}
        if not found:
            terminal = "INFRA_FAILED" if status_row.get("status") == "INFRA_FAILED" else "MISSING_PLANNED"
            rows.append({
                "canonical_key": key_text(key),
                "suite": key[0],
                "task_idx": key[1],
                "state_id": key[2],
                "eval_seed": key[3],
                "condition": key[4],
                "planned_count_for_key": len(planned_rows),
                "discovered_count_for_key": 0,
                "reconciliation_status": terminal,
                "planned_output_dir": p0.get("output_dir", ""),
                "queue_status": status_row.get("status", ""),
                "episode_path": "",
            })
            continue
        for item in found:
            if item.get("status") == "COMPLETE_VALID":
                terminal = "VALID_CLEAN_FAILURE" if item.get("task_success") is False else "VALID_COMPLETE"
            else:
                terminal = "SCHEMA_INVALID"
            rows.append({
                "canonical_key": key_text(key),
                "suite": key[0],
                "task_idx": key[1],
                "state_id": key[2],
                "eval_seed": key[3],
                "condition": key[4],
                "planned_count_for_key": len(planned_rows),
                "discovered_count_for_key": len(found),
                "reconciliation_status": terminal,
                "planned_output_dir": p0.get("output_dir", ""),
                "queue_status": status_row.get("status", ""),
                "episode_path": item.get("episode_path", ""),
                "task_success": item.get("task_success", ""),
                "source_commit": item.get("source_commit", ""),
                "gpu_pair": item.get("gpu_pair", ""),
            })
    for key, found in sorted(ledger_by_key.items(), key=lambda item: key_text(item[0])):
        if key in planned_by_key:
            continue
        for item in found:
            rows.append({
                "canonical_key": key_text(key),
                "suite": key[0],
                "task_idx": key[1],
                "state_id": key[2],
                "eval_seed": key[3],
                "condition": key[4],
                "planned_count_for_key": 0,
                "discovered_count_for_key": len(found),
                "reconciliation_status": "UNEXPECTED_EXTRA_KEY",
                "episode_path": item.get("episode_path", ""),
                "task_success": item.get("task_success", ""),
                "source_commit": item.get("source_commit", ""),
                "gpu_pair": item.get("gpu_pair", ""),
            })

    counts = Counter(str(row.get("reconciliation_status", "")) for row in rows)
    duplicate_planned = [key_text(key) for key, values in planned_by_key.items() if len(values) > 1]
    accounted = (
        counts.get("MISSING_PLANNED", 0)
        + counts.get("VALID_COMPLETE", 0)
        + counts.get("VALID_CLEAN_FAILURE", 0)
        + counts.get("INFRA_FAILED", 0)
        + counts.get("SCHEMA_INVALID", 0)
    )
    summary = {
        "planned_count": len(planned),
        "unique_planned_count": len(planned_by_key),
        "duplicate_planned_key_count": len(duplicate_planned),
        "duplicate_planned_keys": duplicate_planned,
        "discovered_count": len(ledger),
        "unique_discovered_count": len(ledger_by_key),
        "missing_planned_key_count": counts.get("MISSING_PLANNED", 0),
        "unexpected_extra_key_count": counts.get("UNEXPECTED_EXTRA_KEY", 0),
        "valid_complete_count": counts.get("VALID_COMPLETE", 0),
        "valid_clean_failure_count": counts.get("VALID_CLEAN_FAILURE", 0),
        "infra_failed_count": counts.get("INFRA_FAILED", 0),
        "schema_invalid_count": counts.get("SCHEMA_INVALID", 0),
        "replacement_states": counts.get("UNEXPECTED_EXTRA_KEY", 0),
        "hard_gate_planned_300": len(planned) == 300,
        "hard_gate_unique_planned_300": len(planned_by_key) == 300,
        "hard_gate_accounting_matches_planned": accounted == len(planned_by_key),
        "hard_gate_no_extra_denominator_keys": counts.get("UNEXPECTED_EXTRA_KEY", 0) == 0,
        "hard_gate_no_replacement_states": counts.get("UNEXPECTED_EXTRA_KEY", 0) == 0,
    }
    return rows, summary


def write_report(
    path: Path,
    roots: list[Path],
    ledger: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    dupes: list[dict[str, Any]],
    reconciliation_summary: dict[str, Any] | None,
) -> None:
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
        f"- Audit level: metadata postrun ledger audit{' + deep integrity' if any(str(r.get('deep_integrity_enabled')).lower() == 'true' for r in ledger) else ''}",
        "",
    ]
    if reconciliation_summary:
        lines += [
            "## Frozen Manifest Reconciliation",
            "",
            "```json",
            json.dumps(reconciliation_summary, indent=2, sort_keys=True),
            "```",
            "",
        ]
    lines += [
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
    ap.add_argument("--queue-manifest", action="append", default=[], help="Frozen queue_manifest.csv source of truth. May be repeated.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--deep-integrity", action="store_true", help="Recompute artifact SHA and inspect NPZ/video consistency. Run after collection finishes.")
    ap.add_argument("--no-gpu", action="store_true", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.roots]
    out = Path(args.output_dir)
    episodes = iter_episode_dirs(roots)
    ledger = [audit_episode(ep, deep=args.deep_integrity) for ep in episodes]
    summary = aggregate(ledger)
    dupes = duplicate_rows(ledger)
    planned = load_planned_rows([Path(p) for p in args.queue_manifest])
    queue_status = load_queue_status_rows([Path(p) for p in args.queue_manifest])
    reconciliation_rows: list[dict[str, Any]] = []
    reconciliation_summary: dict[str, Any] | None = None
    if planned:
        reconciliation_rows, reconciliation_summary = reconcile_planned(planned, ledger, queue_status)
    write_csv(out / "tables" / "cross_suite_clean_300_master_ledger.csv", ledger)
    write_csv(out / "tables" / "cross_suite_clean_300_summary.csv", summary)
    write_csv(out / "tables" / "cross_suite_clean_300_duplicate_conflicts.csv", dupes)
    if reconciliation_rows:
        write_csv(out / "tables" / "cross_suite_clean_300_reconciliation.csv", reconciliation_rows)
    write_json(out / "reports" / "cross_suite_clean_300_postrun_audit.json", {
        "roots": [str(r) for r in roots],
        "episode_count": len(ledger),
        "valid_complete_count": sum(r.get("status") == "COMPLETE_VALID" for r in ledger),
        "duplicate_count": len(dupes),
        "reconciliation": reconciliation_summary or {},
        "deep_integrity": bool(args.deep_integrity),
    })
    write_report(out / "reports" / "CROSS_SUITE_CLEAN_300_POSTRUN_AUDIT.md", roots, ledger, summary, dupes, reconciliation_summary)
    print(json.dumps({"result": "POSTRUN_AUDIT_DONE", "episodes": len(ledger), "valid": sum(r.get("status") == "COMPLETE_VALID" for r in ledger)}, sort_keys=True))


if __name__ == "__main__":
    main()
