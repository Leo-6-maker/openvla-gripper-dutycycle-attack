#!/usr/bin/env python3
"""Finalize Train300 recovery against the frozen 300-key manifest.

This is a read-only artifact auditor. It selects one primary CLEAN result per
canonical key, with the approved infra retry taking precedence for its key, and
then checks metadata, SHA manifests, row counts, NPZ lengths, and MP4 frame
counts. It does not launch LIBERO, load OpenVLA, train detectors, or run attacks.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


RETRY_KEY = "libero_goal|1|11|0|CLEAN"
EXPECTED_SOURCE_COMMIT = "63793972743f667c6a6bcc12e9700f322f261147"
REQUIRED_SIM_ARRAYS = ["qpos", "qvel", "body_xpos", "body_xquat", "site_xpos", "ctrl"]
SUMMARY_COUNT_FIELDS = [
    "identity_mismatch_count",
    "clean_contract_failure_count",
    "invalid_feature_episode_count",
    "sim_manifest_step_mismatch_count",
    "sim_array_missing_count",
    "sim_array_length_mismatch_count",
    "media_decode_failure_count",
    "artifact_manifest_coverage_failure_count",
    "artifact_sha_mismatch_count",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mp4_frame_count(path: Path) -> tuple[int | None, str]:
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(path)
        try:
            return int(reader.count_frames()), ""
        finally:
            reader.close()
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"


def npz_first_dim(path: Path, member: str) -> tuple[int | None, str]:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            if member not in z.files:
                return None, f"missing_member:{member}"
            return int(z[member].shape[0]), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def as_int(value: Any, default: int = -999) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(value)
    except Exception:
        return default


def canonical_value(summary: dict[str, Any], manifest: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary.get(key)
    return manifest.get(key)


def is_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    if isinstance(value, int):
        return value == 0
    return False


def is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, int):
        return value != 0
    return False


def artifact_entries(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = {}
    for item in artifact.get("files", []) or []:
        rel = item.get("path")
        if rel:
            entries[str(rel)] = item
    return entries


def compare_identity(master: dict[str, Any], manifest: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    mismatches = []
    for field in ["suite", "task_idx", "state_id", "eval_seed", "condition"]:
        expected = str(master.get(field, ""))
        observed = canonical_value(summary, manifest, field)
        if str(observed) != expected:
            mismatches.append(f"{field}:{observed}!={expected}")
    return mismatches


def clean_contract_problems(manifest: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    problems = []
    for source_name, source in [("manifest", manifest), ("summary", summary)]:
        if source.get("condition") != "CLEAN":
            problems.append(f"{source_name}.condition_not_clean:{source.get('condition')}")

    # Train300 manifests have attack_enabled; VIS/RAND flags are checked when
    # present, and summary.vis_or_rand_run guards historical collectors that
    # encoded VIS/RAND status with a combined field.
    if manifest.get("attack_enabled") is not False:
        problems.append(f"manifest.attack_enabled_not_false:{manifest.get('attack_enabled')}")
    for field in ["vis_enabled", "rand_enabled"]:
        if field in manifest and not is_false(manifest.get(field)):
            problems.append(f"manifest.{field}_not_false:{manifest.get(field)}")
        if field in summary and not is_false(summary.get(field)):
            problems.append(f"summary.{field}_not_false:{summary.get(field)}")
    for field in ["attack_enabled", "attack_this", "attack_run", "vis_or_rand_run"]:
        if field in summary and is_true(summary.get(field)):
            problems.append(f"summary.{field}_true:{summary.get(field)}")
    return problems


def audit_primary_episode(
    key: str,
    master: dict[str, str],
    row: dict[str, str],
    required: list[str],
) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    episode_dir = Path(row["output_dir"])
    counters = {field: 0 for field in SUMMARY_COUNT_FIELDS}
    problems: list[str] = []
    details: dict[str, Any] = {
        "canonical_key": key,
        "output_dir": str(episode_dir),
    }

    missing_required = [rel for rel in required if not (episode_dir / rel).exists()]
    if missing_required:
        problems.append("required_artifact_missing:" + "|".join(missing_required))
        counters["artifact_manifest_coverage_failure_count"] = 1

    summary_path = episode_dir / "episode_summary.json"
    manifest_path = episode_dir / "episode_manifest.json"
    artifact_path = episode_dir / "artifact_sha256.json"
    sim_manifest_path = episode_dir / "sim_state_manifest.json"

    summary = read_json(summary_path) if summary_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    identity_mismatches = compare_identity(master, manifest, summary)
    if identity_mismatches:
        problems.append("identity_mismatch:" + "|".join(identity_mismatches))
        counters["identity_mismatch_count"] = 1

    contract = clean_contract_problems(manifest, summary)
    if contract:
        problems.append("clean_contract_failure:" + "|".join(contract))
        counters["clean_contract_failure_count"] = 1

    invalid_feature_steps = as_int(summary.get("invalid_feature_steps"))
    if invalid_feature_steps != 0:
        problems.append(f"invalid_feature_steps:{invalid_feature_steps}")
        counters["invalid_feature_episode_count"] = 1

    if manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        problems.append("source_commit_mismatch")
    if manifest.get("unnorm_key") != master.get("unnorm_key"):
        problems.append("unnorm_mismatch")
    if manifest.get("model_path") != master.get("model_path"):
        problems.append("model_path_mismatch")
    if not (10 <= int(master["state_id"]) <= 19):
        problems.append("state_outside_10_19")

    n_steps = as_int(summary.get("n_steps"))
    step_rows = count_csv_rows(episode_dir / "step_telemetry.csv")
    detector_rows = count_csv_rows(episode_dir / "detector_telemetry.csv")
    frame_rows = count_csv_rows(episode_dir / "frame_index.csv")
    if step_rows != n_steps:
        problems.append(f"step_rows_mismatch:{step_rows}!={n_steps}")
    if detector_rows != n_steps:
        problems.append(f"detector_rows_mismatch:{detector_rows}!={n_steps}")
    if frame_rows != n_steps:
        problems.append(f"frame_rows_mismatch:{frame_rows}!={n_steps}")

    sim_manifest = read_json(sim_manifest_path) if sim_manifest_path.exists() else {}
    sim_steps = as_int(sim_manifest.get("steps"))
    if sim_steps != n_steps:
        problems.append(f"sim_manifest_steps_mismatch:{sim_steps}!={n_steps}")
        counters["sim_manifest_step_mismatch_count"] = 1

    sim_arrays = sim_manifest.get("arrays", {}) if isinstance(sim_manifest.get("arrays", {}), dict) else {}
    missing_arrays = [name for name in REQUIRED_SIM_ARRAYS if name not in sim_arrays]
    length_bad_arrays = []
    unreadable_arrays = []
    for name in REQUIRED_SIM_ARRAYS:
        first_dim, error = npz_first_dim(episode_dir / "sim_state_stream.npz", name)
        if first_dim is None:
            if name not in missing_arrays:
                unreadable_arrays.append(f"{name}:{error}")
            continue
        if first_dim != n_steps:
            length_bad_arrays.append(f"{name}:{first_dim}!={n_steps}")
    if missing_arrays or unreadable_arrays:
        problems.append(
            "sim_array_missing_or_unreadable:"
            + "|".join(missing_arrays + unreadable_arrays)
        )
        counters["sim_array_missing_count"] = 1
    if length_bad_arrays:
        problems.append("sim_array_length_mismatch:" + "|".join(length_bad_arrays))
        counters["sim_array_length_mismatch_count"] = 1

    artifact = read_json(artifact_path) if artifact_path.exists() else {"files": []}
    artifact_map = artifact_entries(artifact)
    required_manifest_entries = [rel for rel in required if rel != "artifact_sha256.json"]
    manifest_missing = [rel for rel in required_manifest_entries if rel not in artifact_map]
    disk_missing = [rel for rel in required if not (episode_dir / rel).exists()]
    if manifest_missing or disk_missing:
        problems.append(
            "artifact_manifest_coverage_failure:"
            + "|".join([f"manifest_missing:{x}" for x in manifest_missing] + [f"disk_missing:{x}" for x in disk_missing])
        )
        counters["artifact_manifest_coverage_failure_count"] = 1

    sha_bad = []
    sha_missing = []
    for rel, item in artifact_map.items():
        file_path = episode_dir / rel
        if not file_path.exists():
            sha_missing.append(f"{rel}:missing")
            counters["artifact_manifest_coverage_failure_count"] = 1
        elif sha256_file(file_path) != item.get("sha256"):
            sha_bad.append(f"{rel}:sha_mismatch")
    if sha_missing:
        problems.append("artifact_manifest_coverage_failure:" + "|".join(sha_missing[:20]))
    if sha_bad:
        problems.append("artifact_sha_mismatch:" + "|".join(sha_bad[:20]))
        counters["artifact_sha_mismatch_count"] = 1

    raw_frames, raw_error = mp4_frame_count(episode_dir / "rollout_raw.mp4")
    overlay_frames, overlay_error = mp4_frame_count(episode_dir / "rollout_overlay.mp4")
    agent_frames, agent_error = npz_first_dim(episode_dir / "agentview_frames_uint8.npz", "agentview")
    media_errors = []
    if raw_frames is None:
        media_errors.append(f"rollout_raw.mp4:{raw_error}")
    elif raw_frames != n_steps:
        media_errors.append(f"rollout_raw.mp4:frame_count:{raw_frames}!={n_steps}")
    if overlay_frames is None:
        media_errors.append(f"rollout_overlay.mp4:{overlay_error}")
    elif overlay_frames != n_steps:
        media_errors.append(f"rollout_overlay.mp4:frame_count:{overlay_frames}!={n_steps}")
    if agent_frames is None:
        media_errors.append(f"agentview_frames_uint8.npz:{agent_error}")
    elif agent_frames != n_steps:
        media_errors.append(f"agentview_frames_uint8.npz:frame_count:{agent_frames}!={n_steps}")
    if media_errors:
        problems.append("media_decode_failure:" + "|".join(media_errors))
        counters["media_decode_failure_count"] = 1

    details.update(
        {
            "n_steps": n_steps,
            "step_rows": step_rows,
            "detector_rows": detector_rows,
            "frame_rows": frame_rows,
            "sim_manifest_steps": sim_steps,
            "raw_mp4_frames": raw_frames,
            "overlay_mp4_frames": overlay_frames,
            "agentview_frames": agent_frames,
            "artifact_file_count": len(artifact.get("files", [])),
            "identity_mismatches": "|".join(identity_mismatches),
            "clean_contract_problems": "|".join(contract),
            "sim_array_missing_or_unreadable": "|".join(missing_arrays + unreadable_arrays),
            "sim_array_length_mismatch": "|".join(length_bad_arrays),
            "artifact_manifest_missing": "|".join(manifest_missing),
            "artifact_disk_missing": "|".join(disk_missing),
            "artifact_sha_missing": "|".join(sha_missing[:20]),
            "artifact_sha_bad": "|".join(sha_bad[:20]),
            "media_errors": "|".join(media_errors),
            "problems": "|".join(problems),
        }
    )
    details.update(counters)
    return details, counters, problems


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    master_path = root / "cross_suite_clean_train300_s10_19_master_manifest.csv"
    status_paths = [
        root / "queue_status_worker_spatial_gpu13.csv",
        root / "queue_status_worker_libero10_gpu54.csv",
        root / "queue_status_worker_goal_gpu26.csv",
        root / "queue_status_worker_goal_recovery_A_gpu13_attempt2.csv",
        root / "queue_status_worker_goal_recovery_B_gpu54.csv",
        root / "queue_status_worker_goal_t01_s11_infra_retry_v1_gpu13.csv",
    ]

    master_rows = read_csv(master_path)
    planned = {row["canonical_key"]: row for row in master_rows}
    status_rows: list[dict[str, str]] = []
    for status_path in status_paths:
        for row in read_csv(status_path):
            row = dict(row)
            row["_status_file"] = str(status_path)
            status_rows.append(row)

    complete_rows = [row for row in status_rows if row.get("status") == "COMPLETE"]
    primary: dict[str, dict[str, str]] = {}
    for row in complete_rows:
        key = row["canonical_key"]
        if key == RETRY_KEY:
            if "retry_v1" in row.get("assigned_worker", ""):
                primary[key] = row
            elif key not in primary:
                primary[key] = row
        elif key not in primary:
            primary[key] = row

    duplicate_primary_keys = []
    for key in planned:
        rows = [row for row in complete_rows if row["canonical_key"] == key]
        if key == RETRY_KEY:
            rows = [row for row in rows if "retry_v1" in row.get("assigned_worker", "")]
        if len(rows) > 1:
            duplicate_primary_keys.append(key)

    missing = sorted(set(planned) - set(primary))
    extra = sorted(set(primary) - set(planned))

    required = [
        "episode_manifest.json",
        "episode_summary.json",
        "step_telemetry.csv",
        "detector_telemetry.csv",
        "frame_index.csv",
        "agentview_frames_uint8.npz",
        "sim_state_stream.npz",
        "sim_state_manifest.json",
        "rollout_raw.mp4",
        "rollout_overlay.mp4",
        "video_manifest.json",
        "artifact_sha256.json",
    ]

    ledger_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    success_count = 0
    clean_failure_count = 0

    for key in sorted(planned):
        master = planned[key]
        row = primary.get(key)
        ledger = dict(master)
        if row is None:
            ledger.update({"primary_status": "MISSING", "primary_output_dir": ""})
            ledger_rows.append(ledger)
            failures.append({"canonical_key": key, "failure": "missing_primary"})
            continue

        episode_dir = Path(row["output_dir"])
        ledger.update(
            {
                "primary_status": "COMPLETE",
                "primary_output_dir": str(episode_dir),
                "primary_source": row.get("_status_file", ""),
                "primary_worker": row.get("assigned_worker", ""),
                "primary_gpu_pair": row.get("assigned_gpu_pair", ""),
            }
        )

        problems: list[str] = []
        for rel in required:
            if not (episode_dir / rel).exists():
                problems.append(f"missing:{rel}")

        summary = read_json(episode_dir / "episode_summary.json") if (episode_dir / "episode_summary.json").exists() else {}
        ledger["task_success"] = summary.get("task_success", "")
        ledger["n_steps"] = summary.get("n_steps", "")
        ledger["invalid_feature_steps"] = summary.get("invalid_feature_steps", "")
        ledger["mlp_emit_step"] = summary.get("mlp_emit_step", "")
        if summary.get("task_success") is True:
            success_count += 1
        elif summary.get("task_success") is False:
            clean_failure_count += 1

        integrity, counters, problems = audit_primary_episode(key, master, row, required)
        integrity_rows.append(integrity)
        if problems:
            failures.append({"canonical_key": key, "failure": "integrity", "problems": problems})
        ledger_rows.append(ledger)

    write_csv(out / "final_primary_ledger.csv", ledger_rows)
    write_csv(out / "deep_integrity_results.csv", integrity_rows)

    state_overlap = sum(1 for row in ledger_rows if str(row.get("state_id")) in {str(i) for i in range(10)})
    counter_totals = {field: sum(int(row.get(field, 0) or 0) for row in integrity_rows) for field in SUMMARY_COUNT_FIELDS}
    summary = {
        "timestamp": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "classification": "PASS_300_VALID"
        if len(primary) == 300
        and not missing
        and not extra
        and not duplicate_primary_keys
        and not failures
        and all(value == 0 for value in counter_totals.values())
        and state_overlap == 0
        else "PARTIAL_OR_FAIL_REVIEW_REQUIRED",
        "planned": len(master_rows),
        "unique_planned": len(planned),
        "primary_complete": len(primary),
        "missing": len(missing),
        "extra": len(extra),
        "duplicate_primary_keys": len(duplicate_primary_keys),
        "integrity_failure_count": len(failures),
        "success_count": success_count,
        "clean_failure_count": clean_failure_count,
        "state_overlap_with_clean300_states_0_9": state_overlap,
        "train_split": "states 10-17",
        "val_split": "states 18-19",
        "test_split": "states 0-9 frozen CLEAN300",
        "retry_key": RETRY_KEY,
        "retry_primary_output": primary.get(RETRY_KEY, {}).get("output_dir", ""),
        "failures": failures[:20],
    }
    summary.update(counter_totals)
    (out / "final_acceptance_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report = f"""# Train300 Final Acceptance Report

```text
classification = {summary['classification']}
planned = {summary['planned']}
primary_complete = {summary['primary_complete']}
missing = {summary['missing']}
extra = {summary['extra']}
duplicate_primary_keys = {summary['duplicate_primary_keys']}
integrity_failure_count = {summary['integrity_failure_count']}
identity_mismatch_count = {summary['identity_mismatch_count']}
clean_contract_failure_count = {summary['clean_contract_failure_count']}
invalid_feature_episode_count = {summary['invalid_feature_episode_count']}
sim_manifest_step_mismatch_count = {summary['sim_manifest_step_mismatch_count']}
sim_array_missing_count = {summary['sim_array_missing_count']}
sim_array_length_mismatch_count = {summary['sim_array_length_mismatch_count']}
media_decode_failure_count = {summary['media_decode_failure_count']}
artifact_manifest_coverage_failure_count = {summary['artifact_manifest_coverage_failure_count']}
artifact_sha_mismatch_count = {summary['artifact_sha_mismatch_count']}
clean_success = {summary['success_count']}
clean_failure = {summary['clean_failure_count']}
state_overlap_with_clean300_states_0_9 = {summary['state_overlap_with_clean300_states_0_9']}
```

Retry key `{RETRY_KEY}` uses primary output:

```text
{summary['retry_primary_output']}
```

Split freeze:

```text
states 10-17 = training
states 18-19 = validation
states 0-9 = frozen CLEAN300 test
```

No Layer2, VIS, RAND, shuffled, oracle, or attack execution was run by this audit.
"""
    (out / "TRAIN300_FINAL_ACCEPTANCE_20260620.md").write_text(report, encoding="utf-8")

    hash_rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.stat().st_size < 50_000_000:
            hash_rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(out / "finalization_small_evidence_sha256.csv", hash_rows, ["path", "size_bytes", "sha256"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
