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
        manifest = read_json(episode_dir / "episode_manifest.json") if (episode_dir / "episode_manifest.json").exists() else {}
        ledger["task_success"] = summary.get("task_success", "")
        ledger["n_steps"] = summary.get("n_steps", "")
        ledger["invalid_feature_steps"] = summary.get("invalid_feature_steps", "")
        ledger["mlp_emit_step"] = summary.get("mlp_emit_step", "")
        if summary.get("task_success") is True:
            success_count += 1
        elif summary.get("task_success") is False:
            clean_failure_count += 1

        if manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT:
            problems.append("source_commit_mismatch")
        if manifest.get("unnorm_key") != master.get("unnorm_key"):
            problems.append("unnorm_mismatch")
        if manifest.get("model_path") != master.get("model_path"):
            problems.append("model_path_mismatch")
        if not (10 <= int(master["state_id"]) <= 19):
            problems.append("state_outside_10_19")

        n_steps = int(summary.get("n_steps", -999)) if str(summary.get("n_steps", "")).strip() else -999
        step_rows = count_csv_rows(episode_dir / "step_telemetry.csv")
        detector_rows = count_csv_rows(episode_dir / "detector_telemetry.csv")
        frame_rows = count_csv_rows(episode_dir / "frame_index.csv")
        if step_rows != n_steps:
            problems.append(f"step_rows_mismatch:{step_rows}!={n_steps}")
        if detector_rows != n_steps:
            problems.append(f"detector_rows_mismatch:{detector_rows}!={n_steps}")
        if frame_rows != n_steps:
            problems.append(f"frame_rows_mismatch:{frame_rows}!={n_steps}")

        artifact_path = episode_dir / "artifact_sha256.json"
        artifact = read_json(artifact_path) if artifact_path.exists() else {"files": []}
        sha_bad = []
        for item in artifact.get("files", []):
            file_path = episode_dir / item["path"]
            if not file_path.exists():
                sha_bad.append(f"{item['path']}:missing")
            elif sha256_file(file_path) != item.get("sha256"):
                sha_bad.append(f"{item['path']}:sha_mismatch")
        if sha_bad:
            problems.append("artifact_sha_mismatch:" + "|".join(sha_bad[:5]))

        raw_frames, raw_error = mp4_frame_count(episode_dir / "rollout_raw.mp4")
        overlay_frames, overlay_error = mp4_frame_count(episode_dir / "rollout_overlay.mp4")
        if raw_frames is not None and raw_frames != n_steps:
            problems.append(f"raw_mp4_frame_mismatch:{raw_frames}!={n_steps}")
        if overlay_frames is not None and overlay_frames != n_steps:
            problems.append(f"overlay_mp4_frame_mismatch:{overlay_frames}!={n_steps}")
        agent_frames, agent_error = npz_first_dim(episode_dir / "agentview_frames_uint8.npz", "agentview")
        if agent_frames is not None and agent_frames != n_steps:
            problems.append(f"agentview_npz_step_mismatch:{agent_frames}!={n_steps}")

        integrity_rows.append(
            {
                "canonical_key": key,
                "output_dir": str(episode_dir),
                "n_steps": n_steps,
                "step_rows": step_rows,
                "detector_rows": detector_rows,
                "frame_rows": frame_rows,
                "raw_mp4_frames": raw_frames,
                "overlay_mp4_frames": overlay_frames,
                "agentview_frames": agent_frames,
                "artifact_file_count": len(artifact.get("files", [])),
                "problems": "|".join(problems),
                "raw_mp4_error": raw_error,
                "overlay_mp4_error": overlay_error,
                "agentview_npz_error": agent_error,
            }
        )
        if problems:
            failures.append({"canonical_key": key, "failure": "integrity", "problems": problems})
        ledger_rows.append(ledger)

    write_csv(out / "final_primary_ledger.csv", ledger_rows)
    write_csv(out / "deep_integrity_results.csv", integrity_rows)

    state_overlap = sum(1 for row in ledger_rows if str(row.get("state_id")) in {str(i) for i in range(10)})
    summary = {
        "timestamp": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "classification": "PASS_300_VALID"
        if len(primary) == 300 and not missing and not extra and not duplicate_primary_keys and not failures
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
