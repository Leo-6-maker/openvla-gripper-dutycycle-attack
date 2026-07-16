#!/usr/bin/env python3
"""Read-only audit of the versioned multi-event 25D reconstruction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve()
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from gripper_attack.b3_causal_25d import B3Causal25DMultieventV1  # noqa: E402
except ModuleNotFoundError:  # standalone server-side audit copy
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from b3_causal_25d import B3Causal25DMultieventV1  # type: ignore[no-redef]


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row: {path}")
                rows.append(value)
    return rows


def audit_artifact(artifact: Path, source_root: Path) -> dict:
    metadata = json.loads((artifact / "episode_metadata.json").read_text(encoding="utf-8"))
    steps = jsonl(artifact / "step_records.jsonl")
    sidecar = jsonl(artifact / "privileged_teacher_sidecar.jsonl")
    if len(steps) != len(sidecar):
        raise ValueError(f"step/sidecar length mismatch: {len(steps)} != {len(sidecar)}")

    # Read only robot fields from the privileged sidecar.  Object state,
    # contact pairs, and any teacher labels are intentionally not copied.
    records = []
    for step, privileged in zip(steps, sidecar):
        if step.get("step") != privileged.get("step"):
            raise ValueError("step/sidecar identity mismatch")
        record = dict(step)
        for name in ("robot0_eef_pos", "robot0_gripper_qpos"):
            if name in privileged:
                record[name] = privileged[name]
        records.append(record)

    rebuilt = B3Causal25DMultieventV1().rebuild(records)
    rows = rebuilt["rows"]
    valid_rows = [row for row in rows if row["valid"]]
    onset_steps = [row["step"] for row in rows if row["close_onset"]]
    reset_steps = [row["step"] for row in rows if row["event_local_state_reset"]]
    later_event_rows = [row for row in rows if row["event_id"] >= 1]
    return {
        "canonical_parent_key": metadata.get("canonical_parent_key"),
        "suite": metadata.get("suite"),
        "task_idx": metadata.get("task_idx"),
        "state_id": metadata.get("state_id"),
        "split": metadata.get("split"),
        "source_artifact_relative": str(artifact.relative_to(source_root)),
        "source_artifact_manifest_sha256": sha256_file(artifact / "artifact_sha256.json"),
        "step_count": len(rows),
        "valid_feature_rows": len(valid_rows),
        "invalid_feature_rows": len(rows) - len(valid_rows),
        "event_count": len(rebuilt["events"]),
        "onset_count": len(onset_steps),
        "reset_count": len(reset_steps),
        "later_event_row_count": len(later_event_rows),
        "onset_steps": json.dumps(onset_steps, separators=(",", ":")),
        "reset_steps": json.dumps(reset_steps, separators=(",", ":")),
        "status": "PASS" if len(valid_rows) == len(rows) else "HOLD",
    }


def run(source_root: Path, output_root: Path, state_max: int, runner_git_head: str) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")

    records = []
    errors = []
    for suite in SUITES:
        for task in range(10):
            for state in range(state_max + 1):
                artifact = source_root / suite / f"task_{task:02d}" / f"state_{state:02d}"
                try:
                    records.append(audit_artifact(artifact, source_root))
                except Exception as exc:  # report every identity, fail closed later
                    records.append({
                        "canonical_parent_key": f"{suite}/task_{task:02d}/state_{state:02d}",
                        "suite": suite, "task_idx": task, "state_id": state,
                        "status": "HOLD", "error": f"{type(exc).__name__}: {exc}",
                    })
                    errors.append(records[-1])

    output_root.mkdir(parents=True, exist_ok=False)
    fields = sorted({key for row in records for key in row})
    census_path = output_root / "B3_CAUSAL_25D_FEATURE_AUDIT_V1.csv"
    with census_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    by_suite = {}
    for suite in SUITES:
        group = [row for row in records if row.get("suite") == suite]
        by_suite[suite] = {
            "identities": len(group),
            "pass": sum(row.get("status") == "PASS" for row in group),
            "hold": sum(row.get("status") != "PASS" for row in group),
            "events": sum(int(row.get("event_count", 0)) for row in group),
            "later_event_rows": sum(int(row.get("later_event_row_count", 0)) for row in group),
        }
    summary = {
        "schema": "B3_CAUSAL_25D_FEATURE_AUDIT_V1",
        "source_root": str(source_root),
        "state_range": [0, state_max],
        "identity_count": len(records),
        "unique_identity_count": len({row.get("canonical_parent_key") for row in records}),
        "pass_count": sum(row.get("status") == "PASS" for row in records),
        "hold_count": sum(row.get("status") != "PASS" for row in records),
        "error_count": len(errors),
        "by_suite": by_suite,
        "runner_git_head": runner_git_head,
        "audit_script_sha256": sha256_file(SCRIPT_DIR),
        "source_teacher_labels_read": False,
        "source_object_state_used": False,
        "source_contact_pairs_used": False,
        "source_unchanged": True,
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_root / "B3_CAUSAL_25D_FEATURE_AUDIT_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = []
    for path in sorted(output_root.iterdir()):
        if path.is_file():
            sums.append(f"{sha256_file(path)}  {path.name}\n")
    (output_root / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-max", type=int, choices=range(50), default=19)
    parser.add_argument("--runner-git-head", required=True)
    args = parser.parse_args()
    summary = run(args.source_root, args.output_root, args.state_max, args.runner_git_head)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["hold_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
