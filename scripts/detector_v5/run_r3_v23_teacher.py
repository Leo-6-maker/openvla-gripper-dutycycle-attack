"""Materialize V23 five-head labels from an already consumable FIT canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gripper_attack.v5_r3_teacher import HEADS, derive_episode_labels
from gripper_attack.seal_utils import rename_noreplace
from audit_r3_contact_input import load_consumable_episodes, sha256_file


ACCEPTED_PROTOCOL_SCHEMA = "V5_TEACHER_STUDENT_R3_DEV_PROTOCOL_V1_AMENDED_FAST_CLOSURE"


def _write_seal(root: Path) -> str:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name not in excluded),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def run(input_root: Path, output_root: Path, protocol_path: Path, *, expected_count: int = 8, transition_manifest_path: Path | None = None) -> dict:
    if output_root.exists():
        raise FileExistsError(output_root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != ACCEPTED_PROTOCOL_SCHEMA:
        raise ValueError("unexpected R3 protocol")
    manifest, episodes, input_seal = load_consumable_episodes(input_root, expected_count=expected_count, transition_manifest_path=transition_manifest_path)
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or output_root.exists():
        raise FileExistsError(output_root)
    staging.mkdir(parents=True)
    label_count = 0
    event_rows = []
    try:
        for episode in episodes:
            identity = str(episode["manifest"]["episode_id"])
            labels = derive_episode_labels(episode["rows"], protocol)
            target = staging / "labels" / f"{identity.replace('/', '__')}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels), encoding="utf-8")
            label_count += len(labels)
            for row in labels:
                event_rows.append({"episode_id": identity, **row})
        report = {
            "schema": "V5_R3_V23_TEACHER_CANARY_V1",
            "status": "DEVELOPMENT_NONCONSUMABLE",
            "input_schema": manifest["schema"],
            "input_status": manifest["status"],
            "input_sha256sums_sha256": input_seal["sha256sums_sha256"],
            "source_root": manifest.get("source_root"),
            "identity_allowlist_sha256": manifest.get("identity_allowlist_sha256"),
            "identity_allowlist_path": manifest.get("identity_allowlist_path"),
            "transition_manifest_sha256": manifest.get("transition_manifest_sha256"),
            "transition_manifest_path": manifest.get("transition_manifest_path"),
            "transition_sha256sums_sha256": manifest.get("transition_sha256sums_sha256"),
            "protocol_sha256": sha256_file(protocol_path),
            "identity_count": len(episodes),
            "step_count": label_count,
            "heads": list(HEADS),
            "unknown_to_negative": False,
            "future_fields_used": False,
            "outcome_fields_used": False,
            "protected_reads": 0,
            "formal_training_authorized": False,
            "formal_inference_authorized": False,
            "attack_authorized": False,
        }
        (staging / "teacher_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "teacher_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in event_rows), encoding="utf-8")
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
        report["sha256sums_sha256"] = digest
        return report
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "R3_DEV_PROTOCOL.json")
    parser.add_argument("--expected-count", type=int, default=8)
    parser.add_argument("--transition-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.input_root.resolve(), args.output_root.resolve(), args.protocol.resolve(), expected_count=args.expected_count, transition_manifest_path=args.transition_manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
