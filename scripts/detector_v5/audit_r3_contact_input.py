"""Fail-closed audit for a sealed FIT670 V2 contact-complete canary.

The production entrypoint is intentionally inert until the source manifest says
PASS_ENGINEERING_CONSUMABLE_INPUT_GATE. This script never discovers or selects
identities from a directory; it consumes only the frozen manifest list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gripper_attack.v5_r3_teacher import R3ContractError, validate_contact_row


CANARY_SCHEMA = "FIT670_V2_CANARY_CONTACT_TELEMETRY_V1"
CONSUMABLE_STATUS = "PASS_ENGINEERING_CONSUMABLE_INPUT_GATE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise ValueError(f"top-level seal missing: {root}")
    if sidecar.read_text(encoding="utf-8").strip() != f"{sha256_file(sums)}  SHA256SUMS":
        raise ValueError(f"top-level seal sidecar mismatch: {root}")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name in listed or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ValueError(f"invalid seal row: {line!r}")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file():
            raise ValueError(f"unsafe or missing sealed file: {name}")
        if sha256_file(root / relative) != digest:
            raise ValueError(f"sealed file mismatch: {name}")
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in sealed root: {path}")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise ValueError(f"sealed file closure mismatch: missing={sorted(actual-listed)} extra={sorted(listed-actual)}")
    return {"sha256sums_sha256": sha256_file(sums), "file_count": len(listed)}


def load_consumable_episodes(input_root: Path, *, expected_count: int = 8) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = input_root.resolve()
    seal = verify_seal(root)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("MANIFEST.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CANARY_SCHEMA:
        raise ValueError(f"unexpected canary schema: {manifest.get('schema')!r}")
    if manifest.get("status") != CONSUMABLE_STATUS:
        raise ValueError(f"input is not consumable: {manifest.get('status')!r}")
    if manifest.get("protected_reads") is not False or manifest.get("attack_enabled") is not False:
        raise ValueError("input manifest authorization is not FIT-only")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_count:
        raise ValueError(f"expected exactly {expected_count} frozen episodes")
    identities: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for item in episodes:
        if not isinstance(item, dict):
            raise ValueError("malformed episode manifest row")
        for key in ("episode_id", "suite", "task_id", "state_id", "seed", "relative_path", "step_count", "source_sha256", "source_commit", "source_tree", "source_command", "environment"):
            if key not in item or item[key] in (None, ""):
                raise ValueError(f"episode manifest missing {key}")
        identity = str(item["episode_id"])
        if identity in identities:
            raise ValueError(f"duplicate episode identity: {identity}")
        identities.add(identity)
        relative = Path(str(item["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe episode path: {relative}")
        path = root / relative
        if not re.fullmatch(r"[0-9a-f]{64}", str(item["source_sha256"])):
            raise ValueError(f"invalid episode source SHA: {identity}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(item["source_commit"])) or not re.fullmatch(r"[0-9a-f]{40}", str(item["source_tree"])):
            raise ValueError(f"invalid source lineage SHA: {identity}")
        if not path.is_file() or sha256_file(path) != str(item["source_sha256"]):
            raise ValueError(f"episode source seal mismatch: {identity}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != int(item["step_count"]):
            raise ValueError(f"episode step count mismatch: {identity}")
        for step, row in enumerate(rows):
            if row.get("episode_id") != identity:
                raise ValueError(f"episode identity mismatch: {identity} step={step}")
            validate_contact_row(row, expected_step=step)
        loaded.append({"manifest": item, "rows": rows})
    return manifest, loaded, seal


def audit(input_root: Path, output_root: Path | None = None, *, expected_count: int = 8) -> dict[str, Any]:
    manifest, episodes, seal = load_consumable_episodes(input_root, expected_count=expected_count)
    report = {
        "schema": "V5_R3_CONTACT_INPUT_AUDIT_V1",
        "status": "PASS_ENGINEERING_CONSUMABLE_INPUT_GATE",
        "input_schema": manifest["schema"],
        "input_status": manifest["status"],
        "identity_count": len(episodes),
        "step_count": sum(len(item["rows"]) for item in episodes),
        "source_sha256s": [item["manifest"]["source_sha256"] for item in episodes],
        "input_sha256sums_sha256": seal["sha256sums_sha256"],
        "protected_reads": 0,
        "attack_enabled": False,
        "teacher_labels_generated": False,
    }
    if output_root is not None:
        output = output_root.resolve()
        if output.exists():
            raise FileExistsError(output)
        staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
        if staging.exists():
            raise FileExistsError(staging)
        staging.mkdir(parents=True)
        (staging / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = [path for path in staging.rglob("*") if path.is_file()]
        (staging / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in sorted(files)), encoding="utf-8")
        (staging / "SHA256SUMS.sha256").write_text(f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
        os.rename(staging, output)
        report["output_root"] = str(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-count", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(audit(args.input_root, args.output_root, expected_count=args.expected_count), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
