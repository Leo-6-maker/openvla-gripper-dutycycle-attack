from __future__ import annotations

import argparse
import hashlib
import shutil
import uuid
from pathlib import Path

from tools.table1_audit.common import (
    REQUIRED_BUNDLE_FILES,
    add_path_arg,
    canonical_digest,
    canonical_json,
    ensure_within_root,
    job_key,
    load_json,
    load_jsonl,
    output_dir,
    parent_key,
    recursive_inventory,
    reject_symlink,
    sha256_file,
)


def _must_match(validation: dict, manifest: Path, condition_root: Path, args: argparse.Namespace) -> None:
    checks = {
        "manifest_sha256": sha256_file(manifest),
        "condition_root_identity": canonical_digest({"condition_root": str(condition_root.resolve())}),
        "state_selection_sha256": sha256_file(args.state_selection.resolve()),
        "global_freeze_sha256": sha256_file(args.global_freeze.resolve()),
        "runtime_lock_sha256": sha256_file(args.runtime_lock.resolve()),
        "retry_policy_sha256": sha256_file(args.retry_policy.resolve()),
        "required_artifact_schema_sha256": sha256_file(args.required_artifact_schema.resolve()),
    }
    for key, actual in checks.items():
        if validation.get(key) != actual:
            raise SystemExit(f"refusing freeze bundle: stale validator report for {key}")
    if not validation.get("closure_pass"):
        raise SystemExit("refusing freeze bundle: closure validator did not pass")
    if validation.get("problems"):
        raise SystemExit("refusing freeze bundle: validator report contains problems")
    accepted = validation.get("accepted_job_keys") or []
    if len(accepted) != 162 or len(set(accepted)) != 162:
        raise SystemExit("refusing freeze bundle: accepted job-key set is not exactly 162 unique keys")


def _artifact_inventory(condition_root: Path, rows: list[dict]) -> list[dict]:
    job_by_dir: dict[Path, str] = {}
    for row in rows:
        out = output_dir(row, Path("."), condition_root)
        job_by_dir[out.resolve()] = job_key(row)
    inv = []
    for item in recursive_inventory(condition_root):
        full = condition_root / item["relative_path"]
        parent = next((job for d, job in job_by_dir.items() if full.resolve().is_relative_to(d)), "")
        inv.append({**item, "job_key": parent})
    return sorted(inv, key=lambda x: x["relative_path"])


def _bundle_files(args: argparse.Namespace, validation: dict, manifest: Path, condition_root: Path, rows: list[dict]) -> dict[str, str]:
    artifacts = _artifact_inventory(condition_root, rows)
    artifact_sums = "".join(f"{a['sha256']}  {a['relative_path']}\n" for a in artifacts)
    pairing = {}
    for row in rows:
        pairing.setdefault("|".join(parent_key(row)), []).append(job_key(row))
    inventory = {
        "schema_version": "condition_result_inventory.v1",
        "condition_id": args.condition_id,
        "manifest_sha256": sha256_file(manifest),
        "row_count": len(rows),
        "accepted_count": len(validation["accepted_job_keys"]),
        "source_root_metadata": str(condition_root),
        "rows": rows,
    }
    bundle_inventory = {
        "schema_version": "bundle_inventory.v1",
        "excludes": ["BUNDLE_SHA256SUMS.txt"],
        "artifact_inventory_digest": validation["artifact_inventory_digest"],
    }
    files = {
        "MANIFEST.jsonl": manifest.read_bytes().decode("utf-8"),
        "MANIFEST.sha256": f"{sha256_file(manifest)}  MANIFEST.jsonl\n",
        "accepted_job_keys.txt": "".join(f"{k}\n" for k in sorted(validation["accepted_job_keys"])),
        "RESULT_INVENTORY.json": canonical_json(inventory),
        "PROVENANCE_AUDIT.json": canonical_json({"schema_version": "provenance_audit.v1", "contracts": validation["contracts"], "provenance": validation.get("provenance", {})}),
        "PAIRING_AUDIT.json": canonical_json({"schema_version": "pairing_audit.v1", "parents": pairing}),
        "ARTIFACT_SHA256SUMS.txt": artifact_sums,
        "CONDITION_RESULTS.json": canonical_json({"schema_version": "condition_results.v1", "condition_id": args.condition_id, "closure_pass": True, "row_classes": validation.get("row_classes", {})}),
        "CONDITION_FREEZE.json": canonical_json({"schema_version": "condition_freeze.v1", "condition_id": args.condition_id, "status": "FREEZE_CANDIDATE", "manifest_sha256": sha256_file(manifest), "validator_report_sha256": sha256_file(args.validator_json.resolve()), "required_files": REQUIRED_BUNDLE_FILES}),
        "BUNDLE_INVENTORY.json": canonical_json(bundle_inventory),
        "README_RESTORE.txt": "Read-only restore check:\npython -m tools.table1_audit.verify_condition_freeze_bundle --bundle . --output-json BUNDLE_VERIFICATION.json --output-md BUNDLE_VERIFICATION.md\n",
    }
    sums = []
    for name, text in sorted(files.items()):
        if name == "BUNDLE_SHA256SUMS.txt":
            continue
        sums.append(f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  {name}\n")
    files["BUNDLE_SHA256SUMS.txt"] = "".join(sums)
    return files


def build(args: argparse.Namespace) -> dict:
    validation = load_json(args.validator_json)
    manifest = args.manifest.resolve()
    condition_root = args.condition_root.resolve()
    freeze_root = args.freeze_root.resolve()
    dest = args.dest.resolve()
    if dest.exists():
        raise SystemExit(f"refusing destination that already exists: {dest}")
    reject_symlink(dest.parent)
    reject_symlink(condition_root)
    try:
        ensure_within_root(dest, freeze_root)
    except ValueError as exc:
        raise SystemExit(f"refusing destination outside freeze root: {dest}") from exc
    try:
        dest.relative_to(condition_root)
        raise SystemExit("refusing destination inside source condition root")
    except ValueError:
        pass
    _must_match(validation, manifest, condition_root, args)
    rows = load_jsonl(manifest)
    files = _bundle_files(args, validation, manifest, condition_root, rows)
    preview = {"dry_run": args.dry_run, "dest": str(dest), "status": "FREEZE_CANDIDATE", "would_write": sorted(files)}
    if args.dry_run:
        return preview
    freeze_root.mkdir(parents=True, exist_ok=True)
    staging = freeze_root / f".{dest.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name, text in files.items():
            (staging / name).write_bytes(text.encode("utf-8"))
        staging.replace(dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return preview


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a condition freeze candidate after closure validation.")
    add_path_arg(ap, "--validator-json", required=True)
    add_path_arg(ap, "--manifest", required=True)
    add_path_arg(ap, "--condition-root", required=True)
    add_path_arg(ap, "--state-selection", required=True)
    add_path_arg(ap, "--global-freeze", required=True)
    add_path_arg(ap, "--runtime-lock", required=True)
    add_path_arg(ap, "--retry-policy", required=True)
    add_path_arg(ap, "--required-artifact-schema", required=True)
    add_path_arg(ap, "--freeze-root", required=True)
    add_path_arg(ap, "--dest", required=True)
    ap.add_argument("--condition-id", required=True)
    ap.add_argument("--write", action="store_true", help="Actually write the candidate. Default is dry-run.")
    args = ap.parse_args()
    args.dry_run = not args.write
    print(canonical_json(build(args)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
