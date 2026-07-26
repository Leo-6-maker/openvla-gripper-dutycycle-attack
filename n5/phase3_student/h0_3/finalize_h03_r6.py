#!/usr/bin/env python3
"""Finalize two H0.3-R6 C1 runs into a reproducible evidence package."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SUMMARY = "ENTITY_REGISTRY_V2_SUMMARY.json"
LEDGER = "ALIAS_LEDGER.json"
SOURCE_FILES = (
    "n5/phase3_student/t2rc1_v2_registry.py",
    "n5/phase3_student/tests/test_c1_v2_resolver.py",
    "n5/phase3_student/h0_3/finalize_h03_r6.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def git(repo_root, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
    ).strip()


def git_symbolic_ref(repo_root):
    """Return the branch name, or empty string for a detached HEAD."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "symbolic-ref", "-q", "--short", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout.strip()


def canonical_summary(summary):
    result = copy.deepcopy(summary)
    result.pop("timestamp", None)
    result.pop("self_sha256", None)
    for row in result.get("per_task", []):
        row.pop("artifact_sha", None)
    return result


def canonical_ledger(ledger):
    result = copy.deepcopy(ledger)
    result.pop("timestamp", None)
    result.pop("self_sha256", None)
    aliases = result.get("aliases", [])
    aliases.sort(key=lambda row: (
        row.get("task_key", ""),
        row.get("entity_role", ""),
        row.get("target", ""),
        row.get("alias", ""),
    ))
    return result


def canonical_run_payload(run_dir):
    summary = load_json(run_dir / SUMMARY)
    ledger = load_json(run_dir / LEDGER)
    return {
        "summary": canonical_summary(summary),
        "alias_ledger": canonical_ledger(ledger),
    }


def inventory(root, label):
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({
            "path": f"{label}/{path.relative_to(root).as_posix()}",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def validate_summary(summary, label):
    expected = {
        "status": "PASS",
        "n_tasks": 40,
        "n_ok": 40,
        "n_env_errors": 0,
        "n_blocked": 0,
        "object_unresolved": 0,
        "object_ambiguous": 0,
        "target_unresolved": 0,
        "target_ambiguous": 0,
    }
    errors = []
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(
                f"{label}:{key} expected {value!r}, got {summary.get(key)!r}")
    per_task = summary.get("per_task", [])
    if len(per_task) != 40:
        errors.append(f"{label}: per_task length {len(per_task)} != 40")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    out = args.out.resolve()

    errors = []
    if run_a == run_b:
        errors.append("run_A and run_B resolve to the same directory")
    if out in (run_a, run_b):
        errors.append("output directory must be distinct from both runs")

    head = git(repo_root, "rev-parse", "HEAD")
    tree = git(repo_root, "rev-parse", "HEAD^{tree}")
    status = git(repo_root, "status", "--porcelain")
    if head != args.source_commit:
        errors.append(f"HEAD {head} != source commit {args.source_commit}")
    if status:
        errors.append("source worktree is dirty")
    branch = git_symbolic_ref(repo_root)
    if branch:
        errors.append(f"worktree is not detached (branch={branch})")

    for label, run_dir in (("run_A", run_a), ("run_B", run_b)):
        if not run_dir.is_dir():
            errors.append(f"{label} missing directory: {run_dir}")
            continue
        for required in (SUMMARY, LEDGER):
            if not (run_dir / required).is_file():
                errors.append(f"{label} missing {required}")
        per_task = run_dir / "per_task"
        if not per_task.is_dir():
            errors.append(f"{label} missing per_task directory")
        elif len(list(per_task.glob("*.json"))) != 40:
            errors.append(
                f"{label} per_task count "
                f"{len(list(per_task.glob('*.json')))} != 40")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    summary_a = load_json(run_a / SUMMARY)
    summary_b = load_json(run_b / SUMMARY)
    errors.extend(validate_summary(summary_a, "run_A"))
    errors.extend(validate_summary(summary_b, "run_B"))

    payload_a = canonical_run_payload(run_a)
    payload_b = canonical_run_payload(run_b)
    payload_a_sha = canonical_sha(payload_a)
    payload_b_sha = canonical_sha(payload_b)
    ledger_a_sha = canonical_sha(payload_a["alias_ledger"])
    ledger_b_sha = canonical_sha(payload_b["alias_ledger"])

    if payload_a_sha != payload_b_sha:
        errors.append("run_A/run_B canonical payload digests differ")
    if ledger_a_sha != ledger_b_sha:
        errors.append("run_A/run_B canonical alias-ledger digests differ")

    source_inventory = []
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            errors.append(f"missing source file {relative}")
        else:
            source_inventory.append({
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 3

    files = inventory(run_a, "run_A") + inventory(run_b, "run_B")
    manifest = {
        "gate": "H0.3-R6",
        "status": "PASS_CANDIDATE_AWAITING_TRANSITION_REVIEW",
        "created_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "commit": head,
            "tree": tree,
            "detached_head": True,
            "git_status_porcelain": "",
            "files": source_inventory,
        },
        "runs": {
            "run_A": {
                "summary_sha256": sha256_file(run_a / SUMMARY),
                "alias_ledger_sha256": sha256_file(run_a / LEDGER),
                "canonical_payload_sha256": payload_a_sha,
                "canonical_alias_ledger_sha256": ledger_a_sha,
            },
            "run_B": {
                "summary_sha256": sha256_file(run_b / SUMMARY),
                "alias_ledger_sha256": sha256_file(run_b / LEDGER),
                "canonical_payload_sha256": payload_b_sha,
                "canonical_alias_ledger_sha256": ledger_b_sha,
            },
            "canonical_payload_identical": True,
            "canonical_alias_ledger_identical": True,
        },
        "files": files,
        "finalization_protocol": {
            "manifest_excludes_its_own_hash": True,
            "sha256sums_excludes_itself": True,
            "transition_receipt_required_in_later_commit": True,
        },
    }

    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "ARTIFACT_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    checksum_rows = list(files)
    checksum_rows.extend(source_inventory)
    checksum_rows.append({
        "path": "ARTIFACT_MANIFEST.json",
        "size_bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    })
    checksums_path = out / "SHA256SUMS"
    with open(checksums_path, "w", encoding="utf-8") as handle:
        for row in sorted(checksum_rows, key=lambda value: value["path"]):
            handle.write(f"{row['sha256']}  {row['path']}\n")

    print(json.dumps({
        "status": "PASS_CANDIDATE_AWAITING_TRANSITION_REVIEW",
        "source_commit": head,
        "source_tree": tree,
        "run_A_canonical_payload_sha256": payload_a_sha,
        "run_B_canonical_payload_sha256": payload_b_sha,
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "sha256sums_sha256": sha256_file(checksums_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
