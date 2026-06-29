from __future__ import annotations

import argparse
from pathlib import Path

from tools.table1_audit.common import (
    REQUIRED_BUNDLE_FILES,
    add_path_arg,
    canonical_json,
    job_key,
    load_json,
    load_jsonl,
    output_dir,
    parent_key,
    sha256_file,
    write_json,
)


def build(args: argparse.Namespace) -> dict:
    validation = load_json(args.validator_json)
    if not validation.get("closure_pass"):
        raise SystemExit("refusing freeze bundle: closure validator did not pass")
    manifest = args.manifest.resolve()
    rows = load_jsonl(manifest)
    dest = args.dest.resolve()
    if dest.exists() and any(dest.iterdir()) and not args.dry_run:
        raise SystemExit(f"refusing non-empty destination: {dest}")
    if dest == args.condition_root.resolve():
        raise SystemExit("refusing to write freeze bundle directly over source condition root")

    accepted = validation.get("accepted_job_keys", [])
    inventory = {
        "condition_id": args.condition_id,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "row_count": len(rows),
        "accepted_count": len(accepted),
        "source_root": str(args.condition_root.resolve()),
        "rows": rows,
    }
    pairing = {}
    for row in rows:
        key = "|".join(parent_key(row))
        pairing.setdefault(key, []).append(job_key(row))
    artifacts = []
    for row in rows:
        out = output_dir(row, manifest.parent)
        for p in sorted(out.glob("*")) if out.exists() else []:
            if p.is_file():
                artifacts.append({"sha256": sha256_file(p), "path": str(p)})
    files = {
        "MANIFEST.sha256": f"{inventory['manifest_sha256']}  MANIFEST.jsonl\n",
        "accepted_job_keys.txt": "".join(f"{k}\n" for k in sorted(accepted)),
        "RESULT_INVENTORY.json": canonical_json(inventory),
        "PROVENANCE_AUDIT.json": canonical_json({"condition_id": args.condition_id, "provenance": validation.get("provenance", {})}),
        "PAIRING_AUDIT.json": canonical_json({"parents": pairing}),
        "ARTIFACT_SHA256SUMS.txt": "".join(f"{a['sha256']}  {a['path']}\n" for a in artifacts),
        "CONDITION_RESULTS.json": canonical_json({"condition_id": args.condition_id, "closure_pass": True, "row_classes": validation.get("row_classes", {})}),
        "CONDITION_FREEZE.json": canonical_json({"condition_id": args.condition_id, "status": "FROZEN", "manifest_sha256": inventory["manifest_sha256"], "required_files": REQUIRED_BUNDLE_FILES}),
        "README_RESTORE.txt": f"Restore by verifying MANIFEST.sha256 and ARTIFACT_SHA256SUMS.txt for {args.condition_id}.\n",
    }
    preview = {"dry_run": args.dry_run, "dest": str(dest), "would_write": sorted(files)}
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=False)
        (dest / "MANIFEST.jsonl").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
        for name, text in files.items():
            (dest / name).write_text(text, encoding="utf-8")
    return preview


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a condition freeze bundle after closure validation.")
    add_path_arg(ap, "--validator-json", required=True)
    add_path_arg(ap, "--manifest", required=True)
    add_path_arg(ap, "--condition-root", required=True)
    add_path_arg(ap, "--dest", required=True)
    ap.add_argument("--condition-id", required=True)
    ap.add_argument("--write", action="store_true", help="Actually write the bundle. Default is dry-run.")
    args = ap.parse_args()
    args.dry_run = not args.write
    print(canonical_json(build(args)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
