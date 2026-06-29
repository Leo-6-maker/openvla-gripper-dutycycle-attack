from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path

from tools.table1_audit.common import (
    add_path_arg,
    canonical_digest,
    canonical_json,
    load_json,
    read_sha256sums,
    reject_symlink,
    safe_relative_path,
    sha256_file,
    write_json,
)


BAD_NAMES = ("credential", "secret", "token", "private_key", "id_rsa", ".pem", ".key", "model", "checkpoint", ".pt", ".pth", ".safetensors")


def _semantic_equal(a: Path, b: Path) -> bool | None:
    if a.suffix.lower() != ".json" or b.suffix.lower() != ".json":
        return None
    return canonical_digest(load_json(a)) == canonical_digest(load_json(b))


def verify(args: argparse.Namespace) -> dict:
    root = args.snapshot_root.resolve()
    github_root = args.github_root.resolve()
    problems: list[dict] = []
    rows: list[dict] = []
    required = ["SNAPSHOT_METADATA.json", "SHA256SUMS.txt", "FILES.json"]
    for name in required:
        if not (root / name).exists():
            problems.append({"class": "missing_snapshot_file", "path": name})
    if problems:
        return {"schema_version": "server_runtime_snapshot_verification.v1", "verification_pass": False, "problems": problems, "rows": rows}
    try:
        reject_symlink(root)
        sums = read_sha256sums(root / "SHA256SUMS.txt")
        files = load_json(root / "FILES.json")["files"]
        metadata = load_json(root / "SNAPSHOT_METADATA.json")
    except Exception as exc:
        return {"schema_version": "server_runtime_snapshot_verification.v1", "verification_pass": False, "problems": [{"class": "snapshot_parse_error", "detail": str(exc)}], "rows": rows}
    for field in ["server_hostname_identifier", "original_repo_path", "branch", "HEAD", "dirty_status", "snapshot_utc_timestamp", "original_relative_paths", "snapshot_creator_command_version"]:
        if field not in metadata:
            problems.append({"class": "metadata_missing_field", "field": field})
    registered = {str(x["relative_path"]) if isinstance(x, dict) else str(x) for x in files}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected_control = set(required)
    extra = sorted(actual - registered - expected_control)
    if extra:
        problems.append({"class": "snapshot_extra_file", "paths": extra})
    for rel in sorted(registered):
        try:
            safe_relative_path(rel)
        except Exception as exc:
            problems.append({"class": "snapshot_traversal", "path": rel, "detail": str(exc)})
            continue
        p = root / rel
        if p.is_symlink():
            problems.append({"class": "snapshot_symlink", "path": rel})
        if not p.exists():
            problems.append({"class": "snapshot_missing_registered_file", "path": rel})
            continue
        if any(bad in rel.lower() for bad in BAD_NAMES) or p.stat().st_size > args.max_file_bytes:
            problems.append({"class": "snapshot_forbidden_file", "path": rel, "size": p.stat().st_size})
        actual_sha = sha256_file(p)
        if sums.get(rel) != actual_sha:
            problems.append({"class": "snapshot_checksum_mismatch", "path": rel, "expected": sums.get(rel), "actual": actual_sha})
        gh = github_root / rel
        row = {
            "relative_path": rel,
            "server_sha256": actual_sha,
            "github_sha256": sha256_file(gh) if gh.exists() and gh.is_file() else "MISSING",
            "byte_identical": gh.exists() and gh.is_file() and sha256_file(gh) == actual_sha,
            "semantic_identical": _semantic_equal(p, gh) if gh.exists() and gh.is_file() else None,
        }
        if gh.exists() and gh.is_file() and not row["byte_identical"] and p.suffix.lower() == ".py":
            row["unified_diff"] = "".join(difflib.unified_diff(
                gh.read_text(encoding="utf-8", errors="replace").splitlines(True),
                p.read_text(encoding="utf-8", errors="replace").splitlines(True),
                fromfile=f"github/{rel}",
                tofile=f"server/{rel}",
            ))
        rows.append(row)
    return {
        "schema_version": "server_runtime_snapshot_verification.v1",
        "verification_pass": not problems,
        "snapshot_root": str(root),
        "github_root": str(github_root),
        "metadata": metadata,
        "problems": problems,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["relative_path", "server_sha256", "github_sha256", "byte_identical", "semantic_identical"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def markdown(result: dict) -> str:
    lines = ["# Server Runtime Snapshot Reconciliation", "", f"Verdict: `{'PASS' if result['verification_pass'] else 'HOLD'}`", "", "## Problems", ""]
    lines += [f"- `{p['class']}`: `{canonical_json(p).strip()}`" for p in result["problems"]] if result["problems"] else ["- none"]
    lines += ["", "## Files", ""]
    for row in result["rows"]:
        lines.append(f"- `{row['relative_path']}`: server `{row['server_sha256']}`, github `{row['github_sha256']}`, byte_identical `{row['byte_identical']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a small Bubble-transported server runtime snapshot.")
    add_path_arg(ap, "--snapshot-root", required=True)
    add_path_arg(ap, "--github-root", required=True)
    add_path_arg(ap, "--output-json", required=True)
    add_path_arg(ap, "--output-csv", required=True)
    add_path_arg(ap, "--output-md", required=True)
    ap.add_argument("--max-file-bytes", type=int, default=1_000_000)
    args = ap.parse_args()
    result = verify(args)
    write_json(args.output_json, result)
    write_csv(args.output_csv, result["rows"])
    args.output_md.write_text(markdown(result), encoding="utf-8")
    print(canonical_json({"verification_pass": result["verification_pass"], "problems": len(result["problems"])}), end="")
    return 0 if result["verification_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
