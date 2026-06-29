from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tools.table1_audit.common import (
    REQUIRED_BUNDLE_FILES,
    add_path_arg,
    canonical_json,
    load_json,
    parent_key,
    read_sha256sums,
    reject_symlink,
    sha256_file,
    write_json,
)


def verify(args: argparse.Namespace) -> dict:
    bundle = args.bundle.resolve()
    problems: list[dict] = []
    if not bundle.exists() or bundle.is_symlink():
        problems.append({"class": "missing_or_symlink_bundle", "path": str(bundle)})
        return {"schema_version": "bundle_verification.v1", "bundle": str(bundle), "verification_pass": False, "problems": problems}
    for name in REQUIRED_BUNDLE_FILES + ["MANIFEST.jsonl"]:
        p = bundle / name
        if not p.exists():
            problems.append({"class": "missing_bundle_file", "path": name})
        elif p.is_symlink():
            problems.append({"class": "symlink_bundle_file", "path": name})
    try:
        reject_symlink(bundle)
    except Exception as exc:
        problems.append({"class": "unexpected_symlink", "detail": str(exc)})
    if not problems:
        sums = read_sha256sums(bundle / "BUNDLE_SHA256SUMS.txt")
        for rel, expected in sums.items():
            if rel == "BUNDLE_SHA256SUMS.txt":
                problems.append({"class": "checksum_cycle", "path": rel})
                continue
            actual = sha256_file(bundle / rel)
            if actual != expected:
                problems.append({"class": "bundle_checksum_mismatch", "path": rel, "expected": expected, "actual": actual})
        manifest_sha = sha256_file(bundle / "MANIFEST.jsonl")
        manifest_sum = (bundle / "MANIFEST.sha256").read_text(encoding="utf-8").split()[0]
        if manifest_sha != manifest_sum:
            problems.append({"class": "manifest_copy_sha_mismatch", "expected": manifest_sum, "actual": manifest_sha})
        freeze = load_json(bundle / "CONDITION_FREEZE.json")
        if freeze.get("status") == "FROZEN":
            problems.append({"class": "candidate_bundle_claims_frozen"})
        if freeze.get("manifest_sha256") != manifest_sha:
            problems.append({"class": "freeze_manifest_sha_mismatch"})
        accepted = [x for x in (bundle / "accepted_job_keys.txt").read_text(encoding="utf-8").splitlines() if x]
        if len(accepted) != 162 or len(set(accepted)) != 162:
            problems.append({"class": "accepted_job_key_count", "count": len(accepted), "unique": len(set(accepted))})
        inventory = load_json(bundle / "RESULT_INVENTORY.json")
        rows = inventory.get("rows", [])
        parents = Counter(parent_key(r) for r in rows)
        if len(parents) != 54 or any(v != 3 for v in parents.values()):
            problems.append({"class": "pairing_not_54x3", "parents": len(parents)})
        if inventory.get("manifest_sha256") != manifest_sha:
            problems.append({"class": "inventory_manifest_sha_mismatch"})
        artifact_sums = read_sha256sums(bundle / "ARTIFACT_SHA256SUMS.txt")
        if not artifact_sums:
            problems.append({"class": "empty_artifact_checksum_inventory"})
    result = {
        "schema_version": "bundle_verification.v1",
        "bundle": str(bundle),
        "bundle_sha256sums_sha256": sha256_file(bundle / "BUNDLE_SHA256SUMS.txt") if (bundle / "BUNDLE_SHA256SUMS.txt").exists() else "",
        "verification_pass": not problems,
        "problems": problems,
    }
    return result


def markdown(result: dict) -> str:
    lines = ["# Condition Freeze Bundle Verification", "", f"Verdict: `{'PASS' if result['verification_pass'] else 'HOLD'}`", "", "## Problems", ""]
    if result["problems"]:
        lines += [f"- `{p['class']}`: `{canonical_json(p).strip()}`" for p in result["problems"]]
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def finalize(args: argparse.Namespace) -> dict:
    verification = load_json(args.bundle_verification)
    if not verification.get("verification_pass"):
        raise SystemExit("refusing finalize: verifier did not pass")
    bundle = args.bundle.resolve()
    final_dir = args.final_dir.resolve()
    if final_dir.exists():
        raise SystemExit("refusing finalize: final envelope destination exists")
    freeze = load_json(bundle / "CONDITION_FREEZE.json")
    if freeze.get("status") != "FREEZE_CANDIDATE":
        raise SystemExit("refusing finalize: bundle is not a freeze candidate")
    final = {
        **freeze,
        "status": "FROZEN",
        "candidate_bundle": str(bundle),
        "candidate_verifier_report_sha256": sha256_file(args.bundle_verification.resolve()),
    }
    final_dir.mkdir(parents=True, exist_ok=False)
    out = final_dir / "CONDITION_FREEZE_FINAL.json"
    write_json(out, final)
    (final_dir / "FINAL_ENVELOPE_SHA256SUMS.txt").write_text(f"{sha256_file(out)}  CONDITION_FREEZE_FINAL.json\n", encoding="utf-8")
    return {"finalized": True, "path": str(out), "status": "FROZEN", "final_dir": str(final_dir)}


def verify_final(args: argparse.Namespace) -> dict:
    final_dir = args.bundle.resolve()
    problems: list[dict] = []
    final = final_dir / "CONDITION_FREEZE_FINAL.json"
    sums = final_dir / "FINAL_ENVELOPE_SHA256SUMS.txt"
    if not final.exists() or not sums.exists():
        problems.append({"class": "missing_final_envelope_file"})
    else:
        expected = sums.read_text(encoding="utf-8").split()[0]
        actual = sha256_file(final)
        if expected != actual:
            problems.append({"class": "final_envelope_checksum_mismatch", "expected": expected, "actual": actual})
        data = load_json(final)
        if data.get("status") != "FROZEN":
            problems.append({"class": "final_envelope_not_frozen"})
        candidate = data.get("candidate_bundle")
        if not candidate or not Path(candidate).exists():
            problems.append({"class": "final_candidate_bundle_missing"})
    return {"schema_version": "final_envelope_verification.v1", "bundle": str(final_dir), "verification_pass": not problems, "final_envelope_sha256": sha256_file(final) if final.exists() else "", "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify or finalize a condition freeze bundle.")
    add_path_arg(ap, "--bundle", required=True)
    add_path_arg(ap, "--output-json")
    add_path_arg(ap, "--output-md")
    add_path_arg(ap, "--bundle-verification")
    add_path_arg(ap, "--final-dir")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--verify-final", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        print(canonical_json(finalize(args)), end="")
        return 0
    if args.verify_final:
        result = verify_final(args)
        if args.output_json:
            write_json(args.output_json, result)
        print(canonical_json(result), end="")
        return 0 if result["verification_pass"] else 2
    result = verify(args)
    if args.output_json:
        write_json(args.output_json, result)
    if args.output_md:
        args.output_md.write_text(markdown(result), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(canonical_json(result), end="")
    return 0 if result["verification_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
