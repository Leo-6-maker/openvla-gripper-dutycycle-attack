"""Seal the prospective Stage VII A/B/C development decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
EXPECTED = {
    "S7-A": "STAGE_VII_S7A_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR",
    "S7-B": "STAGE_VII_S7B_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR",
    "S7-C": "STAGE_VII_S7C_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR",
}
SEAL_FILES = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def verify_seal(root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    sidecar_path = root / "SHA256SUMS.sha256"
    root_sidecar_path = root / "ROOT_SEAL.sha256"
    if not sums_path.is_file() or not sidecar_path.is_file() or not root_sidecar_path.is_file():
        raise SystemExit(f"SEAL_FILES_MISSING:{root}")
    entries: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise SystemExit(f"BAD_SHA256SUMS_LINE:{root}:{line}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"UNSAFE_SHA256SUMS_PATH:{root}:{relative}")
        target = root / path
        if not target.is_file() or sha256_file(target) != digest:
            raise SystemExit(f"SHA256_MISMATCH:{root}:{relative}")
        entries[path.as_posix()] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in SEAL_FILES
    }
    if actual != set(entries):
        raise SystemExit(f"SHA256_FILE_SET_MISMATCH:{root}")
    if sidecar_path.read_text(encoding="utf-8").split()[0] != sha256_file(sums_path):
        raise SystemExit(f"SHA256SUMS_SIDECAR_MISMATCH:{root}")
    root_json = root / "ROOT_SEAL.json"
    if root_sidecar_path.read_text(encoding="utf-8").split()[0] != sha256_file(root_json):
        raise SystemExit(f"ROOT_SEAL_SIDECAR_MISMATCH:{root}")


def require_protected_clean(summary: dict[str, Any], label: str) -> None:
    if summary.get("formal_m4_executed") is not False:
        raise SystemExit(f"FORMAL_M4_NOT_FALSE:{label}")
    if summary.get("eval160") != "UNREAD" or summary.get("protected_evaluation") != "UNREAD":
        raise SystemExit(f"PROTECTED_BOUNDARY_CHANGED:{label}")
    counters = summary.get("protected_counters")
    if not isinstance(counters, dict) or any(value != 0 for value in counters.values()):
        raise SystemExit(f"PROTECTED_COUNTER_NONZERO:{label}")


def candidate_snapshot(label: str, root: Path) -> dict[str, Any]:
    verify_seal(root)
    summary = read_json(root / f"STAGE_VII_{label.replace('-', '')}_CANDIDATE_DEVELOPMENT.json")
    if summary.get("status") != EXPECTED[label]:
        raise SystemExit(f"CANDIDATE_NOT_FAIL_CLOSED:{label}:{summary.get('status')}")
    require_protected_clean(summary, label)
    devtest = summary.get("metrics_by_split", {}).get("DEVTEST", {})
    loso = summary.get("loso", {})
    return {
        "status": summary["status"],
        "root": str(root),
        "source_commit": summary.get("source_commit"),
        "source_tree": summary.get("source_tree"),
        "devtest_pass": devtest.get("pass"),
        "devtest_overall": devtest.get("overall"),
        "devtest_per_suite": devtest.get("per_suite"),
        "loso_mean_identifiable_auroc": loso.get("mean_identifiable_auroc"),
        "loso_per_suite": loso.get("per_suite"),
        "threshold": summary.get("threshold"),
        "protected_counters": summary.get("protected_counters"),
    }


def seal(root: Path, summary: dict[str, Any]) -> None:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in SEAL_FILES):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    write_json(root / "ROOT_SEAL.json", {
        "schema": "STAGE_VII_DEVELOPMENT_DECISION_ROOT_SEAL_V2",
        "status": summary["status"],
        "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sha256sums_sha256": sums_sha,
        "fresh_holdout_authorized": False,
        "new_formal_m4_authorized": False,
        "new_timing_matrix_authorized": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    })
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forensic-root", required=True, type=Path)
    parser.add_argument("--s7a-root", required=True, type=Path)
    parser.add_argument("--s7b-root", required=True, type=Path)
    parser.add_argument("--s7c-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{output}")
    if git("status", "--porcelain"):
        raise SystemExit("WORKTREE_NOT_CLEAN")
    forensic = read_json(args.forensic_root.resolve() / "STAGE_VII_DOMAIN_SHIFT_FORENSIC.json")
    if not str(forensic.get("status", "")).startswith("PASS"):
        raise SystemExit(f"FORENSIC_NOT_PASS:{forensic.get('status')}")
    candidates = {
        label: candidate_snapshot(label, root.resolve())
        for label, root in {
            "S7-A": args.s7a_root,
            "S7-B": args.s7b_root,
            "S7-C": args.s7c_root,
        }.items()
    }
    summary = {
        "schema": "STAGE_VII_DEVELOPMENT_DECISION_V2",
        "status": "STAGE_VII_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "forensic_status": forensic["status"],
        "forensic_root": str(args.forensic_root.resolve()),
        "candidate_statuses": candidates,
        "scientific_conclusion": "NO_GENERALIZABLE_STAGE_VII_DEVELOPMENT_DETECTOR",
        "decision_basis": "S7-A, S7-B, and S7-C each failed at least one frozen development promotion gate; no fresh holdout or formal intervention is authorized.",
        "fresh_holdout_authorized": False,
        "new_formal_m4_authorized": False,
        "new_timing_matrix_authorized": False,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "protected_counters": COUNTERS,
        "frozen_stage_v_vi_changed": False,
        "threshold_retuned": False,
        "rerun_to_pass": False,
    }
    output.mkdir(parents=True)
    write_json(output / "STAGE_VII_DEVELOPMENT_DECISION.json", summary)
    seal(output, summary)
    print(json.dumps({"status": summary["status"], "output_root": str(output), "protected_counters": COUNTERS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
