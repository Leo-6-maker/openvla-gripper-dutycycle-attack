"""Seal the fail-closed Stage VII development decision after candidate gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
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


def seal(root: Path, summary: dict[str, Any]) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    write_json(root / "ROOT_SEAL.json", {
        "schema": "STAGE_VII_DEVELOPMENT_DECISION_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sha256sums_sha256": sums_sha,
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
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{output}")
    forensic = read_json(args.forensic_root.resolve() / "STAGE_VII_DOMAIN_SHIFT_FORENSIC.json")
    s7a = read_json(args.s7a_root.resolve() / "STAGE_VII_S7A_CANDIDATE_DEVELOPMENT.json")
    if s7a.get("status") != "STAGE_VII_S7A_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR":
        raise SystemExit("S7A_NOT_FAIL_CLOSED")
    probes = forensic["context_probes"]
    if probes["P1_25D_plus_language"].get("status") != "UNAVAILABLE_NO_FROZEN_LANGUAGE_EMBEDDING":
        raise SystemExit("P1_STATUS_CHANGED")
    if probes["P4_25D_plus_frozen_visual"].get("status") != "UNAVAILABLE_NO_FROZEN_VISUAL_EMBEDDING":
        raise SystemExit("P4_STATUS_CHANGED")
    summary = {
        "schema": "STAGE_VII_DEVELOPMENT_DECISION_V1",
        "status": "STAGE_VII_DEVELOPMENT_NO_GENERALIZABLE_DETECTOR",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "candidate_statuses": {
            "S7-A": {
                "status": s7a["status"],
                "devtest": s7a["metrics_by_split"]["DEVTEST"],
                "loso_mean_identifiable_auroc": s7a["loso"]["mean_identifiable_auroc"],
            },
            "S7-B": "NOT_TRAINED_MISSING_FROZEN_LANGUAGE_EMBEDDING",
            "S7-C": "NOT_TRAINED_MISSING_FROZEN_LANGUAGE_AND_VISUAL_EMBEDDINGS",
        },
        "forensic_status": forensic["status"],
        "forensic_root": str(args.forensic_root.resolve()),
        "s7a_root": str(args.s7a_root.resolve()),
        "scientific_conclusion": "NO_GENERALIZABLE_STAGE_VII_DEVELOPMENT_DETECTOR",
        "new_formal_m4_authorized": False,
        "new_timing_matrix_authorized": False,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "protected_counters": COUNTERS,
        "frozen_artifacts_changed": False,
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
