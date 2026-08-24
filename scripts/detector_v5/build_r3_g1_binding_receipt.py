"""Create a metadata-only, sealed binding receipt for the frozen G1 splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace


SPLIT_FILES = {
    "episode_train": "EPISODE_TRAIN_MANIFEST.json",
    "episode_validation": "EPISODE_VAL_MANIFEST.json",
    "episode_test": "EPISODE_TEST_MANIFEST.json",
    "task_train": "TASK_TRAIN_MANIFEST.json",
    "task_validation": "TASK_VAL_MANIFEST.json",
    "task_test": "TASK_TEST_MANIFEST.json",
}
FORBIDDEN = {"cal", "check", "g10", "t2r-d", "protected", "attack"}


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _regular_repo_file(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe repository reference")
    current = ROOT
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("symlinked repository reference")
    resolved = current.resolve(strict=True)
    if ROOT.resolve() not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise ValueError("repository reference is not a regular file")
    return resolved


def _seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def build(*, g1_root: Path, output_root: Path, g2_root: Path | None = None) -> dict[str, Any]:
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise ValueError("binding receipt requires a clean checkout")
    if not g1_root.is_absolute() or g1_root.is_symlink() or any(part.casefold() in FORBIDDEN for part in g1_root.parts):
        raise ValueError("unsafe G1 root")
    g1_root = g1_root.resolve(strict=True)
    g1_seal = verify_seal(g1_root)
    audit_path = g1_root / "G1_SPLIT_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_SPLIT_CLOSURE_WITH_HEAD_COVERAGE_FLAGS" or audit.get("checks", {}).get("protected_reads") != 0:
        raise ValueError("G1 is not a passing FIT-only split audit")
    split_files: dict[str, dict[str, Any]] = {}
    for split, filename in SPLIT_FILES.items():
        path = g1_root / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing G1 split file: {filename}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"empty G1 split file: {filename}")
        identities = [row.get("episode_id") for row in rows if isinstance(row, dict)]
        if len(identities) != len(rows) or any(not isinstance(identity, str) for identity in identities) or len(set(identities)) != len(identities):
            raise ValueError(f"G1 split identity closure failed: {filename}")
        identity_sha = hashlib.sha256(("\n".join(identities) + "\n").encode("utf-8")).hexdigest()
        split_files[split] = {"path": filename, "sha256": sha256_file(path), "identity_count": len(identities), "identity_sha256": identity_sha}
    normalization = g1_root / "NORMALIZATION.json"
    if not normalization.is_file() or normalization.is_symlink():
        raise ValueError("missing G1 normalization")
    source_trainer = _regular_repo_file("scripts/detector_v5/run_r3_heldout_development.py")
    full_loader = _regular_repo_file("scripts/detector_v5/run_r3_full670_student_development.py")
    feature_binding = _regular_repo_file("configs/R3_SC5_FEATURE_BINDING_V1.json")
    audit_source = audit.get("builder_source", {})
    payload: dict[str, Any] = {
        "schema": "V5_R3_G1_BINDING_RECEIPT_V1",
        "status": "PASS_G1_REBOUND_READ_ONLY",
        "code_snapshot": {"commit": _git("rev-parse", "HEAD"), "tree": _git("rev-parse", "HEAD^{tree}")},
        "g1": {"root": str(g1_root), "seal_sha256sums_sha256": g1_seal["sha256sums_sha256"], "audit_sha256": sha256_file(audit_path)},
        "split_files": split_files,
        "normalization": {"path": "NORMALIZATION.json", "sha256": sha256_file(normalization), "source": "train_only"},
        "eligible_heads": audit.get("heads", {}),
        "source_bindings": {
            "trainer": {"path": "scripts/detector_v5/run_r3_heldout_development.py", "sha256": sha256_file(source_trainer)},
            "full_loader": {"path": "scripts/detector_v5/run_r3_full670_student_development.py", "sha256": sha256_file(full_loader)},
            "feature_binding": {"path": "configs/R3_SC5_FEATURE_BINDING_V1.json", "sha256": sha256_file(feature_binding)},
            "g1_split_builder_commit": audit_source.get("commit"),
            "g1_split_builder_tree": audit_source.get("tree"),
        },
        "g2": None,
        "permissions": {"metadata_only": True, "teacher_payload_read": False, "student_training": False, "development_inference": False, "test_payload_read": False, "protected_reads": 0, "rollout": False, "attack": False},
    }
    if g2_root is not None:
        if g2_root.is_symlink() or any(part.casefold() in FORBIDDEN for part in g2_root.parts):
            raise ValueError("unsafe G2 root")
        g2_root = g2_root.resolve(strict=True)
        g2_seal = verify_seal(g2_root)
        payload["g2"] = {"root": str(g2_root), "seal_sha256sums_sha256": g2_seal["sha256sums_sha256"], "transition_sha256": sha256_file(g2_root / "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V2.json")}
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink() or output_root.parent.resolve(strict=True) != g1_root.parent:
        raise ValueError("output must be a new G1 sibling")
    if any(part.casefold() in FORBIDDEN for part in output_root.parts):
        raise ValueError("output is under a forbidden path")
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "G1_BINDING_RECEIPT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = _seal(staging)
        rename_noreplace(staging, output_root)
    except Exception as exc:
        (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_G1_BINDING_FAILURE_V1", "error": repr(exc)}, sort_keys=True) + "\n", encoding="utf-8")
        _seal(staging)
        raise
    payload["sha256sums_sha256"] = digest
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--g2-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(g1_root=args.g1_root, g2_root=args.g2_root, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
