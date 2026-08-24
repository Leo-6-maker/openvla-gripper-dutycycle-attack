"""Materialize sealed, read-only G-REC R1A diagnostics without rebuilding geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(root: Path) -> dict[str, str]:
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())), encoding="utf-8")
    sums_sha = sha256_file(sums)
    sidecar = root / "SHA256SUMS.sha256"
    sidecar.write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "sidecar_sha256": sha256_file(sidecar)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--expected-run-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"diagnostic output exists: {output}")
    staging = output.parent / f".{output.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"diagnostic staging exists: {staging}")
    staging.mkdir(parents=True)
    verifier = Path(__file__).with_name("verify_v23_recorded_geometry_independent.py")
    comparator = Path(__file__).with_name("compare_v23_recorded_geometry_runs.py")
    results: dict[str, dict[str, object]] = {}
    try:
        for name, run_root in (("run_A", args.run_a), ("run_B", args.run_b)):
            output_path = staging / f"{name}.json"
            command = [
                sys.executable, str(verifier),
                "--run-root", str(run_root),
                "--pilot-manifest", str(args.pilot_manifest),
                "--index-root", str(args.index_root),
                "--registry-root", str(args.registry_root),
                "--libero-root", str(args.libero_root),
                "--alias-ledger", str(args.alias_ledger),
                "--expected-commit", args.expected_run_commit,
                "--output", str(output_path),
            ]
            proc = subprocess.run(command, check=False, capture_output=True, text=True)
            (staging / f"{name}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (staging / f"{name}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            if output_path.is_file():
                results[name] = json.loads(output_path.read_text(encoding="utf-8"))
            else:
                results[name] = {"status": "HOLD", "error": "verifier produced no result", "returncode": proc.returncode}

        comparison_path = staging / "ab_comparison.json"
        proc = subprocess.run([
            sys.executable, str(comparator), "--run-a", str(args.run_a), "--run-b", str(args.run_b), "--output", str(comparison_path)
        ], check=False, capture_output=True, text=True)
        (staging / "comparison.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (staging / "comparison.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else {"status": "HOLD", "error": "comparator produced no result"}
        status = "PASS" if results["run_A"].get("status") == "PASS" and results["run_B"].get("status") == "PASS" and comparison.get("status") == "PASS" else "HOLD"
        manifest = {
            "schema": "V23_G_REC_R1A_DIAGNOSTIC_BUNDLE_V1",
            "status": status,
            "diagnostic_only": True,
            "geometry_regenerated": False,
            "source_diagnostics_commit": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
            "execution_run_commit": args.expected_run_commit,
            "run_A_source": str(args.run_a.resolve()),
            "run_B_source": str(args.run_b.resolve()),
            "run_A": "run_A.json",
            "run_B": "run_B.json",
            "ab_comparison": "ab_comparison.json",
            "protected_payload_read": False,
            "model_inference": False,
            "action_replay": False,
            "teacher_labeling": False,
            "student_training": False,
            "attack": False,
        }
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal(staging)
        os.rename(staging, output)
        print(json.dumps({"status": status, "output_root": str(output)}, sort_keys=True))
        return 0 if status == "PASS" else 1
    except Exception as exc:
        (staging / "FAILURE.json").write_text(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "HOLD", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
