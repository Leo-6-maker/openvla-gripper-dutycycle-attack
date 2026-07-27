"""Run the recorded-telemetry G-REC A/B evidence closure.

This is a FIT-only geometry job.  It never runs a policy, rollout, action
replay, Teacher, Student, or attack.  The two materializations are checked by
an independent verifier before the final evidence root is published.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


EPISODES = 40
STEPS = 9422
RELATION_ROWS = 11880
ALIAS_ROWS = 217
UNKNOWN_ROWS = 0
SCHEMA = "V23_G_REC_RECORDED_GEOMETRY_EVIDENCE_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_snapshot(repo: Path, expected: str) -> dict[str, str]:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True).strip()
    if head != expected or status:
        raise RuntimeError(f"source worktree is not the expected clean snapshot: {head!r} {status!r}")
    return {"commit": head, "tree": tree}


def run_summary(root: Path) -> dict[str, int]:
    episode_count = step_count = relation_rows = alias_rows = unknown_rows = 0
    episodes = root / "episodes"
    for episode_dir in sorted(episodes.iterdir()):
        if not episode_dir.is_dir():
            continue
        manifest = json.loads((episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (episode_dir / "geometry_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        episode_count += 1
        step_count += len(rows)
        for row in rows:
            for relation in row.get("relations", []):
                relation_rows += 1
                unknown_rows += int(not relation.get("known", False))
                alias_rows += int(relation.get("object", {}).get("registry_identity_status") == "INIT_GEOM_ALIAS_TO_INDEX_BODY")
        if int(manifest.get("step_count", len(rows))) != len(rows):
            raise RuntimeError(f"episode step count mismatch: {episode_dir}")
    return {
        "episode_count": episode_count,
        "step_count": step_count,
        "relation_rows": relation_rows,
        "alias_rows": alias_rows,
        "unknown_rows": unknown_rows,
    }


def payload_files(root: Path) -> dict[str, str]:
    excluded = {"MANIFEST.json", "SHA256SUMS", "SHA256SUMS.sha256"}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name not in excluded
    }


def write_seal(root: Path) -> dict[str, str]:
    files = payload_files(root)
    sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items()))
    sums_path = root / "SHA256SUMS"
    sidecar_path = root / "SHA256SUMS.sha256"
    sums_path.write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(sums_path)
    sidecar_path.write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "sha256sums_file_sha256": sums_sha, "sidecar_sha256": sha256_file(sidecar_path)}


def publish_noreplace(staging: Path, output: Path) -> None:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("renameat2 RENAME_NOREPLACE is not implemented for this architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = getattr(libc, "syscall", None)
    if syscall is None:
        raise RuntimeError("renameat2 syscall unavailable; refusing non-atomic publish")
    # Linux x86_64: __NR_renameat2=316, RENAME_NOREPLACE=1.
    result = syscall(
        ctypes.c_long(316),
        ctypes.c_int(-100), ctypes.c_char_p(os.fsencode(str(staging))),
        ctypes.c_int(-100), ctypes.c_char_p(os.fsencode(str(output))),
        ctypes.c_uint(1),
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(output)
        raise OSError(error, os.strerror(error), str(output))


def command_for(script: Path, args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable, str(script),
        "--pilot-manifest", str(args.pilot_manifest),
        "--index-root", str(args.index_root),
        "--registry-root", str(args.registry_root),
        "--libero-root", str(args.libero_root),
        "--alias-ledger", str(args.alias_ledger),
        "--output-root", str(output),
        "--code-snapshot-commit", args.code_snapshot_commit,
    ]


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-snapshot-commit", required=True)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"output root already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"staging root already exists: {staging}")
    staging.mkdir()

    repo = Path(__file__).resolve().parents[2]
    snapshot = repo_snapshot(repo, args.code_snapshot_commit)
    materializer = Path(__file__).with_name("materialize_v23_recorded_geometry.py")
    verifier = Path(__file__).with_name("verify_v23_recorded_geometry_independent.py")
    run_dirs = {name: staging / name for name in ("run_A", "run_B")}
    reviews = staging / "independent_review"
    comparison_dir = staging / "comparison"
    reviews.mkdir()
    comparison_dir.mkdir()

    try:
        commands: dict[str, list[str]] = {}
        for name, run_dir in run_dirs.items():
            command = command_for(materializer, args, run_dir)
            commands[name] = command
            run_logged(command, staging / f"{name}.stdout.log", staging / f"{name}.stderr.log")

        for name, run_dir in run_dirs.items():
            review_path = reviews / f"{name}.json"
            command = [
                sys.executable, str(verifier),
                "--run-root", str(run_dir),
                "--pilot-manifest", str(args.pilot_manifest),
                "--index-root", str(args.index_root),
                "--registry-root", str(args.registry_root),
                "--libero-root", str(args.libero_root),
                "--alias-ledger", str(args.alias_ledger),
                "--expected-commit", args.code_snapshot_commit,
                "--output", str(review_path),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            (staging / f"{name}.independent.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
            (staging / f"{name}.independent.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if completed.returncode != 0 or review.get("status") != "PASS":
                raise RuntimeError(f"independent review failed for {name}: {review}")

        summaries = {name: run_summary(path) for name, path in run_dirs.items()}
        if summaries["run_A"] != summaries["run_B"]:
            raise RuntimeError(f"A/B summary mismatch: {summaries}")
        if summaries["run_A"] != {
            "episode_count": EPISODES,
            "step_count": STEPS,
            "relation_rows": RELATION_ROWS,
            "alias_rows": ALIAS_ROWS,
            "unknown_rows": UNKNOWN_ROWS,
        }:
            raise RuntimeError(f"closure mismatch: {summaries['run_A']}")
        a_payload = payload_files(run_dirs["run_A"])
        b_payload = payload_files(run_dirs["run_B"])
        if a_payload != b_payload:
            only_a = sorted(set(a_payload) - set(b_payload))
            only_b = sorted(set(b_payload) - set(a_payload))
            changed = sorted(k for k in set(a_payload) & set(b_payload) if a_payload[k] != b_payload[k])
            raise RuntimeError(f"A/B payload mismatch: only_a={only_a} only_b={only_b} changed={changed}")
        comparison = {
            "schema": "V23_G_REC_COMPARISON_V1",
            "status": "PASS",
            "run_A_summary": summaries["run_A"],
            "run_B_summary": summaries["run_B"],
            "canonical_payload_equal": True,
            "canonical_payload_sha256": hashlib.sha256(json.dumps(a_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "independent_review_status": {name: json.loads((reviews / f"{name}.json").read_text())["status"] for name in run_dirs},
            "protected_payload_read": False,
            "action_replay": False,
            "model_inference": False,
            "teacher_labeling": False,
        }
        (comparison_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema": SCHEMA,
            "status": "PASS",
            "source_snapshot_commit": snapshot["commit"],
            "source_snapshot_tree": snapshot["tree"],
            "materializer_source_sha256": sha256_file(materializer),
            "independent_verifier_source_sha256": sha256_file(verifier),
            "pilot_manifest_path": str(args.pilot_manifest.resolve()),
            "index_root_path": str(args.index_root.resolve()),
            "registry_root_path": str(args.registry_root.resolve()),
            "libero_root": str(args.libero_root.resolve()),
            "alias_ledger_path": str(args.alias_ledger.resolve()),
            "run_A": "run_A",
            "run_B": "run_B",
            "comparison": "comparison/comparison.json",
            "independent_reviews": ["independent_review/run_A.json", "independent_review/run_B.json"],
            "summary": summaries["run_A"],
            "canonical_payload_equal": True,
            "protected_payload_read": False,
            "action_replay": False,
            "model_inference": False,
            "teacher_labeling": False,
            "consumer_eligible": False,
            "execution_commands": commands,
        }
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal = write_seal(staging)
        publish_noreplace(staging, output)
        print(json.dumps({"status": "PASS", "output_root": str(output), **seal, "summary": summaries["run_A"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "status": "HOLD",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_snapshot": snapshot,
            "stdout_logs": sorted(p.name for p in staging.glob("*.stdout.log")),
            "stderr_logs": sorted(p.name for p in staging.glob("*.stderr.log")),
        }
        (staging / "FAILURE.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
