#!/usr/bin/env python3
"""Release clean collector with canonical 25D and full model provenance.

The strict four-suite collector is executed one suite per subprocess. This prevents
multiple 7B OpenVLA checkpoints from accumulating in one Python/CUDA process while
preserving a single closed collection root and combined artifact manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stageb.verify_c2g_suite_model_map_strict import verify

REPO = Path(__file__).resolve().parents[2]
STRICT_COLLECTOR = REPO / "scripts" / "stageb" / "collect_c2g_clean_window_rollouts_strict.py"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
FORBIDDEN_CLEAN_KEY_TOKENS = ("attack_outcome", "post_intervention", "counterfactual")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_release_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite-model-map", type=Path, required=True)
    parser.add_argument("--suite-model-report", type=Path, required=True)
    parser.add_argument("--goal-model-manifest", type=Path, required=True)
    parser.add_argument("--model-verification-report", type=Path, required=True)
    return parser.parse_known_args(list(argv))


def parse_forwarded_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--suite", default="")
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_known_args(list(argv))[0]


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("episodes", value) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("manifest must contain a list of objects")
    return [dict(row) for row in rows]


def partition_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_episodes: int = 0,
    suite_filter: str = "",
) -> dict[str, list[dict[str, Any]]]:
    selected = [dict(row) for row in rows]
    if suite_filter:
        if suite_filter not in SUITES:
            raise ValueError(f"unknown suite filter: {suite_filter}")
        selected = [row for row in selected if str(row.get("suite")) == suite_filter]
    if max_episodes > 0:
        selected = selected[:max_episodes]
    partitions = {suite: [] for suite in SUITES}
    for row in selected:
        suite = str(row.get("suite", ""))
        if suite not in partitions:
            raise ValueError(f"manifest row has invalid suite: {suite!r}")
        partitions[suite].append(row)
    return {suite: values for suite, values in partitions.items() if values}


def _remove_option(argv: Sequence[str], name: str, *, takes_value: bool = True) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == name:
            index += 2 if takes_value else 1
            continue
        if takes_value and item.startswith(name + "="):
            index += 1
            continue
        result.append(item)
        index += 1
    return result


def suite_command(forwarded: Sequence[str], manifest: Path, suite: str) -> list[str]:
    cleaned = list(forwarded)
    for option in ("--manifest", "--suite", "--max-episodes"):
        cleaned = _remove_option(cleaned, option)
    return [
        sys.executable,
        str(STRICT_COLLECTOR),
        *cleaned,
        "--manifest", str(manifest),
        "--suite", suite,
        "--max-episodes", "0",
    ]


def forbidden_clean_keys(value: Any, prefix: str = "") -> list[str]:
    """Return forbidden mapping-key paths without inspecting documentation values."""

    problems: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if any(token in key_text.lower() for token in FORBIDDEN_CLEAN_KEY_TOKENS):
                problems.append(path)
            problems.extend(forbidden_clean_keys(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            problems.extend(forbidden_clean_keys(child, f"{prefix}[{index}]"))
    return sorted(set(problems))


def verify_internal_suite_paths(model_map_path: Path) -> None:
    from scripts.stageb.c2f_libero_openvla_adapter import SUITE_MODELS

    value = json.loads(model_map_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("suite model map must be a JSON object")
    mismatches = {}
    for suite, internal in SUITE_MODELS.items():
        if suite not in value:
            continue
        frozen = Path(str(value[suite])).resolve()
        actual = Path(str(internal)).resolve()
        if frozen != actual:
            mismatches[suite] = {"frozen": str(frozen), "collector": str(actual)}
    if mismatches:
        raise ValueError(f"collector SUITE_MODELS differs from frozen map: {mismatches}")


def rebuild_combined_collection_report(
    output_root: Path,
    *,
    expected_git_commit: str,
    suite_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata_paths = sorted(output_root.rglob("episode_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"no episode metadata found under {output_root}")
    results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise ValueError(f"invalid episode metadata: {metadata_path}")
        steps_path = metadata_path.with_name("step_records.jsonl")
        if not steps_path.is_file() or steps_path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing/nonempty step records: {steps_path}")
        if str(metadata.get("git_commit", "")) != expected_git_commit:
            raise ValueError(f"episode commit mismatch: {metadata_path}")
        if str(metadata.get("condition", "")) != "CLEAN":
            raise ValueError(f"non-CLEAN episode in clean collection: {metadata_path}")
        bad_keys = forbidden_clean_keys(metadata)
        if bad_keys:
            raise ValueError(
                f"forbidden outcome keys in clean metadata {metadata_path}: {bad_keys}"
            )
        results.append({
            "parent_key": metadata.get("parent_key"),
            "suite": metadata.get("suite"),
            "task_index": metadata.get("task_index"),
            "state_id": metadata.get("state_id"),
            "n_steps": metadata.get("n_steps"),
            "status": "PASS",
        })
        for artifact in (metadata_path, steps_path):
            artifacts.append({
                "path": artifact.relative_to(output_root).as_posix(),
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            })
    if len({str(row["parent_key"]) for row in results}) != len(results):
        raise ValueError("duplicate parent_key in combined collection")
    artifacts = sorted(artifacts, key=lambda row: row["path"])
    manifest_path = output_root / "c2g_clean_collection_input_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in artifacts),
        encoding="utf-8",
    )
    report = {
        "gate": "C2G_CLEAN_WINDOW_COLLECTION",
        "status": "PASS_CLEAN_COLLECTION",
        "execution_mode": "SUITE_ISOLATED_SUBPROCESSES",
        "episode_count": len(results),
        "results": results,
        "suite_runs": list(suite_runs),
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "openvla_clean_inference_runs": len(results),
        "libero_clean_rollouts": len(results),
        "attacks_launched": 0,
        "attack_outcomes_read": False,
        "git_commit": expected_git_commit,
        "git_clean": True,
    }
    (output_root / "c2g_clean_collection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args, forwarded = parse_release_args(raw)
    forwarded_args = parse_forwarded_args(forwarded)
    verification = verify(
        args.suite_model_map.resolve(),
        args.suite_model_report.resolve(),
        args.goal_model_manifest.resolve(),
    )
    verify_internal_suite_paths(args.suite_model_map.resolve())
    args.model_verification_report.parent.mkdir(parents=True, exist_ok=True)
    args.model_verification_report.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = read_manifest(forwarded_args.manifest.resolve())
    partitions = partition_manifest_rows(
        rows,
        max_episodes=forwarded_args.max_episodes,
        suite_filter=forwarded_args.suite,
    )
    if not partitions:
        raise RuntimeError("no clean episodes selected")
    output_root = forwarded_args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    suite_runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="c2g_clean_suite_manifests_") as td:
        temporary = Path(td)
        for suite in SUITES:
            suite_rows = partitions.get(suite, [])
            if not suite_rows:
                continue
            manifest = temporary / f"{suite}.jsonl"
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in suite_rows),
                encoding="utf-8",
            )
            command = suite_command(forwarded, manifest, suite)
            print(f"[suite={suite}] " + " ".join(command), flush=True)
            completed = subprocess.run(command, cwd=REPO)
            suite_runs.append({
                "suite": suite,
                "episode_count": len(suite_rows),
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "HOLD",
            })
            if completed.returncode != 0:
                raise RuntimeError(f"clean collection failed for {suite}")

    report = rebuild_combined_collection_report(
        output_root,
        expected_git_commit=forwarded_args.expected_git_commit,
        suite_runs=suite_runs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
