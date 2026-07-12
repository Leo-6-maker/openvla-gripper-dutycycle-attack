"""Build immutable A2/B2 FIT/CAL views from the combined R9Q index.

This is an index-only view builder.  It never copies or rewrites source NPZs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OGS_SUITES = ("libero_spatial", "libero_object", "libero_goal")
ALL_SUITES = OGS_SUITES + ("libero_10",)
SCHEMA = "c2g.r9q.training_views.2026-07-13.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _validate_source(rows: list[dict[str, Any]], combined_root: Path) -> dict[str, Any]:
    index_path = combined_root / "dataset_index.jsonl"
    norm_path = combined_root / "normalization.json"
    if not index_path.is_file() or not norm_path.is_file():
        raise FileNotFoundError("combined root requires dataset_index.jsonl and normalization.json")
    if not rows:
        raise ValueError("combined index is empty")
    source_files = [
        {"path": index_path.resolve().as_posix(), "sha256": sha256_file(index_path)},
        {"path": norm_path.resolve().as_posix(), "sha256": sha256_file(norm_path)},
    ]
    seen: set[str] = set()
    for row in rows:
        parent_key = str(row.get("parent_key", ""))
        if not parent_key or parent_key in seen:
            raise ValueError(f"duplicate or empty parent_key: {parent_key!r}")
        seen.add(parent_key)
        if row.get("cohort") != "DETECTOR_TRAIN":
            raise ValueError(f"non-train cohort in combined source: {parent_key}")
        if row.get("preview_split") not in {"FIT", "CAL", "CHECK"}:
            raise ValueError(f"invalid preview_split for {parent_key}")
        if not str(row.get("task_language", "")).strip():
            raise ValueError(f"empty task_language for {parent_key}")
        npz_path = Path(str(row.get("npz_path", "")))
        if not npz_path.is_absolute() or not npz_path.is_file():
            raise FileNotFoundError(f"missing absolute NPZ for {parent_key}: {npz_path}")
        actual = sha256_file(npz_path)
        if row.get("npz_sha256") != actual:
            raise ValueError(f"NPZ hash mismatch for {parent_key}")
        source_files.append({"path": npz_path.resolve().as_posix(), "sha256": actual})
    return {
        "schema": "c2g.r9q.dataset_source_closure.2026-07-13.v1",
        "combined_root": combined_root.resolve().as_posix(),
        "source_file_count": len(source_files),
        "files": sorted(source_files, key=lambda item: item["path"]),
    }


def build_views(
    *,
    combined_root: Path,
    output_root: Path,
    mode: str,
    expected_fit: int | None = None,
    expected_cal: int | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    if mode not in {"a2", "b2"}:
        raise ValueError("mode must be a2 or b2")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    rows = read_jsonl(combined_root / "dataset_index.jsonl")
    closure = _validate_source(rows, combined_root)
    allowed = set(OGS_SUITES if mode == "a2" else ALL_SUITES)
    selected = [row for row in rows if row["suite"] in allowed and row["preview_split"] in {"FIT", "CAL"}]
    fit_rows = [dict(row, eligible_for_detector_fit=True, eligible_for_calibration=False) for row in selected if row["preview_split"] == "FIT"]
    cal_rows = [dict(row, eligible_for_detector_fit=False, eligible_for_calibration=True) for row in selected if row["preview_split"] == "CAL"]
    if expected_fit is not None and len(fit_rows) != expected_fit:
        raise ValueError(f"{mode} FIT count {len(fit_rows)} != {expected_fit}")
    if expected_cal is not None and len(cal_rows) != expected_cal:
        raise ValueError(f"{mode} CAL count {len(cal_rows)} != {expected_cal}")
    if mode == "a2" and any(row["suite"] == "libero_10" for row in fit_rows + cal_rows):
        raise AssertionError("A2 contains libero_10")
    if mode == "b2" and not any(row["suite"] == "libero_10" for row in fit_rows):
        raise AssertionError("B2 has no libero_10 FIT support")

    output_root.mkdir(parents=True)
    _write_jsonl(output_root / "fit_manifest.jsonl", fit_rows)
    _write_jsonl(output_root / "cal_manifest.jsonl", cal_rows)
    closure_path = output_root / "dataset_source_closure.json"
    _write_json(closure_path, closure)
    manifest = {
        "schema": SCHEMA,
        "mode": mode.upper(),
        "combined_root": combined_root.resolve().as_posix(),
        "expected_git_commit": expected_commit,
        "dataset_index_sha256": sha256_file(combined_root / "dataset_index.jsonl"),
        "normalization_sha256": sha256_file(combined_root / "normalization.json"),
        "source_closure_sha256": sha256_file(closure_path),
        "fit_count": len(fit_rows),
        "cal_count": len(cal_rows),
        "check_rows_excluded": sum(1 for row in rows if row["preview_split"] == "CHECK"),
        "suite_counts_fit": dict(sorted(Counter(row["suite"] for row in fit_rows).items())),
        "suite_counts_cal": dict(sorted(Counter(row["suite"] for row in cal_rows).items())),
        "fit_manifest_sha256": sha256_file(output_root / "fit_manifest.jsonl"),
        "cal_manifest_sha256": sha256_file(output_root / "cal_manifest.jsonl"),
        "class_balance_source": "FIT_ONLY",
        "check_consumption_count": 0,
    }
    manifest_path = output_root / "training_manifest.json"
    _write_json(manifest_path, manifest)
    files = [p for p in output_root.iterdir() if p.is_file()]
    sums = output_root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in sorted(files, key=lambda p: p.name)), encoding="utf-8")
    sums_sha = sha256_file(sums)
    (output_root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    manifest["sha256sums_sha256"] = sums_sha
    _write_json(manifest_path, manifest)
    # Rebuild the closure after the manifest update, then freeze the final list.
    files = [p for p in output_root.iterdir() if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}]
    sums.write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in sorted(files, key=lambda p: p.name)), encoding="utf-8")
    sums_sha = sha256_file(sums)
    (output_root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {**manifest, "sha256sums_sha256": sums_sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build R9Q A2/B2 FIT/CAL manifests")
    parser.add_argument("--combined-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=["a2", "b2"], required=True)
    parser.add_argument("--expected-fit", type=int)
    parser.add_argument("--expected-cal", type=int)
    parser.add_argument("--expected-commit")
    args = parser.parse_args(argv)
    try:
        report = build_views(
            combined_root=args.combined_root,
            output_root=args.output_root,
            mode=args.mode,
            expected_fit=args.expected_fit,
            expected_cal=args.expected_cal,
            expected_commit=args.expected_commit,
        )
    except Exception as exc:
        print(f"HOLD_C2G_R9Q_MANIFEST: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
