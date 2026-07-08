#!/usr/bin/env python3
"""Strict hygiene checks for C2f collection roots.

This complements audit_c2f_observation_collection.py with file-system parity
checks that are easy to miss in semantic audits:

- one episode_metadata.json per manifest episode;
- every step rgb_path exists;
- every rgb/*.png is referenced by exactly one step row;
- total referenced RGB rows == total step rows == total RGB png files;
- task_language is non-empty in metadata and every row;
- no attack/outcome/privileged fields appear in step_records.

Use before materialization.  In particular, if a smoke run reports 732 step rows
but 1464 RGB files, this script should FAIL with ORPHAN_RGB_FILES unless the
extra files are outside the C2f root being audited.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

FORBIDDEN_STEP_FIELDS = {
    "object_pose", "target_pose", "object_x", "object_y", "object_z",
    "target_x", "target_y", "target_z", "object_to_target_distance",
    "manual_failure_label", "attack_outcome", "vis_success", "rand_success",
    "cmd_open_success", "task_success_after_attack",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict C2f collection hygiene/RGB parity check")
    ap.add_argument("--c2f-root", required=True)
    ap.add_argument("--output-dir", default="", help="default: <c2f-root>/hygiene")
    ap.add_argument("--expected-episodes", type=int, default=0)
    ap.add_argument("--allow-primary-all-zero", action="store_true", help="allow all-zero primary labels, useful for smoke only")
    args = ap.parse_args()

    root = Path(args.c2f_root)
    out = Path(args.output_dir) if args.output_dir else root / "hygiene"
    out.mkdir(parents=True, exist_ok=True)

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    def err(code: str, detail: str, episode: str = "<collection>") -> None:
        errors.append({"code": code, "detail": detail, "episode": episode})

    def warn(code: str, detail: str, episode: str = "<collection>") -> None:
        warnings.append({"code": code, "detail": detail, "episode": episode})

    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not manifest_path.exists():
        err("MISSING_MANIFEST", "manifest.json missing")
    elif manifest.get("boundaries", {}).get("attack") != "NOT_PERFORMED":
        err("BOUNDARY_ATTACK_NOT_CLEAN", str(manifest.get("boundaries", {})))

    meta_paths = sorted((root / "episodes").glob("*/*/episode_metadata.json"))
    if args.expected_episodes and len(meta_paths) != args.expected_episodes:
        err("EPISODE_COUNT_MISMATCH", f"expected={args.expected_episodes} found={len(meta_paths)}")

    total_steps = 0
    total_rgb_files = 0
    total_referenced_rgb = 0
    total_primary = 0
    language_sources: Counter[str] = Counter()

    for meta_path in meta_paths:
        ep_dir = meta_path.parent
        ep_name = ep_dir.as_posix()
        meta = read_json(meta_path)
        if not str(meta.get("task_language", "")).strip():
            err("EMPTY_META_TASK_LANGUAGE", "metadata task_language is empty", ep_name)
        language_sources[str(meta.get("task_language_source", ""))] += 1

        step_path = ep_dir / "step_records.jsonl"
        if not step_path.exists():
            err("MISSING_STEP_RECORDS", "step_records.jsonl missing", ep_name)
            continue
        rows = read_jsonl(step_path)
        step_count = len(rows)
        total_steps += step_count

        rgb_files: Set[str] = {p.relative_to(ep_dir).as_posix() for p in sorted((ep_dir / "rgb").glob("*.png"))}
        total_rgb_files += len(rgb_files)
        referenced: List[str] = []
        duplicate_steps = [k for k, v in Counter(int(r.get("step", -1)) for r in rows).items() if v > 1]
        if duplicate_steps:
            err("DUPLICATE_STEP_ROWS", f"steps={duplicate_steps[:20]}", ep_name)

        for i, row in enumerate(rows):
            if not str(row.get("task_language", "")).strip():
                err("EMPTY_ROW_TASK_LANGUAGE", f"row_index={i} step={row.get('step')}", ep_name)
            forbidden = FORBIDDEN_STEP_FIELDS.intersection(row.keys())
            if forbidden:
                err("FORBIDDEN_STEP_FIELDS", ",".join(sorted(forbidden)), ep_name)
            feats = row.get("features_25d", [])
            if len(feats) != 25:
                err("FEATURE_LEN", f"row_index={i} step={row.get('step')} len={len(feats)}", ep_name)
            try:
                total_primary += int(row.get("teacher_primary_attackable", 0))
            except Exception:
                err("BAD_PRIMARY_LABEL", f"row_index={i} value={row.get('teacher_primary_attackable')}", ep_name)
            rel = str(row.get("rgb_path", ""))
            referenced.append(rel)
            total_referenced_rgb += 1
            if rel not in rgb_files:
                err("RGB_PATH_NOT_FOUND", f"row_index={i} step={row.get('step')} rgb_path={rel}", ep_name)

        ref_counts = Counter(referenced)
        dup_refs = [k for k, v in ref_counts.items() if v > 1]
        if dup_refs:
            err("DUPLICATE_RGB_REFERENCES", f"examples={dup_refs[:20]}", ep_name)
        orphans = sorted(rgb_files - set(referenced))
        if orphans:
            err("ORPHAN_RGB_FILES", f"count={len(orphans)} examples={orphans[:20]}", ep_name)
        if len(rgb_files) != step_count:
            err("RGB_FILE_STEP_COUNT_MISMATCH", f"rgb_files={len(rgb_files)} step_rows={step_count}", ep_name)

        episode_rows.append({
            "episode_dir": ep_name,
            "suite": meta.get("suite"),
            "task_index": meta.get("task_index"),
            "parent_key": meta.get("parent_key"),
            "task_language_source": meta.get("task_language_source", ""),
            "n_steps": step_count,
            "n_rgb_files": len(rgb_files),
            "n_orphan_rgb": len(rgb_files - set(referenced)),
            "primary_steps": sum(int(r.get("teacher_primary_attackable", 0)) for r in rows),
            "clean_success": meta.get("clean_success"),
            "clean_success_observed": meta.get("clean_success_observed"),
        })

    if total_rgb_files != total_steps:
        err("ROOT_RGB_STEP_COUNT_MISMATCH", f"rgb_files={total_rgb_files} step_rows={total_steps}")
    if total_referenced_rgb != total_steps:
        err("ROOT_REFERENCED_RGB_STEP_COUNT_MISMATCH", f"referenced_rgb={total_referenced_rgb} step_rows={total_steps}")
    if total_primary == 0 and not args.allow_primary_all_zero:
        err("PRIMARY_ALL_ZERO", "teacher_primary_attackable has zero positive steps")
    elif total_primary == 0:
        warn("PRIMARY_ALL_ZERO_ALLOWED", "all-zero primary labels allowed by flag; smoke only, not trainable")

    report = {
        "gate": "C2F_COLLECTION_HYGIENE",
        "status": "PASS" if not errors else "FAIL",
        "c2f_root": str(root),
        "n_episodes": len(meta_paths),
        "n_steps": total_steps,
        "n_rgb_files": total_rgb_files,
        "n_referenced_rgb": total_referenced_rgb,
        "primary_steps": total_primary,
        "task_language_sources": dict(language_sources),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "boundaries": {"attack": "NOT_PERFORMED", "d7b2_outcome_read": False},
    }
    write_json(out / "c2f_collection_hygiene_report.json", report)
    write_csv(out / "episode_hygiene.csv", episode_rows)
    write_csv(out / "errors.csv", errors)
    write_csv(out / "warnings.csv", warnings)
    print(json.dumps({"status": report["status"], "n_steps": total_steps, "n_rgb_files": total_rgb_files, "n_errors": len(errors), "output_dir": str(out)}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
