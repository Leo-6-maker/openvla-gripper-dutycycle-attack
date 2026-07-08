#!/usr/bin/env python3
"""Audit C2f observation-rich CLEAN collection artifacts.

This is a lightweight post-collection gate for smoke3 and pilot200 runs.  It
checks the schema written by collect_c2f_observation_clean_rollouts.py without
running LIBERO/OpenVLA and without reading D7 attack outcomes.

Intended use
------------
Smoke3:
  python scripts/stageb/audit_c2f_observation_collection.py \
    --c2f-root <l10_smoke3_obs_clean_ROOT> \
    --expected-episodes 3 \
    --mode smoke3 \
    --output-dir <ROOT>/audit

Pilot200:
  python scripts/stageb/audit_c2f_observation_collection.py \
    --c2f-root <l10_pilot200_obs_clean_ROOT> \
    --expected-episodes 200 \
    --mode pilot200 \
    --output-dir <ROOT>/audit \
    --strict-primary-nondegenerate

The audit is intentionally conservative on schema/RGB/feature validity and
configurable on teacher-label degeneracy because a 3-episode smoke can
legitimately miss primary events if all three episodes fail early or abstain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - the script will report a clear error at runtime.
    Image = None

SCHEMA_COLLECTION = "C2F_OBS_LANG_CLEAN_COLLECTION_V1"
SCHEMA_EPISODE = "C2F_OBS_LANG_CLEAN_EPISODE_V1"
EVENT_ROLES = {
    "primary_attackable",
    "auxiliary_manipulation",
    "distractor_or_setup",
    "unsupported_or_abstain",
}
FORBIDDEN_STEP_FIELDS = {
    "object_pose",
    "target_pose",
    "object_x",
    "object_y",
    "object_z",
    "target_x",
    "target_y",
    "target_z",
    "object_to_target_distance",
    "manual_failure_label",
    "attack_outcome",
    "task_success_after_attack",
    "vis_success",
    "rand_success",
    "cmd_open_success",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            raise ValueError(f"Invalid JSONL at {path}:{lineno}: {e}") from e
    return rows


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def image_stats(path: Path) -> Dict[str, Any]:
    if Image is None:
        raise RuntimeError("Pillow is required for RGB audit")
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return {
        "shape": list(arr.shape),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "blank": bool(arr.size == 0 or arr.max() < 5 or arr.std() < 1e-6),
    }


def audit_episode(meta_path: Path, sample_images: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    ep_dir = meta_path.parent
    meta = read_json(meta_path)
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def err(code: str, detail: str) -> None:
        errors.append({"episode": str(ep_dir), "code": code, "detail": detail})

    def warn(code: str, detail: str) -> None:
        warnings.append({"episode": str(ep_dir), "code": code, "detail": detail})

    if meta.get("schema") != SCHEMA_EPISODE:
        err("EPISODE_SCHEMA_MISMATCH", f"schema={meta.get('schema')!r}")
    if meta.get("condition") != "CLEAN":
        err("NON_CLEAN_EPISODE", f"condition={meta.get('condition')!r}")

    step_path = ep_dir / "step_records.jsonl"
    if not step_path.exists():
        err("MISSING_STEP_RECORDS", "step_records.jsonl missing")
        return ({"episode_dir": str(ep_dir), "n_steps": 0, "n_rgb": 0}, errors, warnings)

    rows = read_jsonl(step_path)
    if not rows:
        err("EMPTY_STEP_RECORDS", "no step rows")

    role_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    primary = hazard = release = 0
    nan_feature_rows = 0
    inf_feature_rows = 0
    zero_feature_rows = 0
    missing_rgb = 0
    blank_rgb = 0
    rgb_means: List[float] = []
    rgb_stds: List[float] = []
    feature_means: List[float] = []
    forbidden_hits: Counter[str] = Counter()

    # Sample first/middle/last images plus a few evenly spaced rows.
    img_indices = set()
    if rows:
        base = [0, len(rows) // 2, len(rows) - 1]
        img_indices.update(i for i in base if 0 <= i < len(rows))
        if sample_images > len(img_indices):
            for i in np.linspace(0, len(rows) - 1, num=min(sample_images, len(rows)), dtype=int):
                img_indices.add(int(i))

    seen_steps = set()
    for i, row in enumerate(rows):
        step = row.get("step", i)
        if step in seen_steps:
            err("DUPLICATE_STEP", f"step={step}")
        seen_steps.add(step)

        forbidden = FORBIDDEN_STEP_FIELDS.intersection(row.keys())
        for f in forbidden:
            forbidden_hits[f] += 1

        feats = row.get("features_25d", [])
        if len(feats) != 25:
            err("FEATURE_LENGTH", f"step={step} len={len(feats)}")
        farr = np.asarray([safe_float(x) for x in feats], dtype=np.float64)
        if farr.size == 25:
            if np.isnan(farr).any():
                nan_feature_rows += 1
            if np.isinf(farr).any():
                inf_feature_rows += 1
            if np.all(np.nan_to_num(farr, nan=0.0, posinf=0.0, neginf=0.0) == 0.0):
                zero_feature_rows += 1
            feature_means.append(float(np.nanmean(farr)))

        role = str(row.get("teacher_event_role", ""))
        if role not in EVENT_ROLES:
            err("INVALID_EVENT_ROLE", f"step={step} role={role!r}")
        role_counts[role] += 1
        phase_counts[str(row.get("teacher_phase", ""))] += 1

        for key in ["teacher_hazard", "teacher_primary_attackable", "teacher_release_safe"]:
            if int(row.get(key, -1)) not in (0, 1):
                err("INVALID_BINARY_LABEL", f"step={step} {key}={row.get(key)!r}")
        primary += int(row.get("teacher_primary_attackable", 0))
        hazard += int(row.get("teacher_hazard", 0))
        release += int(row.get("teacher_release_safe", 0))

        if not str(row.get("task_language", "")).strip():
            err("EMPTY_TASK_LANGUAGE", f"step={step}")

        rel_rgb = str(row.get("rgb_path", ""))
        rgb_path = ep_dir / rel_rgb
        if not rel_rgb or not rgb_path.exists():
            missing_rgb += 1
            if i in img_indices:
                err("MISSING_RGB", f"step={step} rgb_path={rel_rgb!r}")
        elif i in img_indices:
            try:
                st = image_stats(rgb_path)
                rgb_means.append(st["mean"])
                rgb_stds.append(st["std"])
                if st["blank"]:
                    blank_rgb += 1
                    err("BLANK_RGB", f"step={step} rgb_path={rel_rgb!r} stats={st}")
            except Exception as e:
                err("RGB_READ_ERROR", f"step={step} rgb_path={rel_rgb!r}: {e}")

    if forbidden_hits:
        err("FORBIDDEN_STUDENT_FIELD", json.dumps(dict(forbidden_hits), sort_keys=True))
    if nan_feature_rows:
        err("NAN_FEATURE_ROWS", f"rows={nan_feature_rows}")
    if inf_feature_rows:
        err("INF_FEATURE_ROWS", f"rows={inf_feature_rows}")
    if zero_feature_rows:
        warn("ZERO_FEATURE_ROWS", f"rows={zero_feature_rows}")
    if missing_rgb:
        err("MISSING_RGB_ROWS", f"rows={missing_rgb}")

    n = len(rows)
    summary = {
        "episode_dir": str(ep_dir),
        "suite": meta.get("suite"),
        "task_index": meta.get("task_index"),
        "task_name": meta.get("task_name", ""),
        "parent_key": meta.get("parent_key"),
        "clean_success": bool(meta.get("clean_success", False)),
        "n_steps": n,
        "n_rgb_rows": n - missing_rgb,
        "missing_rgb_rows": missing_rgb,
        "blank_rgb_sampled": blank_rgb,
        "zero_feature_rows": zero_feature_rows,
        "nan_feature_rows": nan_feature_rows,
        "inf_feature_rows": inf_feature_rows,
        "primary_attackable_steps": primary,
        "hazard_steps": hazard,
        "release_safe_steps": release,
        "primary_rate": primary / max(1, n),
        "hazard_rate": hazard / max(1, n),
        "release_rate": release / max(1, n),
        "event_role_counts": dict(role_counts),
        "teacher_phase_counts": dict(phase_counts),
        "sampled_rgb_mean_min": min(rgb_means) if rgb_means else None,
        "sampled_rgb_mean_max": max(rgb_means) if rgb_means else None,
        "sampled_rgb_std_min": min(rgb_stds) if rgb_stds else None,
        "sampled_rgb_std_max": max(rgb_stds) if rgb_stds else None,
        "feature_mean_mean": statistics.mean(feature_means) if feature_means else None,
    }
    return summary, errors, warnings


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def render_markdown(report: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# C2f Observation Collection Audit — {report['mode']}\n")
    lines.append(f"Status: **{report['status']}**\n")
    lines.append(f"Root: `{report['c2f_root']}`\n")
    lines.append("\n## Summary\n")
    for k in [
        "n_episodes", "n_steps", "n_rgb_rows", "n_errors", "n_warnings",
        "primary_attackable_steps", "hazard_steps", "release_safe_steps",
        "primary_rate", "hazard_rate", "release_rate",
    ]:
        lines.append(f"- `{k}`: {report.get(k)}")
    lines.append("\n## Event roles\n")
    for k, v in sorted(report.get("event_role_counts", {}).items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("\n## Teacher phases\n")
    for k, v in sorted(report.get("teacher_phase_counts", {}).items()):
        lines.append(f"- `{k}`: {v}")
    if report.get("errors"):
        lines.append("\n## Errors\n")
        for e in report["errors"][:50]:
            lines.append(f"- `{e['code']}`: {e['detail']} ({e.get('episode','')})")
    if report.get("warnings"):
        lines.append("\n## Warnings\n")
        for w in report["warnings"][:50]:
            lines.append(f"- `{w['code']}`: {w['detail']} ({w.get('episode','')})")
    lines.append("\n## Boundaries\n")
    for k, v in sorted(report.get("boundaries", {}).items()):
        lines.append(f"- `{k}`: {v}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit C2f observation-rich CLEAN collection artifacts")
    ap.add_argument("--c2f-root", required=True)
    ap.add_argument("--output-dir", default="", help="default: <c2f-root>/audit")
    ap.add_argument("--expected-episodes", type=int, default=0)
    ap.add_argument("--mode", choices=["smoke3", "pilot200", "generic"], default="generic")
    ap.add_argument("--sample-images", type=int, default=5)
    ap.add_argument("--strict-primary-nondegenerate", action="store_true", help="fail if all primary labels are 0 or all steps are primary")
    ap.add_argument("--max-primary-rate", type=float, default=0.50)
    ap.add_argument("--max-unsupported-rate", type=float, default=0.90)
    args = ap.parse_args()

    root = Path(args.c2f_root)
    out = Path(args.output_dir) if args.output_dir else root / "audit"
    out.mkdir(parents=True, exist_ok=True)

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def err(code: str, detail: str) -> None:
        errors.append({"episode": "<collection>", "code": code, "detail": detail})

    def warn(code: str, detail: str) -> None:
        warnings.append({"episode": "<collection>", "code": code, "detail": detail})

    manifest_path = root / "manifest.json"
    manifest: Dict[str, Any] = {}
    if not manifest_path.exists():
        err("MISSING_MANIFEST", "manifest.json missing")
    else:
        manifest = read_json(manifest_path)
        if manifest.get("schema") != SCHEMA_COLLECTION:
            err("COLLECTION_SCHEMA_MISMATCH", f"schema={manifest.get('schema')!r}")
        b = manifest.get("boundaries", {})
        if b.get("condition") != "CLEAN_ONLY" or b.get("attack") != "NOT_PERFORMED" or b.get("d7b2_outcome_read") is not False:
            err("BOUNDARY_VIOLATION", json.dumps(b, sort_keys=True))

    if not (root / "SHA256SUMS").exists():
        warn("MISSING_SHA256SUMS", "SHA256SUMS missing")
    if not (root / "SHA256SUMS.sha256").exists():
        warn("MISSING_SHA256SUMS_SHA256", "SHA256SUMS.sha256 missing")

    meta_paths = sorted((root / "episodes").glob("*/*/episode_metadata.json"))
    if args.expected_episodes and len(meta_paths) != args.expected_episodes:
        err("EPISODE_COUNT_MISMATCH", f"expected={args.expected_episodes} found={len(meta_paths)}")
    if not meta_paths:
        err("NO_EPISODES", f"no episode_metadata.json files under {root / 'episodes'}")

    episode_summaries: List[Dict[str, Any]] = []
    for mp in meta_paths:
        s, es, ws = audit_episode(mp, args.sample_images)
        episode_summaries.append(s)
        errors.extend(es)
        warnings.extend(ws)

    role_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    n_steps = n_rgb = primary = hazard = release = 0
    for s in episode_summaries:
        n_steps += int(s.get("n_steps", 0))
        n_rgb += int(s.get("n_rgb_rows", 0))
        primary += int(s.get("primary_attackable_steps", 0))
        hazard += int(s.get("hazard_steps", 0))
        release += int(s.get("release_safe_steps", 0))
        role_counts.update(s.get("event_role_counts", {}))
        phase_counts.update(s.get("teacher_phase_counts", {}))

    primary_rate = primary / max(1, n_steps)
    unsupported_rate = role_counts.get("unsupported_or_abstain", 0) / max(1, n_steps)
    if args.strict_primary_nondegenerate:
        if primary == 0:
            err("PRIMARY_ALL_ZERO", "teacher_primary_attackable has zero positive steps")
        if primary_rate > args.max_primary_rate:
            err("PRIMARY_RATE_TOO_HIGH", f"primary_rate={primary_rate:.4f} > max={args.max_primary_rate:.4f}")
    else:
        if primary == 0:
            warn("PRIMARY_ALL_ZERO", "teacher_primary_attackable has zero positive steps; acceptable for smoke only if episodes do not reach target carry")
        if primary_rate > args.max_primary_rate:
            warn("PRIMARY_RATE_HIGH", f"primary_rate={primary_rate:.4f} > max={args.max_primary_rate:.4f}")
    if unsupported_rate > args.max_unsupported_rate:
        warn("UNSUPPORTED_RATE_HIGH", f"unsupported_rate={unsupported_rate:.4f} > max={args.max_unsupported_rate:.4f}")

    status = "PASS" if not errors else "FAIL"
    report = {
        "gate": "C2F_OBSERVATION_COLLECTION_AUDIT",
        "mode": args.mode,
        "status": status,
        "c2f_root": str(root),
        "output_dir": str(out),
        "created_at_unix": time.time(),
        "expected_episodes": args.expected_episodes,
        "n_episodes": len(meta_paths),
        "n_steps": n_steps,
        "n_rgb_rows": n_rgb,
        "primary_attackable_steps": primary,
        "hazard_steps": hazard,
        "release_safe_steps": release,
        "primary_rate": primary_rate,
        "hazard_rate": hazard / max(1, n_steps),
        "release_rate": release / max(1, n_steps),
        "unsupported_rate": unsupported_rate,
        "event_role_counts": dict(role_counts),
        "teacher_phase_counts": dict(phase_counts),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "episode_summaries_csv": str(out / "episode_summaries.csv"),
        "boundaries": {
            "attack": "NOT_PERFORMED",
            "d7b2_outcome_read": False,
            "student_forbidden_privileged_fields_checked": True,
            "rgb_required": True,
            "features_25d_required": True,
        },
    }
    write_json(out / "c2f_observation_collection_audit.json", report)
    write_csv(out / "episode_summaries.csv", episode_summaries)
    write_csv(out / "errors.csv", errors)
    write_csv(out / "warnings.csv", warnings)
    (out / "C2F_OBSERVATION_COLLECTION_AUDIT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": status, "n_episodes": len(meta_paths), "n_errors": len(errors), "n_warnings": len(warnings), "output_dir": str(out)}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
