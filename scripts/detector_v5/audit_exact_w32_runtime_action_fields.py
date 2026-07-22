#!/usr/bin/env python3
"""Read-only audit of runtime action fields in Exact-W32 split roots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.action_contract import CanonicalActionState  # noqa: E402
from gripper_attack.b3_training_protocol import seal_directory, verify_sealed_directory  # noqa: E402
from gripper_attack.factorized_scheduler_bridge import sha256_file  # noqa: E402

EPSILON = 1e-6
CERT_FIELDS = {
    "field_semantics": "OPENVLA_RAW_ACTION",
    "field_stage": "CLEAN_PRE_ATTACK_DECODE",
    "field_dimension": 7,
    "gripper_index": 6,
    "postprocessed": False,
    "attacked": False,
}


class AuditError(ValueError):
    pass


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_dicts(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _manifest(root: Path) -> tuple[dict[str, Any], str | None]:
    candidates = [root / name for name in ("input_manifest.json", "source_binding.json", "manifest.json")]
    candidates.extend(sorted(root.rglob("input_manifest.json")))
    candidates.extend(sorted(root.rglob("source_binding.json")))
    candidates.extend(sorted(root.rglob("manifest.json")))
    seen: set[Path] = set()
    merged: dict[str, Any] = {}
    digest = None
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            value = _json(path)
        except Exception:
            continue
        if isinstance(value, dict):
            merged.update(value)
            digest = digest or sha256_file(path)
    return merged, digest


def _certified(manifest: Mapping[str, Any]) -> bool:
    return any(all(item.get(key) == expected for key, expected in CERT_FIELDS.items()) for item in _walk_dicts(manifest))


def _raw(value: Any) -> float | None:
    if not isinstance(value, (list, tuple)) or len(value) < 7:
        return None
    try:
        result = float(value[6])
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and -0.1 <= result <= 1.1 else None


def _raw_field_invalid(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, (list, tuple)) or len(value) < 7:
        return True
    try:
        result = float(value[6])
    except (TypeError, ValueError):
        return True
    return not math.isfinite(result) or not -0.1 <= result <= 1.1


def _step(row: Mapping[str, Any]) -> int:
    value = row.get("step", row.get("step_index"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError("STEP_INVALID")
    return value


def _identity(row: Mapping[str, Any], path: Path) -> str:
    value = row.get("canonical_parent_key", row.get("episode"))
    if not isinstance(value, str):
        raise AuditError(f"IDENTITY_MISSING:{path}")
    return value


def _load_rows(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("step_records.jsonl"))
    if not paths and (root / "step_records.jsonl").is_file():
        paths = [root / "step_records.jsonl"]
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_source_path"] = str(path)
                rows.append(row)
    return rows


def audit_split(split_key: str, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    verify_sealed_directory(root)
    manifest, manifest_sha = _manifest(root)
    certified = _certified(manifest)
    rows = _load_rows(root)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_identity(row, Path(row["_source_path"]))].append(row)
    counters = Counter()
    records: list[dict[str, Any]] = []
    streaks: list[int] = []
    route_counts: Counter[str] = Counter()
    event_role_counts: Counter[str] = Counter()
    source_commits: set[str] = set()
    contract_shas: set[str] = set()
    for identity, episode_rows in sorted(grouped.items()):
        episode_rows.sort(key=_step)
        if [_step(row) for row in episode_rows] != list(range(len(episode_rows))):
            counters["invalid_steps"] += len(episode_rows)
        current = 0
        max_streak = 0
        previous_close = False
        for row in episode_rows:
            clean = _raw(row.get("clean_action_raw_7d"))
            fallback = _raw(row.get("action_raw"))
            if _raw_field_invalid(row.get("clean_action_raw_7d")) or _raw_field_invalid(row.get("action_raw")):
                counters["nonfinite_or_invalid_raw_steps"] += 1
            attacked = any(name in row for name in ("attacked_action", "attack_action", "mutated_action"))
            if attacked:
                counters["invalid_steps"] += 1
                status = "ATTACKED_ACTION_FORBIDDEN"
                raw = None
            elif clean is not None and fallback is not None and abs(clean - fallback) > EPSILON:
                counters["mismatch_steps"] += 1
                status = "RAW_FIELD_MISMATCH"
                raw = None
            elif clean is not None:
                counters["preferred_field_steps"] += 1
                status = "PREFERRED"
                raw = clean
            elif fallback is not None and certified:
                counters["fallback_field_steps"] += 1
                status = "CERTIFIED_FALLBACK"
                raw = fallback
            elif fallback is not None:
                counters["fallback_uncertified_steps"] += 1
                status = "FALLBACK_UNCERTIFIED"
                raw = None
            else:
                counters["missing_steps"] += 1
                status = "MISSING_RAW_FIELD"
                raw = None
            candidate = False
            known = False
            if raw is not None:
                state = CanonicalActionState.from_step({"clean_action_raw_7d": [0.0] * 6 + [raw]})
                candidate = state.candidate_close
                known = state.action_known
                if not known:
                    counters["boundary_steps"] += 1
                if candidate:
                    counters["candidate_close_steps"] += 1
            if candidate:
                current += 1
                max_streak = max(max_streak, current)
                for target, known_name, positive_name in (
                    ("grasp", "grasp_known_mask", "grasp_target"),
                    ("manipulation", "manipulation_known_mask", "manipulation_target"),
                    ("release", "release_known_mask", "release_target"),
                ):
                    if bool(row.get(known_name, False)) and bool(row.get(positive_name, False)):
                        counters[f"candidate_close_{target}_positive_steps"] += 1
            else:
                if previous_close:
                    counters["candidate_resets"] += 1
                current = 0
            previous_close = candidate
            if row.get("route") is not None or row.get("mechanism_route") is not None:
                route_counts[str(row.get("route", row.get("mechanism_route")))] += 1
            if row.get("event_role") is not None:
                event_role_counts[str(row["event_role"])] += 1
            if row.get("source_commit"):
                source_commits.add(str(row["source_commit"]))
            if row.get("action_contract_sha256"):
                contract_shas.add(str(row["action_contract_sha256"]))
        if max_streak:
            streaks.append(max_streak)
            for threshold in (3, 5, 10):
                if max_streak >= threshold:
                    counters[f"episodes_streak_ge_{threshold}"] += 1
    total = len(rows)
    invalid = counters["invalid_steps"] + counters["mismatch_steps"] + counters["fallback_uncertified_steps"] + counters["missing_steps"]
    status = "PASS" if total and invalid == 0 and len(source_commits) <= 1 and len(contract_shas) <= 1 else "HOLD_INCOMPLETE_RUNTIME_FIELD_COVERAGE"
    result = {
        "split_key": split_key,
        "root": str(root),
        "episodes": len(grouped),
        "steps": total,
        "preferred_field_steps": counters["preferred_field_steps"],
        "fallback_field_steps": counters["fallback_field_steps"],
        "missing_steps": counters["missing_steps"],
        "boundary_steps": counters["boundary_steps"],
        "mismatch_steps": counters["mismatch_steps"],
        "invalid_steps": counters["invalid_steps"],
        "nonfinite_or_invalid_raw_steps": counters["nonfinite_or_invalid_raw_steps"],
        "fallback_uncertified_steps": counters["fallback_uncertified_steps"],
        "candidate_close_steps": counters["candidate_close_steps"],
        "candidate_close_rate": counters["candidate_close_steps"] / max(1, total),
        "candidate_close_max_streak": max(streaks, default=0),
        "candidate_close_median_streak": sorted(streaks)[len(streaks) // 2] if streaks else 0,
        "episodes_with_streak_ge_3": counters["episodes_streak_ge_3"],
        "episodes_with_streak_ge_5": counters["episodes_streak_ge_5"],
        "episodes_with_streak_ge_10": counters["episodes_streak_ge_10"],
        "candidate_resets": counters["candidate_resets"],
        "candidate_close_grasp_positive_steps": counters["candidate_close_grasp_positive_steps"],
        "candidate_close_manipulation_positive_steps": counters["candidate_close_manipulation_positive_steps"],
        "candidate_close_release_positive_steps": counters["candidate_close_release_positive_steps"],
        "source_commit": manifest.get("source_commit") or (sorted(source_commits)[0] if len(source_commits) == 1 else None),
        "source_commit_count": len(source_commits),
        "action_contract_sha": manifest.get("action_contract_sha256") or (sorted(contract_shas)[0] if len(contract_shas) == 1 else None),
        "action_contract_sha_count": len(contract_shas),
        "input_manifest_sha": manifest_sha,
        "fallback_manifest_certified": certified,
        "route_counts": dict(route_counts),
        "event_role_counts": dict(event_role_counts),
        "status": status,
    }
    compatibility = {
        "split_key": split_key,
        "episodes": len(grouped),
        "candidate_close_rate": result["candidate_close_rate"],
        "max_streak": result["candidate_close_max_streak"],
        "median_streak": result["candidate_close_median_streak"],
        "episodes_streak_ge_3": result["episodes_with_streak_ge_3"],
        "episodes_streak_ge_5": result["episodes_with_streak_ge_5"],
        "episodes_streak_ge_10": result["episodes_with_streak_ge_10"],
        "classification": "COMPATIBLE_WITH_DWELL10" if result["episodes_with_streak_ge_10"] else "INCOMPATIBLE_WITH_DWELL10",
        "status": status,
    }
    distribution = {"split_key": split_key, "max_streaks": streaks, "classification": compatibility["classification"]}
    return result, [compatibility], distribution


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", required=True, help="split_key=sealed_root")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"OUTPUT_EXISTS:{output}")
    splits = []
    for item in args.split:
        if "=" not in item:
            raise SystemExit("SPLIT_FORMAT_MUST_BE_KEY_EQUALS_ROOT")
        key, value = item.split("=", 1)
        splits.append((key, Path(value).resolve()))
    results, compatibility, distributions = [], [], []
    try:
        for key, root in splits:
            result, compat, distribution = audit_split(key, root)
            results.append(result)
            compatibility.extend(compat)
            distributions.append(distribution)
    except Exception as exc:
        raise SystemExit(f"HOLD:{type(exc).__name__}:{exc}")
    status = "PASS" if len(results) == 12 and all(item["status"] == "PASS" for item in results) else "HOLD_INCOMPLETE_RUNTIME_FIELD_COVERAGE"
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        (staging / "EXACT_W32_RUNTIME_ACTION_FIELD_AUDIT.json").write_text(json.dumps({"schema": "EXACT_W32_RUNTIME_ACTION_FIELD_AUDIT_V1", "status": status, "split_count": len(results), "splits": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "EXACT_W32_RUNTIME_ACTION_FIELD_AUDIT.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in results for key in row if not isinstance(row[key], dict)})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow({key: row.get(key) for key in fields})
        with (staging / "candidate_close_compatibility.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(compatibility[0]) if compatibility else ["split_key", "status"])
            writer.writeheader()
            writer.writerows(compatibility)
        (staging / "candidate_close_streak_distribution.json").write_text(json.dumps(distributions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": status, "output_root": str(output), "split_count": len(results)}, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
