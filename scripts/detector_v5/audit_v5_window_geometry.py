"""FIT-only audit of V5 candidate-window geometry.

This is read-only with respect to source roots.  It writes one new sealed
report root and never promotes UNKNOWN or non-candidate rows to windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episode_root(root: Path, row: dict[str, Any]) -> Path:
    return root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"


def _fit_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("canonical_parent_key", ""))
        state = int(row.get("state_id", -1))
        task = int(row.get("task_idx", -1))
        if state not in range(20) or task not in range(10):
            continue
        if key in seen or key != f"{row['suite']}/task_{task:02d}/state_{state:02d}":
            raise ValueError(f"invalid or duplicate FIT identity: {key}")
        seen.add(key)
        result.append(dict(row, task_idx=task, state_id=state, canonical_parent_key=key))
    if len(result) != 800:
        raise ValueError(f"expected 800 FIT identities, got {len(result)}")
    return sorted(result, key=lambda row: row["canonical_parent_key"])


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    x = [a for a, _ in pairs]
    y = [b for _, b in pairs]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom = (sum(value * value for value in dx) * sum(value * value for value in dy)) ** 0.5
    return None if denom == 0 else sum(a * b for a, b in zip(dx, dy)) / denom


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = _fit_rows(args.registry_csv.resolve())
    window_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    tier_steps = Counter()
    tier_windows = Counter()
    length_by_tier: dict[str, list[int]] = defaultdict(list)
    utility_length_pairs: list[tuple[float, float]] = []
    utility_time_pairs: list[tuple[float, float]] = []
    non_contiguous = 0
    overlap_count = 0
    for row in rows:
        identity = row["canonical_parent_key"]
        student = _jsonl(_episode_root(args.s1_root.resolve(), row) / "student_input_records.jsonl")
        teacher = _jsonl(_episode_root(args.teacher_root.resolve(), row) / "v5_teacher_utility.jsonl")
        if len(student) != len(teacher) or not student:
            raise ValueError(f"stream mismatch: {identity}")
        base_counts: Counter[str] = Counter()
        seen_base: set[str] = set()
        rankable_runs: list[dict[str, Any]] = []
        active: dict[str, Any] | None = None
        for index, (srow, trow) in enumerate(zip(student, teacher)):
            if int(srow.get("step", -1)) != index or int(trow.get("step", -1)) != index:
                raise ValueError(f"step mismatch: {identity}:{index}")
            phase = str(trow.get("phase_name", "UNKNOWN"))
            base_id = str(trow.get("window_id", ""))
            tier = trow.get("utility_tier")
            tier = None if tier is None else int(tier)
            known = bool(trow.get("known_mask", False))
            candidate = bool(trow.get("candidate_close", False))
            valid = bool(srow.get("valid", False))
            if phase == "UNKNOWN":
                tier_steps["UNKNOWN"] += 1
            if base_id.startswith("none:"):
                tier_steps["none_window_rows"] += 1
            rankable = valid and known and candidate and phase != "UNKNOWN" and not base_id.startswith("none:")
            if candidate and known and phase != "UNKNOWN" and not valid:
                tier_steps["student_invalid_candidate_overlap"] += 1
            current = {
                "index": index,
                "base_id": base_id,
                "phase": phase,
                "tier": tier,
                "rankable": rankable,
                "start": int(trow.get("window_start", index)),
                "end": int(trow.get("window_end", index)),
                "time_since_close": float(srow.get("features_25d", [0.0] * 25)[17]),
            }
            same = bool(
                active and rankable and index == active["indices"][-1] + 1
                and base_id == active["base_id"] and phase == active["phase"] and tier == active["tier"]
            )
            if same:
                active["indices"].append(index)
                continue
            if active:
                rankable_runs.append(active)
                active = None
            if rankable:
                active = {**current, "indices": [index]}
        if active:
            rankable_runs.append(active)
        for run in rankable_runs:
            base_id = str(run["base_id"])
            ordinal = base_counts[base_id]
            base_counts[base_id] += 1
            if ordinal:
                non_contiguous += 1
                seen_base.add(base_id)
            indices = tuple(run["indices"])
            tier = run["tier"]
            window_id = base_id if ordinal == 0 else f"{base_id}#segment{ordinal}"
            length = len(indices)
            tier_key = "unknown" if tier is None else str(tier)
            tier_windows[tier_key] += 1
            length_by_tier[tier_key].append(length)
            if tier is not None:
                tier_steps[tier_key] += length
                utility_length_pairs.append((float(tier), float(length)))
                utility_time_pairs.append((float(tier), float(run["time_since_close"])))
            window_rows.append({
                "canonical_parent_key": identity,
                "window_id": window_id,
                "source_window_id": base_id,
                "phase_name": run["phase"],
                "utility_tier": tier,
                "start_step": indices[0],
                "end_step": indices[-1],
                "step_count": length,
                "decision_anchor_step": indices[min(9, length - 1)],
                "minimum_dwell_met": length >= 10,
                "contiguous": True,
            })
            if run["end"] >= run["start"] and run["start"] != indices[0]:
                overlap_count += 1
        episode_tiers = [run["tier"] for run in rankable_runs if run["tier"] is not None]
        has_positive = any(tier >= 2 for tier in episode_tiers)
        has_negative = any(tier <= 1 for tier in episode_tiers)
        if not rankable_runs:
            category = "NO_CANDIDATE"
        elif has_positive and has_negative:
            category = "TRUE_MIXED"
        elif has_positive:
            category = "POSITIVE_ONLY"
        else:
            category = "PURE_NEGATIVE"
        episode_rows.append({
            "canonical_parent_key": identity,
            "suite": row["suite"],
            "task_idx": int(row["task_idx"]),
            "state_id": int(row["state_id"]),
            "candidate_window_count": len(rankable_runs),
            "positive_window_count": sum(tier >= 2 for tier in episode_tiers),
            "negative_window_count": sum(tier <= 1 for tier in episode_tiers),
            "category": category,
        })
    summary = {
        "schema": "DETECTOR_V5_WINDOW_GEOMETRY_AUDIT_V2",
        "identity_count": len(rows),
        "candidate_window_count": len(window_rows),
        "known_candidate_window_count": len(window_rows),
        "tier_window_counts": dict(tier_windows),
        "tier_step_counts": dict(tier_steps),
        "positive_only_episode_count": sum(row["category"] == "POSITIVE_ONLY" for row in episode_rows),
        "true_mixed_episode_count": sum(row["category"] == "TRUE_MIXED" for row in episode_rows),
        "tier3_tier2_episode_count": sum(row["positive_window_count"] > 1 for row in episode_rows),
        "pure_negative_episode_count": sum(row["category"] == "PURE_NEGATIVE" for row in episode_rows),
        "no_candidate_episode_count": sum(row["category"] == "NO_CANDIDATE" for row in episode_rows),
        "non_contiguous_same_window_id_count": non_contiguous,
        "overlap_count": overlap_count,
        "unknown_rows": tier_steps["UNKNOWN"],
        "none_window_rows": tier_steps["none_window_rows"],
        "student_invalid_candidate_overlap": tier_steps["student_invalid_candidate_overlap"],
        "utility_vs_window_length_pearson": _pearson(utility_length_pairs),
        "utility_vs_time_since_close_pearson": _pearson(utility_time_pairs),
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        with (staging / "DETECTOR_V5_WINDOW_GEOMETRY.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(window_rows[0]) if window_rows else ["canonical_parent_key"])
            writer.writeheader()
            writer.writerows(window_rows)
        with (staging / "DETECTOR_V5_EPISODE_GEOMETRY.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]) if episode_rows else ["canonical_parent_key"])
            writer.writeheader()
            writer.writerows(episode_rows)
        _atomic_text(staging / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _atomic_text(staging / "audit_report.md", "# Official V5 window geometry\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
        payload = sorted(path.name for path in staging.iterdir() if path.is_file())
        _atomic_text(staging / "SHA256SUMS", "".join(f"{_sha(staging / name)}  {name}\n" for name in payload))
        _atomic_text(staging / "SHA256SUMS.sha256", f"{_sha(staging / 'SHA256SUMS')}  SHA256SUMS\n")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
