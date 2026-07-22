#!/usr/bin/env python3
"""Read-only closure audit for the Factorized V2 production inputs.

This command never loads a model and never writes to an input root.  It is
deliberately stricter than the historical W32 child-root checks: a production
chain requires a sealed top-level source root, exact identity/step joins, and
an independently certified clean raw action source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.action_contract import CanonicalActionState  # noqa: E402

EXPECTED_SPLITS = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))
SPLIT_RE = re.compile(r"o[0-3]_i[0-2]")
RAW_FIELDS = ("clean_action_raw_7d", "action_raw")


class ProductionInputAuditError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductionInputAuditError(f"DUPLICATE_JSON_KEY:{key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def _identity(row: Mapping[str, Any]) -> str:
    value = row.get("episode") or row.get("canonical_parent_key") or row.get("identity")
    if not isinstance(value, str) or value.count("/") != 2:
        raise ProductionInputAuditError("IDENTITY_INVALID")
    return value


def _step(row: Mapping[str, Any]) -> int:
    value = row.get("step", row.get("step_index"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionInputAuditError("STEP_INVALID")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProductionInputAuditError(f"MISSING_JSONL:{path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProductionInputAuditError(f"JSONL_ROW_NOT_OBJECT:{path}")
            rows.append(value)
    return rows


def verify_sealed_root(path: Path) -> dict[str, Any]:
    """Verify an exact recursive SHA256SUMS closure without changing the root."""
    result: dict[str, Any] = {"path": str(path), "exists": path.is_dir(), "pass": False}
    if not path.is_dir():
        result["reason"] = "MISSING_ROOT"
        return result
    sums = path / "SHA256SUMS"
    sidecar = path / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        result["reason"] = "SEAL_FILES_MISSING"
        return result
    sums_sha = sha256_file(sums)
    result["sha256s_sha256"] = sums_sha
    if sidecar.read_text(encoding="utf-8").strip() != f"{sums_sha}  SHA256SUMS":
        result["reason"] = "SEAL_SIDECAR_MISMATCH"
        return result
    listed: dict[str, str] = {}
    try:
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, sep, name = line.partition("  ")
            relative = Path(name)
            if not sep or not re.fullmatch(r"[0-9a-fA-F]{64}", digest) or not name:
                raise ProductionInputAuditError("SEAL_ROW_INVALID")
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in listed:
                raise ProductionInputAuditError("SEAL_PATH_INVALID")
            target = path / relative
            if not target.is_file() or target.is_symlink() or sha256_file(target) != digest.lower():
                raise ProductionInputAuditError(f"SEAL_FILE_MISMATCH:{name}")
            listed[relative.as_posix()] = digest.lower()
        actual = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()}
        expected = set(listed) | {"SHA256SUMS", "SHA256SUMS.sha256"}
        if actual != expected:
            raise ProductionInputAuditError(
                f"SEAL_FILE_SET_MISMATCH:extra={sorted(actual - expected)}:missing={sorted(expected - actual)}"
            )
    except ProductionInputAuditError as exc:
        result["reason"] = str(exc)
        return result
    result["pass"] = True
    result["reason"] = "PASS"
    return result


def _find_split(root: Path, split: str, *, prefix: str | None = None) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [item for item in root.iterdir() if item.is_dir() and split in item.name]
    if prefix:
        candidates = [item for item in candidates if item.name.startswith(prefix)]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _inner_sets(splits_root: Path, split: str) -> tuple[set[str], set[str]]:
    value = _strict_json(splits_root / "inner_cv_splits.json")
    split_map = value.get("splits", value)
    outer = split_map.get(f"fold_{split[1]}") if isinstance(split_map, Mapping) else None
    if not isinstance(outer, Mapping):
        raise ProductionInputAuditError(f"INNER_OUTER_FOLD_MISSING:{split}")
    inner_folds = outer.get("inner_folds")
    if not isinstance(inner_folds, list) or len(inner_folds) != 3:
        raise ProductionInputAuditError(f"INNER_FOLD_CLOSURE:{split}")
    inner = int(split[4])
    heldout = set(str(item) for item in inner_folds[inner].get("identities", []))
    train = set()
    for index, item in enumerate(inner_folds):
        if index != inner:
            train.update(str(identity) for identity in item.get("identities", []))
    if not heldout or not train or train & heldout:
        raise ProductionInputAuditError(f"INNER_IDENTITY_CONTRACT:{split}")
    return train, heldout


def _metadata_clean(path: Path) -> tuple[bool, str]:
    for name in ("episode_metadata.json", "condition_config.json"):
        candidate = path / name
        if candidate.is_file():
            try:
                value = _strict_json(candidate)
            except ProductionInputAuditError:
                raise
            except Exception:
                continue
            if value.get("attack_enabled") is False and value.get("condition", "CLEAN") == "CLEAN":
                return True, name
    return False, "MISSING_OR_UNCERTIFIED_CLEAN_METADATA"


def audit_raw_action(identity_root: Path) -> dict[str, Any]:
    clean, metadata_source = _metadata_clean(identity_root)
    step_path = identity_root / "step_records.jsonl"
    rows = _rows(step_path)
    source_counts: dict[str, int] = {field: 0 for field in RAW_FIELDS}
    counts = {"close": 0, "open": 0, "boundary": 0, "unknown": 0}
    streaks: list[int] = []
    current = 0
    used_fields: set[str] = set()
    invalid_raw = False
    for row in rows:
        raw_field = next((field for field in RAW_FIELDS if field in row), None)
        if raw_field is None:
            counts["unknown"] += 1
            current = 0
            continue
        used_fields.add(raw_field)
        source_counts[raw_field] += 1
        value = row.get(raw_field)
        if raw_field == "action_raw" and not clean:
            # A fallback is only allowed when the clean pre-attack contract was
            # explicitly certified.  This audit never guesses that certification.
            counts["unknown"] += 1
            current = 0
            continue
        try:
            if not isinstance(value, (list, tuple)) or len(value) < 7:
                raise ValueError("RAW_ACTION_SHAPE")
            raw_value = float(value[6])
            if not math.isfinite(raw_value) or not -0.1 <= raw_value <= 1.1:
                raise ValueError("RAW_ACTION_VALUE")
            state = CanonicalActionState.from_step({"clean_action_raw_7d": value})
        except Exception:
            invalid_raw = True
            state = None
        if state is None or not state.action_known and state.action_intent == "UNKNOWN":
            counts["unknown"] += 1
            current = 0
        elif state.action_intent == "CLOSE":
            counts["close"] += 1
            current += 1
            streaks.append(current)
        elif state.action_intent == "OPEN":
            counts["open"] += 1
            current = 0
        else:
            counts["boundary"] += 1
            current = 0
    return {
        "step_count": len(rows),
        "source_fields": sorted(used_fields),
        "source_field_counts": source_counts,
        "counts": counts,
        "close_rate": counts["close"] / len(rows) if rows else 0.0,
        "unknown_rate": counts["unknown"] / len(rows) if rows else 0.0,
        "boundary_rate": counts["boundary"] / len(rows) if rows else 0.0,
        "max_close_streak": max(streaks, default=0),
        "episodes_with_streak_ge_3": int(max(streaks, default=0) >= 3),
        "episodes_with_streak_ge_5": int(max(streaks, default=0) >= 5),
        "episodes_with_streak_ge_10": int(max(streaks, default=0) >= 10),
        "clean_metadata_certified": clean,
        "clean_metadata_source": metadata_source,
        "semantic_certification": "DIRECT_CLEAN_OPENVLA_RAW_ACTION" if "clean_action_raw_7d" in used_fields and clean and not invalid_raw else "HOLD",
        "invalid_raw_action": invalid_raw,
    }


def _index_rows(path: Path, *, require_identity: bool = True) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in _rows(path):
        key = (_identity(row), _step(row)) if require_identity else _step(row)
        if key in result:
            raise ProductionInputAuditError(f"DUPLICATE_STEP:{path}:{key}")
        result[key] = row
    return result


def _student_path(s1_root: Path, identity: str) -> Path:
    return s1_root.joinpath(*identity.split("/")) / "student_input_records.jsonl"


def _clean_path(clean_root: Path, identity: str) -> Path:
    return clean_root.joinpath(*identity.split("/")) / "step_records.jsonl"


def _teacher_index(teacher_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(teacher_root.rglob("*.jsonl")):
        if "factorized_teacher" not in path.name:
            continue
        for row in _rows(path):
            key = (_identity(row), _step(row))
            if key in index:
                raise ProductionInputAuditError(f"DUPLICATE_TEACHER_STEP:{key}")
            index[key] = row
    return index


def _split_audit(
    split: str,
    *,
    prediction_dir: Path | None,
    run_dir: Path | None,
    s1_root: Path,
    clean_root: Path,
    teacher_index: Mapping[tuple[str, int], Mapping[str, Any]],
    expected_heldout: set[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {"split": split, "pass": False}
    if prediction_dir is None or run_dir is None:
        row["reason"] = "SPLIT_ROOT_MISSING"
        return row
    row["prediction_root"] = str(prediction_dir)
    row["run_root"] = str(run_dir)
    row["prediction_seal"] = verify_sealed_root(prediction_dir)
    row["run_seal"] = verify_sealed_root(run_dir)
    prediction_manifest = prediction_dir / "prediction_manifest.json"
    run_binding = run_dir / "source_binding.json"
    run_config = run_dir / "run_config.json"
    if not prediction_manifest.is_file() or not run_binding.is_file() or not run_config.is_file():
        row["reason"] = "SPLIT_MANIFEST_INCOMPLETE"
        return row
    try:
        manifest = _strict_json(prediction_manifest)
        binding = _strict_json(run_binding)
        config = _strict_json(run_config)
        predictions = _index_rows(prediction_dir / "heldout_step_predictions.jsonl")
    except Exception as exc:
        row["reason"] = f"PREDICTION_PARSE:{exc}"
        return row
    actual_ids = {identity for identity, _ in predictions}
    row["identity_count"] = len(actual_ids)
    row["expected_identity_count"] = len(expected_heldout)
    row["identity_exact"] = actual_ids == expected_heldout
    row["record_count"] = len(predictions)
    row["formal_selection_eligible"] = manifest.get("formal_selection_eligible") is False and config.get("formal_selection_eligible") is False
    row["checkpoint_sha256"] = sha256_file(run_dir / "checkpoint.pt") if (run_dir / "checkpoint.pt").is_file() else None
    row["source_commit"] = binding.get("source_commit")
    row["feature_order_sha256"] = config.get("feature_order_sha256") or binding.get("feature_order_sha256")
    all_join = True
    raw_audit: dict[str, Any] = {"episodes": 0, "steps": 0, "source_fields": set(), "counts": {"close": 0, "open": 0, "boundary": 0, "unknown": 0}, "max_close_streak": 0, "episodes_with_streak_ge_3": 0, "episodes_with_streak_ge_5": 0, "episodes_with_streak_ge_10": 0, "certified": True}
    feature_orders: set[str] = set()
    student_missing = runtime_missing = teacher_missing = 0
    for identity in sorted(actual_ids):
        try:
            student = _index_rows(_student_path(s1_root, identity))
            runtime = _index_rows(_clean_path(clean_root, identity), require_identity=False)
            for student_row in student.values():
                value = student_row.get("feature_order_sha256")
                if isinstance(value, str):
                    feature_orders.add(value.lower())
            action = audit_raw_action(_clean_path(clean_root, identity).parent)
            raw_audit["episodes"] += 1
            raw_audit["steps"] += action["step_count"]
            raw_audit["source_fields"].update(action["source_fields"])
            for key in raw_audit["counts"]:
                raw_audit["counts"][key] += action["counts"][key]
            raw_audit["max_close_streak"] = max(raw_audit["max_close_streak"], action["max_close_streak"])
            raw_audit["episodes_with_streak_ge_3"] += action["episodes_with_streak_ge_3"]
            raw_audit["episodes_with_streak_ge_5"] += action["episodes_with_streak_ge_5"]
            raw_audit["episodes_with_streak_ge_10"] += action["episodes_with_streak_ge_10"]
            raw_audit["certified"] &= action["semantic_certification"] != "HOLD"
        except Exception:
            student_missing += 1
            runtime_missing += 1
            all_join = False
            continue
        for key in [key for key in predictions if key[0] == identity]:
            if key not in student:
                student_missing += 1
                all_join = False
            if key[1] not in runtime:
                runtime_missing += 1
                all_join = False
            if key not in teacher_index:
                teacher_missing += 1
                all_join = False
    row.update({
        "student_exact_join": student_missing == 0,
        "runtime_action_exact_join": runtime_missing == 0,
        "teacher_exact_join": teacher_missing == 0,
        "student_missing_steps": student_missing,
        "runtime_missing_steps": runtime_missing,
        "teacher_missing_steps": teacher_missing,
        "feature_order_sha256": sorted(feature_orders)[0] if len(feature_orders) == 1 else None,
        "feature_order_consistent": len(feature_orders) == 1,
        "candidate_close_audit": {**raw_audit, "source_fields": sorted(raw_audit["source_fields"])},
        "pass": bool(row["prediction_seal"]["pass"] and row["run_seal"]["pass"] and row["identity_exact"] and all_join and raw_audit["certified"] and row["formal_selection_eligible"] and len(feature_orders) == 1),
    })
    row["reason"] = "PASS" if row["pass"] else "SPLIT_INPUT_CLOSURE_HOLD"
    return row


def audit_roots(*, w32_root: Path, splits_root: Path, s1_root: Path, clean_root: Path, teacher_root: Path) -> dict[str, Any]:
    root_seals = {
        "w32_root": verify_sealed_root(w32_root),
        "splits_root": verify_sealed_root(splits_root),
        "s1_root": verify_sealed_root(s1_root),
        "teacher_root": verify_sealed_root(teacher_root),
    }
    teacher_index: dict[tuple[str, int], dict[str, Any]] = {}
    teacher_error = None
    try:
        teacher_index = _teacher_index(teacher_root)
    except Exception as exc:
        teacher_error = str(exc)
    rows: list[dict[str, Any]] = []
    for split in EXPECTED_SPLITS:
        prediction_dir = _find_split(w32_root, split, prefix="predict_")
        run_dir = _find_split(w32_root, split, prefix="V2B_")
        try:
            _, heldout = _inner_sets(splits_root, split)
        except Exception:
            heldout = set()
        rows.append(_split_audit(split, prediction_dir=prediction_dir, run_dir=run_dir, s1_root=s1_root, clean_root=clean_root, teacher_index=teacher_index, expected_heldout=heldout))
    child_pass = all(row["pass"] for row in rows)
    result = {
        "schema": "FACTORIZED_V2_PRODUCTION_INPUT_AUDIT_V1",
        "expected_split_keys": list(EXPECTED_SPLITS),
        "split_count": len(rows),
        "exact_split_closure": len(rows) == 12 and all(row["split"] in EXPECTED_SPLITS for row in rows),
        "root_seals": root_seals,
        "teacher_index_error": teacher_error,
        "splits": rows,
        "candidate_close_source": "clean/*/step_records.jsonl:clean_action_raw_7d[6]",
        "candidate_close_status": "DIRECTLY_AVAILABLE_FROM_SEALED_CLEAN_RUNTIME" if all(row.get("candidate_close_audit", {}).get("certified", False) for row in rows) else "HOLD",
        "independent_identity_sources_present": False,
        "production_chain_ready": bool(all(item["pass"] for item in root_seals.values()) and child_pass and teacher_error is None),
        "model_inference": False,
        "training": False,
        "full_fit": False,
        "cal_check_read": False,
        "rollout": False,
        "attack": False,
    }
    if not result["production_chain_ready"]:
        result["blockers"] = [
            name.upper() + "_SEAL_HOLD" for name, status in root_seals.items() if not status["pass"]
        ]
        if teacher_error:
            result["blockers"].append("TEACHER_INDEX_HOLD")
        result["blockers"].append("INDEPENDENT_CALIBRATION_AND_POLICY_IDENTITY_SOURCES_MISSING")
    return result


def _atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def _seal_flat(root: Path) -> str:
    names = sorted(path.name for path in root.iterdir() if path.is_file())
    sums = "".join(f"{sha256_file(root / name)}  {name}\n" for name in names)
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def write_audit_root(result: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Atomically write a new, sealed audit root; never touch an existing root."""
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        (staging / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        columns = ["split", "pass", "identity_exact", "student_exact_join", "runtime_action_exact_join", "teacher_exact_join", "record_count", "reason"]
        with (staging / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in result.get("splits", []):
                writer.writerow({key: row.get(key, "") for key in columns})
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "FACTORIZED_V2_PRODUCTION_INPUT_AUDIT_ROOT_V1",
            "summary_filename": "summary.json",
            "split_table_filename": "splits.csv",
            "production_chain_ready": bool(result.get("production_chain_ready")),
            "model_inference": False,
            "training": False,
            "attack": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = _seal_flat(staging)
        os.replace(staging, output_root)
        return {"status": "PASS", "output_root": str(output_root), "sha256s_sha256": digest}
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w32-root", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.output_root is None and args.output_json is None:
        parser.error("one of --output-root or --output-json is required")
    result = audit_roots(
        w32_root=args.w32_root.resolve(), splits_root=args.splits_root.resolve(),
        s1_root=args.s1_root.resolve(), clean_root=args.clean_root.resolve(),
        teacher_root=args.teacher_root.resolve(),
    )
    if args.output_root is not None:
        write_audit_root(result, args.output_root.resolve())
        print(json.dumps({"status": "PASS" if result["production_chain_ready"] else "HOLD", "output_root": str(args.output_root.resolve()), "production_chain_ready": result["production_chain_ready"]}, sort_keys=True))
        return 0 if result["production_chain_ready"] else 2
    _atomic_text(args.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_csv:
        columns = ["split", "pass", "identity_exact", "student_exact_join", "runtime_action_exact_join", "teacher_exact_join", "record_count", "candidate_close_audit", "reason"]
        if args.output_csv.exists():
            raise FileExistsError(f"OUTPUT_EXISTS:{args.output_csv}")
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output_csv.with_name(f".{args.output_csv.name}.{uuid.uuid4().hex}.staging")
        with staging.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in result["splits"]:
                writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key, "") for key in columns})
        os.replace(staging, args.output_csv)
    print(json.dumps({"status": "PASS" if result["production_chain_ready"] else "HOLD", "production_chain_ready": result["production_chain_ready"]}, sort_keys=True))
    return 0 if result["production_chain_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
