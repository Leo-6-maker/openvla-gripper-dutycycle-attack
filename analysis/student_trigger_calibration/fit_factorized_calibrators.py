#!/usr/bin/env python3
"""Fit Factorized V2 calibrators on a sealed, identity-scoped calibration bundle.

FAIL-CLOSED:
- canonical ``calibration_records.jsonl`` only;
- exact episode/step closure against the calibration-fit manifest;
- sealed split bundle required;
- checkpoint, split, feature and source bindings must match;
- fit, heldout and checkpoint-training identity provenance is explicit;
- missing fields, duplicate JSON keys, bool-as-number, NaN/Inf and
  logit/probability mismatches are rejected.

Methods: RAW, INTERCEPT_ONLY, PLATT.
This script never selects scheduler thresholds and never authorizes training,
full fit, rollout or attack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

LOGIT_PROB_TOLERANCE = 0.01
SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEADS = ("grasp", "manipulation", "release")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_text(text: str, label: str) -> Any:
    duplicate_keys: list[str] = []

    def hook(pairs):
        seen: set[str] = set()
        result = {}
        for key, value in pairs:
            if key in seen:
                duplicate_keys.append(str(key))
            seen.add(key)
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=hook)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON_PARSE_ERROR:{label}:{exc}") from exc
    if duplicate_keys:
        raise SystemExit(f"DUPLICATE_JSON_KEY:{label}:{sorted(set(duplicate_keys))}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"JSON_MISSING:{path}")
    value = _strict_json_text(path.read_text(encoding="utf-8"), str(path))
    if not isinstance(value, dict):
        raise SystemExit(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _identity_set(manifest: dict[str, Any], keys: tuple[str, ...], label: str) -> set[str]:
    values = None
    for key in keys:
        if key in manifest:
            values = manifest[key]
            break
    if not isinstance(values, list) or not values:
        raise SystemExit(f"{label}_IDENTITIES_MISSING")
    identities: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item:
            raise SystemExit(f"{label}_IDENTITY_INVALID:{item!r}")
        if item in identities:
            raise SystemExit(f"{label}_IDENTITY_DUPLICATE:{item}")
        identities.add(item)
    return identities


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise SystemExit(f"{label}_SHA_INVALID")
    return value.lower()


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise SystemExit(f"{label}_COMMIT_INVALID")
    return value.lower()


def verify_sealed_directory(root: Path) -> None:
    """Verify an exact recursive SHA256SUMS/SHA256SUMS.sha256 file set."""
    root = root.resolve()
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"BUNDLE_SEAL_MISSING:{root}")
    expected_sidecar = f"{sha256_file(sums)}  SHA256SUMS"
    if sidecar.read_text(encoding="utf-8").strip() != expected_sidecar:
        raise SystemExit(f"BUNDLE_SEAL_SIDECAR_INVALID:{root}")

    listed: set[str] = set()
    for line_no, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), start=1):
        digest, separator, name = line.partition("  ")
        rel = Path(name)
        if (
            not separator
            or not SHA_RE.fullmatch(digest)
            or rel.is_absolute()
            or ".." in rel.parts
            or rel.as_posix() in listed
        ):
            raise SystemExit(f"BUNDLE_CHECKSUM_ROW_INVALID:{root}:{line_no}")
        target = root / rel
        if not target.is_file() or sha256_file(target) != digest.lower():
            raise SystemExit(f"BUNDLE_CHECKSUM_MISMATCH:{root}:{name}")
        listed.add(rel.as_posix())

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != listed | {"SHA256SUMS", "SHA256SUMS.sha256"}:
        raise SystemExit(f"BUNDLE_FILE_SET_MISMATCH:{root}")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value))))


def validate_record(record: dict[str, Any], head: str, index: int) -> tuple[float, float, bool, bool]:
    """Validate one canonical calibration row for one head."""
    for field in (
        "episode",
        "step",
        f"{head}_logit",
        f"{head}_probability",
        f"{head}_known_mask",
        f"{head}_target",
    ):
        if field not in record:
            raise SystemExit(f"FIELD_MISSING:record={index}:field={field}")

    episode = record["episode"]
    step = record["step"]
    logit = record[f"{head}_logit"]
    probability = record[f"{head}_probability"]
    known = record[f"{head}_known_mask"]
    target = record[f"{head}_target"]

    if not isinstance(episode, str) or not episode:
        raise SystemExit(f"EPISODE_INVALID:record={index}")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise SystemExit(f"STEP_INVALID:record={index}:step={step!r}")
    if isinstance(logit, bool) or not isinstance(logit, (int, float)) or not math.isfinite(float(logit)):
        raise SystemExit(f"LOGIT_INVALID:record={index}:head={head}:value={logit!r}")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise SystemExit(f"PROBABILITY_INVALID:record={index}:head={head}:value={probability!r}")
    if not isinstance(known, bool):
        raise SystemExit(f"KNOWN_MASK_NOT_BOOL:record={index}:head={head}")
    if not isinstance(target, bool):
        raise SystemExit(f"TARGET_NOT_BOOL:record={index}:head={head}")
    return float(logit), float(probability), known, target


def load_and_validate(
    bundle_root: Path,
    split: str,
    *,
    fit_identities: set[str],
    checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    split_root = Path(bundle_root).resolve() / split
    verify_sealed_directory(split_root)

    manifest_path = split_root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "FACTORIZED_V2_OFFLINE_CALIBRATION_BUNDLE_V1":
        raise SystemExit(f"CALIBRATION_BUNDLE_SCHEMA_INVALID:{split}")
    if manifest.get("split") != split:
        raise SystemExit(f"CALIBRATION_BUNDLE_SPLIT_MISMATCH:{split}")
    if str(manifest.get("checkpoint_sha256", "")).lower() != checkpoint_sha256:
        raise SystemExit(f"CALIBRATION_BUNDLE_CHECKPOINT_MISMATCH:{split}")
    stream_name = manifest.get("record_stream", "calibration_records.jsonl")
    if stream_name != "calibration_records.jsonl":
        raise SystemExit(f"CALIBRATION_RECORD_STREAM_INVALID:{stream_name}")

    stream = split_root / "calibration_records.jsonl"
    if not stream.is_file():
        raise SystemExit(f"CALIBRATION_RECORDS_MISSING:{stream}")

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, line in enumerate(stream.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = _strict_json_text(line, f"{stream}:{index + 1}")
        if not isinstance(value, dict):
            raise SystemExit(f"CALIBRATION_RECORD_OBJECT_REQUIRED:{index}")
        episode = value.get("episode")
        step = value.get("step")
        key = (episode, step)
        if key in seen:
            raise SystemExit(f"DUPLICATE_KEY:record={index}:key={key}")
        seen.add(key)
        records.append(value)

    if not records:
        raise SystemExit(f"CALIBRATION_RECORDS_EMPTY:{split}")

    record_identities = {str(row.get("episode")) for row in records}
    missing = fit_identities - record_identities
    extra = record_identities - fit_identities
    if missing or extra:
        raise SystemExit(
            f"CALIBRATION_IDENTITY_CLOSURE_FAIL:{split}:"
            f"missing={sorted(missing)}:extra={sorted(extra)}"
        )

    for head in HEADS:
        for index, record in enumerate(records):
            validate_record(record, head, index)

    return records, manifest, sha256_file(split_root / "SHA256SUMS")


def check_logit_prob_consistency(records: list[dict[str, Any]], head: str) -> tuple[bool, float]:
    max_error = 0.0
    for record in records:
        if record[f"{head}_known_mask"]:
            expected = sigmoid(float(record[f"{head}_logit"]))
            error = abs(expected - float(record[f"{head}_probability"]))
            max_error = max(max_error, error)
    return max_error <= LOGIT_PROB_TOLERANCE, max_error


def _head_examples(records: list[dict[str, Any]], head: str) -> tuple[list[float], list[float]]:
    positives: list[float] = []
    negatives: list[float] = []
    for record in records:
        if not record[f"{head}_known_mask"]:
            continue
        value = float(record[f"{head}_logit"])
        (positives if record[f"{head}_target"] else negatives).append(value)
    return positives, negatives


def fit_raw(records: list[dict[str, Any]], head: str) -> dict[str, Any]:
    positives, negatives = _head_examples(records, head)
    binding_ok, max_error = check_logit_prob_consistency(records, head)
    valid = binding_ok and len(positives) >= 1 and len(negatives) >= 1
    return {
        "head": head,
        "method": "RAW",
        "a": 1.0,
        "b": 0.0,
        "n_fit_pos": len(positives),
        "n_fit_neg": len(negatives),
        "method_valid": valid,
        "method_status": "PASS" if valid else "HOLD_INSUFFICIENT_OR_BINDING",
        "logit_prob_max_error": round(max_error, 8),
    }


def fit_intercept(records: list[dict[str, Any]], head: str) -> dict[str, Any]:
    positives, negatives = _head_examples(records, head)
    binding_ok, max_error = check_logit_prob_consistency(records, head)
    if not binding_ok or len(positives) < 5 or len(negatives) < 5:
        return {
            "head": head,
            "method": "INTERCEPT_ONLY",
            "a": 1.0,
            "b": 0.0,
            "n_fit_pos": len(positives),
            "n_fit_neg": len(negatives),
            "method_valid": False,
            "method_status": "HOLD_INSUFFICIENT_OR_BINDING",
            "logit_prob_max_error": round(max_error, 8),
        }

    best_b, best_loss = 0.0, float("inf")
    for index in range(61):
        b = -3.0 + index * 0.1
        loss = 0.0
        for value in positives:
            loss -= math.log(max(1e-7, sigmoid(value + b)))
        for value in negatives:
            loss -= math.log(max(1e-7, 1.0 - sigmoid(value + b)))
        loss /= len(positives) + len(negatives)
        if loss < best_loss:
            best_loss, best_b = loss, b

    return {
        "head": head,
        "method": "INTERCEPT_ONLY",
        "a": 1.0,
        "b": round(best_b, 6),
        "n_fit_pos": len(positives),
        "n_fit_neg": len(negatives),
        "class_prevalence": len(positives) / (len(positives) + len(negatives)),
        "fit_loss": round(best_loss, 6),
        "method_valid": True,
        "method_status": "PASS",
        "logit_prob_max_error": round(max_error, 8),
    }


def fit_platt(records: list[dict[str, Any]], head: str) -> dict[str, Any]:
    positives, negatives = _head_examples(records, head)
    binding_ok, max_error = check_logit_prob_consistency(records, head)
    if not binding_ok or len(positives) < 5 or len(negatives) < 5:
        return {
            "head": head,
            "method": "PLATT",
            "a": 1.0,
            "b": 0.0,
            "n_fit_pos": len(positives),
            "n_fit_neg": len(negatives),
            "method_valid": False,
            "method_status": "HOLD_INSUFFICIENT_OR_BINDING",
            "logit_prob_max_error": round(max_error, 8),
        }

    best_a, best_b, best_loss = 1.0, 0.0, float("inf")
    for a_index in range(15):
        a = 0.2 + a_index * 0.2
        for b_index in range(61):
            b = -3.0 + b_index * 0.1
            loss = 0.0
            for value in positives:
                loss -= math.log(max(1e-7, sigmoid(a * value + b)))
            for value in negatives:
                loss -= math.log(max(1e-7, 1.0 - sigmoid(a * value + b)))
            loss /= len(positives) + len(negatives)
            if loss < best_loss:
                best_loss, best_a, best_b = loss, a, b

    return {
        "head": head,
        "method": "PLATT",
        "a": round(best_a, 6),
        "b": round(best_b, 6),
        "n_fit_pos": len(positives),
        "n_fit_neg": len(negatives),
        "class_prevalence": len(positives) / (len(positives) + len(negatives)),
        "fit_loss": round(best_loss, 6),
        "method_valid": True,
        "method_status": "PASS",
        "logit_prob_max_error": round(max_error, 8),
    }


def validate_fit_heldout_disjoint(
    fit_manifest: dict[str, Any],
    heldout_manifest: dict[str, Any],
) -> None:
    fit_ids = _identity_set(fit_manifest, ("fit_identities", "identities"), "FIT")
    heldout_ids = _identity_set(
        heldout_manifest,
        ("heldout_identities", "evaluation_identities", "identities"),
        "HELDOUT",
    )
    overlap = fit_ids & heldout_ids
    if overlap:
        raise ValueError(f"CALIBRATION_LEAKAGE:{len(overlap)}:{sorted(overlap)}")


def classify_provenance(
    fit_manifest: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
) -> str:
    fit_ids = _identity_set(fit_manifest, ("fit_identities", "identities"), "FIT")
    training_ids = _identity_set(
        checkpoint_manifest,
        ("training_identities", "train_identities", "identities"),
        "TRAINING",
    )
    if fit_ids & training_ids:
        return "TRAIN_RESUBSTITUTION_CALIBRATION"
    return "INDEPENDENT_CALIBRATION"


def _write_contract(output_root: Path, contract: dict[str, Any]) -> None:
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    (staging / "calibration_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = sorted(
        path for path in staging.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (staging / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    (staging / "SHA256SUMS.sha256").write_text(
        f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n",
        encoding="utf-8",
    )
    os.replace(staging, output_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-bundle-root", "--inner-train-bundle-root", dest="bundle_root", type=Path, required=True)
    parser.add_argument("--calibration-fit-manifest", "--inner-train-manifest", dest="fit_manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--method", choices=("RAW", "INTERCEPT_ONLY", "PLATT"), default="PLATT")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--student-source-commit", required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"OUTPUT_EXISTS:{output_root}")

    checkpoint_sha = _require_sha(args.checkpoint_sha256, "CHECKPOINT")
    student_commit = _require_commit(args.student_source_commit, "STUDENT_SOURCE")

    fit_manifest = load_json(args.fit_manifest)
    heldout_manifest = load_json(args.heldout_manifest)
    checkpoint_manifest = load_json(args.checkpoint_manifest)

    fit_ids = _identity_set(fit_manifest, ("fit_identities", "identities"), "FIT")
    heldout_ids = _identity_set(
        heldout_manifest,
        ("heldout_identities", "evaluation_identities", "identities"),
        "HELDOUT",
    )
    if fit_ids & heldout_ids:
        raise SystemExit(f"CALIBRATION_LEAKAGE:{sorted(fit_ids & heldout_ids)}")

    declared_checkpoint = _require_sha(checkpoint_manifest.get("checkpoint_sha256"), "CHECKPOINT_MANIFEST")
    if declared_checkpoint != checkpoint_sha:
        raise SystemExit("CHECKPOINT_MANIFEST_BINDING_MISMATCH")

    provenance = classify_provenance(fit_manifest, checkpoint_manifest)
    records, bundle_manifest, bundle_seal = load_and_validate(
        args.bundle_root,
        args.split,
        fit_identities=fit_ids,
        checkpoint_sha256=checkpoint_sha,
    )

    calibrators: list[dict[str, Any]] = []
    for head in HEADS:
        if args.method == "RAW":
            calibrator = fit_raw(records, head)
        elif args.method == "INTERCEPT_ONLY":
            calibrator = fit_intercept(records, head)
        else:
            calibrator = fit_platt(records, head)
        calibrator.update(
            checkpoint_sha256=checkpoint_sha,
            split=args.split,
            provenance=provenance,
            formal_selection_eligible=False,
        )
        calibrators.append(calibrator)

    all_valid = all(item["method_valid"] is True for item in calibrators)
    authoritative = provenance == "INDEPENDENT_CALIBRATION" and all_valid
    contract = {
        "schema": "FACTORIZED_V2_CALIBRATION_CONTRACT_V2",
        "split": args.split,
        "method": args.method,
        "checkpoint_sha256": checkpoint_sha,
        "student_source_commit": student_commit,
        "fit_manifest_sha256": sha256_file(args.fit_manifest),
        "heldout_manifest_sha256": sha256_file(args.heldout_manifest),
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint_manifest),
        "calibration_bundle_manifest_sha256": sha256_file(
            Path(args.bundle_root).resolve() / args.split / "manifest.json"
        ),
        "calibration_bundle_seal_sha256": bundle_seal,
        "feature_input_seal_sha256": _require_sha(
            bundle_manifest.get("feature_input_seal_sha256"),
            "FEATURE_INPUT_SEAL",
        ),
        "provenance": provenance,
        "authoritative": authoritative,
        "all_heads_valid": all_valid,
        "fit_identity_count": len(fit_ids),
        "heldout_identity_count": len(heldout_ids),
        "calibrators": calibrators,
        "formal_selection_eligible": False,
        "training_authorized": False,
        "full_fit_authorized": False,
        "attack_authorized": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_contract(output_root, contract)
    print(
        f"Calibration contract sealed:{output_root} "
        f"provenance={provenance} authoritative={authoritative}"
    )
    return 0 if authoritative else 2


if __name__ == "__main__":
    raise SystemExit(main())
