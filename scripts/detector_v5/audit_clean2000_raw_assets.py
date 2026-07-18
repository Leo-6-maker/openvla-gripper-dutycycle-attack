#!/usr/bin/env python3
"""Read-only CLEAN2000 raw-asset recovery census.

This tool inventories existing files and FIT metadata only.  It never runs a
model, replays an episode, materializes a Teacher, or changes a source root.
RGB found in a parallel collection is deliberately reported separately from
the Official V3 artifact source; discovery is not authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "CLEAN2000_ARTIFACT_RECOVERY_AUDIT_V1"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
MODEL_EXTENSIONS = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}
TASK_RE = re.compile(r"^task_(\d+)$")
STATE_RE = re.compile(r"^state_(\d+)$")
SMALL_HASH_LIMIT = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _canonical_key(suite: str, task: int, state: int) -> str:
    return f"{suite}/task_{task:02d}/state_{state:02d}"


def _parse_key_from_parts(parts: tuple[str, ...]) -> tuple[str, int, int] | None:
    suite_positions = [index for index, part in enumerate(parts) if part in SUITES]
    for suite_index in reversed(suite_positions):
        task_match = next((TASK_RE.match(part) for part in parts[suite_index + 1 :]), None)
        if task_match is None:
            continue
        task_index = parts.index(task_match.group(0), suite_index + 1)
        state_match = next((STATE_RE.match(part) for part in parts[task_index + 1 :]), None)
        if state_match is None:
            continue
        state_index = parts.index(state_match.group(0), task_index + 1)
        task = int(task_match.group(1))
        state = int(state_match.group(1))
        return _canonical_key(parts[suite_index], task, state), task, state
    return None


def _identity_root_parts(parts: tuple[str, ...]) -> tuple[str, ...] | None:
    suite_positions = [index for index, part in enumerate(parts) if part in SUITES]
    for suite_index in reversed(suite_positions):
        task_index = next((index for index in range(suite_index + 1, len(parts)) if TASK_RE.match(parts[index])), None)
        if task_index is None:
            continue
        state_index = next((index for index in range(task_index + 1, len(parts)) if STATE_RE.match(parts[index])), None)
        if state_index is not None:
            return parts[: state_index + 1]
    return None


def _parse_asset_root(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError(f"asset root must be LABEL=PATH, got {value!r}")
    return label.strip(), Path(path).expanduser().resolve()


def _read_registry(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    malformed: list[str] = []
    for row in rows:
        key = str(row.get("canonical_parent_key", ""))
        parts = key.split("/")
        try:
            suite = parts[0]
            task = int(str(row.get("task_idx", "-1")))
            state = int(str(row.get("state_id", "-1")))
        except ValueError:
            malformed.append(key)
            continue
        if len(parts) != 3 or suite not in SUITES or task < 0 or task >= 10 or state < 0 or state >= 50:
            malformed.append(key)
            continue
        canonical = _canonical_key(suite, task, state)
        if key != canonical or canonical in seen:
            malformed.append(key)
            continue
        seen.add(canonical)
        parsed.append(
            {
                "canonical_parent_key": canonical,
                "suite": suite,
                "task_idx": task,
                "state_id": state,
                "split": str(row.get("split", "")),
                "task_success": str(row.get("task_success", "")),
            }
        )
    if malformed:
        raise ValueError(f"registry contains malformed or duplicate identities: {malformed[:5]}")
    expected = {_canonical_key(suite, task, state) for suite in SUITES for task in range(10) for state in range(50)}
    if set(seen) != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"registry is not the canonical 2000 universe: missing={missing[:5]} extra={extra[:5]}")
    summary = {
        "path": str(path),
        "sha256": sha256_file(path),
        "identity_count": len(parsed),
        "fit_identity_count": sum(row["state_id"] < 20 for row in parsed),
        "protected_identity_count": sum(row["state_id"] >= 20 for row in parsed),
        "suite_counts": dict(Counter(row["suite"] for row in parsed)),
    }
    return parsed, summary


def _new_info(key: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_parent_key": key,
        "split": row.get("split", ""),
        "state_id": row["state_id"],
        "identity_root_relatives": set(),
        "file_count": 0,
        "bytes": 0,
        "image_files": [],
        "video_files": [],
        "policy_files": [],
        "privileged_files": [],
        "model_files": [],
        "metadata_files": [],
        "file_names": set(),
        "extension_counts": Counter(),
        "semantic_read": False,
        "semantic_parse_errors": 0,
        "step_count_hint": 0,
        "rgb_paths": [],
        "rgb_path_missing": 0,
        "rgb_path_duplicate": 0,
        "direct_logits": False,
        "direct_policy_intent": False,
        "language": False,
        "model_pointer": False,
        "privileged_step_data": False,
        "teacher_fields": False,
        "source_schema": "",
        "collector_commits": set(),
        "source_commits": set(),
        "sample_content_fields": set(),
    }


def _contains_model_pointer(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_model_pointer(key) or _contains_model_pointer(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_model_pointer(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in ("model", "processor", "checkpoint", "safetensor", "tokenizer"))
    return False


def _relative_suffix_exists(image_relatives: set[str], rgb_path: str) -> bool:
    normalized = rgb_path.replace("\\", "/").lstrip("./")
    return any(candidate == normalized or candidate.endswith("/" + normalized) for candidate in image_relatives)


def _inspect_fit_file(path: Path, info: dict[str, Any], state_root: Path, image_relatives: set[str]) -> None:
    """Read content only for FIT identities; protected split files stay metadata-only."""
    info["semantic_read"] = True
    name = path.name.lower()
    if name == "episode_metadata.json":
        value = _json_object(path)
        if value is None:
            info["semantic_parse_errors"] += 1
            return
        info["source_schema"] = str(value.get("schema", ""))
        for field in ("collector_commit", "source_commit"):
            if value.get(field):
                info[f"{field}s"].add(str(value[field]))
        info["language"] = bool(value.get("task_language") or value.get("prompt"))
        info["model_pointer"] = _contains_model_pointer(value)
        if value.get("n_steps") is not None:
            try:
                info["step_count_hint"] = int(value["n_steps"])
            except (TypeError, ValueError):
                info["semantic_parse_errors"] += 1
        info["sample_content_fields"].update(str(key) for key in value)
        return
    if name not in {"step_records.jsonl", "policy_intent_records.jsonl"}:
        if "privileged" in name or "contact" in name or "object_state" in name:
            info["privileged_step_data"] = True
        return
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSON object expected")
                info["step_count_hint"] = max(info["step_count_hint"], int(value.get("step", -1)) + 1)
                fields = {str(key) for key in value}
                info["sample_content_fields"].update(fields)
                if value.get("rgb_path"):
                    info["rgb_paths"].append(str(value["rgb_path"]))
                if any("logit" in field.lower() for field in fields) or "score_head_summary" in fields:
                    info["direct_logits"] = True
                if "clean_policy_intent_9d" in fields or "policy_intent_9d" in fields:
                    info["direct_policy_intent"] = True
                if value.get("task_language") or value.get("prompt"):
                    info["language"] = True
                if any(field.startswith("teacher_") for field in fields):
                    info["teacher_fields"] = True
                if any(token in field.lower() for field in fields for token in ("contact", "object_state", "qpos", "eef")):
                    info["privileged_step_data"] = True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        info["semantic_parse_errors"] += 1
        return
    if name == "policy_intent_records.jsonl":
        info["direct_policy_intent"] = True
    if info["rgb_paths"]:
        info["rgb_path_missing"] = sum(not _relative_suffix_exists(image_relatives, path) for path in info["rgb_paths"])
        info["rgb_path_duplicate"] = len(info["rgb_paths"]) - len(set(info["rgb_paths"]))


def _scan_asset_root(label: str, root: Path, registry: dict[str, dict[str, Any]], max_semantic_state: int) -> dict[str, Any]:
    infos = {key: _new_info(key, row) for key, row in registry.items()}
    discovered: dict[str, set[str]] = defaultdict(set)
    path_size_digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    extension_counts: Counter[str] = Counter()
    symlink_count = 0
    metadata_hashes: list[dict[str, Any]] = []
    if not root.is_dir():
        return {"label": label, "root": str(root), "status": "NOT_FOUND", "identity_rows": infos, "error": "root missing"}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        symlink_count += sum((Path(directory) / name).is_symlink() for name in dirnames)
        dirnames[:] = [name for name in dirnames if not (Path(directory) / name).is_symlink()]
        for filename in filenames:
            path = Path(directory) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            relative = path.relative_to(root).as_posix()
            size = int(stat.st_size)
            file_count += 1
            total_bytes += size
            suffix = path.suffix.lower()
            extension_counts[suffix or "<none>"] += 1
            path_size_digest.update(relative.encode("utf-8"))
            path_size_digest.update(b"\0")
            path_size_digest.update(str(size).encode("ascii"))
            path_size_digest.update(b"\n")
            parsed = _parse_key_from_parts(tuple(Path(relative).parts))
            if parsed is None:
                continue
            key, _task, state = parsed
            if key not in infos:
                discovered[key].add(relative)
                continue
            identity_root_parts = _identity_root_parts(tuple(Path(relative).parts))
            if identity_root_parts is None:
                continue
            identity_root = Path(*identity_root_parts)
            info = infos[key]
            info["identity_root_relatives"].add(identity_root.as_posix())
            info["file_count"] += 1
            info["bytes"] += size
            info["file_names"].add(filename.lower())
            info["extension_counts"][suffix or "<none>"] += 1
            if suffix in IMAGE_EXTENSIONS:
                info["image_files"].append(relative)
            if suffix in VIDEO_EXTENSIONS:
                info["video_files"].append(relative)
            lower_path = relative.lower()
            if "policy_intent" in lower_path or "logit" in lower_path or "score_head" in lower_path:
                info["policy_files"].append(relative)
            if any(token in lower_path for token in ("privileged", "contact", "object_state", "qpos", "eef")):
                info["privileged_files"].append(relative)
            if suffix in MODEL_EXTENSIONS or any(token in lower_path for token in ("processor", "tokenizer", "model_tree", "checkpoint")):
                info["model_files"].append(relative)
            if suffix in {".json", ".jsonl", ".csv", ".yaml", ".yml"}:
                info["metadata_files"].append(relative)
            if state <= max_semantic_state and (path.name.lower() in {"episode_metadata.json", "step_records.jsonl", "policy_intent_records.jsonl"}):
                _inspect_fit_file(path, info, root / identity_root, set())
    for key, info in infos.items():
        image_relatives = {path[len(next(iter(info["identity_root_relatives"]))) + 1 :] if info["identity_root_relatives"] else path for path in info["image_files"]}
        if info["rgb_paths"] and info["image_files"]:
            info["rgb_path_missing"] = sum(not _relative_suffix_exists(image_relatives, path) for path in info["rgb_paths"])
        if info["privileged_files"]:
            info["privileged_step_data"] = True
        discovered[key].update(info["identity_root_relatives"])
    for key in infos:
        if infos[key]["identity_root_relatives"]:
            discovered[key].update(infos[key]["identity_root_relatives"])
    return {
        "label": label,
        "root": str(root),
        "status": "PASS",
        "identity_rows": infos,
        "discovered": discovered,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "extension_counts": extension_counts,
        "symlink_count": symlink_count,
        "path_size_inventory_sha256": path_size_digest.hexdigest(),
        "metadata_hashes": metadata_hashes,
    }


def _source_origin(label: str) -> tuple[str, str]:
    lowered = label.lower()
    if "official" in lowered or lowered in {"v3", "official_v3_clean"}:
        return "OFFICIAL_V3_CLEAN_ARTIFACT", "OFFICIAL_V3_BOUND"
    if "derived" in lowered or "embedding" in lowered or "siglip" in lowered:
        return "DERIVED_MODEL_INPUT_OUTPUT", "DERIVED_NOT_RAW"
    if "c2f" in lowered or "raw" in lowered:
        return "PARALLEL_C2F_RAW_COLLECTION", "UNBOUND_PARALLEL_SOURCE"
    return "UNCLASSIFIED_SOURCE", "UNVERIFIED"


def _classify(info: dict[str, Any], source_origin: str, compatibility: str, max_semantic_state: int) -> dict[str, Any]:
    image_count = len(info["image_files"])
    video_count = len(info["video_files"])
    if image_count:
        rgb_class = "RGB_DIRECT"
        rgb_alignment = "PASS" if info["rgb_paths"] and info["rgb_path_missing"] == 0 and info["rgb_path_duplicate"] == 0 else ("HOLD" if info["rgb_paths"] else "NOT_CHECKED")
    elif video_count:
        alignment_evidence = any(token in name for name in info["file_names"] for token in ("index", "timestamp", "alignment", "step"))
        rgb_class = "RGB_VIDEO_ALIGNED" if alignment_evidence else "RGB_MISSING"
        rgb_alignment = "PASS" if alignment_evidence else "HOLD_UNALIGNED_VIDEO"
    else:
        rgb_class, rgb_alignment = "RGB_MISSING", "NOT_FOUND"
    if info["direct_logits"] or info["policy_files"]:
        logits_class = "LOGITS_DIRECT"
    elif rgb_class != "RGB_MISSING" and info["model_files"] and info["language"]:
        logits_class = "LOGITS_REINFERABLE"
    else:
        logits_class = "LOGITS_MISSING"
    if info["privileged_files"] and info["semantic_read"]:
        privileged_class = "PRIVILEGED_READY"
    elif info["privileged_files"]:
        privileged_class = "PRIVILEGED_PARTIAL"
    else:
        privileged_class = "PRIVILEGED_MISSING"
    if compatibility == "OFFICIAL_V3_BOUND":
        disposition = "OFFICIAL_V3_SOURCE_CANDIDATE"
    elif source_origin == "PARALLEL_C2F_RAW_COLLECTION":
        disposition = "DISCOVERED_UNBOUND_PARALLEL_SOURCE"
    elif source_origin == "DERIVED_MODEL_INPUT_OUTPUT":
        disposition = "DERIVED_NOT_RAW_SOURCE"
    else:
        disposition = "UNVERIFIED_SOURCE"
    return {
        "rgb_class": rgb_class,
        "rgb_alignment_status": rgb_alignment,
        "logits_class": logits_class,
        "privileged_class": privileged_class,
        "source_origin": source_origin,
        "source_compatibility": compatibility,
        "selection_disposition": disposition,
        "semantic_read": bool(info["semantic_read"] and info["state_id"] <= max_semantic_state),
    }


def _root_seal_pointer(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"path": "", "exists": False, "seal_status": "NOT_SUPPLIED"}
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.exists():
        return {"path": str(root), "exists": False, "seal_status": "NOT_FOUND"}
    if not sums.is_file() or not sidecar.is_file():
        return {"path": str(root), "exists": True, "seal_status": "UNSEALED"}
    try:
        expected = f"{sha256_file(sums)}  SHA256SUMS"
        actual = sidecar.read_text(encoding="utf-8").strip()
        return {"path": str(root), "exists": True, "seal_status": "PASS" if actual == expected else "HOLD_SIDECAR"}
    except OSError as exc:
        return {"path": str(root), "exists": True, "seal_status": "HOLD_READ", "error": str(exc)}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    registry_rows, registry_summary = _read_registry(Path(args.registry_csv).resolve())
    registry = {row["canonical_parent_key"]: row for row in registry_rows}
    roots = [_parse_asset_root(value) for value in args.asset_root]
    root_results: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for label, root in roots:
        result = _scan_asset_root(label, root, registry, int(args.max_semantic_state))
        origin, compatibility = _source_origin(label)
        discovered_keys = {key for key, info in result["identity_rows"].items() if info["identity_root_relatives"]}
        extra_keys = sorted(set(result.get("discovered", {})) - set(registry))
        missing_keys = sorted(set(registry) - discovered_keys)
        duplicate_keys = sorted(key for key, info in result["identity_rows"].items() if len(info["identity_root_relatives"]) > 1)
        result_summary = {
            "label": label,
            "root": str(root),
            "status": result["status"],
            "source_origin": origin,
            "source_compatibility": compatibility,
            "file_count": result.get("file_count", 0),
            "total_bytes": result.get("total_bytes", 0),
            "identity_count": len(discovered_keys),
            "missing_identity_count": len(missing_keys),
            "extra_identity_count": len(extra_keys),
            "duplicate_identity_root_count": len(duplicate_keys),
            "missing_identity_examples": missing_keys[:20],
            "extra_identity_examples": extra_keys[:20],
            "duplicate_identity_examples": duplicate_keys[:20],
            "path_size_inventory_sha256": result.get("path_size_inventory_sha256", ""),
            "extension_counts": dict(result.get("extension_counts", {})),
            "symlink_count": result.get("symlink_count", 0),
            "protected_split_semantic_reads": 0,
            "fit_semantic_reads": sum(bool(info["semantic_read"]) for info in result["identity_rows"].values()),
        }
        counts = Counter()
        for key, row in registry.items():
            info = result["identity_rows"][key]
            classification = _classify(info, origin, compatibility, int(args.max_semantic_state))
            counts[classification["rgb_class"]] += 1
            counts[classification["logits_class"]] += 1
            counts[classification["privileged_class"]] += 1
            identity_rows.append(
                {
                    "root_label": label,
                    "root": str(root),
                    "canonical_parent_key": key,
                    "split": row["split"],
                    "state_id": row["state_id"],
                    "discovered": bool(info["identity_root_relatives"]),
                    "identity_root_relatives": ";".join(sorted(info["identity_root_relatives"])),
                    "file_count": info["file_count"],
                    "bytes": info["bytes"],
                    "image_count": len(info["image_files"]),
                    "video_count": len(info["video_files"]),
                    "rgb_class": classification["rgb_class"],
                    "rgb_alignment_status": classification["rgb_alignment_status"],
                    "logits_class": classification["logits_class"],
                    "privileged_class": classification["privileged_class"],
                    "policy_intent_direct": info["direct_policy_intent"],
                    "direct_logits_evidence": info["direct_logits"],
                    "language_evidence": info["language"],
                    "model_evidence": bool(info["model_files"] or info["model_pointer"]),
                    "teacher_or_privileged_fields_detected": info["teacher_fields"],
                    "content_field_examples": ";".join(sorted(info["sample_content_fields"])[:80]),
                    "source_schema": info["source_schema"],
                    "collector_commits": ";".join(sorted(info["collector_commits"])),
                    "source_commits": ";".join(sorted(info["source_commits"])),
                    "semantic_read": classification["semantic_read"],
                    "source_origin": origin,
                    "source_compatibility": compatibility,
                    "selection_disposition": classification["selection_disposition"],
                    "rgb_examples": ";".join(info["image_files"][:3]),
                    "policy_examples": ";".join(info["policy_files"][:3]),
                    "privileged_examples": ";".join(info["privileged_files"][:3]),
                }
            )
        result_summary["classification_counts"] = dict(counts)
        fit_infos = [info for info in result["identity_rows"].values() if info["state_id"] <= int(args.max_semantic_state)]
        result_summary["fit_source_schema_counts"] = dict(Counter(info["source_schema"] for info in fit_infos if info["source_schema"]))
        result_summary["fit_collector_commit_counts"] = dict(Counter(commit for info in fit_infos for commit in info["collector_commits"]))
        result_summary["fit_content_field_union"] = sorted({field for info in fit_infos for field in info["sample_content_fields"]})
        root_results.append(result_summary)
    official_rows = [row for row in identity_rows if row["source_compatibility"] == "OFFICIAL_V3_BOUND" and row["state_id"] < int(args.max_semantic_state) + 1]
    official_summary = {
        "fit_identity_count": len(official_rows),
        "fit_policy_intent_direct": sum(bool(row["policy_intent_direct"]) for row in official_rows),
        "fit_direct_logits": sum(bool(row["direct_logits_evidence"]) for row in official_rows),
        "fit_rgb_direct": sum(row["rgb_class"] == "RGB_DIRECT" for row in official_rows),
        "fit_rgb_video_aligned": sum(row["rgb_class"] == "RGB_VIDEO_ALIGNED" for row in official_rows),
        "fit_rgb_missing": sum(row["rgb_class"] == "RGB_MISSING" for row in official_rows),
        "fit_privileged_ready": sum(row["privileged_class"] == "PRIVILEGED_READY" for row in official_rows),
        "fit_teacher_or_privileged_field_leakage_sources": sum(row["teacher_or_privileged_fields_detected"] for row in official_rows),
    }
    return {
        "schema": SCHEMA,
        "status": "PASS_METADATA_ONLY",
        "captured_by": "read_only_asset_inventory",
        "registry": registry_summary,
        "official_s1_root": _root_seal_pointer(Path(args.official_s1_root).resolve() if args.official_s1_root else None),
        "asset_roots": root_results,
        "official_v3_fit_view": official_summary,
        "protected_split_semantic_reads": [],
        "source_mutation_count": 0,
        "model_inference_run": False,
        "replay_run": False,
        "teacher_materialization_run": False,
        "training_authorized": False,
        "attack_authorized": False,
        "selection_or_promotion_performed": False,
        "interpretation": {
            "RGB_DIRECT": "per-identity image files were found; alignment is reported separately",
            "RGB_VIDEO_ALIGNED": "video plus explicit step/alignment evidence was found",
            "RGB_MISSING": "no direct aligned image/video source was found in that root",
            "LOGITS_DIRECT": "raw policy-intent/top-logit telemetry was present in source records",
            "LOGITS_REINFERABLE": "RGB plus local model/language evidence exists, but no inference was run",
            "LOGITS_MISSING": "no direct logits or justified reinference path was found",
            "PRIVILEGED_READY": "per-step privileged/sidecar evidence is present; it remains Teacher-only",
            "PRIVILEGED_PARTIAL": "privileged pointers exist but completeness was not established",
            "PRIVILEGED_MISSING": "no privileged source evidence was found",
        },
    }, identity_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _seal(root: Path) -> None:
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    sums = "".join(f"{sha256_file(root / name)}  {name}\n" for name in names).encode("utf-8")
    _atomic_write(root / "SHA256SUMS", sums)
    _atomic_write(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n".encode("utf-8"))


def write_output(output: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        _atomic_write(staging / "summary.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        fields = [
            "root_label", "root", "canonical_parent_key", "split", "state_id", "discovered",
            "identity_root_relatives", "file_count", "bytes", "image_count", "video_count",
            "rgb_class", "rgb_alignment_status", "logits_class", "privileged_class",
            "policy_intent_direct", "direct_logits_evidence", "language_evidence", "model_evidence",
            "teacher_or_privileged_fields_detected", "content_field_examples", "source_schema", "collector_commits", "source_commits",
            "semantic_read", "source_origin", "source_compatibility", "selection_disposition",
            "rgb_examples", "policy_examples", "privileged_examples",
        ]
        _write_csv(staging / "identity_asset_rows.csv", rows, fields)
        _write_csv(staging / "source_inventory.csv", report["asset_roots"], [
            "label", "root", "status", "source_origin", "source_compatibility", "file_count", "total_bytes",
            "identity_count", "missing_identity_count", "extra_identity_count", "duplicate_identity_root_count",
            "path_size_inventory_sha256", "symlink_count", "fit_semantic_reads", "classification_counts",
        ])
        markdown = [
            "# CLEAN2000 artifact recovery audit",
            "",
            "This is a metadata-only, read-only inventory. No model inference, replay, Teacher materialization, training, or attack was run.",
            "",
            "```json",
            json.dumps(report, indent=2, sort_keys=True),
            "```",
            "",
            "RGB discovered in a non-Official-V3 root is reported as an unbound parallel source and is not promoted to a V5 input.",
            "The path-size inventory digest is not a content tree hash; existing source seals remain the authority for sealed Official V3 artifacts.",
        ]
        _atomic_write(staging / "audit_report.md", ("\n".join(markdown) + "\n").encode("utf-8"))
        _seal(staging)
        if output.exists():
            raise FileExistsError(f"output appeared during audit: {output}")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--official-s1-root")
    parser.add_argument("--asset-root", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--max-semantic-state", type=int, default=19)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report, rows = audit(args)
    write_output(Path(args.output_root).resolve(), report, rows)
    print(json.dumps({"schema": SCHEMA, "status": report["status"], "output_root": str(Path(args.output_root).resolve()), "asset_roots": report["asset_roots"], "official_v3_fit_view": report["official_v3_fit_view"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
