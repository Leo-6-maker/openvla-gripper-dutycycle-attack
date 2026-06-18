#!/usr/bin/env python3
"""Freeze a server-wide SC5 source census without building a training corpus.

The census is intentionally conservative:
- it scans roots for ``step_records.jsonl``;
- records manifest/source hashes and provenance fields;
- classifies each trajectory as candidate, conditional, OOD, or excluded;
- emits compact CSV/JSON/report artifacts.

It does not train models, calibrate Teacher thresholds, segment events, or
launch GPU/LIBERO jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except Exception:  # pragma: no cover - tests install pyyaml in the project env
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]

OBJECT_TASKS = (
    "alphabet_soup",
    "cream_cheese",
    "salad_dressing",
    "bbq_sauce",
    "ketchup",
    "tomato_sauce",
    "butter",
    "milk",
    "chocolate_pudding",
    "orange_juice",
)

PLACE_HINTS = ("bowl", "mug", "wine", "rack", "plate", "basket")
OOD_HINTS = ("drawer", "stove", "button", "turn", "push", "open_", "close_")
ATTACK_HINTS = ("vis", "attack", "oracle", "rand", "random", "pgd", "zero_margin")


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the census config")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames or []})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_output(args: list[str]) -> tuple[bool, str]:
    try:
        value = subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return True, value
    except Exception:
        return False, ""


def git_value(args: list[str]) -> str:
    ok, value = git_output(args)
    if ok:
        return value
    return ""


def git_branch_name() -> str:
    branch = git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch and branch != "HEAD":
        return branch
    if branch == "HEAD":
        return "DETACHED_HEAD"
    return git_value(["symbolic-ref", "--short", "HEAD"]) or "UNKNOWN_BRANCH"


def git_dirty_status() -> str:
    ok, value = git_output(["status", "--porcelain"])
    if not ok:
        return "GIT_STATUS_UNAVAILABLE"
    if not value:
        return "CLEAN"
    return f"DIRTY:{sha256_text(value)}"


def git_provenance_fields() -> dict[str, str]:
    head = git_value(["rev-parse", "HEAD"])
    if not head:
        return {
            "repo_head": "",
            "repo_branch": "",
            "repo_dirty": "GIT_STATUS_UNAVAILABLE",
            "repo_provenance": "GIT_PROVENANCE_UNAVAILABLE",
        }
    return {
        "repo_head": head,
        "repo_branch": git_branch_name(),
        "repo_dirty": git_dirty_status(),
        "repo_provenance": "PASS",
    }


def normalize_task(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("-", "_").replace(" ", "_")
    matches = [(text.index(task), task) for task in OBJECT_TASKS if task in text]
    if matches:
        return sorted(matches, key=lambda item: (item[0], item[1]))[0][1]
    return text


def infer_state_id(manifest: Mapping[str, Any], step_dir: Path) -> str:
    for key in ("state_id", "state", "init_state_id"):
        if key in manifest and manifest.get(key) not in (None, ""):
            return str(manifest.get(key))
    command = str(manifest.get("command", ""))
    match = re.search(r"--state_ids\s+([0-9]+)", command)
    if match:
        return match.group(1)
    for part in [step_dir.name, str(step_dir)]:
        match = re.search(r"(?:^|[_-])s(?:tate)?([0-9]+)(?:$|[_-])", part.lower())
        if match:
            return match.group(1)
        match = re.search(r"state([0-9]+)", part.lower())
        if match:
            return match.group(1)
    return ""


def infer_task(manifest: Mapping[str, Any], first_record: Mapping[str, Any], step_dir: Path) -> str:
    candidates = [
        manifest.get("task_name"),
        manifest.get("task"),
        manifest.get("task_id"),
        first_record.get("task_name"),
        first_record.get("task"),
        first_record.get("task_id"),
        manifest.get("command"),
        step_dir.name,
        str(step_dir),
    ]
    for cand in candidates:
        task = normalize_task(cand)
        if task:
            return task
    return "unknown"


def infer_suite(manifest: Mapping[str, Any], first_record: Mapping[str, Any], step_dir: Path) -> str:
    candidates = [
        manifest.get("suite"),
        first_record.get("suite"),
        manifest.get("task_id"),
        manifest.get("model_checkpoint_path"),
        manifest.get("command"),
        str(step_dir),
    ]
    text = " ".join(str(x or "").lower() for x in candidates)
    if "libero_object" in text:
        return "libero_object"
    if "libero_spatial" in text:
        return "libero_spatial"
    if "libero_goal" in text:
        return "libero_goal"
    if "libero_10" in text or "libero-10" in text:
        return "libero_10"
    if "libero_90" in text or "libero-90" in text:
        return "libero_90"
    return "unknown"


def truthy(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value in (1, "1", "true", "True", "success", "SUCCESS"):
        return "1"
    if value in (0, "0", "false", "False", "failed", "FAIL", "failed"):
        return "0"
    return ""


def infer_success(manifest: Mapping[str, Any], records: list[Mapping[str, Any]]) -> str:
    for key in ("success", "clean_success", "success_done", "success_check"):
        val = truthy(manifest.get(key))
        if val:
            return val
    status = str(manifest.get("status", "")).lower()
    if status == "success":
        return "1"
    if status in {"failed", "fail", "timeout"}:
        # Some failed manifests still contain success_check rows, so fall through.
        pass
    for row in reversed(records):
        for key in ("success_done", "success_check", "success"):
            val = truthy(row.get(key))
            if val == "1":
                return "1"
    if status in {"failed", "fail", "timeout"}:
        return "0"
    return ""


def infer_clean_status(manifest: Mapping[str, Any], records: list[Mapping[str, Any]], step_dir: Path) -> str:
    text = " ".join(
        str(x or "").lower()
        for x in [
            manifest.get("run_id"),
            manifest.get("trigger"),
            manifest.get("command"),
            manifest.get("attack_objective"),
            manifest.get("attack_config_path"),
            step_dir.name,
        ]
    )
    sampled = records[: min(len(records), 25)]
    if sampled:
        attack_active = any(bool(r.get("attack_active")) for r in sampled)
        attack_methods = {str(r.get("attack_method", "")).lower() for r in sampled}
        if attack_active:
            return "ATTACK_OR_INTERVENTION"
        if attack_methods and attack_methods <= {"", "none", "nan"}:
            return "CLEAN"
    if " clean" in f" {text}" or "_clean" in text or "trigger clean" in text:
        if not any(hint in text for hint in ATTACK_HINTS if hint != "attack"):
            return "CLEAN"
    if any(hint in text for hint in ATTACK_HINTS):
        return "ATTACK_OR_INTERVENTION"
    return "UNKNOWN_CLEAN_PROVENANCE"


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value in (None, ""):
        return False
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x)


def schema_status(records: list[Mapping[str, Any]], alias_config: Mapping[str, Any]) -> tuple[str, str]:
    if not records:
        return "NO_RECORDS", "no_records"
    aliases = alias_config.get("required_deployment_fields", {})
    if not isinstance(aliases, Mapping):
        return "BAD_ALIAS_CONFIG", "bad_alias_config"
    present_sources: list[str] = []
    missing: list[str] = []
    # Check first few records because some early rows may be partially initialized.
    sample = records[: min(10, len(records))]
    for field, spec in aliases.items():
        if isinstance(spec, dict):
            names = spec.get("aliases", [field])
        elif isinstance(spec, list):
            names = spec
        else:
            names = [str(spec)]
        found = False
        for row in sample:
            for name in names:
                if is_finite_number(row.get(str(name))):
                    found = True
                    present_sources.append(f"{field}:{name}")
                    break
            if found:
                break
        if not found:
            missing.append(str(field))
    if missing:
        return "MISSING_FIELDS", ",".join(missing)
    return "PASS", ";".join(present_sources)


def task_tier(task: str, suite: str, clean_status: str, success: str, schema: str) -> tuple[str, str]:
    lower = task.lower()
    if clean_status != "CLEAN":
        return "EXCLUDED_AUDIT_ONLY", clean_status
    if success != "1":
        return "EXCLUDED_AUDIT_ONLY", "CLEAN_FAIL_OR_UNKNOWN_SUCCESS"
    if schema != "PASS":
        return "EXCLUDED_AUDIT_ONLY", f"SCHEMA_{schema}"
    if suite == "libero_object" and task in OBJECT_TASKS:
        return "PRIMARY_SC5_POSITIVE_CANDIDATE", "LIBERO_OBJECT_SINGLE_OBJECT_CANDIDATE"
    if suite in {"libero_goal", "libero_90"} and (task in OBJECT_TASKS or any(h in lower for h in PLACE_HINTS)):
        return "CONDITIONAL_PLACE_CANDIDATE", "REQUIRES_OBJECT_TARGET_VALIDATION"
    if suite == "libero_10":
        return "CONDITIONAL_MULTI_STAGE_CANDIDATE", "REQUIRES_EVENT_SEGMENTATION"
    if any(h in lower for h in OOD_HINTS):
        return "OOD_ABSTAIN", "INCOMPATIBLE_ARTICULATION_OR_PUSH"
    return "EXCLUDED_AUDIT_ONLY", "UNKNOWN_TASK_OR_UNSUPPORTED_SUITE"


def read_manifest(step_dir: Path) -> tuple[str, dict[str, Any], str]:
    for name in ("run_manifest.json", "episode_manifest.json", "manifest.json", "summary.json"):
        path = step_dir / name
        if path.is_file():
            return str(path), read_json(path), sha256_file(path)
    return "", {}, ""


def read_records(path: Path, *, max_records_for_memory: int = 500) -> tuple[list[Mapping[str, Any]], int, str]:
    records: list[Mapping[str, Any]] = []
    errors = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
                if isinstance(row, Mapping) and len(records) < max_records_for_memory:
                    records.append(row)
            except Exception:
                errors += 1
    status = "PASS" if errors == 0 else f"JSON_ERRORS_{errors}"
    return records, errors, status


def should_skip_dir(path: Path, skip_substrings: Iterable[str]) -> bool:
    text = str(path)
    return any(part and part in text for part in skip_substrings)


def scan_roots(config: Mapping[str, Any], aliases: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    skip = list(config.get("exclude_dir_name_substrings", []) or [])
    for root_cfg in config.get("roots", []) or []:
        root = Path(str(root_cfg.get("path", "")))
        dirs_scanned = 0
        step_records = 0
        manifests = 0
        exists = root.is_dir()
        if exists:
            for dirpath, dirnames, filenames in os.walk(root):
                path = Path(dirpath)
                dirnames[:] = [d for d in dirnames if not should_skip_dir(path / d, skip)]
                if should_skip_dir(path, skip):
                    continue
                dirs_scanned += 1
                if "step_records.jsonl" not in filenames:
                    continue
                step_records += 1
                step_path = path / "step_records.jsonl"
                manifest_path, manifest, manifest_sha = read_manifest(path)
                if manifest_path:
                    manifests += 1
                file_size = step_path.stat().st_size
                source_sha = sha256_file(step_path) if file_size else ""
                records, json_errors, json_status = read_records(step_path)
                first_record = records[0] if records else {}
                task = infer_task(manifest, first_record, path)
                state_id = infer_state_id(manifest, path)
                suite = infer_suite(manifest, first_record, path)
                success = infer_success(manifest, records)
                clean_status = infer_clean_status(manifest, records, path)
                schema, schema_note = schema_status(records, aliases)
                tier, exclusion = task_tier(task, suite, clean_status, success, schema)
                episode_rows.append(
                    {
                        "episode_id": sha256_text(str(step_path))[:16],
                        "root_path": str(root),
                        "episode_dir": str(path),
                        "step_records_path": str(step_path),
                        "step_records_sha256": source_sha,
                        "step_records_size_bytes": file_size,
                        "json_status": json_status,
                        "manifest_path": manifest_path,
                        "manifest_sha256": manifest_sha,
                        "task": task,
                        "state_id": state_id,
                        "suite": suite,
                        "success": success,
                        "clean_status": clean_status,
                        "schema_status": schema,
                        "schema_note": schema_note,
                        "tier": tier,
                        "exclusion_reason": exclusion,
                        "run_id": manifest.get("run_id", ""),
                        "code_git_commit": manifest.get("code_git_commit", ""),
                        "cuda_visible_devices": manifest.get("cuda_visible_devices", ""),
                        "command_sha256": sha256_text(str(manifest.get("command", ""))) if manifest.get("command") else "",
                    }
                )
        root_rows.append(
            {
                "root_path": str(root),
                "role": root_cfg.get("role", ""),
                "exists": exists,
                "dirs_scanned": dirs_scanned,
                "step_records_found": step_records,
                "manifests_found": manifests,
            }
        )
    return root_rows, episode_rows


def summarize(config: Mapping[str, Any], root_rows: list[Mapping[str, Any]], episode_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    counts["directories_scanned"] = sum(int(r["dirs_scanned"]) for r in root_rows)
    counts["step_records"] = len(episode_rows)
    counts["manifests"] = sum(1 for r in episode_rows if r.get("manifest_path"))
    counts["known_clean_success"] = sum(1 for r in episode_rows if r.get("clean_status") == "CLEAN" and r.get("success") == "1")
    counts["clean_fail"] = sum(1 for r in episode_rows if r.get("clean_status") == "CLEAN" and r.get("success") == "0")
    counts["unknown_task_names"] = sum(1 for r in episode_rows if r.get("task") in {"", "unknown"})
    tier_counts = Counter(str(r.get("tier", "")) for r in episode_rows)
    exclusion_counts = Counter(str(r.get("exclusion_reason", "")) for r in episode_rows)
    historical = config.get("expected_historical_counts", {}) or {}
    drift = {}
    mapping = {
        "directories_scanned": "directories_scanned",
        "step_records": "step_records",
        "manifests": "manifests",
        "known_clean_success": "known_clean_success",
        "clean_fail": "clean_fail",
        "initially_unknown_task_names": "unknown_task_names",
    }
    for hist_key, current_key in mapping.items():
        if hist_key in historical:
            drift[hist_key] = int(counts[current_key]) - int(historical[hist_key])
    status = "SC5_SOURCE_CENSUS_FROZEN"
    if any(v != 0 for v in drift.values()):
        status = "SC5_SOURCE_CENSUS_FROZEN_WITH_CURRENT_SCAN_DRIFT"
    repo_fields = git_provenance_fields()
    return {
        "status": status,
        **repo_fields,
        "counts": dict(counts),
        "tier_counts": dict(tier_counts),
        "exclusion_counts": dict(exclusion_counts),
        "historical_expected_counts": historical,
        "current_minus_historical": drift,
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# V2 SC5 Source Census",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a source census only. It does not build a canonical corpus, train an MLP/TCN, run GPU jobs, or launch LIBERO.",
        "",
        "## Counts",
        "",
    ]
    for key, value in sorted(summary["counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Tier Counts", ""])
    for key, value in sorted(summary["tier_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Current Minus Historical", ""])
    for key, value in sorted(summary["current_minus_historical"].items()):
        lines.append(f"- {key}: {value:+d}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Allowed: source availability, provenance, schema and exclusion census.",
            "Forbidden: canonical usable count, expanded MLP pass/fail, TCN need, Student trigger success, VIS bridge success.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots_config", type=Path, default=REPO_ROOT / "configs" / "v2_sc5_data_roots.yaml")
    ap.add_argument("--schema_aliases", type=Path, default=REPO_ROOT / "configs" / "v2_sc5_schema_aliases.yaml")
    ap.add_argument("--tables_dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--artifacts_dir", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--reports_dir", type=Path, default=REPO_ROOT / "reports")
    args = ap.parse_args()

    config = read_yaml(args.roots_config)
    aliases = read_yaml(args.schema_aliases)
    root_rows, episode_rows = scan_roots(config, aliases)
    summary = summarize(config, root_rows, episode_rows)
    exclusion_rows = [
        {
            "exclusion_reason": reason,
            "count": count,
        }
        for reason, count in sorted(summary["exclusion_counts"].items())
    ]
    write_csv(args.tables_dir / "v2_sc5_source_roots.csv", root_rows)
    write_csv(args.tables_dir / "v2_sc5_episode_inventory.csv", episode_rows)
    write_csv(args.tables_dir / "v2_sc5_exclusion_reasons.csv", exclusion_rows)
    write_json(args.artifacts_dir / "v2_sc5_source_inventory.json", summary)
    write_report(args.reports_dir / "V2_SC5_SOURCE_CENSUS.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
