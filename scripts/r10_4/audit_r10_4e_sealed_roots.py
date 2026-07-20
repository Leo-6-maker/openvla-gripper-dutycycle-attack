#!/usr/bin/env python3
"""Independent fail-closed auditor for the R10.4E E-R3a output.

The auditor verifies:
- the panel aggregate SHA256 seal and every nested digest;
- the external task00 reuse binding and original sealed root;
- the task01 fresh episode root, JSONL invariants, and metadata bindings;
- the append-only ledger revision hash chain;
- the final ledger and panel summary against independently supplied bindings.

The audit report must be written outside the sealed panel root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
TASK00 = "libero_10/task_00/state_20"
TASK01 = "libero_10/task_01/state_20"
PHASE = "E_R3A_TASK01_CANARY"
PASS_STATUSES = {"PASS_RUNTIME_NO_EMIT", "PASS_RUNTIME_EMIT_OBSERVED"}
VALID_TERMINATIONS = {
    "SUCCESS_TERMINATION",
    "HORIZON_TERMINATION",
    "FULL_LOOP_TASK_FAILURE",
}
FRESH_REQUIRED = {
    "step_records.jsonl",
    "detector_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
    "episode_metadata.json",
    "episode_summary.json",
    "runtime_audit.json",
    "ROOT_SEAL_RECEIPT.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R10.4E E-R3a sealed output")
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--task00-root", required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-authorization-comment-id", required=True, type=int)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL_ROW_NOT_OBJECT:{path.name}:{line_number}")
        rows.append(value)
    return rows


def _safe_relative_name(name: str) -> bool:
    candidate = Path(name)
    return bool(name) and not candidate.is_absolute() and ".." not in candidate.parts


def verify_sealed_tree(root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"root": str(root), "checks": []}
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir():
        report["checks"].append(("ROOT_EXISTS", False, str(root)))
        report["valid"] = False
        return report
    if not sums.is_file() or not sidecar.is_file():
        report["checks"].append(("SEAL_FILES_EXIST", False, "missing SHA256SUMS or sidecar"))
        report["valid"] = False
        return report

    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    sidecar_ok = len(tokens) == 2 and tokens[1] == "SHA256SUMS" and SHA256_RE.fullmatch(tokens[0])
    report["checks"].append(("SEAL_SIDECAR_FORMAT", bool(sidecar_ok), tokens[:2]))
    actual_sums_sha = sha256_file(sums)
    report["sha256sums_sha256"] = actual_sums_sha
    report["checks"].append(
        ("SEAL_SIDECAR_DIGEST", bool(sidecar_ok and tokens[0] == actual_sums_sha), actual_sums_sha)
    )

    listed: dict[str, str] = {}
    parse_errors: list[str] = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or not SHA256_RE.fullmatch(pieces[0]) or not _safe_relative_name(pieces[1].strip()):
            parse_errors.append(line[:160])
            continue
        name = pieces[1].strip()
        if name in listed:
            parse_errors.append(f"duplicate:{name}")
            continue
        listed[name] = pieces[0]
    report["checks"].append(("SHA256SUMS_PARSE", not parse_errors, parse_errors[:5]))

    mismatches: list[str] = []
    for name, expected in listed.items():
        path = root / name
        if not path.is_file():
            mismatches.append(f"missing:{name}")
        else:
            actual = sha256_file(path)
            if actual != expected:
                mismatches.append(f"digest:{name}:{actual}")
    report["checks"].append(("ALL_LISTED_DIGESTS", not mismatches, mismatches[:5]))

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path not in {sums, sidecar}
    }
    extra = sorted(actual_files - set(listed))
    missing = sorted(set(listed) - actual_files)
    report["checks"].append(
        ("FILE_SET_EXACT", not extra and not missing, {"extra": extra[:5], "missing": missing[:5]})
    )
    report["listed_files"] = sorted(listed)
    report["valid"] = all(ok for _, ok, *_ in report["checks"])
    return report


def audit_external_task00(root: Path) -> dict[str, Any]:
    report = verify_sealed_tree(root)
    report["type"] = "external_task00"
    summary_path = root / "episode_summary.json"
    try:
        summary = read_json(summary_path)
        report["identity"] = summary.get("identity")
        report["status"] = summary.get("status")
        report["n_steps"] = summary.get("n_steps")
        report["emit_count"] = summary.get("emit_count")
        report["summary_sha256"] = sha256_file(summary_path)
        report["checks"].append(("TASK00_IDENTITY", summary.get("identity") == TASK00, summary.get("identity")))
        report["checks"].append(("TASK00_STATUS", summary.get("status") in PASS_STATUSES, summary.get("status")))
    except Exception as exc:
        report["checks"].append(("TASK00_SUMMARY", False, f"{type(exc).__name__}:{exc}"))
    report["valid"] = all(ok for _, ok, *_ in report["checks"])
    return report


def audit_reuse_root(root: Path, external: dict[str, Any], expected_external_root: Path) -> dict[str, Any]:
    report = verify_sealed_tree(root)
    report["type"] = "reuse_binding"
    expected_local = {"REUSE_BINDING.json"}
    local_content = set(report.get("listed_files", []))
    report["checks"].append(("REUSE_FILE_SET", local_content == expected_local, sorted(local_content)))
    try:
        binding = read_json(root / "REUSE_BINDING.json")
        report["identity"] = binding.get("identity")
        checks = {
            "REUSE_SCHEMA": binding.get("schema") == "R10_4E_TASK00_REUSE_BINDING_V1",
            "REUSE_IDENTITY": binding.get("identity") == TASK00,
            "REUSE_EXTERNAL_PATH": binding.get("external_root") == str(expected_external_root.resolve()),
            "REUSE_EXTERNAL_SEAL": binding.get("external_sha256sums_sha256")
            == external.get("sha256sums_sha256"),
            "REUSE_EXTERNAL_SUMMARY": binding.get("external_summary_sha256")
            == external.get("summary_sha256"),
            "REUSE_STATUS": binding.get("original_status") == external.get("status"),
            "REUSE_STEPS": binding.get("n_steps") == external.get("n_steps"),
            "REUSE_EMITS": binding.get("emit_count") == external.get("emit_count"),
        }
        for name, ok in checks.items():
            report["checks"].append((name, ok, binding.get(name.lower())))
    except Exception as exc:
        report["checks"].append(("REUSE_BINDING_PARSE", False, f"{type(exc).__name__}:{exc}"))
    report["valid"] = all(ok for _, ok, *_ in report["checks"])
    return report


def _all_finite_25d(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 25:
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except Exception:
        return False


def audit_fresh_task01(
    root: Path,
    *,
    expected_head: str,
    expected_comment_id: int,
    expected_receipt_sha: str,
) -> dict[str, Any]:
    report = verify_sealed_tree(root)
    report["type"] = "fresh_task01"
    listed = set(report.get("listed_files", []))
    report["checks"].append(("FRESH_FILE_SET", listed == FRESH_REQUIRED, sorted(listed)))

    try:
        summary = read_json(root / "episode_summary.json")
        metadata = read_json(root / "episode_metadata.json")
        runtime_audit = read_json(root / "runtime_audit.json")
        root_receipt = read_json(root / "ROOT_SEAL_RECEIPT.json")
        steps = parse_jsonl(root / "step_records.jsonl")
        detectors = parse_jsonl(root / "detector_records.jsonl")
        privileged = parse_jsonl(root / "privileged_teacher_sidecar.jsonl")
    except Exception as exc:
        report["checks"].append(("FRESH_CONTENT_PARSE", False, f"{type(exc).__name__}:{exc}"))
        report["valid"] = False
        return report

    report["identity"] = summary.get("identity")
    report["status"] = summary.get("status")
    report["n_steps"] = summary.get("n_steps")
    report["emit_count"] = summary.get("emit_count")
    checks = {
        "SUMMARY_SCHEMA": summary.get("schema") == "R10_4E_SINGLE_EPISODE_PASSIVE_RESULT_V1",
        "SUMMARY_IDENTITY": summary.get("identity") == TASK01,
        "SUMMARY_STATUS": summary.get("status") in PASS_STATUSES,
        "SUMMARY_TERMINATION": summary.get("termination_reason") in VALID_TERMINATIONS,
        "SUMMARY_NO_VIOLATIONS": summary.get("violations") == [],
        "SUMMARY_NO_MUTATION": summary.get("action_mutation") is False,
        "SUMMARY_PRIVILEGED_ISOLATED": summary.get("privileged_runtime_input") is False,
        "STEP_COUNT": len(steps) == summary.get("n_steps"),
        "DETECTOR_COUNT": len(detectors) == summary.get("n_steps"),
        "PRIVILEGED_COUNT": len(privileged) == summary.get("n_steps"),
        "STEP_INDEX_CONTIGUOUS": [row.get("step") for row in steps] == list(range(len(steps))),
        "DETECTOR_INDEX_CONTIGUOUS": [row.get("step") for row in detectors] == list(range(len(detectors))),
        "GENERATION_EXACT_ONE": all(
            isinstance(row.get("generation_passes_per_step"), int)
            and not isinstance(row.get("generation_passes_per_step"), bool)
            and row.get("generation_passes_per_step") == 1
            for row in steps
        ),
        "ACTION_ZERO_ERROR": all(float(row.get("action_max_abs_error", -1.0)) == 0.0 for row in steps),
        "ACTION_EXACT_COPY": all(
            row.get("clean_env_action_7d") == row.get("executed_action_7d") for row in steps
        ),
        "FEATURES_VALID_25D": all(_all_finite_25d(row.get("features_25d")) for row in steps),
        "EMIT_COUNT_MATCH": sum(1 for row in detectors if row.get("emit") is True)
        == summary.get("emit_count"),
        "MAX_ONE_EMIT": int(summary.get("emit_count", -1)) in {0, 1},
        "PRIVILEGED_NOT_INPUT": all(row.get("detector_input") is False for row in privileged),
        "METADATA_SCHEMA": metadata.get("schema") == "R10_4E_SINGLE_EPISODE_PASSIVE_METADATA_V1",
        "METADATA_IDENTITY": metadata.get("identity") == TASK01,
        "METADATA_PARENT": metadata.get("parent", {}).get("identity") == TASK01,
        "METADATA_HEAD": metadata.get("source_commit") == expected_head,
        "METADATA_RECEIPT": metadata.get("panel_receipt_sha256") == expected_receipt_sha,
        "METADATA_COMMENT": metadata.get("authorization_comment_id") == expected_comment_id,
        "METADATA_NO_MUTATION": metadata.get("action_mutation") is False,
        "METADATA_NO_ATTACK": all(
            metadata.get(key) is False
            for key in (
                "attack_enabled",
                "command_open_enabled",
                "visual_attack_enabled",
                "random_attack_enabled",
                "privileged_runtime_input",
            )
        ),
        "RUNTIME_AUDIT_VALID": runtime_audit.get("runtime_valid") is True,
        "RUNTIME_AUDIT_STATUS": runtime_audit.get("status") == summary.get("status"),
        "RUNTIME_AUDIT_TERMINATION": runtime_audit.get("termination_reason")
        == summary.get("termination_reason"),
        "RUNTIME_AUDIT_NO_MUTATION": runtime_audit.get("action_mutation") is False,
        "RUNTIME_AUDIT_PRIVILEGED": runtime_audit.get("privileged_runtime_input") is False,
        "ROOT_RECEIPT_SCHEMA": root_receipt.get("schema") == "R10_4E_ROOT_SEAL_RECEIPT_V1",
        "ROOT_RECEIPT_IDENTITY": root_receipt.get("identity") == TASK01,
        "ROOT_RECEIPT_HEAD": root_receipt.get("source_commit") == expected_head,
        "ROOT_RECEIPT_AUTH": root_receipt.get("panel_receipt_sha256") == expected_receipt_sha,
    }
    for name, ok in checks.items():
        report["checks"].append((name, ok, None))
    report["valid"] = all(ok for _, ok, *_ in report["checks"])
    return report


def audit_ledger_chain(panel_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"checks": []}
    revision_files = sorted(panel_root.glob("panel_ledger_rev*.json"))
    report["checks"].append(("LEDGER_REVISIONS_PRESENT", bool(revision_files), len(revision_files)))
    previous_sha: str | None = None
    latest: dict[str, Any] | None = None
    for index, path in enumerate(revision_files):
        try:
            value = read_json(path)
        except Exception as exc:
            report["checks"].append(("LEDGER_REVISION_PARSE", False, f"{path.name}:{exc}"))
            continue
        expected_name = f"panel_ledger_rev{index:04d}.json"
        report["checks"].append(("LEDGER_FILENAME", path.name == expected_name, path.name))
        report["checks"].append(("LEDGER_SCHEMA", value.get("schema") == "R10_4E_PANEL_LEDGER_REVISION_V1", path.name))
        report["checks"].append(("LEDGER_REVISION", value.get("revision") == index, path.name))
        report["checks"].append(("LEDGER_PREVIOUS_SHA", value.get("previous_ledger_sha256") == previous_sha, path.name))
        report["checks"].append(("LEDGER_COUNT", value.get("n_attempts") == len(value.get("attempts", [])), path.name))
        previous_sha = sha256_file(path)
        latest = value

    try:
        final = read_json(panel_root / "panel_ledger.json")
        report["checks"].append(("FINAL_LEDGER_SCHEMA", final.get("schema") == "R10_4E_PANEL_LEDGER_V1", None))
        report["checks"].append(("FINAL_LEDGER_REVISION", latest is not None and final.get("revision") == latest.get("revision"), None))
        report["checks"].append(("FINAL_LEDGER_CHAIN", final.get("previous_ledger_sha256") == previous_sha, None))
        report["checks"].append(("FINAL_LEDGER_ATTEMPTS", latest is not None and final.get("attempts") == latest.get("attempts"), None))
        attempts = final.get("attempts", [])
        report["checks"].append(("FINAL_LEDGER_COUNT", final.get("n_attempts") == len(attempts) == 2, None))
        report["checks"].append(("FINAL_LEDGER_IDENTITIES", [row.get("identity") for row in attempts] == [TASK00, TASK01], None))
        report["checks"].append(("FINAL_LEDGER_REUSE", attempts and attempts[0].get("reuse") is True, None))
        report["checks"].append(("FINAL_LEDGER_FRESH_PASS", len(attempts) == 2 and attempts[1].get("status") in PASS_STATUSES, None))
        report["checks"].append(("FINAL_LEDGER_PANEL_OK", final.get("panel_ok") is True and final.get("all_runtime_valid") is True, None))
        report["final"] = final
    except Exception as exc:
        report["checks"].append(("FINAL_LEDGER_PARSE", False, f"{type(exc).__name__}:{exc}"))
    report["valid"] = all(ok for _, ok, *_ in report["checks"])
    return report


def main() -> int:
    args = parse_args()
    if not HEAD_RE.fullmatch(args.expected_head):
        raise SystemExit("EXPECTED_HEAD_INVALID")
    if not SHA256_RE.fullmatch(args.expected_receipt_sha256):
        raise SystemExit("EXPECTED_RECEIPT_SHA_INVALID")
    if args.expected_authorization_comment_id <= 0:
        raise SystemExit("EXPECTED_COMMENT_ID_INVALID")

    panel_root = args.panel_root.resolve()
    task00_root = args.task00_root.resolve()
    output = args.output.resolve()
    if panel_root == output or panel_root in output.parents:
        raise SystemExit("AUDIT_OUTPUT_MUST_BE_OUTSIDE_PANEL_ROOT")

    external = audit_external_task00(task00_root)
    aggregate = verify_sealed_tree(panel_root)
    reuse = audit_reuse_root(panel_root / TASK00.replace("/", "_"), external, task00_root)
    fresh = audit_fresh_task01(
        panel_root / TASK01.replace("/", "_"),
        expected_head=args.expected_head,
        expected_comment_id=args.expected_authorization_comment_id,
        expected_receipt_sha=args.expected_receipt_sha256,
    )
    ledger = audit_ledger_chain(panel_root)

    summary_checks: list[tuple[str, bool, Any]] = []
    try:
        summary = read_json(panel_root / "panel_summary.json")
        summary_checks.extend(
            [
                ("PANEL_NAME", summary.get("panel") == "R10_4E", summary.get("panel")),
                ("PANEL_PHASE", summary.get("phase") == PHASE, summary.get("phase")),
                ("PANEL_HEAD", summary.get("source_commit") == args.expected_head, summary.get("source_commit")),
                ("PANEL_RECEIPT", summary.get("panel_receipt_sha256") == args.expected_receipt_sha256, summary.get("panel_receipt_sha256")),
                ("PANEL_COMMENT", summary.get("authorization_comment_id") == args.expected_authorization_comment_id, summary.get("authorization_comment_id")),
                ("PANEL_COUNTS", summary.get("n_tasks_attempted") == 2 and summary.get("n_reuse") == 1 and summary.get("n_fresh") == 1, None),
                ("PANEL_OK", summary.get("panel_ok") is True and summary.get("all_runtime_valid") is True, None),
                ("PANEL_TASKS", [row.get("identity") for row in summary.get("per_task", [])] == [TASK00, TASK01], None),
            ]
        )
    except Exception as exc:
        summary_checks.append(("PANEL_SUMMARY_PARSE", False, f"{type(exc).__name__}:{exc}"))

    top_level_expected = {
        TASK00.replace("/", "_"),
        TASK01.replace("/", "_"),
    }
    actual_dirs = {path.name for path in panel_root.iterdir() if path.is_dir() and not path.name.startswith(".")}
    structure_ok = actual_dirs == top_level_expected

    components = {
        "external_task00": external,
        "panel_aggregate_seal": aggregate,
        "reuse_binding": reuse,
        "fresh_task01": fresh,
        "ledger_chain": ledger,
        "panel_summary": {"checks": summary_checks, "valid": all(ok for _, ok, *_ in summary_checks)},
        "panel_structure": {
            "checks": [("EXACT_EPISODE_DIRS", structure_ok, sorted(actual_dirs))],
            "valid": structure_ok,
        },
    }
    overall = all(component.get("valid", False) for component in components.values())
    report = {
        "schema": "R10_4E_E_R3A_INDEPENDENT_AUDIT_V1",
        "panel_root": str(panel_root),
        "task00_root": str(task00_root),
        "expected_head": args.expected_head,
        "expected_authorization_comment_id": args.expected_authorization_comment_id,
        "expected_receipt_sha256": args.expected_receipt_sha256,
        "overall": "PASS" if overall else "FAIL",
        "components": components,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "output": str(output)}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
