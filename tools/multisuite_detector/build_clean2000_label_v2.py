#!/usr/bin/env python3
"""Hardened entrypoint for CLEAN2000 Label V2 construction.

The reviewed implementation body is loaded from its immutable parent commit.
This entrypoint applies the final Gate-A1 closeout rules before delegating to
that implementation. Formal execution remains separately authorized.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path


_CORE_COMMIT = "35e17855c57277f866142c34129425e0259ece5b"
_CORE_REPO_PATH = "tools/multisuite_detector/build_clean2000_label_v2.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    _CORE_SOURCE = subprocess.check_output(
        ["git", "show", f"{_CORE_COMMIT}:{_CORE_REPO_PATH}"],
        cwd=_REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception as exc:
    raise RuntimeError(
        f"unable to load immutable Label V2 core {_CORE_COMMIT}:{_CORE_REPO_PATH}"
    ) from exc

_core = types.ModuleType("_label_v2_core")
_core.__file__ = __file__
exec(compile(_CORE_SOURCE, __file__, "exec"), _core.__dict__)

# Re-export the reviewed API so existing imports/tests retain the same interface.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


MANUAL_FALLBACK_ORDER = {
    "positive_clean_success": [
        "positive_clean_success",
        "eligible_no_event",
        "failure_or_boundary",
        "abstention_or_ineligible",
    ],
    "eligible_no_event": [
        "eligible_no_event",
        "failure_or_boundary",
        "abstention_or_ineligible",
        "positive_clean_success",
    ],
    "failure_or_boundary": [
        "failure_or_boundary",
        "abstention_or_ineligible",
        "positive_clean_success",
        "eligible_no_event",
    ],
    "abstention_or_ineligible": [
        "abstention_or_ineligible",
        "positive_clean_success",
        "eligible_no_event",
        "failure_or_boundary",
    ],
}
_core.MANUAL_FALLBACK_ORDER = MANUAL_FALLBACK_ORDER

_original_validate_source_disposition_invariants = _core.validate_source_disposition_invariants


def validate_source_disposition_invariants(
    availability: dict[str, str],
    census: dict[str, str],
    clean_success: bool,
    event_present: bool,
) -> None:
    """Enforce a total, cohort-specific source disposition partition."""

    _original_validate_source_disposition_invariants(
        availability,
        census,
        clean_success,
        event_present,
    )
    episode = census["episode_key"]
    cohort = census["cohort_class"]
    source_no_event = _core.bool_field(availability, "source_no_event")
    explicit_abstention = _core.bool_field(
        availability,
        "source_explicit_abstention",
    )
    clean_failure_no_event = _core.bool_field(
        availability,
        "source_clean_failure_no_event",
    )

    if event_present:
        if source_no_event or explicit_abstention or clean_failure_no_event:
            _core.fail(f"{episode}: positive event has a no-event disposition flag")
        return

    if cohort == "PRIMARY_SUCCESS_ELIGIBLE":
        if not source_no_event or explicit_abstention or clean_failure_no_event:
            _core.fail(
                f"{episode}: primary-success no-event requires the generic "
                "source_no_event disposition only"
            )
        return

    if cohort == "ELIGIBLE_CLEAN_FAILURE":
        if not clean_failure_no_event or explicit_abstention:
            _core.fail(
                f"{episode}: eligible clean-failure no-event requires "
                "source_clean_failure_no_event"
            )
        return

    if cohort == "MECHANISM_INELIGIBLE_ABSTENTION":
        if not explicit_abstention or clean_failure_no_event:
            _core.fail(
                f"{episode}: mechanism-ineligible row requires explicit source "
                "abstention"
            )
        return

    _core.fail(f"{episode}: unsupported disposition cohort: {cohort}")


_core.validate_source_disposition_invariants = validate_source_disposition_invariants


def _arg_value(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_disposition_summary(label_path: Path) -> dict[str, object]:
    counts = {
        "primary_success_positive": 0,
        "primary_success_no_event": 0,
        "eligible_clean_failure_positive": 0,
        "eligible_clean_failure_no_event": 0,
        "mechanism_ineligible_abstention": 0,
    }
    with label_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        cohort = row["cohort_class"]
        event_present = row["event_present"] == "true"
        if cohort == "PRIMARY_SUCCESS_ELIGIBLE":
            key = (
                "primary_success_positive"
                if event_present
                else "primary_success_no_event"
            )
        elif cohort == "ELIGIBLE_CLEAN_FAILURE":
            key = (
                "eligible_clean_failure_positive"
                if event_present
                else "eligible_clean_failure_no_event"
            )
        elif cohort == "MECHANISM_INELIGIBLE_ABSTENTION":
            key = "mechanism_ineligible_abstention"
        else:
            raise RuntimeError(f"unexpected cohort in formal output: {cohort}")
        counts[key] += 1
    return {
        "unexplained_disposition_rows": 0,
        "disposition_subtype_counts": counts,
    }


def _postprocess_formal_outputs(output_root: Path) -> None:
    label_path = output_root / "label_v2.csv"
    summary_path = output_root / "validation_summary.json"
    manifest_path = output_root / "build_manifest.json"
    manual_path = output_root / "manual_audit_sample_manifest.csv"
    sums_path = output_root / "SHA256SUMS"

    disposition = _formal_disposition_summary(label_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(disposition)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manual_fallback_policy"] = MANUAL_FALLBACK_ORDER
    manifest["source_disposition_closure"] = disposition
    manifest["builder_entrypoint"] = str(Path(__file__).resolve())
    manifest["core_source_commit"] = _CORE_COMMIT
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sums = [
        f"{_sha256_file(path)}  {path.name}"
        for path in [label_path, manifest_path, summary_path, manual_path]
    ]
    sums_path.write_text("\n".join(sums) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    result = _core.main(args)
    if result != 0:
        return result

    mode = _arg_value(args, "--mode")
    if mode == "formal-ledger-build":
        output_text = _arg_value(args, "--output-root")
        if output_text is None:
            raise RuntimeError("formal output root disappeared after validation")
        _postprocess_formal_outputs(Path(output_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
