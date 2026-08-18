"""CPU-only T1-D0R timing and fresh-parent authority audit.

This module reads only frozen metadata/identity registries.  It never imports a
model library, opens a GPU, steps an environment, or consumes an outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
KEY_RE = re.compile(r"^(libero_(?:10|goal|object|spatial))/task_(\d{2})/state_(\d{2})$")
DIR_KEY_RE = re.compile(r"(?:^|_)(libero_(?:10|goal|object|spatial))_task(\d{2})_state_(\d{2})(?:_|$)")


class AuthorityError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def identity_digest(keys: Iterable[str]) -> str:
    """Stable digest: sorted canonical keys joined without a trailing newline."""
    return sha256_bytes("\n".join(sorted(keys)).encode("utf-8"))


def rank_sha256(salt: str, key: str) -> str:
    return sha256_bytes(f"{salt}::{key}".encode("utf-8"))


def parse_key(value: Any) -> str:
    if not isinstance(value, str):
        raise AuthorityError(f"IDENTITY_NOT_STRING:{value!r}")
    match = KEY_RE.fullmatch(value)
    if not match:
        raise AuthorityError(f"NON_CANONICAL_IDENTITY:{value}")
    suite, task, state = match.groups()
    if not 0 <= int(task) <= 9 or not 20 <= int(state) <= 49:
        raise AuthorityError(f"IDENTITY_OUTSIDE_G10_CORRIDOR:{value}")
    return value


def get_path(value: Any, path: Sequence[str]) -> Any:
    current = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise AuthorityError(f"JSON_PATH_MISSING:{'.'.join(path)}")
        current = current[part]
    return current


def require_file(spec: Mapping[str, Any]) -> tuple[Path, str]:
    path = Path(str(spec["path"]))
    if not path.is_file():
        raise AuthorityError(f"SOURCE_FILE_MISSING:{path}")
    actual = sha256_file(path)
    expected = str(spec.get("sha256", ""))
    if expected and actual != expected:
        raise AuthorityError(f"SOURCE_SHA256_MISMATCH:{spec['name']}:{actual}!={expected}")
    return path, actual


def list_json_identities(spec: Mapping[str, Any], universe: set[str] | None = None) -> tuple[set[str], dict[str, Any]]:
    path, actual = require_file(spec)
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = get_path(payload, tuple(spec.get("json_path", ())))
    if not isinstance(values, list):
        raise AuthorityError(f"IDENTITY_LIST_REQUIRED:{spec['name']}")
    field = spec.get("item_field")
    keys: list[str] = []
    for item in values:
        value = item if field is None else get_path(item, tuple(str(field).split(".")))
        keys.append(parse_key(value))
    if len(keys) != len(set(keys)):
        raise AuthorityError(f"DUPLICATE_IDENTITIES:{spec['name']}")
    key_set = set(keys)
    if universe is not None and not key_set <= universe:
        unknown = sorted(key_set - universe)
        raise AuthorityError(f"IDENTITY_JOIN_OUTSIDE_G10:{spec['name']}:{unknown[:3]}")
    return key_set, {
        "name": spec["name"],
        "kind": spec["kind"],
        "path": str(path),
        "sha256": actual,
        "json_path": list(spec.get("json_path", ())),
        "item_field": field,
        "identity_count": len(key_set),
        "identity_digest": identity_digest(key_set),
    }


def parse_directory_identity(name: str) -> str | None:
    match = DIR_KEY_RE.search(name)
    if not match:
        return None
    suite, task, state = match.groups()
    return parse_key(f"{suite}/task_{task}/state_{state}")


def list_directory_identities(spec: Mapping[str, Any], universe: set[str] | None = None) -> tuple[set[str], dict[str, Any]]:
    base = Path(str(spec["base_path"]))
    if not base.is_dir():
        raise AuthorityError(f"SOURCE_DIRECTORY_MISSING:{base}")
    names = sorted(path.name for path in base.glob(str(spec["glob"])) if path.is_dir())
    listing_sha = identity_digest(names)
    expected = str(spec.get("listing_sha256", ""))
    if expected and listing_sha != expected:
        raise AuthorityError(f"DIRECTORY_LISTING_SHA256_MISMATCH:{listing_sha}!={expected}")
    ignored: list[str] = []
    keys: list[str] = []
    ignored_patterns = [re.compile(pattern) for pattern in spec.get("ignored_name_regexes", ())]
    for name in names:
        key = parse_directory_identity(name)
        if key is None:
            if any(pattern.fullmatch(name) for pattern in ignored_patterns):
                ignored.append(name)
                continue
            raise AuthorityError(f"DIRECTORY_IDENTITY_AMBIGUOUS:{name}")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise AuthorityError(f"DUPLICATE_DIRECTORY_IDENTITIES:{spec['name']}")
    key_set = set(keys)
    if universe is not None and not key_set <= universe:
        raise AuthorityError(f"DIRECTORY_IDENTITY_JOIN_OUTSIDE_G10:{sorted(key_set - universe)[:3]}")
    return key_set, {
        "name": spec["name"],
        "kind": spec["kind"],
        "base_path": str(base),
        "glob": str(spec["glob"]),
        "directory_count": len(names),
        "listing_sha256": listing_sha,
        "identity_count": len(key_set),
        "identity_digest": identity_digest(key_set),
        "ignored_noncanonical_directories": ignored,
    }


def load_source(spec: Mapping[str, Any], universe: set[str] | None = None) -> tuple[set[str], dict[str, Any]]:
    if spec.get("kind") == "directory_identity_listing":
        return list_directory_identities(spec, universe)
    return list_json_identities(spec, universe)


def parse_timing_source(protocol: Mapping[str, Any]) -> dict[str, Any]:
    timing = protocol["timing_freeze"]
    source = timing["scheduler_source"]
    path, actual = require_file(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = {
        "status": payload.get("status"),
        "attack_enabled": payload.get("attack_enabled"),
        "t5_steps": payload.get("t5_steps", payload.get("T5")),
        "h_phys": payload.get("h_phys", payload.get("H_phys")),
        "one_shot": payload.get("one_shot"),
        "explicit_attack_start_field_present": any(
            key in payload for key in ("attack_start_step", "attack_start_offset", "attack_anchor")
        ),
    }
    errors: list[str] = []
    if observed["status"] != "FROZEN":
        errors.append("SCHEDULER_NOT_FROZEN")
    if observed["attack_enabled"] is not False:
        errors.append("SCHEDULER_ATTACK_ENABLED")
    if observed["t5_steps"] != 5 or observed["h_phys"] != 10 or observed["one_shot"] is not True:
        errors.append("SCHEDULER_T5_HORIZON_ONE_SHOT_MISMATCH")
    if observed["explicit_attack_start_field_present"]:
        errors.append("HISTORICAL_ATTACK_START_FIELD_UNEXPECTED")
    return {
        "path": str(path),
        "sha256": actual,
        "observed_fields": observed,
        "historical_attack_start_identifiable": False,
        "errors": errors,
    }


def local_binding_hashes(protocol: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for binding in protocol.get("local_bindings", []):
        relative = Path(str(binding["path"]))
        path = repo_root / relative
        if not path.is_file():
            raise AuthorityError(f"LOCAL_BINDING_MISSING:{relative}")
        actual = sha256_file(path)
        expected = str(binding["sha256"])
        if actual != expected:
            raise AuthorityError(f"LOCAL_BINDING_SHA256_MISMATCH:{relative}:{actual}!={expected}")
        rows[str(relative)] = {"sha256": actual, "role": binding.get("role")}
    return rows


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def derive_population(
    g10_keys: Iterable[str],
    exclusion_sets: Mapping[str, set[str]],
    salt: str,
    suites: Sequence[str] = SUITES,
    tasks: Iterable[int] = range(10),
) -> dict[str, Any]:
    g10 = sorted({parse_key(key) for key in g10_keys})
    if len(g10) != len(set(g10)):
        raise AuthorityError("G10_DUPLICATE_IDENTITIES")
    union = set().union(*(set(values) for values in exclusion_sets.values())) if exclusion_sets else set()
    if not union <= set(g10):
        raise AuthorityError(f"EXCLUSION_UNION_OUTSIDE_G10:{sorted(union - set(g10))[:3]}")
    rows: list[dict[str, Any]] = []
    for key in g10:
        match = KEY_RE.fullmatch(key)
        assert match is not None
        suite, task, state = match.groups()
        flags = {name: key in values for name, values in sorted(exclusion_sets.items())}
        fresh = not any(flags.values())
        rows.append({
            "canonical_parent_key": key,
            "suite": suite,
            "task_idx": int(task),
            "state_id": int(state),
            **flags,
            "excluded_union": not fresh,
            "fresh_after_exclusion": fresh,
            "rank_sha256": rank_sha256(salt, key) if fresh else None,
            "rank_within_suite_task": None,
            "selected": False,
        })
    for suite in suites:
        for task in tasks:
            candidates = [row for row in rows if row["suite"] == suite and row["task_idx"] == int(task) and row["fresh_after_exclusion"]]
            candidates.sort(key=lambda row: (row["rank_sha256"], row["canonical_parent_key"]))
            for index, row in enumerate(candidates, 1):
                row["rank_within_suite_task"] = index
                row["selected"] = index == 1
    design: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []
    for suite in suites:
        for task in tasks:
            candidates = [row for row in rows if row["suite"] == suite and row["task_idx"] == int(task) and row["fresh_after_exclusion"]]
            selected = [row for row in candidates if row["selected"]]
            if len(selected) > 1:
                raise AuthorityError(f"MULTIPLE_SELECTED_PARENTS:{suite}/task_{int(task):02d}")
            cell = {
                "design_cell": f"{suite}/task_{int(task):02d}",
                "suite": suite,
                "task_idx": int(task),
                "candidate_count": len(candidates),
                "selected": bool(selected),
                "selected_parent_key": selected[0]["canonical_parent_key"] if selected else None,
                "status": "EXECUTABLE_FRESH_PARENT" if selected else "STRUCTURALLY_UNAVAILABLE_FRESH_IDENTITY",
                "missing_reason": None if selected else "NO_NEW_UNEXPOSED_G10_IDENTITY_AFTER_FROZEN_EXCLUSIONS",
            }
            design.append(cell)
            if selected:
                parents.append({
                    "ordinal": len(parents) + 1,
                    "canonical_parent_key": selected[0]["canonical_parent_key"],
                    "suite": suite,
                    "task_idx": int(task),
                    "source": "SOURCE_DERIVED_FRESH_G10_FIRST_RANK",
                    "selection_rank_sha256": selected[0]["rank_sha256"],
                    "rank_within_suite_task": 1,
                    "future_state": "IDENTITY_FROZEN",
                })
    return {
        "g10_rows": rows,
        "design_rows": design,
        "parent_rows": parents,
        "exclusion_union": sorted(union),
        "fresh_rows": [row for row in rows if row["fresh_after_exclusion"]],
        "source_counts": {name: len(values) for name, values in sorted(exclusion_sets.items())},
    }


def protected_counters() -> dict[str, int]:
    return {
        "pgd_calls": 0,
        "adversarial_images": 0,
        "env_step_calls": 0,
        "physical_interventions": 0,
        "vphys_reads": 0,
        "attack_outcome_reads": 0,
        "eval160_reads": 0,
        "protected_reads": 0,
    }


def recompute(protocol: Mapping[str, Any], repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = (repo_root or ROOT).resolve()
    errors: list[str] = []
    local_bindings: dict[str, Any] = {}
    timing_source: dict[str, Any] = {}
    source_meta: list[dict[str, Any]] = []
    exclusion_sets: dict[str, set[str]] = {}
    derived: dict[str, Any] | None = None
    try:
        local_bindings = local_binding_hashes(protocol, repo_root)
        timing_source = parse_timing_source(protocol)
        errors.extend(timing_source["errors"])
        source_specs = protocol["source_recompute"]["sources"]
        g10_spec = next(spec for spec in source_specs if spec["kind"] == "g10_identities")
        g10_set, g10_meta = load_source(g10_spec)
        source_meta.append(g10_meta)
        if len(g10_set) != 1200:
            errors.append(f"G10_IDENTITY_COUNT_DERIVED:{len(g10_set)}")
        for spec in source_specs:
            if spec is g10_spec:
                continue
            values, meta = load_source(spec, g10_set)
            exclusion_sets[str(spec["name"])] = values
            source_meta.append(meta)
        derived = derive_population(
            g10_set,
            exclusion_sets,
            str(protocol["source_recompute"]["selection_salt"]),
        )
    except (AuthorityError, KeyError, json.JSONDecodeError, OSError) as exc:
        errors.append(str(exc))
    if derived is None:
        derived = {"g10_rows": [], "design_rows": [], "parent_rows": [], "exclusion_union": [], "fresh_rows": [], "source_counts": {}}

    design = derived["design_rows"]
    parents = derived["parent_rows"]
    missing = [row["design_cell"] for row in design if not row["selected"]]
    suite_counts = {suite: sum(row["suite"] == suite for row in parents) for suite in SUITES}
    if len(design) != 40:
        errors.append(f"DESIGN_CELL_COUNT_DERIVED:{len(design)}")
    if len(parents) != 39:
        errors.append(f"EXECUTABLE_PARENT_COUNT_DERIVED:{len(parents)}")
    if missing != ["libero_goal/task_01"]:
        errors.append(f"MISSING_CELL_DERIVED:{missing}")
    if suite_counts != {"libero_10": 10, "libero_goal": 9, "libero_object": 10, "libero_spatial": 10}:
        errors.append(f"SUITE_PARENT_COUNTS_DERIVED:{suite_counts}")
    timing = protocol["timing_freeze"]
    required_end = int(timing["attack_window"]["length"]) + int(timing["physical_followup"]["length"])
    if required_end != 15:
        errors.append(f"TIMING_WINDOW_LENGTH_DERIVED:{required_end}")
    status = "STAGE_X_X1R_T1D0R_AUTHORITY_PASS" if not errors else "STAGE_X_X1R_T1D0R_HOLD"
    return {
        "schema": "STAGE_X_X1R_T1D0R_SOURCE_RECOMPUTE_AUDIT_V1",
        "status": status,
        "errors": errors,
        "source_binding": {
            "reviewed_pr125_head": protocol["reviewed_pr125"]["head_commit"],
            "reviewed_pr125_tree": protocol["reviewed_pr125"]["head_tree"],
            "d0r_runtime_source_commit": git_value(repo_root, "rev-parse", "HEAD"),
            "d0r_runtime_source_tree": git_value(repo_root, "rev-parse", "HEAD^{tree}"),
            "worktree_status_before_evidence": git_value(repo_root, "status", "--porcelain"),
            "official_environment": protocol["official_environment"],
        },
        "timing": {
            **timing,
            "scheduler_source_observation": timing_source,
            "runtime_implementation_status": "PROSPECTIVE_RUNNER_NOT_YET_BOUND",
            "historical_attack_start_identifiable": False,
        },
        "source_files": source_meta,
        "local_bindings": local_bindings,
        "derived": {
            "g10_identity_count": len(derived["g10_rows"]),
            "g10_identity_digest": identity_digest(row["canonical_parent_key"] for row in derived["g10_rows"]),
            "exclusion_source_counts": derived["source_counts"],
            "exclusion_union_count": len(derived["exclusion_union"]),
            "exclusion_union_digest": identity_digest(derived["exclusion_union"]),
            "fresh_candidate_count": len(derived["fresh_rows"]),
            "fresh_candidate_digest": identity_digest(row["canonical_parent_key"] for row in derived["fresh_rows"]),
            "design_cell_count": len(design),
            "executable_parent_count": len(parents),
            "missing_design_cells": missing,
            "executable_parent_counts_by_suite": suite_counts,
            "zero_replacement": True,
        },
        "outcome_firewall": {
            "selection_outcomes_read": False,
            "clean_success_read": False,
            "emit_read": False,
            "vphys_read": False,
            "attack_outcome_read": False,
            "protected_registry_read": False,
            "eval160_read": False,
        },
        "authorization": {
            "model_inference_authorized": False,
            "clean_parent_materialization_authorized": False,
            "pgd_authorized": False,
            "env_step_authorized": False,
            "physical_intervention_authorized": False,
            "attack_outcome_authorized": False,
            "protected_authorized": False,
            "next_gate": "CLEAN_PARENT_MATERIALIZATION_REVIEW_REQUIRED",
        },
        "protected_boundary": {
            "eval160": "UNREAD",
            "protected_evaluation": "UNREAD",
            "counters": protected_counters(),
        },
        "execution_started": False,
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def repo_artifact_hashes(repo_root: Path, relative_paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise AuthorityError(f"GENERATED_ARTIFACT_MISSING:{relative}")
        result[relative] = sha256_file(path)
    return result


def write_evidence(protocol_path: Path, receipt: dict[str, Any], derived: dict[str, Any], repo_root: Path) -> None:
    reports = repo_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    source_report = reports / "STAGE_X_X1R_T1D0R_SOURCE_RECOMPUTE_AUDIT_V1.json"
    g10_ledger = reports / "STAGE_X_X1R_T1D0R_G10_IDENTITY_EXCLUSION_LEDGER_V1.json"
    design_ledger = reports / "STAGE_X_X1R_T1D0R_DESIGN_CELL_LEDGER_V1.json"
    parent_ledger = reports / "STAGE_X_X1R_T1D0R_PARENT_LEDGER_V1.json"
    sums = reports / "STAGE_X_X1R_T1D0R_SHA256SUMS.txt"
    root_seal = reports / "STAGE_X_X1R_T1D0R_ROOT_SEAL.json"
    root_seal_sha = reports / "STAGE_X_X1R_T1D0R_ROOT_SEAL.sha256"

    source_report.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(g10_ledger, derived["g10_rows"])
    write_jsonl(design_ledger, derived["design_rows"])
    write_jsonl(parent_ledger, derived["parent_rows"])

    authority_files = [
        "configs/STAGE_X_X1R_T1D0R_TIMING_PARENT_AUTHORITY_V1.json",
        "scripts/stage_x/audit_stage_x1r_t1d0r_authority.py",
        "tests/stage_x/test_stage_x1r_t1d0r_authority.py",
        "docs/handoffs/STAGE_X_X1R_T1D0R_TIMING_PARENT_AUTHORITY_HANDOFF_20260818.md",
        "reports/STAGE_X_X1R_T1D0R_SOURCE_RECOMPUTE_AUDIT_V1.json",
        "reports/STAGE_X_X1R_T1D0R_G10_IDENTITY_EXCLUSION_LEDGER_V1.json",
        "reports/STAGE_X_X1R_T1D0R_DESIGN_CELL_LEDGER_V1.json",
        "reports/STAGE_X_X1R_T1D0R_PARENT_LEDGER_V1.json",
    ]
    generated_hashes = repo_artifact_hashes(repo_root, authority_files)
    seal = {
        "schema": "STAGE_X_X1R_T1D0R_ROOT_SEAL_V1",
        "status": receipt["status"],
        "reviewed_pr125": receipt["source_binding"],
        "d0r_runtime_source": {
            "commit": receipt["source_binding"]["d0r_runtime_source_commit"],
            "tree": receipt["source_binding"]["d0r_runtime_source_tree"],
            "worktree_status_before_evidence": receipt["source_binding"]["worktree_status_before_evidence"],
        },
        "external_source_bindings": receipt["source_files"],
        "local_source_bindings": receipt["local_bindings"],
        "prospective_timing": receipt["timing"],
        "derived_population": receipt["derived"],
        "generated_artifact_hashes": generated_hashes,
        "final_branch_seal": {
            "commit": "NOT_SELF_REFERENTIAL_LIVE_GITHUB_HANDOFF",
            "tree": "NOT_SELF_REFERENTIAL_LIVE_GITHUB_HANDOFF",
            "runtime_commit_pre_evidence": receipt["source_binding"]["d0r_runtime_source_commit"],
            "runtime_tree_pre_evidence": receipt["source_binding"]["d0r_runtime_source_tree"],
            "note": "The final pushed HEAD/tree is recorded after commit in the handoff; embedding it in this containing commit would be self-referential.",
        },
        "authorization": receipt["authorization"],
        "protected_boundary": receipt["protected_boundary"],
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
        "sha256sums_scope": {
            "covers_root_seal_json": True,
            "excludes_self": True,
            "excludes_root_seal_sha256_sidecar": True,
        },
    }
    root_seal.write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums_hashes = repo_artifact_hashes(repo_root, authority_files + ["reports/STAGE_X_X1R_T1D0R_ROOT_SEAL.json"])
    sums.write_text("".join(f"{digest}  {path}\n" for path, digest in sorted(sums_hashes.items())), encoding="utf-8")
    root_seal_sha.write_text(f"{sha256_file(root_seal)}  {root_seal.name}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only T1-D0R authority recomputation")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/STAGE_X_X1R_T1D0R_TIMING_PARENT_AUTHORITY_V1.json")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--write-evidence", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = recompute(protocol, repo_root)
    source_specs = protocol["source_recompute"]["sources"]
    g10_spec = next(spec for spec in source_specs if spec["kind"] == "g10_identities")
    g10_set, _ = load_source(g10_spec)
    exclusion_sets = {}
    for spec in source_specs:
        if spec is not g10_spec:
            values, _ = load_source(spec, g10_set)
            exclusion_sets[spec["name"]] = values
    derived = derive_population(g10_set, exclusion_sets, protocol["source_recompute"]["selection_salt"])
    if args.write_evidence:
        write_evidence(protocol_path, receipt, derived, repo_root)
    print(json.dumps({"status": receipt["status"], "errors": receipt["errors"], "derived": receipt["derived"]}, sort_keys=True))
    return 0 if not receipt["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
