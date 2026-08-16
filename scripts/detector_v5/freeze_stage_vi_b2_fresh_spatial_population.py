"""Freeze the outcome-blind Stage VI-B2 fresh spatial candidate universe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
IDENTITY_RE = re.compile(r"^libero_(?:10|goal|object|spatial)/task_\d{2}/state_\d{2}$")
DIR_IDENTITY_RE = re.compile(r"(libero_(?:10|goal|object|spatial))_task_?(\d+)_state_?(\d+)")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _check_input(spec: Mapping[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(spec["path"])).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"INPUT_MISSING:{label}:{path}")
    actual = _sha(path)
    expected = spec.get("expected_sha256")
    if expected is not None and actual != str(expected):
        raise ValueError(f"INPUT_SHA_MISMATCH:{label}:{actual}:{expected}")
    return path, {"path": str(path), "sha256": actual, "role": str(spec.get("role", label))}


def _check_boundary(value: Mapping[str, Any], label: str) -> None:
    counters = value.get("protected_counters")
    if counters is not None and counters != COUNTERS:
        raise ValueError(f"PROTECTED_COUNTERS:{label}")
    for field in ("branch_results_read", "outcomes_read", "m4_outcomes_read"):
        if value.get(field) is True:
            raise ValueError(f"OUTCOME_READ:{label}:{field}")


def _identity(value: Any, label: str) -> str:
    if isinstance(value, str):
        key = value
    elif isinstance(value, Mapping):
        key = str(value.get("canonical_parent_key", ""))
    else:
        key = ""
    if not IDENTITY_RE.fullmatch(key):
        raise ValueError(f"INVALID_IDENTITY:{label}:{key}")
    return key


def _manifest_keys(path: Path, fields: tuple[str, ...], label: str) -> set[str]:
    value = _load(path)
    _check_boundary(value, label)
    found: set[str] = set()
    for field in fields:
        raw = value.get(field)
        if not isinstance(raw, list):
            continue
        for item in raw:
            found.add(_identity(item, label))
    if not found:
        raise ValueError(f"IDENTITY_LIST_MISSING:{label}")
    return found


def _g10_keys(path: Path) -> set[str]:
    value = _load(path)
    _check_boundary(value, "g10_manifest")
    raw = value.get("identities")
    if not isinstance(raw, list) or not raw:
        raise ValueError("G10_IDENTITIES_MISSING")
    keys = {_identity(item, "g10_manifest") for item in raw}
    if len(keys) != len(raw):
        raise ValueError("G10_DUPLICATE_IDENTITIES")
    return keys


def _directory_identities(spec: Mapping[str, Any], label: str) -> tuple[set[str], dict[str, Any]]:
    base = Path(str(spec["base"])).expanduser().resolve()
    pattern = str(spec["glob"])
    if not base.is_dir():
        raise ValueError(f"PHYSICAL_REGISTRY_BASE_MISSING:{label}:{base}")
    matches = sorted(path.name for path in base.glob(pattern) if path.is_dir())
    identities: set[str] = set()
    for name in matches:
        match = DIR_IDENTITY_RE.search(name)
        if match:
            identities.add(f"{match.group(1)}/task_{int(match.group(2)):02d}/state_{int(match.group(3)):02d}")
    listing_sha = hashlib.sha256("\n".join(matches).encode("utf-8")).hexdigest()
    return identities, {"base": str(base), "glob": pattern, "matched_directories": matches, "listing_sha256": listing_sha}


def _git_binding(source_repo: str | None) -> dict[str, str | None]:
    if not source_repo:
        return {"commit": None, "tree": None}
    repo = Path(source_repo).resolve()
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"SOURCE_BINDING_READ_FAIL:{repo}:{exc}") from exc
    return {"commit": commit, "tree": tree}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _seal(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append(f"{_sha(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{_sha(root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def freeze(protocol_path: Path, output_root: Path, *, source_repo: str | None = None) -> dict[str, Any]:
    protocol = _load(protocol_path.resolve())
    if protocol.get("schema") != "STAGE_VI_B2_FRESH_SPATIAL_POPULATION_PROTOCOL_V1":
        raise ValueError("FRESH_SPATIAL_PROTOCOL_SCHEMA")
    if protocol.get("status") != "FROZEN_OUTCOME_BLIND_PRE_CLEAN_ROLLOUT":
        raise ValueError("FRESH_SPATIAL_PROTOCOL_NOT_FROZEN")
    output_root = output_root.resolve()
    if output_root.exists():
        raise ValueError(f"REFUSE_OVERWRITE:{output_root}")

    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("FRESH_SPATIAL_INPUTS_MISSING")

    bindings: dict[str, Any] = {}
    exclusion_sets: dict[str, set[str]] = {}

    g10_path, bindings["g10_manifest"] = _check_input(inputs["g10_manifest"], "g10_manifest")
    g10 = _g10_keys(g10_path)
    spatial = {key for key in g10 if key.startswith("libero_spatial/")}

    for group, field, schema_fields in (
        ("prior_exposure", "exposure_union", ("excluded_parent_keys", "parent_keys")),
        ("prior_clean_attempt", "clean_attempt_union", ("excluded_parent_keys", "parent_keys")),
    ):
        path, bindings[field] = _check_input(inputs[field], field)
        exclusion_sets[group] = _manifest_keys(path, schema_fields, field)

    for group, field, fields in (
        ("stage_v_formal_population", "stage_v_formal_manifests", ("parents", "selected_parents")),
        ("stage_v_physical_matrix", "stage_v_physical_matrix_manifests", ("parents", "selected_parents")),
        ("stage_vi_b2_population", "stage_vi_b2_manifests", ("parents", "selected_parents")),
        ("stage_vi_b2_development", "stage_vi_b2_development_manifests", ("parents", "selected_parents")),
    ):
        specs = inputs.get(field, [])
        if not isinstance(specs, list) or not specs:
            raise ValueError(f"INPUT_LIST_MISSING:{field}")
        keys: set[str] = set()
        records = []
        for index, spec in enumerate(specs):
            path, record = _check_input(spec, f"{field}[{index}]")
            keys |= _manifest_keys(path, fields, f"{field}[{index}]")
            records.append(record)
        bindings[field] = records
        exclusion_sets[group] = keys

    path, bindings["rejected_v2_manifest"] = _check_input(inputs["rejected_v2_manifest"], "rejected_v2_manifest")
    exclusion_sets["rejected_v2_candidate"] = _manifest_keys(path, ("parents", "selected_parents"), "rejected_v2_manifest")

    physical_registry = []
    exclusion_sets["prior_physical_intervention_named_roots"] = set()
    for index, spec in enumerate(inputs.get("prior_physical_named_roots", [])):
        identities, record = _directory_identities(spec, f"prior_physical_named_roots[{index}]")
        exclusion_sets["prior_physical_intervention_named_roots"] |= identities
        physical_registry.append(record)
    bindings["prior_physical_named_roots"] = physical_registry

    if not spatial:
        raise ValueError("NO_SPATIAL_G10_IDENTITIES")
    salt = str(protocol["selection"]["salt"])
    rank = lambda key: hashlib.sha256(f"{salt}::{key}".encode("utf-8")).hexdigest()
    excluded_by_key: dict[str, list[str]] = {}
    for key in sorted(spatial):
        excluded_by_key[key] = sorted(group for group, keys in exclusion_sets.items() if key in keys)
    candidates = sorted((key for key in spatial if not excluded_by_key[key]), key=lambda key: (rank(key), key))
    if len(candidates) < 2:
        status = "HOLD_NO_TWO_FRESH_SPATIAL_IDENTITIES"
    else:
        status = "PASS_FROZEN_FRESH_SPATIAL_UNIVERSE"

    output_root.mkdir(parents=True)
    universe_rows = []
    for key in sorted(spatial):
        suite, task, state = key.split("/")
        reasons = excluded_by_key[key]
        universe_rows.append({
            "canonical_parent_key": key,
            "suite": suite,
            "task_index": int(task.removeprefix("task_")),
            "state_index": int(state.removeprefix("state_")),
            "rank_sha256": rank(key),
            "status": "EXCLUDED" if reasons else "FRESH_STATIC_ELIGIBILITY_CANDIDATE",
            "exclusion_groups": reasons,
        })
    manifest = {
        "schema": "STAGE_VI_B2_FRESH_SPATIAL_POPULATION_FREEZE_V1",
        "status": status,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha(protocol_path.resolve()),
        "source_binding": _git_binding(source_repo),
        "selection": {
            "salt": salt,
            "order": "SHA256(salt::canonical_parent_key), then canonical_parent_key",
            "selection_is_outcome_blind": True,
            "freeze_before_new_clean_rollouts": True,
            "clean_rollouts_started": False,
            "candidate_outcomes_read": False,
            "target_new_spatial_parents": 2,
        },
        "source_universe": {
            "name": "G10 held-out non-protected LIBERO universe",
            "identity_count": len(g10),
            "spatial_identity_count": len(spatial),
            "protected_registry_read": False,
            "eval160_read": False,
        },
        "input_bindings": bindings,
        "exclusion_sets": {
            group: {"count": len(keys), "spatial_overlap_count": len(keys & spatial), "identities": sorted(keys)}
            for group, keys in sorted(exclusion_sets.items())
        },
        "universe": universe_rows,
        "fresh_candidate_count": len(candidates),
        "fresh_candidate_order": [
            {"order": index, "canonical_parent_key": key, "rank_sha256": rank(key)}
            for index, key in enumerate(candidates, start=1)
        ],
        "required_population": {
            "retained_parent_count": 14,
            "new_spatial_parent_count": 2,
            "total_parent_count": 16,
            "suite_counts": {"libero_10": 7, "libero_goal": 3, "libero_object": 2, "libero_spatial": 4},
        },
        "protected_counters": COUNTERS,
        "outcomes_read": False,
        "intervention_executed": False,
        "labels_generated": False,
        "v_phys_generated": False,
    }
    _write(output_root / "FRESH_SPATIAL_POPULATION_MANIFEST.json", manifest)
    _seal(output_root)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-repo")
    args = parser.parse_args()
    try:
        result = freeze(args.protocol, args.output_root, source_repo=args.source_repo)
        print(json.dumps({"status": result["status"], "root": str(args.output_root.resolve()), "fresh_candidate_count": result["fresh_candidate_count"]}, sort_keys=True))
        return 0 if result["status"] == "PASS_FROZEN_FRESH_SPATIAL_UNIVERSE" else 2
    except Exception as exc:
        print(json.dumps({"status": "HOLD_FRESH_SPATIAL_POPULATION", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
