"""Shared read-only helpers for the C2g R8R Clean2000 reuse audit."""
from __future__ import annotations
import csv, hashlib, json, math, re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.multisuite_detector.plan_c2g_scientific_corpus import (
    COHORTS, PASS_STATUS as PLAN_PASS_STATUS, SCHEMA as PLAN_SCHEMA, SUITES,
)

SOURCE_SPEC_SCHEMA = "c2g.r8r.clean2000_source_spec.2026-07-11.v1"
SOURCE_CLASSES = {
    "RAW_COLLECTION_SOURCE", "REPLACEMENT_SOURCE", "MERGED_VIEW",
    "CAVEAT_OR_OVERRIDE_VIEW", "DERIVED_MATERIALIZATION", "TRAINING_OUTPUT",
    "ATTACK_OR_INTERVENTION_OUTPUT_EXCLUDED", "UNKNOWN",
}
EPISODE_CLASSES = {
    "RAW_COLLECTION_SOURCE", "REPLACEMENT_SOURCE", "MERGED_VIEW",
    "CAVEAT_OR_OVERRIDE_VIEW",
}
IDENTITY_RE = re.compile(
    r"(?P<suite>libero_(?:object|spatial|goal|10)).*?"
    r"task[_/-](?P<task>\d+).*?state[_/-](?P<state>\d+)", re.I,
)

@dataclass(frozen=True)
class SourceView:
    name: str
    root: Path
    source_class: str
    canonical_suites: tuple[str, ...]
    priority: int
    clean_only: bool
    runtime_valid_by_manifest: bool
    model_provenance_bound: bool
    processor_provenance_bound: bool
    feature_25d_order_bound: bool
    evidence_paths: tuple[Path, ...]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} must contain an object")
        rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def assert_sha(path: Path, expected: str, label: str) -> str:
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"{label} expected SHA256 invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")
    return actual


def new_output(path: Path, repo: Path) -> Path:
    output = path.resolve()
    if output.exists():
        raise FileExistsError(output)
    repo = repo.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("output must be outside repository")
    output.mkdir(parents=True)
    return output


def load_registry(registry: Path, report: Path, expected_sha: str):
    registry, report = registry.resolve(), report.resolve()
    assert_sha(report, expected_sha, "R7 plan")
    plan = read_json(report)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != PLAN_PASS_STATUS:
        raise ValueError("invalid R7 plan")
    if Path(str(plan.get("registry", ""))).resolve() != registry:
        raise ValueError("R7 plan binds another registry")
    if plan.get("registry_sha256") != sha256_file(registry):
        raise ValueError("R7 registry binding mismatch")
    rows, seen = read_jsonl(registry), set()
    for row in rows:
        identity = (
            str(row.get("suite", "")), int(row.get("task_index", -1)),
            int(row.get("state_id", -1)),
        )
        if (
            identity[0] not in SUITES or min(identity[1:]) < 0 or identity in seen
            or row.get("cohort") not in COHORTS
        ):
            raise ValueError(f"invalid registry row: {identity}")
        seen.add(identity)
    return rows, plan


def load_source_spec(path: Path):
    raw = read_json(path.resolve())
    if raw.get("schema") != SOURCE_SPEC_SCHEMA:
        raise ValueError("source spec schema mismatch")
    views, names, roots = [], set(), set()
    for value in raw.get("views", []):
        name = str(value.get("name", "")).strip()
        root = Path(str(value.get("root", ""))).resolve()
        source_class = str(value.get("source_class", ""))
        suites = tuple(map(str, value.get("canonical_suites", ())))
        if (
            not name or name in names or root in roots or source_class not in SOURCE_CLASSES
            or any(suite not in SUITES for suite in suites)
        ):
            raise ValueError(f"invalid source view: {name}")
        if not root.exists() or (source_class in EPISODE_CLASSES and not root.is_dir()):
            raise FileNotFoundError(root)
        evidence = tuple(Path(str(item)).resolve() for item in value.get("evidence_paths", ()))
        assertions = tuple(
            bool(value.get(key, False)) for key in (
                "clean_only", "runtime_valid_by_manifest", "model_provenance_bound",
                "processor_provenance_bound", "feature_25d_order_bound",
            )
        )
        if any(assertions) and not evidence:
            raise ValueError(f"{name} has assertions without evidence")
        expected = {
            str(Path(str(key)).resolve()): str(digest)
            for key, digest in dict(value.get("evidence_sha256", {})).items()
        }
        for item in evidence:
            if not item.is_file():
                raise FileNotFoundError(item)
            if str(item) in expected:
                assert_sha(item, expected[str(item)], f"evidence {item}")
        views.append(SourceView(
            name, root, source_class, suites, int(value.get("priority", 0)),
            *assertions, evidence,
        ))
        names.add(name)
        roots.add(root)
    if not views:
        raise ValueError("source spec has no views")
    predecessors = [Path(str(item)).resolve() for item in raw.get("predecessor_roots", ())]
    return views, predecessors, raw


def _nested(metadata: Mapping[str, Any]):
    output = [("metadata", metadata)]
    for key in ("adapter_episode_info", "episode_info", "identity", "source_identity"):
        if isinstance(metadata.get(key), Mapping):
            output.append((f"metadata.{key}", metadata[key]))
    return output


def resolve_identity(metadata: Mapping[str, Any], metadata_path: Path):
    found, suites, tasks, states = [], [], [], []
    for prefix, mapping in _nested(metadata):
        for key in ("suite", "suite_name", "benchmark_suite"):
            if mapping.get(key) is not None:
                suites.append((str(mapping[key]).lower(), f"{prefix}.{key}"))
        for key, target in (
            (("task_index", "task_id", "task_idx"), tasks),
            (("state_id", "init_state_id", "initial_state_id", "state_idx"), states),
        ):
            for name in key:
                try:
                    if mapping.get(name) is not None:
                        target.append((int(mapping[name]), f"{prefix}.{name}"))
                except Exception:
                    pass
    for suite, suite_source in suites:
        for task, task_source in tasks:
            for state, state_source in states:
                found.append(((suite, task, state), f"{suite_source}+{task_source}+{state_source}"))
    texts = [(str(metadata_path), "metadata_path")]
    for prefix, mapping in _nested(metadata):
        for key in ("parent_key", "episode_key", "registry_parent_key"):
            if mapping.get(key):
                texts.append((str(mapping[key]), f"{prefix}.{key}"))
    for text, source in texts:
        match = IDENTITY_RE.search(text)
        if match:
            found.append(((
                match.group("suite").lower(), int(match.group("task")),
                int(match.group("state")),
            ), source))
    found = [item for item in found if item[0][0] in SUITES and min(item[0][1:]) >= 0]
    identities = {item[0] for item in found}
    if not identities:
        return None, "UNRESOLVED", "no valid identity"
    if len(identities) > 1:
        return None, "CONFLICT", repr(sorted(identities))
    identity = next(iter(identities))
    return identity, ";".join(sorted({source for item, source in found if item == identity})), None


def _identity_key(row: Mapping[str, Any]):
    suite = str(row.get("suite", ""))
    return (
        SUITES.index(suite) if suite in SUITES else len(SUITES),
        int(row.get("task_index", -1)), int(row.get("state_id", -1)),
        str(row.get("source_view_name", "")), str(row.get("metadata_path", "")),
    )


def build_source_view_ledger(views: Sequence[SourceView], lookup):
    rows, by_name = [], {view.name: view for view in views}
    for view in views:
        if view.source_class not in EPISODE_CLASSES:
            continue
        for steps in sorted(view.root.rglob("step_records.jsonl")):
            metadata_path = steps.with_name("episode_metadata.json")
            if not metadata_path.is_file():
                continue
            metadata = read_json(metadata_path)
            identity, method, error = resolve_identity(metadata, metadata_path)
            registered = lookup.get(identity) if identity else None
            rows.append({
                "source_view_name": view.name, "source_root": str(view.root),
                "source_class": view.source_class, "priority": view.priority,
                "canonical_for_suite": bool(identity and identity[0] in view.canonical_suites),
                "metadata_path": str(metadata_path.resolve()),
                "step_records_path": str(steps.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
                "step_records_sha256": sha256_file(steps),
                "identity_resolution_method": method, "identity_resolution_error": error,
                "suite": identity[0] if identity else None,
                "task_index": identity[1] if identity else None,
                "state_id": identity[2] if identity else None,
                "registered": bool(registered),
                "parent_key": registered.get("parent_key") if registered else None,
                "cohort": registered.get("cohort") if registered else None,
                "split": registered.get("split") if registered else None,
                "selected_canonical": False, "canonical_conflict": False,
            })
    rows.sort(key=_identity_key)
    return rows, by_name


def select_canonical(source_rows, registry_rows):
    grouped = defaultdict(list)
    for row in source_rows:
        if row["registered"]:
            grouped[(row["suite"], row["task_index"], row["state_id"])].append(row)
    selected, reconciliation, conflicts = {}, [], 0
    for registry in registry_rows:
        identity = (registry["suite"], registry["task_index"], registry["state_id"])
        views = grouped.get(identity, [])
        candidates = sorted(
            [row for row in views if row["canonical_for_suite"]],
            key=lambda row: (-row["priority"], row["source_view_name"], row["metadata_path"]),
        )
        if len(candidates) != 1:
            conflicts += 1
            for row in candidates:
                row["canonical_conflict"] = True
        else:
            candidates[0]["selected_canonical"] = True
            selected[identity] = candidates[0]
        reconciliation.append({
            "suite": identity[0], "task_index": identity[1], "state_id": identity[2],
            "parent_key": registry["parent_key"], "cohort": registry["cohort"],
            "split": registry["split"], "physical_view_count": len(views),
            "canonical_candidate_count": len(candidates),
            "selected_source_view": candidates[0]["source_view_name"] if len(candidates) == 1 else None,
            "selected_metadata_path": candidates[0]["metadata_path"] if len(candidates) == 1 else None,
            "identity_status": "CANONICAL_SELECTED" if len(candidates) == 1 else (
                "MISSING" if not candidates else "CANONICAL_CONFLICT"
            ),
        })
    reconciliation.sort(key=_identity_key)
    return selected, reconciliation, conflicts

IDENTITY_KEY = _identity_key
