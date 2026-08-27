#!/usr/bin/env python3
"""Build the Stage-AC official-init inventory and freshness firewall.

This is static only.  It loads official LIBERO ``.pruned_init`` files, records
each row's canonical numeric bytes, and removes every identity present in the
explicit historical authority inputs before applying the frozen 30 x 8 rule.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


SCHEMA = "STAGE_AC_AC1R1_OFFICIAL_INIT_STATE_UNIVERSE_V1"
GATE = "STAGE_AC_AC1R1_OFFICIAL_INIT_STATE_UNIVERSE_EXPANSION_AND_MODEL_SPECIFIC_REPLICATION_PROGRAM_V1"
SELECTION_SALT = "STAGE_AC_AC2_FRESH_PARENT_SELECTION_V1_20260827"
TARGET_SUITES = ("libero_10", "libero_object", "libero_spatial")
STATES_PER_TASK = 50
PARENTS_PER_TASK = 8
TARGET_PARENT_COUNT = 30 * PARENTS_PER_TASK
CANONICAL_ENCODING = "STAGE_AC_INIT_STATE_CANONICAL_NUMERIC_V1|dtype=<f8|order=C|shape=47"
KEY_RE = re.compile(r"^libero_(?:10|object|spatial|goal)/task_\d{2}/state_\d{2}$")


HISTORY_SOURCES = (
    ("stage_x_g10_exclusion", "reports/STAGE_X_X1R_T1D0R1_G10_IDENTITY_EXCLUSION_LEDGER_V1.json", "g10"),
    (
        "stage_x_f1a3_frozen_identity",
        "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_EXPOSURE_CLASSIFICATION_V3.json",
        "f1a3_frozen",
    ),
    (
        "stage_x_f1a3_split_ledger",
        "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_SPLIT_LEDGER_V3.json",
        "listed",
    ),
    ("stage_v_m4", "reports/server_evidence/STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json", "listed"),
    ("stage_vi_b2", "configs/STAGE_VI_B2_FRESH_PARENT_MANIFEST_V3.json", "listed"),
    ("stage_z_z0_40", "reports/STAGE_Z_Z0_SHARED_40_IDENTITY_PANEL_V1.json", "listed"),
    ("stage_z_z0r1_36", "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json", "listed"),
    ("stage_z_z2_exposure", "reports/STAGE_Z_Z2_SCIENTIFIC_EXPOSURE_LEDGER_V2.json", "listed"),
    ("stage_aa_aa0", "reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json", "listed"),
    ("stage_aa_aa2", "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json", "listed"),
    ("stage_aa_aa2r2", "reports/STAGE_AA_AA2R2_PHASE_B_V2_CENSUS_TERMINAL_V1.json", "listed"),
    ("stage_ac_ac0", "reports/STAGE_AC_AC0_CONSTRUCT_VALIDATION_TERMINAL_V1.json", "listed"),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_binding(path: Path, display_path: str) -> dict:
    data = path.read_bytes()
    return {"path": display_path, "bytes": len(data), "sha256": sha256_bytes(data)}


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: object) -> dict:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": path.name, "bytes": len(data), "sha256": sha256_bytes(data)}


def git_binding(repo: Path) -> dict:
    def run(fmt: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), "show", "-s", f"--format={fmt}", "HEAD"], text=True).strip()

    return {"commit": run("%H"), "tree": run("%T")}


def source_root_binding(repo: Path) -> dict:
    try:
        return {"path": str(repo), "git": git_binding(repo)}
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(repo), "git": None, "status": "NOT_A_GIT_CHECKOUT"}


def load_task_map(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError("libero_task_map is not a dict")
            return value
    raise ValueError("libero_task_map assignment not found")


def load_init(path: Path) -> np.ndarray:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    array = np.asarray(value)
    if array.ndim != 2 or array.shape != (STATES_PER_TASK, 47):
        raise ValueError(f"{path}: expected {(STATES_PER_TASK, 47)}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path}: expected floating state array, got {array.dtype}")
    return np.asarray(array, dtype="<f8", order="C")


def task_language(task: str) -> str:
    filename = f"{task}.bddl"
    if filename[0].isupper():
        start = filename.find("SCENE") + (8 if "SCENE10" in filename else 7)
        text = " ".join(filename[start:].split("_"))
    else:
        text = " ".join(filename.split("_"))
    return text[:-5] if text.endswith(".bddl") else text


def canonical_row_bytes(row: np.ndarray) -> bytes:
    return row.astype("<f8", copy=False).tobytes(order="C")


def state_digest(raw: bytes) -> str:
    return sha256_bytes((CANONICAL_ENCODING + "\n").encode("ascii") + raw)


def build_official_inventory(official_root: Path) -> tuple[dict, dict[str, dict], list[dict]]:
    task_map_path = official_root / "libero/libero/benchmark/libero_suite_task_map.py"
    task_map = load_task_map(task_map_path)
    repo_git = git_binding(official_root)
    task_map_binding = file_binding(task_map_path, "libero/libero/benchmark/libero_suite_task_map.py")
    tasks = []
    rows = []
    by_key = {}
    for suite in TARGET_SUITES:
        suite_tasks = task_map.get(suite)
        if not isinstance(suite_tasks, list) or len(suite_tasks) != 10:
            raise ValueError(f"{suite}: official task map must contain 10 tasks")
        for task_index, task_name in enumerate(suite_tasks):
            task = f"task_{task_index:02d}"
            init_rel = f"libero/libero/init_files/{suite}/{task_name}.pruned_init"
            bddl_rel = f"libero/libero/bddl_files/{suite}/{task_name}.bddl"
            init_path = official_root / init_rel
            bddl_path = official_root / bddl_rel
            if not init_path.is_file() or not bddl_path.is_file():
                raise FileNotFoundError(f"missing official source: {init_path} or {bddl_path}")
            init_binding = file_binding(init_path, init_rel)
            bddl_binding = file_binding(bddl_path, bddl_rel)
            array = load_init(init_path)
            task_entry = {
                "suite": suite,
                "task": task,
                "task_index": task_index,
                "task_name": task_name,
                "language": task_language(task_name),
                "bddl": bddl_binding,
                "init_states": init_binding,
                "row_count": int(array.shape[0]),
                "state_width": int(array.shape[1]),
            }
            tasks.append(task_entry)
            seen = {}
            for state_index, row in enumerate(array):
                raw = canonical_row_bytes(row)
                digest = state_digest(raw)
                key = f"{suite}/{task}/state_{state_index:02d}"
                duplicate_of = seen.get(digest)
                if duplicate_of is None:
                    seen[digest] = state_index
                record = {
                    "canonical_parent_key": key,
                    "suite": suite,
                    "task": task,
                    "task_index": task_index,
                    "task_name": task_name,
                    "language": task_language(task_name),
                    "official_init_index": state_index,
                    "state": f"state_{state_index:02d}",
                    "state_dtype": "<f8",
                    "state_shape": [47],
                    "canonical_encoding": CANONICAL_ENCODING,
                    "state_sha256": digest,
                    "state_bytes_base64": base64.b64encode(raw).decode("ascii"),
                    "source_init_file": init_binding,
                    "source_bddl_file": bddl_binding,
                    "duplicate_of_state_index": duplicate_of,
                }
                rows.append(record)
                by_key[key] = record
    authority = {
        "libero_source": {
            "root": str(official_root),
            "git": repo_git,
            "task_map": task_map_binding,
        },
        "target_suites": list(TARGET_SUITES),
        "task_count": len(tasks),
        "row_count": len(rows),
        "states_per_task": STATES_PER_TASK,
        "tasks": tasks,
        "canonical_encoding": CANONICAL_ENCODING,
    }
    return authority, by_key, rows


def walk_canonical_records(value: object):
    if isinstance(value, dict):
        key = value.get("canonical_parent_key")
        if isinstance(key, str) and KEY_RE.fullmatch(key):
            yield value
        for child in value.values():
            yield from walk_canonical_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_canonical_records(child)


def parse_history_file(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        yield None, json.loads(text)
        return
    except json.JSONDecodeError:
        # Some historical authorities use JSONL content with a .json suffix.
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                yield line_number, json.loads(line)


def history_hits(project_root: Path, official_by_key: dict[str, dict]) -> tuple[dict[str, list[dict]], list[dict], list[dict], list[dict]]:
    hits: dict[str, list[dict]] = defaultdict(list)
    source_bindings = []
    errors = []
    unresolved = []
    for source_id, relative_path, rule in HISTORY_SOURCES:
        path = project_root / relative_path
        if not path.is_file():
            errors.append({"source": source_id, "path": relative_path, "error": "MISSING_REQUIRED_HISTORY_AUTHORITY"})
            continue
        source_bindings.append({"source_id": source_id, "kind": rule, "file": file_binding(path, relative_path)})
        try:
            records = list(parse_history_file(path))
        except Exception as exc:
            errors.append({"source": source_id, "path": relative_path, "error": f"PARSE_ERROR:{type(exc).__name__}:{exc}"})
            continue
        for line_number, payload in records:
            for record in walk_canonical_records(payload):
                key = record["canonical_parent_key"]
                include = True
                basis = "LISTED_IN_HISTORICAL_AUTHORITY"
                if rule == "g10":
                    include = bool(record.get("excluded_union"))
                    basis = "STAGE_X_G10_EXCLUDED_UNION" if include else "STAGE_X_G10_NOT_IN_EXCLUDED_UNION"
                elif rule == "f1a3_frozen":
                    include = bool(record.get("permanent_exclusion_after_freeze"))
                    basis = "STAGE_X_F1A3_PERMANENT_FROZEN_IDENTITY" if include else "STAGE_X_F1A3_NOT_PERMANENTLY_FROZEN"
                if not include:
                    continue
                if key not in official_by_key:
                    unresolved.append({"source_id": source_id, "path": relative_path, "line": line_number, "canonical_parent_key": key, "reason": "NOT_IN_TARGET_OFFICIAL_UNIVERSE"})
                    continue
                observed = []
                for field in ("state_sha256", "init_state_sha256", "initial_state_sha256", "exact_state_sha256"):
                    value = record.get(field)
                    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                        observed.append({"field": field, "sha256": value})
                        if value != official_by_key[key]["state_sha256"]:
                            errors.append({"source_id": source_id, "path": relative_path, "line": line_number, "canonical_parent_key": key, "error": "HISTORICAL_STATE_DIGEST_MISMATCH", "field": field, "observed": value, "official": official_by_key[key]["state_sha256"]})
                hits[key].append({"source_id": source_id, "path": relative_path, "line": line_number, "basis": basis, "observed_state_digests": observed})
    return hits, source_bindings, errors, unresolved


def select_fresh(rows: list[dict], hits: dict[str, list[dict]]) -> tuple[list[dict], list[dict], dict]:
    fresh = []
    duplicates = []
    by_task = defaultdict(list)
    for row in rows:
        key = row["canonical_parent_key"]
        if hits.get(key):
            continue
        if row["duplicate_of_state_index"] is not None:
            duplicates.append({"canonical_parent_key": key, "duplicate_of_state_index": row["duplicate_of_state_index"], "state_sha256": row["state_sha256"]})
            continue
        rank_input = f"{SELECTION_SALT}|{row['suite']}|{row['task']}|{row['state_sha256']}".encode("utf-8")
        row_copy = dict(row)
        row_copy["selection_rank_sha256"] = sha256_bytes(rank_input)
        fresh.append(row_copy)
        by_task[(row["suite"], row["task"])].append(row_copy)
    selected = []
    task_counts = {}
    for suite in TARGET_SUITES:
        for task_index in range(10):
            task = f"task_{task_index:02d}"
            candidates = sorted(by_task[(suite, task)], key=lambda x: x["selection_rank_sha256"])
            task_counts[f"{suite}/{task}"] = len(candidates)
            if len(candidates) >= PARENTS_PER_TASK:
                selected.extend(candidates[:PARENTS_PER_TASK])
    capacity_ok = len(fresh) >= TARGET_PARENT_COUNT and all(v >= PARENTS_PER_TASK for v in task_counts.values())
    return fresh, selected if capacity_ok else [], {
        "fresh_count": len(fresh),
        "duplicate_count": len(duplicates),
        "task_counts": task_counts,
        "target_parent_count": TARGET_PARENT_COUNT,
        "parents_per_task": PARENTS_PER_TASK,
        "all_tasks_have_target_capacity": all(v >= PARENTS_PER_TASK for v in task_counts.values()),
        "selection_capacity_ok": capacity_ok,
        "selection_salt": SELECTION_SALT,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output root: {out}")
    out.mkdir(parents=True, exist_ok=True)

    authority, official_by_key, official_rows = build_official_inventory(args.official_root)
    script_rel = Path("scripts/stage_ac/build_stage_ac1r1_static_population.py")
    script_path = args.project_root / script_rel
    if not script_path.is_file():
        script_path = Path(__file__).resolve()
    script_binding = file_binding(script_path, str(script_rel).replace("\\", "/"))
    project_source = source_root_binding(args.project_root)
    hits, history_sources, errors, unresolved = history_hits(args.project_root, official_by_key)
    fresh_rows, selected_rows, selection = select_fresh(official_rows, hits)
    target_history = {key: value for key, value in hits.items() if key in official_by_key}
    history_keys = sorted(target_history)
    history_set_sha = sha256_bytes(("\n".join(history_keys) + "\n").encode("utf-8")) if history_keys else sha256_bytes(b"")
    fresh_keys = sorted(row["canonical_parent_key"] for row in fresh_rows)
    fresh_set_sha = sha256_bytes(("\n".join(fresh_keys) + "\n").encode("utf-8")) if fresh_keys else sha256_bytes(b"")
    status = (
        "STAGE_AC_AC1R1_NEW_POPULATION_AUTHORITY_HOLD_STOP_FOR_PI"
        if errors
        else "STAGE_AC_AC1R1_NEW_POPULATION_CAPACITY_HOLD_STOP_FOR_PI"
        if not selection["selection_capacity_ok"]
        else "STAGE_AC_AC1R1_NEW_POPULATION_FREEZE_PASS_STOP_FOR_PI"
    )

    universe_report = {
        "schema": SCHEMA,
        "status": status,
        "gate": GATE,
        "claim_boundary": "Static official init-state inventory only; no model inference, env.step, treatment, endpoint, or scientific outcome.",
        "official_authority": authority,
        "history_policy": {
            "blacklist_static_frozen_identity_reservations": True,
            "blacklist_reason": "No historical identity becomes fresh by renaming or moving namespace.",
            "digest_reconstruction": "Each target historical key is resolved against the exact official row digest in this inventory.",
        },
        "inventory_counts": {
            "official_rows": len(official_rows),
            "historical_target_keys": len(history_keys),
            "fresh_rows_before_duplicate_filter": len(fresh_rows) + selection["duplicate_count"],
            "fresh_unique_rows": len(fresh_rows),
            "selected_rows": len(selected_rows),
        },
        "rows": official_rows,
        "errors": errors,
    }
    blacklist_report = {
        "schema": "STAGE_AC_AC1R1_HISTORICAL_EXPOSURE_BLACKLIST_V1",
        "status": status,
        "gate": GATE,
        "claim_boundary": "Historical freshness firewall; entries include runtime exposure and prior frozen identity reservations, with basis preserved.",
        "source_authorities": history_sources,
        "history_key_set_sha256": history_set_sha,
        "target_key_count": len(history_keys),
        "unresolved_out_of_scope_records": unresolved,
        "errors": errors,
        "records": [
            {
                "canonical_parent_key": key,
                "official_state_sha256": official_by_key[key]["state_sha256"],
                "basis": sorted({hit["basis"] for hit in target_history[key]}),
                "source_hits": target_history[key],
            }
            for key in history_keys
        ],
    }
    fresh_report = {
        "schema": "STAGE_AC_AC1R1_FRESH_UNIVERSE_V1",
        "status": status,
        "gate": GATE,
        "claim_boundary": "Fresh official init-state candidates after the complete historical firewall; no model inference or scientific exposure.",
        "selection": selection,
        "fresh_set_sha256": fresh_set_sha,
        "selected_parent_keys": [row["canonical_parent_key"] for row in selected_rows],
        "fresh_rows": fresh_rows,
        "selected_rows": selected_rows,
    }
    outputs = {}
    outputs["universe"] = write_json(out / "STAGE_AC_AC1R1_OFFICIAL_INIT_STATE_UNIVERSE_V1.json", universe_report)
    outputs["blacklist"] = write_json(out / "STAGE_AC_AC1R1_HISTORICAL_EXPOSURE_BLACKLIST_V1.json", blacklist_report)
    outputs["fresh"] = write_json(out / "STAGE_AC_AC1R1_FRESH_UNIVERSE_V1.json", fresh_report)
    root_payload = {
        "schema": "STAGE_AC_AC1R1_ROOT_SEAL_V1",
        "status": status,
        "gate": GATE,
        "source_authorities": {
            "official_repo": authority["libero_source"],
            "project_source": project_source,
            "builder_script": script_binding,
            "history_sources": history_sources,
            "selection_salt": SELECTION_SALT,
        },
        "outputs": outputs,
        "population": {
            "official_rows": len(official_rows),
            "historical_target_keys": len(history_keys),
            "historical_key_set_sha256": history_set_sha,
            "fresh_unique_rows": len(fresh_rows),
            "fresh_set_sha256": fresh_set_sha,
            "selected_rows": len(selected_rows),
            "selection": selection,
            "unresolved_out_of_scope_count": len(unresolved),
            "error_count": len(errors),
        },
        "scientific_firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "open_intervention": 0,
            "pgd": 0,
            "v_phys": 0,
            "protected_or_eval160": 0,
            "fresh_scientific_identity_exposure": 0,
        },
        "next_legal_action": (
            "STOP_FOR_PI_REVIEW_AND_SEPARATE_NEW_POPULATION_AUTHORITY"
            if status != "STAGE_AC_AC1R1_NEW_POPULATION_FREEZE_PASS_STOP_FOR_PI"
            else "STOP_FOR_PI_REVIEW_BEFORE_AC2_CLEAN_SCREEN"
        ),
    }
    root_payload_hash = sha256_bytes(canonical_json(root_payload))
    root = dict(root_payload)
    root["root_payload_sha256"] = root_payload_hash
    root_binding = write_json(out / "STAGE_AC_AC1R1_ROOT_SEAL_V1.json", root)
    sidecar = f"{root_binding['sha256']}  STAGE_AC_AC1R1_ROOT_SEAL_V1.json\n".encode("ascii")
    (out / "STAGE_AC_AC1R1_ROOT_SEAL_V1.sha256").write_bytes(sidecar)
    print(json.dumps({"status": status, "outputs": {**outputs, "root": root_binding}, "root_payload_sha256": root_payload_hash, "selection": selection}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
