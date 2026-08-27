#!/usr/bin/env python3
"""Build the Stage-AC treatment-naive exposure taxonomy.

Static/offline only.  The old AC1R1 ``any historical listing`` blacklist is
intentionally not reused: a static mention is recorded as ``listed`` but does
not by itself make an identity non-treatment-naive.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from build_stage_ac1r1_static_population import (
    canonical_encoding,
    canonical_json,
    canonical_row_bytes,
    file_binding,
    git_binding,
    load_init,
    load_task_map,
    sha256_bytes,
    state_digest,
    task_language,
)


SCHEMA = "STAGE_AC_AC1R2_OFFICIAL_STATE_EXPOSURE_TAXONOMY_V1"
POPULATION_SCHEMA = "STAGE_AC_AC1R2_TREATMENT_NAIVE_POPULATION_V1"
INDEX_SCHEMA = "STAGE_AC_AC1R2_HISTORICAL_EXPOSURE_AUTHORITY_INDEX_V1"
ROOT_SCHEMA = "STAGE_AC_AC1R2_ROOT_SEAL_V1"
GATE = "STAGE_AC_AC1R2_TREATMENT_NAIVE_OFFICIAL_POPULATION_RECLASSIFICATION_V1"
SUITES = ("libero_10", "libero_object", "libero_spatial", "libero_goal")
PRIMARY_SUITES = ("libero_10", "libero_object", "libero_spatial")
STATES_PER_TASK = 50
KEY_RE = re.compile(r"^libero_(?:10|object|spatial|goal)/task_\d{2}/state_\d{2}$")

FLAGS = (
    "ever_listed_in_historical_population",
    "ever_model_inference",
    "ever_clean_env_step",
    "ever_clean_eligibility_screen",
    "ever_engineering_canary",
    "ever_open_intervention",
    "ever_oracle_open_intervention",
    "ever_visual_or_pgd_intervention",
    "ever_attacked_env_step",
    "ever_physical_endpoint_read",
    "ever_v_phys_read",
    "ever_aa_v_phys_read",
    "ever_manual_or_ai_physical_outcome_review",
    "ever_task_success_used_for_attack_analysis",
    "protected_or_eval160_exposure",
)

CLASSES = (
    "H0_UNTOUCHED",
    "HC_CLEAN_ONLY",
    "HE_ENGINEERING_ONLY",
    "HT_OPEN_TREATMENT_EXPOSED",
    "HY_PHYSICAL_OUTCOME_EXPOSED",
    "HV_VISUAL_OR_PGD_EXPOSED",
    "HP_PROTECTED",
    "HU_UNRESOLVED",
)

ENGINEERING_CANARIES = {
    "libero_10/task_04/state_20",
    "libero_object/task_02/state_42",
    "libero_spatial/task_05/state_34",
}

SOURCE_SPECS = (
    ("stage_x_g10", "reports/STAGE_X_X1R_T1D0R1_G10_IDENTITY_EXCLUSION_LEDGER_V1.json", "g10"),
    (
        "stage_x_f1a3_classification",
        "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_EXPOSURE_CLASSIFICATION_V3.json",
        "f1a3",
    ),
    (
        "stage_x_f1a3_split_ledger",
        "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_SPLIT_LEDGER_V3.json",
        "listed",
    ),
    ("stage_x_m4_manifest", "reports/server_evidence/STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json", "m4"),
    ("stage_x_vi_b2_manifest", "configs/STAGE_VI_B2_FRESH_PARENT_MANIFEST_V3.json", "b2"),
    ("stage_x_physical_alias", "reports/STAGE_X_X1R_T1D0R1_PHYSICAL_ALIAS_LEDGER_V1.json", "alias"),
    ("stage_x_x0_result", "docs/handoffs/STAGE_X_X0_RESULT_20260817.md", "x0_authority"),
    ("stage_x_x0_protocol", "configs/STAGE_X_X0_DUTY_CYCLE_MECHANISM_PROTOCOL_V1.json", "authority"),
    ("stage_x_f1a3_contract", "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_SOURCE_LEAKAGE_CONTRACT_V3.json", "authority"),
    ("stage_x_e3_pool", "reports/STAGE_X_X1R2_E3_SELECTIVE_REALIZABILITY_POOL_V1.json", "e3"),
    ("stage_x_e3_decision", "reports/STAGE_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_20260821/E3_DECISION_TABLE_V1.json", "e3"),
    ("stage_x_e3_root", "reports/STAGE_X1R2_E3_FACTORIZED_SELECTIVE_REALIZABILITY_20260821/E3_ROOT_SEAL_V1.json", "authority"),
    ("stage_x_f1c4_ledger", "reports/STAGE_X1R2_F1C4_FRESH_CANARY_NAMESPACE_V1_20260822/F1C4_FRESH_CANARY_LEDGER_V1.json", "f1c4"),
    ("stage_x_f1c4_runtime", "reports/STAGE_X1R2_F1C4_FRESH_CANARY_RESULT_V1_R3_20260822/F1C4_RUNTIME_AUDIT_V1.json", "f1c4_runtime"),
    ("stage_x_ac1r1_universe", "reports/STAGE_AC_AC1R1_OFFICIAL_INIT_STATE_UNIVERSE_V1.json", "listed"),
    ("stage_x_ac1r1_blacklist", "reports/STAGE_AC_AC1R1_HISTORICAL_EXPOSURE_BLACKLIST_V1.json", "listed"),
    ("stage_x_ac1r1_fresh", "reports/STAGE_AC_AC1R1_FRESH_UNIVERSE_V1.json", "listed"),
    ("stage_z_z0", "reports/STAGE_Z_Z0_SHARED_40_IDENTITY_PANEL_V1.json", "listed"),
    ("stage_z_z0r1", "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json", "listed"),
    ("stage_z_z2", "reports/STAGE_Z_Z2_SCIENTIFIC_EXPOSURE_LEDGER_V2.json", "z2"),
    ("stage_z_z3c_index", "reports/STAGE_Z_Z3C_BRANCH_RECEIPT_INDEX_V1.json", "z3c"),
    ("stage_z_z3c_terminal", "reports/STAGE_Z_Z3C_TERMINAL_SYNTHESIS_V1.json", "z3c_terminal"),
    ("stage_z_z3c_root", "reports/STAGE_Z_Z3C_ROOT_SEAL_V1.json", "authority"),
    ("stage_z_z3d_mapping", "reports/server_evidence/STAGE_Z_Z3D_BLINDED_VIDEO_MAPPING_V1.json", "z3d"),
    (
        "stage_z_z3d_reconciliation",
        "reports/server_evidence/STAGE_Z_Z3D_AI_SECONDARY_RECONCILIATION_V1R1/STAGE_Z_Z3D_AI_SECONDARY_UNBLIND_RECONCILIATION_V1.json",
        "z3d",
    ),
    ("stage_z_z3dh_panel", "reports/STAGE_Z_Z3DH_AI_PANEL_CROSS_AUDIT_V1.json", "z3dh"),
    ("stage_z_z4_synthesis", "reports/STAGE_Z_Z4_STATIC_CROSS_MODEL_SYNTHESIS_V1.json", "authority"),
    ("stage_aa_aa0", "reports/STAGE_AA_AA0_FRESH_CAPACITY_INVENTORY_V1.json", "aa0"),
    ("stage_aa_aa1_terminal", "reports/STAGE_AA_AA1_ENGINEERING_CANARY_TERMINAL_V1.json", "aa1"),
    ("stage_aa_aa1_receipt_index", "reports/STAGE_AA_AA1_ENGINEERING_CANARY_RECEIPT_INDEX_V1.json", "aa1"),
    ("stage_aa_aa1r1_terminal", "reports/STAGE_AA_AA1R1_ENGINEERING_BRANCH_TERMINAL_V1.json", "aa1"),
    ("stage_aa_aa1r1_receipt_index", "reports/STAGE_AA_AA1R1_ENGINEERING_BRANCH_RECEIPT_INDEX_V1.json", "aa1"),
    ("stage_aa_aa2_launch", "reports/STAGE_AA_AA2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json", "listed"),
    ("stage_aa_aa2r2_census", "reports/STAGE_AA_AA2R2_PHASE_B_V2_CENSUS_TERMINAL_V1.json", "aa2r2"),
    ("stage_ac_ac0", "reports/STAGE_AC_AC0_CONSTRUCT_VALIDATION_TERMINAL_V1.json", "ac0"),
)


def parse_document(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def walk_records(value):
    if isinstance(value, dict):
        key = value.get("canonical_parent_key")
        if isinstance(key, str) and KEY_RE.fullmatch(key):
            yield value
        for child in value.values():
            yield from walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_records(child)


def positive_counter(value, names: set[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in names and isinstance(child, (int, float)) and child > 0:
                return True
            if positive_counter(child, names):
                return True
    elif isinstance(value, list):
        return any(positive_counter(child, names) for child in value)
    return False


def true_value(value, names: set[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in names and child is True:
                return True
            if true_value(child, names):
                return True
    elif isinstance(value, list):
        return any(true_value(child, names) for child in value)
    return False


def build_official_inventory(official_root: Path):
    task_map_path = official_root / "libero/libero/benchmark/libero_suite_task_map.py"
    task_map = load_task_map(task_map_path)
    repo_git = git_binding(official_root)
    task_map_binding = file_binding(task_map_path, "libero/libero/benchmark/libero_suite_task_map.py")
    tasks = []
    rows = []
    by_key = {}
    for suite in SUITES:
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
                encoding = canonical_encoding(array.shape[1])
                digest = state_digest(raw, encoding)
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
                    "state_shape": [int(array.shape[1])],
                    "canonical_encoding": encoding,
                    "state_sha256": digest,
                    "state_bytes_base64": base64.b64encode(raw).decode("ascii"),
                    "source_init_file": init_binding,
                    "source_bddl_file": bddl_binding,
                    "duplicate_of_state_index": duplicate_of,
                }
                rows.append(record)
                by_key[key] = record
    return {
        "libero_source": {
            "root": str(official_root),
            "git": repo_git,
            "task_map": task_map_binding,
        },
        "target_suites": list(SUITES),
        "task_count": len(tasks),
        "row_count": len(rows),
        "states_per_task": STATES_PER_TASK,
        "tasks": tasks,
        "canonical_encoding_family": "STAGE_AC_INIT_STATE_CANONICAL_NUMERIC_V1|dtype=<f8|order=C|shape=<per-task-width>",
    }, by_key, rows


class Exposure:
    def __init__(self, keys: set[str], source_paths: dict[str, str]):
        self.keys = keys
        self.source_paths = source_paths
        self.flags = {key: {name: False for name in FLAGS} for key in keys}
        self.hits = {key: {} for key in keys}
        self.unresolved = {key: set() for key in keys}
        self.out_of_scope = Counter()

    def add(self, key: str, source_id: str, *flags: str, note: str | None = None):
        if key not in self.keys:
            self.out_of_scope[source_id] += 1
            return
        for flag in flags:
            self.flags[key][flag] = True
        hit = self.hits[key].setdefault(source_id, {"rows": 0, "flags": set(), "notes": set()})
        hit["rows"] += 1
        hit["flags"].update(flags)
        if note:
            hit["notes"].add(note)

    def unresolved_key(self, key: str, source_id: str, reason: str):
        if key not in self.keys:
            self.out_of_scope[source_id] += 1
            return
        self.unresolved[key].add(f"{source_id}:{reason}")
        self.add(key, source_id, note=reason)

    def apply_explicit_counters(self, key: str, source_id: str, record: dict):
        if positive_counter(record, {"model_inference_calls", "inference_calls"}) or true_value(record, {"model_inference", "model_inference_read"}):
            self.add(key, source_id, "ever_model_inference")
        if positive_counter(record, {"env_step_calls", "clean_env_step_calls"}):
            self.add(key, source_id, "ever_clean_env_step")
        if positive_counter(record, {"open_intervention_steps", "physical_interventions"}) or true_value(record, {"open_intervention", "physical_intervention"}):
            self.add(key, source_id, "ever_open_intervention", "ever_oracle_open_intervention")
        if positive_counter(record, {"pgd_calls", "visual_pgd_calls"}) or true_value(record, {"visual_or_pgd_intervention", "visual_attack"}):
            self.add(key, source_id, "ever_visual_or_pgd_intervention")
        if positive_counter(record, {"attacked_env_steps"}) or true_value(record, {"attacked_env_step"}):
            self.add(key, source_id, "ever_attacked_env_step")
        if positive_counter(record, {"physical_telemetry_reads", "physical_endpoint_reads"}) or true_value(record, {"physical_endpoint_read"}):
            self.add(key, source_id, "ever_physical_endpoint_read")
        if positive_counter(record, {"v_phys_reads", "vphys_reads"}) or true_value(record, {"v_phys_read", "vphys_read"}):
            self.add(key, source_id, "ever_v_phys_read")
        if positive_counter(record, {"aa_v_phys_reads"}) or true_value(record, {"aa_v_phys_read"}):
            self.add(key, source_id, "ever_aa_v_phys_read")
        if positive_counter(record, {"task_success_reads"}) or true_value(record, {"task_success_used_for_attack_analysis"}):
            self.add(key, source_id, "ever_task_success_used_for_attack_analysis")
        if positive_counter(record, {"protected_reads", "eval160_reads", "protected_semantic_reads"}) or true_value(record, {"protected_or_eval160_exposure", "protected_payload_read", "eval160_read"}):
            self.add(key, source_id, "protected_or_eval160_exposure")

    def finalize_hits(self):
        result = {}
        for key, source_hits in self.hits.items():
            result[key] = []
            for source_id, hit in sorted(source_hits.items()):
                result[key].append(
                    {
                        "source_id": source_id,
                        "path": self.source_paths[source_id],
                        "rows": hit["rows"],
                        "flags": sorted(hit["flags"]),
                        "notes": sorted(hit["notes"]),
                    }
                )
        return result


def official_counter(value, names: set[str]) -> bool:
    return positive_counter(value, names) or true_value(value, names)


def apply_source(source_id: str, mode: str, document, exposure: Exposure):
    records = list(walk_records(document))
    for record in records:
        key = record["canonical_parent_key"]
        exposure.apply_explicit_counters(key, source_id, record)
        if mode in {"listed", "g10", "m4", "b2", "alias", "z2", "z3c", "z3c_terminal", "z3d", "z3dh", "aa0", "aa2r2", "ac0", "aa1", "e3", "f1c4", "f1c4_runtime"}:
            exposure.add(key, source_id, "ever_listed_in_historical_population")
        if mode == "g10":
            if record.get("prior_clean_attempt"):
                exposure.add(key, source_id, "ever_model_inference", "ever_clean_env_step", "ever_clean_eligibility_screen", note="G10 prior clean attempt")
            if record.get("prior_exposure"):
                exposure.unresolved_key(key, source_id, "prior_exposure_semantics_not_specific_enough")
            if record.get("stage_v_physical_matrix_6d39860") or record.get("stage_v_physical_matrix_f696f582"):
                exposure.add(key, source_id, "ever_open_intervention", "ever_oracle_open_intervention", "ever_physical_endpoint_read", "ever_v_phys_read", note="G10 explicit Stage-V physical-matrix membership")
            if record.get("stage_v_formal_population") or record.get("stage_vi_b2_development") or record.get("stage_vi_b2_population"):
                exposure.add(key, source_id, "ever_clean_eligibility_screen", note="historical clean/development population marker")
        elif mode == "f1a3":
            source_class = record.get("source_class")
            if source_class == "V3_PRISTINE_NO_RELEVANT_IDENTITY_EXPOSURE":
                pass
            elif source_class in {"V3_DETECTOR_TRAIN_ONLY_ALLOWED_WITH_FLAG", "V3_HARD_EXECUTION_EXPOSURE_EXCLUDE"}:
                exposure.add(key, source_id, "ever_engineering_canary", note=source_class)
                if source_class == "V3_HARD_EXECUTION_EXPOSURE_EXCLUDE":
                    exposure.add(key, source_id, "ever_model_inference", "ever_clean_env_step", note="F1A3 hard-execution exclusion")
            else:
                exposure.unresolved_key(key, source_id, f"unrecognized_f1a3_source_class:{source_class}")
        elif mode == "m4":
            if record.get("status_pair") == "PASS/PASS":
                exposure.add(key, source_id, "ever_model_inference", "ever_clean_env_step", "ever_clean_eligibility_screen", note="M4 clean-only PASS/PASS corridor")
        elif mode == "b2":
            exposure.add(key, source_id, "ever_clean_eligibility_screen", note="B2 clean rollout population manifest")
        elif mode == "alias":
            if record.get("physical_intervention_semantics_from_directory_name") not in {"IDENTITY_ONLY", "NOT_IDENTIFIABLE"}:
                exposure.unresolved_key(key, source_id, "unknown_alias_semantics")
        elif mode == "x0_authority":
            # The handoff is an authority binding; its parent population is
            # attached below from the exact M4 and B2 manifests.
            pass
        elif mode == "z2":
            if record.get("model_parent_cells") or record.get("exposing_models") or record.get("model_family"):
                exposure.add(key, source_id, "ever_model_inference", "ever_clean_env_step", "ever_clean_eligibility_screen", note="Z2 scientific clean reference")
        elif mode == "z3c":
            if record.get("arm") and record.get("arm") != "CLEAN_BRANCH_CRITICAL":
                exposure.add(key, source_id, "ever_open_intervention", "ever_oracle_open_intervention", note="Z3-C command-OPEN branch")
            exposure.add(key, source_id, "ever_physical_endpoint_read", note="Z3-C physical branch telemetry")
        elif mode == "z3c_terminal":
            # The terminal's parent_rows are the identity-level authority for
            # V_phys labels; the branch index alone contains no such field.
            exposure.add(key, source_id, "ever_physical_endpoint_read", "ever_v_phys_read", note="Z3-C parent-level V_phys diagnostic")
        elif mode == "z3d":
            exposure.add(key, source_id, "ever_manual_or_ai_physical_outcome_review", note="Z3-D AI-secondary unblind/audit mapping")
            exposure.add(key, source_id, "ever_physical_endpoint_read")
            exposure.add(key, source_id, "ever_v_phys_read")
            if str(record.get("arm", "")).startswith("COMMAND_OPEN"):
                exposure.add(key, source_id, "ever_open_intervention", "ever_oracle_open_intervention")
        elif mode == "z3dh":
            exposure.add(key, source_id, "ever_manual_or_ai_physical_outcome_review", note="Z3-DH AI panel cross-audit")
        elif mode == "aa0":
            if key in ENGINEERING_CANARIES:
                exposure.add(key, source_id, "ever_engineering_canary", note="AA0 reserved engineering canary")
        elif mode == "aa1":
            if key not in ENGINEERING_CANARIES:
                exposure.unresolved_key(key, source_id, "AA1 source contains non-canary identity")
            else:
                exposure.add(key, source_id, "ever_engineering_canary", note="AA1 engineering canary")
        elif mode == "aa2r2":
            exposure.add(key, source_id, "ever_model_inference", "ever_clean_env_step", "ever_clean_eligibility_screen", note="AA2R2 clean-only census")
        elif mode == "ac0":
            if key not in ENGINEERING_CANARIES:
                exposure.unresolved_key(key, source_id, "AC0 source contains non-canary identity")
            else:
                exposure.add(key, source_id, "ever_model_inference", "ever_clean_env_step", "ever_clean_eligibility_screen", "ever_engineering_canary", "ever_physical_endpoint_read", note="AC0 consumed-only construct calibration")
        elif mode == "e3":
            exposure.add(key, source_id, "ever_engineering_canary", "ever_model_inference", "ever_visual_or_pgd_intervention", note="E3 permanently excluded visual/PGD engineering probe")
        elif mode == "f1c4":
            exposure.add(key, source_id, "ever_engineering_canary", "ever_visual_or_pgd_intervention", note="F1C4 permanently excluded visual/PGD canary namespace")
        elif mode == "f1c4_runtime":
            exposure.add(key, source_id, "ever_engineering_canary", "ever_model_inference", "ever_clean_env_step", "ever_visual_or_pgd_intervention", note="F1C4 runtime canary")
            if official_counter(record, {"attacked_env_steps"}):
                exposure.add(key, source_id, "ever_attacked_env_step")


def classify(flags: dict[str, bool], unresolved: set[str]) -> str:
    if unresolved:
        return "HU_UNRESOLVED"
    if flags["protected_or_eval160_exposure"]:
        return "HP_PROTECTED"
    if flags["ever_v_phys_read"] or flags["ever_manual_or_ai_physical_outcome_review"]:
        return "HY_PHYSICAL_OUTCOME_EXPOSED"
    if flags["ever_visual_or_pgd_intervention"] or flags["ever_attacked_env_step"]:
        return "HV_VISUAL_OR_PGD_EXPOSED"
    if flags["ever_open_intervention"] or flags["ever_oracle_open_intervention"]:
        return "HT_OPEN_TREATMENT_EXPOSED" if not flags["ever_engineering_canary"] else "HE_ENGINEERING_ONLY"
    if flags["ever_engineering_canary"]:
        return "HE_ENGINEERING_ONLY"
    if flags["ever_model_inference"] or flags["ever_clean_env_step"] or flags["ever_clean_eligibility_screen"]:
        return "HC_CLEAN_ONLY"
    return "H0_UNTOUCHED"


def canonical_key_set(keys: list[str]) -> str:
    return sha256_bytes(("\n".join(sorted(keys)) + "\n").encode("utf-8")) if keys else sha256_bytes(b"")


def write_json(path: Path, value: object) -> dict:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": path.name, "bytes": len(data), "sha256": sha256_bytes(data)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output root: {out}")
    out.mkdir(parents=True, exist_ok=True)

    authority, official_by_key, official_rows = build_official_inventory(args.official_root)
    source_paths = {source_id: relative_path for source_id, relative_path, _ in SOURCE_SPECS}
    exposure = Exposure(set(official_by_key), source_paths)
    source_bindings = []
    source_errors = []
    source_record_counts = {}
    for source_id, relative_path, mode in SOURCE_SPECS:
        project_path = args.project_root / relative_path
        overlay_path = args.history_root / relative_path if args.history_root else None
        path = project_path if project_path.is_file() else overlay_path
        if path is None or not path.is_file():
            source_errors.append({"source_id": source_id, "path": relative_path, "error": "MISSING_REQUIRED_HISTORY_AUTHORITY"})
            continue
        binding = file_binding(path, relative_path)
        if path != project_path:
            binding["resolved_path"] = str(path)
        if mode == "x0_authority":
            source_bindings.append({"source_id": source_id, "mode": mode, "file": binding, "parse_status": "TEXT_AUTHORITY", "canonical_record_count": 0})
            continue
        try:
            document = parse_document(path)
            records = list(walk_records(document))
        except Exception as exc:
            source_errors.append({"source_id": source_id, "path": relative_path, "error": f"PARSE_ERROR:{type(exc).__name__}:{exc}"})
            source_bindings.append({"source_id": source_id, "mode": mode, "file": binding, "parse_status": "ERROR"})
            continue
        source_record_counts[source_id] = len(records)
        source_bindings.append({"source_id": source_id, "mode": mode, "file": binding, "parse_status": "PASS", "canonical_record_count": len(records)})
        if mode != "authority":
            apply_source(source_id, mode, document, exposure)

    # Resolve the G10 name-only physical-root marker with the explicit clean-only
    # M4 corridor authority.  The one key outside M4 is already explicit in the
    # Stage-V physical matrix and therefore does not remain ambiguous.
    g10_path = args.project_root / "reports/STAGE_X_X1R_T1D0R1_G10_IDENTITY_EXCLUSION_LEDGER_V1.json"
    m4_path = args.project_root / "reports/server_evidence/STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json"
    g10_rows = list(walk_records(parse_document(g10_path))) if g10_path.is_file() else []
    m4_keys = {row["canonical_parent_key"] for row in walk_records(parse_document(m4_path))} if m4_path.is_file() else set()
    b2_path = args.project_root / "configs/STAGE_VI_B2_FRESH_PARENT_MANIFEST_V3.json"
    b2_keys = {row["canonical_parent_key"] for row in walk_records(parse_document(b2_path))} if b2_path.is_file() else set()
    physical_matrix_keys = {
        row["canonical_parent_key"]
        for row in g10_rows
        if row.get("stage_v_physical_matrix_6d39860") or row.get("stage_v_physical_matrix_f696f582")
    }
    for row in g10_rows:
        if row.get("prior_physical_intervention_named_roots") and row["canonical_parent_key"] not in physical_matrix_keys and row["canonical_parent_key"] not in m4_keys:
            exposure.unresolved_key(row["canonical_parent_key"], "stage_x_g10", "name_only_physical_root_not_resolved_by_clean_corridor")

    # X0 is retrospective and execution-free, but its sealed handoff states
    # that it consumed exact physical/V_phys evidence for all 40 Stage-V
    # parents and all 16 Stage-VI-B2 parents.  Bind those populations through
    # their exact manifests rather than guessing keys from prose.
    x0_population = m4_keys | b2_keys
    for key in sorted(x0_population):
        exposure.add(
            key,
            "stage_x_x0_result",
            "ever_open_intervention",
            "ever_oracle_open_intervention",
            "ever_physical_endpoint_read",
            "ever_v_phys_read",
            note="X0 retrospective V_phys evidence for Stage-V/B2 parent population",
        )

    evidence = exposure.finalize_hits()
    rows = []
    class_counts = Counter()
    suite_class_counts = defaultdict(Counter)
    for official in official_rows:
        key = official["canonical_parent_key"]
        cls = classify(exposure.flags[key], exposure.unresolved[key])
        class_counts[cls] += 1
        suite_class_counts[official["suite"]][cls] += 1
        rows.append(
            {
                **official,
                "exposure_flags": exposure.flags[key],
                "exposure_status": "EXPOSURE_STATUS_UNRESOLVED" if exposure.unresolved[key] else "EXPOSURE_STATUS_RESOLVED",
                "unresolved_reasons": sorted(exposure.unresolved[key]),
                "exposure_class": cls,
                "treatment_naive": cls in {"H0_UNTOUCHED", "HC_CLEAN_ONLY"},
                "primary_stage_ac_candidate": official["suite"] in PRIMARY_SUITES and cls in {"H0_UNTOUCHED", "HC_CLEAN_ONLY"},
                "goal_ac0g_pending": official["suite"] == "libero_goal",
                "evidence": evidence[key],
            }
        )

    treatment_naive = [row for row in rows if row["treatment_naive"]]
    primary = [row for row in rows if row["primary_stage_ac_candidate"]]
    primary_keys = [row["canonical_parent_key"] for row in primary]
    all_tn_keys = [row["canonical_parent_key"] for row in treatment_naive]
    status = "STAGE_AC_AC1R2_TREATMENT_NAIVE_TAXONOMY_PASS_STOP_FOR_PI" if not source_errors else "STAGE_AC_AC1R2_EXPOSURE_AUTHORITY_HOLD_STOP_FOR_PI"
    claim_boundary = (
        "Static official-init exposure reclassification only; no model inference, env.step, simulator, treatment, endpoint, "
        "protected read, or scientific outcome was executed by AC1R2. H0/HC are treatment-naive under the explicit sources; "
        "HU is never primary. Goal remains outside the primary AC population pending consumed-only AC0G."
    )
    taxonomy = {
        "schema": SCHEMA,
        "status": status,
        "gate": GATE,
        "claim_boundary": claim_boundary,
        "official_authority": authority,
        "exposure_classes": list(CLASSES),
        "exposure_flags": list(FLAGS),
        "classification_precedence": ["HU_UNRESOLVED", "HP_PROTECTED", "HY_PHYSICAL_OUTCOME_EXPOSED", "HV_VISUAL_OR_PGD_EXPOSED", "HT_OPEN_TREATMENT_EXPOSED", "HE_ENGINEERING_ONLY", "HC_CLEAN_ONLY", "H0_UNTOUCHED"],
        "source_authorities": source_bindings,
        "source_errors": source_errors,
        "counts": {
            "official_rows": len(rows),
            "official_duplicate_rows_within_task": sum(row["duplicate_of_state_index"] is not None for row in rows),
            "treatment_naive_rows_all_suites": len(treatment_naive),
            "primary_stage_ac_candidate_rows": len(primary),
            "goal_treatment_naive_rows_held_pending_ac0g": sum(row["suite"] == "libero_goal" for row in treatment_naive),
            "unresolved_rows": class_counts["HU_UNRESOLVED"],
        },
        "class_counts": dict(sorted(class_counts.items())),
        "suite_class_counts": {suite: dict(sorted(suite_class_counts[suite].items())) for suite in SUITES},
        "rows": rows,
    }
    population = {
        "schema": POPULATION_SCHEMA,
        "status": status,
        "gate": GATE,
        "claim_boundary": claim_boundary,
        "selection_policy": {
            "old_ac1r1_240_parent_rule_retired": True,
            "complete_candidate_universe_frozen_before_ac2": True,
            "primary_suites": list(PRIMARY_SUITES),
            "goal_excluded_until_ac0g_pass": True,
            "no_replacement": True,
            "no_top_up": True,
            "no_selection_by_treatment_outcome": True,
        },
        "candidate_universe": {
            "all_treatment_naive_keys": sorted(all_tn_keys),
            "all_treatment_naive_key_set_sha256": canonical_key_set(all_tn_keys),
            "primary_stage_ac_keys": sorted(primary_keys),
            "primary_stage_ac_key_set_sha256": canonical_key_set(primary_keys),
        },
        "counts_by_suite": {
            suite: {
                "treatment_naive": sum(row["treatment_naive"] and row["suite"] == suite for row in rows),
                "primary_stage_ac_candidate": sum(row["primary_stage_ac_candidate"] and row["suite"] == suite for row in rows),
                "goal_ac0g_pending": suite == "libero_goal",
            }
            for suite in SUITES
        },
        "rows": [row for row in rows if row["treatment_naive"]],
    }
    history_index = {
        "schema": INDEX_SCHEMA,
        "status": status,
        "gate": GATE,
        "claim_boundary": claim_boundary,
        "official_source": authority["libero_source"],
        "source_authorities": source_bindings,
        "source_record_counts": source_record_counts,
        "source_errors": source_errors,
        "out_of_scope_record_counts": dict(sorted(exposure.out_of_scope.items())),
        "classification_policy": {
            "static_listing_only_does_not_equal_treatment": True,
            "prior_clean_is_allowed_as_historical_observation": True,
            "prior_open_or_physical_outcome_is_excluded": True,
            "ambiguous_evidence_is_hu_unresolved": True,
            "protected_or_eval160_never_read_by_ac1r2": True,
        },
        "identity_records": [
            {
                "canonical_parent_key": row["canonical_parent_key"],
                "exposure_class": row["exposure_class"],
                "exposure_status": row["exposure_status"],
                "exposure_flags": row["exposure_flags"],
                "unresolved_reasons": row["unresolved_reasons"],
                "evidence": row["evidence"],
            }
            for row in rows
        ],
    }
    script_rel = Path("scripts/stage_ac/build_stage_ac1r2_treatment_naive_taxonomy.py")
    script_path = args.project_root / script_rel
    script_binding = file_binding(script_path if script_path.is_file() else Path(__file__).resolve(), str(script_rel).replace("\\", "/"))
    try:
        project_git = git_binding(args.project_root)
    except (OSError, subprocess.CalledProcessError, ValueError):
        project_git = None
    root_payload = {
        "schema": ROOT_SCHEMA,
        "status": status,
        "gate": GATE,
        "source_authorities": {
            "official_repo": authority["libero_source"],
            "project_source": {"path": str(args.project_root), "git": project_git, "status": "NOT_A_GIT_CHECKOUT" if project_git is None else "BOUND"},
            "builder_script": script_binding,
            "history_sources": source_bindings,
        },
        "outputs": {},
        "population": {
            "official_rows": len(rows),
            "class_counts": dict(sorted(class_counts.items())),
            "treatment_naive_all_suites": len(treatment_naive),
            "primary_stage_ac_candidates": len(primary),
            "goal_pending_ac0g": sum(row["suite"] == "libero_goal" for row in treatment_naive),
            "primary_key_set_sha256": canonical_key_set(primary_keys),
            "all_treatment_naive_key_set_sha256": canonical_key_set(all_tn_keys),
            "source_error_count": len(source_errors),
        },
        "scientific_firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "simulator": 0,
            "open_intervention": 0,
            "visual_or_pgd": 0,
            "physical_endpoint_read": 0,
            "v_phys": 0,
            "protected_or_eval160": 0,
            "new_identity": 0,
        },
        "next_legal_action": "STOP_FOR_PI_REVIEW_BEFORE_AC0G_OR_AC2" if not source_errors else "STOP_FOR_PI_AUTHORITY_REPAIR",
    }
    outputs = {}
    outputs["taxonomy"] = write_json(out / "STAGE_AC_AC1R2_OFFICIAL_STATE_EXPOSURE_TAXONOMY_V1.json", taxonomy)
    outputs["population"] = write_json(out / "STAGE_AC_AC1R2_TREATMENT_NAIVE_POPULATION_V1.json", population)
    outputs["history_index"] = write_json(out / "STAGE_AC_AC1R2_HISTORICAL_EXPOSURE_AUTHORITY_INDEX_V1.json", history_index)
    root_payload["outputs"] = outputs
    root_payload_hash = sha256_bytes(canonical_json(root_payload))
    root = dict(root_payload)
    root["root_payload_sha256"] = root_payload_hash
    outputs["root"] = write_json(out / "STAGE_AC_AC1R2_ROOT_SEAL_V1.json", root)
    sidecar = f"{outputs['root']['sha256']}  STAGE_AC_AC1R2_ROOT_SEAL_V1.json\n".encode("ascii")
    (out / "STAGE_AC_AC1R2_ROOT_SEAL_V1.sha256").write_bytes(sidecar)
    print(json.dumps({"status": status, "outputs": outputs, "class_counts": dict(sorted(class_counts.items())), "primary_count": len(primary), "goal_pending": sum(row["suite"] == "libero_goal" for row in treatment_naive), "source_errors": source_errors, "root_payload_sha256": root_payload_hash}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
