#!/usr/bin/env python3
"""Freeze the Stage-AC2 treatment-naive clean-screen population.

Static only.  No model, simulator, GPU, or outcome access is performed here.
The live PI authorization fixes 30 task strata x 8 parents per stratum; the
selection is made only from the sealed AC1R2 H0/HC taxonomy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
TASKS_PER_SUITE = 10
PARENTS_PER_TASK = 8
PARENT_COUNT = len(SUITES) * TASKS_PER_SUITE * PARENTS_PER_TASK
CELL_COUNT = PARENT_COUNT * len(MODELS)
SELECTION_SALT = "STAGE_AC_AC2_TREATMENT_NAIVE_PARENT_SELECTION_V1_20260827"
CLEAN_SEED_SALT = "STAGE_AC_AC2_CLEAN_SCREEN_SEED_V1_20260827"
DENOMINATOR_SALT = "STAGE_AC_AC2_MODEL_SPECIFIC_DENOMINATOR_V1_20260827"
PI_COMMENT_ID = 5434166412
Z1_PROTOCOL = "configs/STAGE_Z_Z1_RUNTIME_PROTOCOL_V11.json"
AC0_PROTOCOL = "configs/STAGE_AC_AC0_CONSTRUCT_VALIDATION_PROTOCOL_V1.json"
AC0_ROOT = "reports/STAGE_AC_AC0_ROOT_SEAL_V1.json"
AC0_TERMINAL = "reports/STAGE_AC_AC0_CONSTRUCT_VALIDATION_TERMINAL_V1.json"
AC1R2_ROOT = "reports/STAGE_AC_AC1R2_ROOT_SEAL_V1.json"
AC1R2_POPULATION = "reports/STAGE_AC_AC1R2_TREATMENT_NAIVE_POPULATION_V1.json"
AC1R2_TAXONOMY = "reports/STAGE_AC_AC1R2_OFFICIAL_STATE_EXPOSURE_TAXONOMY_V1.json"
ELIGIBILITY = "src/stage_ac/eligibility_v2.py"
RUNNER = "scripts/stage_ac/run_stage_ac2_clean_screen.py"
WORKER = "scripts/stage_ac/run_stage_ac2_family_worker.py"
INHERITED_RUNTIME = (
    "scripts/stage_aa/run_stage_aa2r2_engineering_canary.py",
    "scripts/stage_aa/run_stage_aa1_engineering_canary.py",
    "scripts/stage_z/run_stage_z_z1_runtime_canary.py",
    "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py",
    "src/stage_aa/action_semantics_v2.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": path.as_posix(), "bytes": len(data), "sha256": sha256_bytes(data)}


def binding(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def seed_for(family: str, parent_key: str) -> tuple[int, str]:
    digest = sha256_bytes(f"{CLEAN_SEED_SALT}|{family}|{parent_key}".encode("utf-8"))
    return int(digest[:15], 16) % (2**31 - 1), digest


def parent_rank(row: dict[str, Any]) -> str:
    return sha256_bytes(f"{SELECTION_SALT}|{row['canonical_parent_key']}|{row['state_sha256']}".encode("utf-8"))


def checkpoint_spec(z1: dict[str, Any], family: str, suite: str) -> dict[str, Any]:
    spec = z1["model_families"][family]
    if family == "M0_OPENVLA":
        return {"path": spec["paths"][suite], "manifest_sha256": spec.get("checkpoint_manifest_sha256")}
    if family == "M1_OPENVLA_OFT":
        return {
            "path": f"{spec['checkpoint_root'].rstrip('/')}/{suite}",
            "manifest_sha256": spec.get("checkpoint_manifests_sha256"),
        }
    return {"path": spec["checkpoint"], "manifest_sha256": spec.get("checkpoint_manifest_sha256")}


def compact_parent(row: dict[str, Any], rank: str) -> dict[str, Any]:
    required = (
        "canonical_parent_key",
        "suite",
        "task",
        "task_index",
        "task_name",
        "official_init_index",
        "state",
        "state_dtype",
        "state_shape",
        "canonical_encoding",
        "state_sha256",
        "state_bytes_base64",
        "source_init_file",
        "source_bddl_file",
        "exposure_class",
    )
    value = {key: row[key] for key in required}
    value.update(
        {
            "treatment_naive": True,
            "exposure_status": row.get("exposure_status"),
            "selection_rank_sha256": rank,
            "selection_salt": SELECTION_SALT,
        }
    )
    return value


def build(*, write: bool) -> dict[str, Any]:
    population_path = ROOT / AC1R2_POPULATION
    taxonomy_path = ROOT / AC1R2_TAXONOMY
    ac1r2_root_path = ROOT / AC1R2_ROOT
    ac0_root_path = ROOT / AC0_ROOT
    ac0_protocol_path = ROOT / AC0_PROTOCOL
    ac0_terminal_path = ROOT / AC0_TERMINAL
    z1_path = ROOT / Z1_PROTOCOL
    population = json.loads(population_path.read_text(encoding="utf-8"))
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    z1 = json.loads(z1_path.read_text(encoding="utf-8"))
    ac0 = json.loads(ac0_protocol_path.read_text(encoding="utf-8"))
    ac0_terminal = json.loads(ac0_terminal_path.read_text(encoding="utf-8"))

    if population.get("status") != "STAGE_AC_AC1R2_TREATMENT_NAIVE_TAXONOMY_PASS_STOP_FOR_PI":
        raise RuntimeError("AC1R2_POPULATION_NOT_SEALED_PASS")
    if taxonomy.get("status") != "STAGE_AC_AC1R2_TREATMENT_NAIVE_TAXONOMY_PASS_STOP_FOR_PI":
        raise RuntimeError("AC1R2_TAXONOMY_NOT_SEALED_PASS")
    if z1.get("status") != "STAGE_Z_Z1_RUNTIME_SOURCE_AUTHORITY_FROZEN":
        raise RuntimeError("Z1_RUNTIME_AUTHORITY_NOT_FROZEN")
    if ac0.get("status") != "STAGE_AC_AC0_PROVISIONAL_ENGINEERING_CALIBRATION_ONLY":
        raise RuntimeError("AC0_PROTOCOL_NOT_FROZEN")
    if ac0.get("fresh_science_authorized") is not False:
        raise RuntimeError("AC0_FRESH_SCIENCE_FIREWALL_INVALID")
    if ac0_terminal.get("status") != "STAGE_AC_AC0_CONSTRUCT_VALIDATION_PASS_STOP_FOR_PI" or ac0_terminal.get("control_validation", {}).get("flicker_variant_selected") != "STRICT_NO_FLICKER":
        raise RuntimeError("AC0_TERMINAL_FLICKER_SELECTION_NOT_FROZEN")
    if not (population_path.is_file() and taxonomy_path.is_file() and ac1r2_root_path.is_file() and ac0_root_path.is_file() and ac0_terminal_path.is_file()):
        raise RuntimeError("AC1R2_OR_AC0_ROOT_AUTHORITY_MISSING")

    rows = [
        row
        for row in population.get("rows", [])
        if row.get("suite") in SUITES
        and row.get("exposure_class") in {"H0_UNTOUCHED", "HC_CLEAN_ONLY"}
        and row.get("treatment_naive") is True
        and row.get("primary_stage_ac_candidate") is True
        and row.get("exposure_status") == "EXPOSURE_STATUS_RESOLVED"
    ]
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[(str(row["suite"]), str(row["task"]))].append(row)
    expected_tasks = {(suite, f"task_{index:02d}") for suite in SUITES for index in range(TASKS_PER_SUITE)}
    if set(by_task) != expected_tasks:
        missing = sorted(expected_tasks - set(by_task))
        extra = sorted(set(by_task) - expected_tasks)
        raise RuntimeError(f"AC2_TASK_STRATA_INVALID:missing={missing}:extra={extra}")

    selected: list[dict[str, Any]] = []
    task_summary: list[dict[str, Any]] = []
    for suite, task in sorted(expected_tasks):
        candidates = by_task[(suite, task)]
        ranked = sorted(((parent_rank(row), row) for row in candidates), key=lambda item: (item[0], item[1]["canonical_parent_key"]))
        if len(ranked) < PARENTS_PER_TASK:
            raise RuntimeError(f"STAGE_AC_AC2_STATIC_POPULATION_CAPACITY_HOLD_STOP_FOR_PI:{suite}:{task}:{len(ranked)}")
        chosen = [compact_parent(row, rank) for rank, row in ranked[:PARENTS_PER_TASK]]
        selected.extend(chosen)
        task_summary.append(
            {
                "suite": suite,
                "task": task,
                "available_treatment_naive": len(ranked),
                "selected_count": len(chosen),
                "selected_parent_keys": [row["canonical_parent_key"] for row in chosen],
                "selection_ranks": [row["selection_rank_sha256"] for row in chosen],
            }
        )
    if len(selected) != PARENT_COUNT or len({row["canonical_parent_key"] for row in selected}) != PARENT_COUNT:
        raise RuntimeError("AC2_SELECTED_PARENT_COUNT_OR_UNIQUENESS_INVALID")

    code_commit = git_value("rev-parse", "HEAD")
    code_tree = git_value("rev-parse", "HEAD^{tree}")
    runtime_files = [binding(RUNNER), binding(WORKER), binding(ELIGIBILITY), *(binding(path) for path in INHERITED_RUNTIME)]
    parent_binding = binding(AC1R2_POPULATION)
    taxonomy_binding = binding(AC1R2_TAXONOMY)
    ac1r2_root_binding = binding(AC1R2_ROOT)
    ac0_root_binding = binding(AC0_ROOT)
    ac0_terminal_binding = binding(AC0_TERMINAL)
    ac0_protocol_binding = binding(AC0_PROTOCOL)
    z1_binding = binding(Z1_PROTOCOL)

    eligibility_v2 = ac0["eligibility_v2"]
    protocol = {
        "schema": "STAGE_AC_AC2_CLEAN_SCREEN_PROTOCOL_V1",
        "status": "STAGE_AC_AC2_CLEAN_SCREEN_PROTOCOL_AUTHORIZED_PRE_EXPOSURE",
        "gate": "STAGE_AC_AC2_TREATMENT_NAIVE_MODEL_SPECIFIC_GPU_SCREENING_AND_CONDITIONAL_AC3_V1",
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "claim_boundary": "Treatment-naive official-state AC2 clean-only screening; no OPEN treatment, endpoint, V_phys, protected read, or AC3 promotion.",
        "clean_only": True,
        "model_inference_allowed": True,
        "env_step_allowed": True,
        "open_intervention_allowed": False,
        "attack_or_pgd_allowed": False,
        "physical_endpoint_read_allowed": False,
        "v_phys_read_allowed": False,
        "task_success_read_allowed": False,
        "protected_or_eval160_allowed": False,
        "population": {
            "parent_count": PARENT_COUNT,
            "task_strata": len(task_summary),
            "parents_per_task": PARENTS_PER_TASK,
            "model_families": list(MODELS),
            "suites": list(SUITES),
            "model_parent_cell_count": CELL_COUNT,
            "full_census_required": True,
            "selection_outcome_blind": True,
            "goal_in_primary": False,
            "replacement_or_top_up": False,
        },
        "selection": {
            "parent_selection_salt": SELECTION_SALT,
            "parent_rank_rule": "sha256(selection_salt|canonical_parent_key|state_sha256), ascending within each suite/task stratum",
            "clean_seed_salt": CLEAN_SEED_SALT,
            "clean_seed_rule": "uint31(sha256(clean_seed_salt|model_family|canonical_parent_key)[:15])",
            "model_specific_denominator_salt": DENOMINATOR_SALT,
            "selection_inputs_exclude": ["AC2 eligibility result", "OPEN outcome", "V_phys", "task success", "future model response"],
        },
        "eligibility": {
            "implementation": binding(ELIGIBILITY),
            "source_protocol": ac0_protocol_binding,
            "stable_grasp_window_steps": int(eligibility_v2["stable_grasp_window_steps"]),
            "clean_continuation_steps": int(eligibility_v2["clean_continuation_steps"]),
            "minimum_lift_m": float(eligibility_v2["minimum_lift_m"]),
            "absolute_object_eef_distance_max_m": float(eligibility_v2["absolute_object_eef_distance_max_m"]),
            "relative_carry_displacement_max_m": 0.04,
            "relative_vector": "object_position - eef_position",
            "relative_reference": "first row of local continuation",
            "support_contact_forbidden": True,
            "strict_no_flicker": True,
            "full_episode_horizon_required": False,
            "local_continuation_only": True,
            "terminal_rule": "terminal_after allowed only on final local row; terminal_before or earlier terminal_after invalidates the window",
        },
        "model_runtime": {
            "action_dim": 7,
            "boundaries": {
                "M0_OPENVLA": "FRESH_PER_STEP",
                "M1_OPENVLA_OFT": "FRESH_OFT_ACTION_QUEUE",
                "M2_PI05_LIBERO": "FRESH_PI05_REPLAN",
            },
            "queue_lengths": {"M0_OPENVLA": 1, "M1_OPENVLA_OFT": 8, "M2_PI05_LIBERO": 5},
            "action_semantics": "inherited frozen official three-state/PI05 boundary adapter",
            "final_action": "authoritative 7-D action delivered by existing clean runtime; no gripper overwrite",
        },
        "evidence_requirements": {
            "every_cell_persisted": True,
            "complete_clean_rows": True,
            "raw_and_final_7d_actions": True,
            "action_boundary_markers": True,
            "object_eef_positions_and_distances": True,
            "relative_carry_displacement": True,
            "contact_and_support": True,
            "lift": True,
            "terminal_markers": True,
            "queue_or_replan_metadata": True,
            "candidate_rejection_intersections": True,
            "selected_anchor_evidence": True,
            "boundary_state_digest_and_snapshot": True,
            "failure_receipt_before_raise": True,
        },
        "scientific_firewall": {
            "open_intervention_steps": 0,
            "pgd_calls": 0,
            "attacked_env_steps": 0,
            "physical_endpoint_reads": 0,
            "v_phys_reads": 0,
            "aa_v_phys_reads": 0,
            "task_success_reads": 0,
            "protected_reads": 0,
            "eval160_reads": 0,
        },
        "source_bindings": {
            "ac1r2_population": parent_binding,
            "ac1r2_taxonomy": taxonomy_binding,
            "ac1r2_root": ac1r2_root_binding,
            "ac0_root": ac0_root_binding,
            "ac0_terminal": ac0_terminal_binding,
            "z1_runtime_protocol": z1_binding,
        },
        "next_legal_action": "EXECUTE_ALL_720_CLEAN_CELLS_ONLY",
    }

    protocol_path = ROOT / "configs/STAGE_AC_AC2_CLEAN_SCREEN_PROTOCOL_V1.json"
    if write:
        write_json(protocol_path, protocol)
    protocol_binding = binding(protocol_path.relative_to(ROOT).as_posix()) if write else {"path": protocol_path.relative_to(ROOT).as_posix()}

    source_authority = {
        "schema": "STAGE_AC_AC2_RUNTIME_SOURCE_AUTHORITY_V1",
        "status": "STAGE_AC_AC2_RUNTIME_SOURCE_AUTHORITY_FROZEN",
        "gate": protocol["gate"],
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "claim_boundary": protocol["claim_boundary"],
        "git_binding": {
            "repository": "Leo-6-maker/openvla-gripper-dutycycle-attack",
            "branch": git_value("branch", "--show-current"),
            "commit": code_commit,
            "tree": code_tree,
        },
        "runtime_environment": {
            "python": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python",
            "environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
            "common_libero_checkout": z1["environment"]["common_libero_checkout"],
            "common_libero_commit": z1["environment"]["common_libero_commit"],
            "common_libero_tree": z1["environment"]["common_libero_tree"],
            "libero_config_path": z1["environment"]["libero_config_path"],
            "dummy_wait_steps": z1["environment"]["dummy_wait_steps"],
            "camera": z1["environment"]["camera"],
            "camera_height": z1["environment"]["camera_height"],
            "camera_width": z1["environment"]["camera_width"],
            "resize": z1["environment"]["resize"],
            "control_freq": z1["environment"]["control_freq"],
            "gpu_admission": "free_memory_mib > 20480 immediately before worker launch",
            "max_project_workers_per_gpu": 1,
            "foreign_processes": "allowed and recorded; never touched",
        },
        "model_authorities": z1["model_families"],
        "runtime_files": runtime_files,
        "input_authorities": {
            "protocol": protocol_binding,
            "ac0_protocol": ac0_protocol_binding,
            "ac0_root": ac0_root_binding,
            "ac0_terminal": ac0_terminal_binding,
            "ac1r2_population": parent_binding,
            "ac1r2_taxonomy": taxonomy_binding,
            "ac1r2_root": ac1r2_root_binding,
            "z1_protocol": z1_binding,
        },
        "scientific_firewall": protocol["scientific_firewall"],
        "next_legal_action": "EXECUTE_ONLY_THE_FROZEN_720_CELL_MANIFEST",
    }
    source_path = ROOT / "reports/STAGE_AC_AC2_RUNTIME_SOURCE_AUTHORITY_V1.json"
    if write:
        write_json(source_path, source_authority)
    source_binding = binding(source_path.relative_to(ROOT).as_posix()) if write else {"path": source_path.relative_to(ROOT).as_posix()}

    cells: list[dict[str, Any]] = []
    for family in MODELS:
        for parent in selected:
            seed, seed_digest = seed_for(family, parent["canonical_parent_key"])
            checkpoint = checkpoint_spec(z1, family, parent["suite"])
            index = len(cells) + 1
            cells.append(
                {
                    "cell_id": f"AC2-{index:04d}",
                    "cell_index": index,
                    "model_family": family,
                    "canonical_parent_key": parent["canonical_parent_key"],
                    "suite": parent["suite"],
                    "task": parent["task"],
                    "source_task_idx": int(parent["task_index"]),
                    "state": parent["state"],
                    "state_id": int(parent["official_init_index"]),
                    "state_sha256": parent["state_sha256"],
                    "parent_exposure_class": parent["exposure_class"],
                    "parent_selection_rank_sha256": parent["selection_rank_sha256"],
                    "checkpoint": checkpoint["path"],
                    "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
                    "env_seed": int(z1["environment"]["env_seed"]),
                    "seed": seed,
                    "seed_digest": seed_digest,
                    "clean_only_authorization": "AC2_CLEAN_ONLY_NO_OPEN_NO_ATTACK_NO_PROTECTED",
                }
            )
    if len(cells) != CELL_COUNT or len({cell["cell_id"] for cell in cells}) != CELL_COUNT:
        raise RuntimeError("AC2_CELL_MANIFEST_INVALID")

    manifest = {
        "schema": "STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1",
        "status": "STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_FROZEN_PRE_EXPOSURE",
        "gate": protocol["gate"],
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "cell_count": CELL_COUNT,
        "full_census_required": True,
        "population": {
            "parent_count": PARENT_COUNT,
            "model_count": len(MODELS),
            "parent_key_set_sha256": sha256_bytes(canonical_json([row["canonical_parent_key"] for row in selected])),
            "selection_salt": SELECTION_SALT,
            "task_strata": task_summary,
        },
        "git_binding": {"repository": "Leo-6-maker/openvla-gripper-dutycycle-attack", "branch": git_value("branch", "--show-current"), "commit": code_commit, "tree": code_tree},
        "source_bindings": {
            "protocol": protocol_binding,
            "runtime_source_authority": source_binding,
            "ac1r2_population": parent_binding,
            "ac1r2_taxonomy": taxonomy_binding,
            "ac1r2_root": ac1r2_root_binding,
            "ac0_root": ac0_root_binding,
            "ac0_terminal": ac0_terminal_binding,
            "z1_protocol": z1_binding,
        },
        "parents": selected,
        "cells": cells,
        "pre_exposure_counters": {
            "model_inference_calls": 0,
            "env_step_calls": 0,
            "open_intervention_steps": 0,
            "pgd_calls": 0,
            "physical_endpoint_reads": 0,
            "v_phys_reads": 0,
            "protected_reads": 0,
            "task_success_reads": 0,
        },
        "next_legal_action": "EXECUTE_ALL_CELLS_EXACTLY_ONCE_UNLESS_PI_APPROVES_A_TECHNICAL_RECOVERY",
    }
    manifest_path = ROOT / "reports/STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
    if write:
        write_json(manifest_path, manifest)
    manifest_binding = binding(manifest_path.relative_to(ROOT).as_posix()) if write else {"path": manifest_path.relative_to(ROOT).as_posix()}

    root_payload = {
        "schema": "STAGE_AC_AC2_PRE_GPU_ROOT_SEAL_V1",
        "status": "STAGE_AC_AC2_PRE_GPU_STATIC_FREEZE_PASS",
        "gate": protocol["gate"],
        "authorization_pi_comment_id": PI_COMMENT_ID,
        "git_binding": {"repository": "Leo-6-maker/openvla-gripper-dutycycle-attack", "branch": git_value("branch", "--show-current"), "commit": code_commit, "tree": code_tree},
        "official_libero_authority": {
            "commit": z1["environment"]["common_libero_commit"],
            "tree": z1["environment"]["common_libero_tree"],
            "checkout": z1["environment"]["common_libero_checkout"],
        },
        "population": {
            "parent_count": PARENT_COUNT,
            "cell_count": CELL_COUNT,
            "suite_counts": dict(sorted(Counter(row["suite"] for row in selected).items())),
            "task_counts": {f"{suite}/{task}": sum(item["suite"] == suite and item["task"] == task for item in selected) for suite, task in sorted({(row["suite"], row["task"]) for row in selected})},
            "h0_count": sum(row["exposure_class"] == "H0_UNTOUCHED" for row in selected),
            "hc_count": sum(row["exposure_class"] == "HC_CLEAN_ONLY" for row in selected),
            "parent_key_set_sha256": manifest["population"]["parent_key_set_sha256"],
        },
        "source_bindings": {
            "protocol": protocol_binding,
            "runtime_source_authority": source_binding,
            "manifest": manifest_binding,
            "ac1r2_population": parent_binding,
            "ac1r2_taxonomy": taxonomy_binding,
            "ac1r2_root": ac1r2_root_binding,
            "ac0_root": ac0_root_binding,
            "ac0_terminal": ac0_terminal_binding,
            "z1_protocol": z1_binding,
        },
        "scientific_firewall": {
            "model_inference": 0,
            "env_step": 0,
            "gpu_worker": 0,
            "open_intervention": 0,
            "pgd": 0,
            "physical_endpoint_read": 0,
            "v_phys": 0,
            "protected_or_eval160": 0,
            "task_success_read": 0,
            "new_identity": 0,
        },
        "next_legal_action": "GPU_AC2_CLEAN_ONLY_AFTER_THIS_SEAL_IS_COMMITTED_AND_VERIFIED",
    }
    root_path = ROOT / "reports/STAGE_AC_AC2_PRE_GPU_ROOT_SEAL_V1.json"
    root_payload_hash = sha256_bytes(canonical_json(root_payload))
    root = {**root_payload, "root_payload_sha256": root_payload_hash}
    if write:
        root_binding = write_json(root_path, root)
        sidecar = ROOT / "reports/STAGE_AC_AC2_PRE_GPU_ROOT_SEAL_V1.sha256"
        sidecar.write_bytes(f"{root_binding['sha256']}  STAGE_AC_AC2_PRE_GPU_ROOT_SEAL_V1.json\n".encode("ascii"))
    else:
        root_binding = {"path": root_path.relative_to(ROOT).as_posix()}

    return {
        "status": "STAGE_AC_AC2_PRE_GPU_STATIC_FREEZE_PASS",
        "parent_count": PARENT_COUNT,
        "cell_count": CELL_COUNT,
        "suite_counts": dict(sorted(Counter(row["suite"] for row in selected).items())),
        "h0_count": sum(row["exposure_class"] == "H0_UNTOUCHED" for row in selected),
        "hc_count": sum(row["exposure_class"] == "HC_CLEAN_ONLY" for row in selected),
        "root_payload_sha256": root_payload_hash,
        "outputs": {
            "protocol": protocol_binding,
            "source_authority": source_binding,
            "manifest": manifest_binding,
            "root": root_binding,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = build(write=args.write)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
