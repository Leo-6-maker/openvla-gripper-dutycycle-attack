#!/usr/bin/env python3
"""Build the read-only Stage Z Z0 authority and shared-panel audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SALT = "STAGE_Z_CROSS_MODEL_PANEL_V1_20260822"
SNAPSHOT_UTC = "2026-08-22T15:27:41Z"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TASKS = tuple(range(10))
STAGES = {
    "f1a3_root": "reports/STAGE_X_X1R2_F1A3_SOURCE_SPLIT_AND_POPULATION_FREEZE_V3_20260821/F1A3_ROOT_SEAL_V3.json",
    "f1t_root": "reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json",
    "identity_ledger": "reports/STAGE_X_X1R_T1D0R1_G10_IDENTITY_EXCLUSION_LEDGER_V1.json",
    "q3_pool": "reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json",
    "e2_pool": "reports/STAGE_X_X1R2_Q3R3_E2_SUCCESSOR_ENGINEERING_POOL_V1.json",
    "e3_pool": "reports/STAGE_X_X1R2_E3_SELECTIVE_REALIZABILITY_POOL_V1.json",
    "m0_authority": "configs/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_V1.json",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_jsonl(rel: str):
    return [json.loads(line) for line in (ROOT / rel).read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row: dict) -> str:
    return row["canonical_parent_key"]


def write_json(rel: str, value: dict) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def source_binding(rel: str) -> dict:
    path = ROOT / rel
    return {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def selected_rows(rel: str) -> list[dict]:
    value = load_json(rel)
    return list(value["selected"])


def build_panel() -> tuple[dict, dict]:
    ledger = load_jsonl(STAGES["identity_ledger"])
    assert len(ledger) == 1200
    assert len({key(row) for row in ledger}) == len(ledger)
    base_fresh = {
        key(row)
        for row in ledger
        if row["fresh_after_exclusion"] and row["state_id"] >= 20
    }
    consumed = {}
    for label in ("q3_pool", "e2_pool", "e3_pool"):
        rows = selected_rows(STAGES[label])
        consumed[label] = {key(row) for row in rows}
    consumed_union = set().union(*consumed.values())
    remaining = base_fresh - consumed_union
    assert len(base_fresh) == 210
    assert len(consumed["q3_pool"]) == 48
    assert len(consumed["e2_pool"]) == 12
    assert len(consumed["e3_pool"]) == 12
    assert len(consumed_union) == 63
    assert len(remaining) == 147

    by_key = {key(row): row for row in ledger}
    panel_rows = []
    selected = []
    for suite in SUITES:
        for task_idx in TASKS:
            candidates = [
                row for row in by_key.values()
                if row["suite"] == suite and row["task_idx"] == task_idx and key(row) in remaining
            ]
            ranked = sorted(
                (
                    {
                        "canonical_parent_key": key(row),
                        "rank_sha256": hashlib.sha256(f"{SALT}|{key(row)}".encode()).hexdigest(),
                    }
                    for row in candidates
                ),
                key=lambda row: (row["rank_sha256"], row["canonical_parent_key"]),
            )
            chosen = ranked[0] if ranked else None
            if chosen:
                selected.append(chosen["canonical_parent_key"])
            panel_rows.append(
                {
                    "suite": suite,
                    "task_idx": task_idx,
                    "candidate_count_after_union_exclusion": len(ranked),
                    "ranked_fresh_candidates": ranked,
                    "selected_parent_key": chosen["canonical_parent_key"] if chosen else None,
                    "selected_rank_sha256": chosen["rank_sha256"] if chosen else None,
                    "structurally_missing_without_replacement": chosen is None,
                }
            )
    missing = [
        f"{row['suite']}/task_{row['task_idx']:02d}"
        for row in panel_rows
        if row["structurally_missing_without_replacement"]
    ]
    assert len(selected) == 36
    assert missing == [
        "libero_goal/task_01",
        "libero_goal/task_04",
        "libero_goal/task_06",
        "libero_goal/task_09",
    ]
    panel = {
        "schema": "STAGE_Z_Z0_SHARED_40_IDENTITY_PANEL_V1",
        "status": "HOLD_STAGE_Z_Z0_CROSS_MODEL_AUTHORITY_NOT_ESTABLISHED",
        "panel_contract": {
            "salt": SALT,
            "selection_order": "sha256(salt|suite|task|state), then canonical_parent_key",
            "suite_count": 4,
            "tasks_per_suite": 10,
            "requested_identity_count": 40,
            "candidate_state_range": [0, 49],
            "no_replacement_or_top_up": True,
            "outcome_independent": True,
        },
        "source_bindings": {
            name: source_binding(rel)
            for name, rel in STAGES.items()
            if name in {"f1a3_root", "f1t_root", "identity_ledger", "q3_pool", "e2_pool", "e3_pool"}
        },
        "population_accounting": {
            "identity_ledger_rows": len(ledger),
            "base_fresh_after_prior_exclusion": len(base_fresh),
            "q3_selected": len(consumed["q3_pool"]),
            "e2_selected": len(consumed["e2_pool"]),
            "e3_selected": len(consumed["e3_pool"]),
            "consumed_union": len(consumed_union),
            "remaining_fresh_after_union": len(remaining),
            "panel_selected": len(selected),
            "panel_missing": len(missing),
        },
        "missing_tasks_without_replacement": missing,
        "rows": panel_rows,
        "protected_boundary": {
            "scientific_outcome_read": False,
            "gpu_rollout": False,
            "model_inference": False,
            "physical_intervention": False,
            "protected_eval160": False,
        },
    }
    return panel, {"base_fresh": base_fresh, "consumed_union": consumed_union, "remaining": remaining}


def main() -> None:
    commit = git_value("rev-parse", "HEAD")
    tree = git_value("rev-parse", "HEAD^{tree}")
    panel, _ = build_panel()

    protocol = {
        "schema": "STAGE_Z_CROSS_MODEL_OPEN_DUTY_PROTOCOL_V1",
        "status": "HOLD_STAGE_Z_Z0_CROSS_MODEL_AUTHORITY_NOT_ESTABLISHED",
        "stage": "STAGE_Z_Z0_CROSS_MODEL_STATIC_AUTHORITY_AND_PROTOCOL_FREEZE",
        "audit_scope": "CPU/static/offline authority audit only; no scientific rollout was authorized or executed",
        "git_binding": {"head_commit": commit, "head_tree": tree},
        "models": [
            {
                "id": "Z-M0",
                "name": "existing suite-matched OpenVLA authority",
                "role": "paper/X0 reference authority",
                "runtime_authority": STAGES["m0_authority"],
            },
            {
                "id": "Z-M1",
                "name": "official OpenVLA-OFT",
                "source_commit": "e4287e94541f459edc4feabc4e181f537cd569a8",
                "source_tree": "0ae110ee28943b9e46feffad84429d2d6e026a32",
                "checkpoint_revisions": {
                    "libero_spatial": "6d0231af0e48c5985f1ff86908f4674b84bc049b",
                    "libero_object": "4c89574e1c538b6c102f43f0526d60a9d3650148",
                    "libero_goal": "c2d0f9fbbd82674683b397ff923168a12f6a307b",
                    "libero_10": "95220f9a3421a7ff12d4218e73d09ade830fa9a3",
                },
                "checkpoint_materialized_for_Z0": False,
            },
            {
                "id": "Z-M2",
                "name": "official OpenPI pi0.5-LIBERO",
                "source_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
                "source_tree": "a7f18af2745255b5fa98c86d6031f858bf73d1be",
                "config": "pi05_libero",
                "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
                "local_checkpoint_path": "/llm_jzm/mt/models/openpi-assets/checkpoints/pi05_libero",
                "checkpoint_object_count": 16,
                "checkpoint_materialized_for_Z0": True,
                "official_source_checkout_materialized_for_Z0": False,
            },
        ],
        "common_task_contract": {
            "suites": list(SUITES),
            "tasks_per_suite": 10,
            "common_action_dim": 7,
            "candidate_state_range": [0, 49],
            "fresh_panel": "4 suites x 10 tasks x 1 state, frozen before outcomes",
        },
        "action_contract": {
            "arm_indices": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "libero_native_open": -1.0,
            "libero_native_close": 1.0,
            "open_command": "replace only final gripper action after model output; no sign guess; no decode/re-encode",
            "intervention_is_not_run_in_Z0": True,
        },
        "branch_decision_contract": {
            "Z-M0": "per-step policy decision as bound by existing authority",
            "Z-M1": "fresh policy decision at official OFT action-queue boundary; NUM_ACTIONS_CHUNK=8 / num_open_loop_steps=8 requires runtime verification",
            "Z-M2": "fresh policy decision at official LIBERO replan boundary; replan_steps=5 with action_horizon=10 requires runtime verification",
            "exact_runtime_boundary_sealed": False,
        },
        "frozen_parameters": {
            "horizons": {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520},
            "dummy_wait_steps": 10,
            "camera_size": 256,
            "intervention_horizon": 10,
        },
        "prohibited_until_Z0_pass": [
            "GPU/model inference",
            "Z1/Z2/Z3/Z4",
            "VIS-PGD/RAND/SHUFFLED",
            "Student/detector timing selection",
            "protected Eval160 or V_phys reads",
            "new identity replacement/top-up",
        ],
        "blocking_reasons": [
            "shared 40-identity panel is structurally incomplete: four Goal tasks have no remaining fresh identity under the frozen exclusion union",
            "official OFT checkpoints are not materialized on the server for exact byte-level authority sealing",
            "official OpenPI source checkout is not materialized separately from the modified fork",
            "current M0 server model trees require an exact re-audit against the immutable runtime authority",
        ],
        "references": {
            "official_oft": "https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8",
            "official_openpi": "https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac",
        },
    }

    model_authority = {
        "schema": "STAGE_Z_Z0_MODEL_AUTHORITY_MAP_V1",
        "status": "HOLD_STAGE_Z_Z0_CROSS_MODEL_AUTHORITY_NOT_ESTABLISHED",
        "git_binding": {"head_commit": commit, "head_tree": tree},
        "source_authorities": {
            "openvla_oft": {
                "remote": "https://github.com/moojink/openvla-oft.git",
                "commit": "e4287e94541f459edc4feabc4e181f537cd569a8",
                "tree": "0ae110ee28943b9e46feffad84429d2d6e026a32",
                "checkout_present": False,
                "checkpoint_revisions": protocol["models"][1]["checkpoint_revisions"],
                "checkpoint_bytes_sealed": False,
            },
            "openpi": {
                "remote": "https://github.com/Physical-Intelligence/openpi.git",
                "commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
                "tree": "a7f18af2745255b5fa98c86d6031f858bf73d1be",
                "checkout_present": False,
                "checkpoint_uri": "gs://openpi-assets/checkpoints/pi05_libero",
                "checkpoint_local_path": "/llm_jzm/mt/models/openpi-assets/checkpoints/pi05_libero",
                "checkpoint_object_count": 16,
                "checkpoint_local_bytes": 12439122345,
                "checkpoint_bytes_sealed": False,
                "modified_fork_excluded": "/llm_jzm/mt/openpi",
            },
        },
        "m0_reaudit": {
            "immutable_authority": source_binding(STAGES["m0_authority"]),
            "server_model_dirs": {
                "libero_10": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10", "recursive_bytes": 15085097823},
                "libero_goal": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-goal", "recursive_bytes": 15085099486},
                "libero_object": {"path": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object", "recursive_bytes": 15085099978},
                "libero_spatial": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620", "recursive_bytes": 15085134953},
            },
            "exact_tree_and_weight_reaudit_complete": False,
        },
        "runtime_environment": {
            "server": "pm-364c0001",
            "user": "dty_user",
            "env": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
            "python": "3.10.16",
            "torch": "2.2.2+cu118",
            "numpy": "1.26.4",
            "transformers": "4.40.1",
            "snapshot_utc": SNAPSHOT_UTC,
        },
    }

    parity = {
        "schema": "STAGE_Z_Z0_ENVIRONMENT_ACTION_PARITY_V1",
        "status": "HOLD_STAGE_Z_Z0_CROSS_MODEL_AUTHORITY_NOT_ESTABLISHED",
        "git_binding": {"head_commit": commit, "head_tree": tree},
        "environment": model_authority["runtime_environment"],
        "gpu_admission_snapshot": {
            "contract": "free_memory_mib > 20480; max one project worker per physical GPU; foreign processes coexist untouched",
            "snapshot_utc": SNAPSHOT_UTC,
            "devices": [
                {"index": 0, "free_memory_mib": 4732, "utilization_pct": 99, "admissible_by_memory": False},
                {"index": 1, "free_memory_mib": 4732, "utilization_pct": 94, "admissible_by_memory": False},
                {"index": 2, "free_memory_mib": 4732, "utilization_pct": 99, "admissible_by_memory": False},
                {"index": 3, "free_memory_mib": 4732, "utilization_pct": 98, "admissible_by_memory": False},
                {"index": 4, "free_memory_mib": 81210, "utilization_pct": 0, "admissible_by_memory": True},
                {"index": 5, "free_memory_mib": 29578, "utilization_pct": 94, "admissible_by_memory": True, "foreign_processes_untouched": True},
                {"index": 6, "free_memory_mib": 29578, "utilization_pct": 91, "admissible_by_memory": True, "foreign_processes_untouched": True},
                {"index": 7, "free_memory_mib": 81210, "utilization_pct": 0, "admissible_by_memory": True},
            ],
            "project_workers_launched": 0,
        },
        "disk_snapshot": {
            "snapshot_utc": SNAPSHOT_UTC,
            "/mnt/sdc": {"available_1024_blocks": 27000148, "capacity_pct": 100},
            "/llm_jzm": {"available_1024_blocks": 43588920, "capacity_pct": 99},
        },
        "action_semantics": {
            "common_dim": 7,
            "arm_indices": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "native_open": -1.0,
            "native_close": 1.0,
            "m0": "existing project authority; exact runtime source/tree re-audit pending",
            "m1_oft": "official source normalizes gripper [0,1] to [-1,+1] and inverts for LIBERO; exact installed runtime re-audit pending",
            "m2_openpi": "official LIBERO path emits 7D actions directly to env; exact installed runtime/source checkout re-audit pending",
            "silent_fallback_allowed": False,
            "decode_reencode_allowed": False,
        },
        "parity_checks": {
            "common_libero_assets_and_bddl_bound": False,
            "preprocessing_bound_for_all_models": False,
            "normalization_bound_for_all_models": False,
            "chunk_and_replan_bound_for_all_models": False,
            "fresh_branch_boundary_bound_for_all_models": False,
            "checkpoint_bytes_bound_for_all_models": False,
        },
        "protected_boundary": {"eval160": "UNREAD", "v_phys": "UNREAD", "scientific_outcomes": "UNREAD"},
    }

    panel["git_binding"] = {"head_commit": commit, "head_tree": tree}
    write_json("configs/STAGE_Z_CROSS_MODEL_OPEN_DUTY_PROTOCOL_V1.json", protocol)
    write_json("reports/STAGE_Z_Z0_MODEL_AUTHORITY_MAP_V1.json", model_authority)
    write_json("reports/STAGE_Z_Z0_ENVIRONMENT_ACTION_PARITY_V1.json", parity)
    write_json("reports/STAGE_Z_Z0_SHARED_40_IDENTITY_PANEL_V1.json", panel)

    artifact_rels = [
        "configs/STAGE_Z_CROSS_MODEL_OPEN_DUTY_PROTOCOL_V1.json",
        "reports/STAGE_Z_Z0_MODEL_AUTHORITY_MAP_V1.json",
        "reports/STAGE_Z_Z0_ENVIRONMENT_ACTION_PARITY_V1.json",
        "reports/STAGE_Z_Z0_SHARED_40_IDENTITY_PANEL_V1.json",
    ]
    manifest = [
        {"path": rel, "bytes": (ROOT / rel).stat().st_size, "sha256": sha256_file(ROOT / rel)}
        for rel in artifact_rels
    ]
    root_seal = {
        "schema": "STAGE_Z_Z0_ROOT_SEAL_V1",
        "status": "HOLD_STAGE_Z_Z0_CROSS_MODEL_AUTHORITY_NOT_ESTABLISHED",
        "git_binding": {"head_commit": commit, "head_tree": tree},
        "snapshot_utc": SNAPSHOT_UTC,
        "artifact_manifest": manifest,
        "panel_status": panel["status"],
        "blocking_reasons": protocol["blocking_reasons"],
        "scientific_rollout_started": False,
        "gpu_workers_started": 0,
        "protected_boundary": {"eval160": "UNREAD", "v_phys": "UNREAD", "scientific_outcomes": "UNREAD"},
        "next_legal_action": "PI review of Z0 HOLD; no Z1/Z2/Z3/Z4 until authority and exact 40-panel contract are re-established",
    }
    write_json("reports/STAGE_Z_Z0_ROOT_SEAL_V1.json", root_seal)
    root_hash = sha256_file(ROOT / "reports/STAGE_Z_Z0_ROOT_SEAL_V1.json")
    (ROOT / "reports/STAGE_Z_Z0_ROOT_SEAL_V1.sha256").write_text(root_hash + "  STAGE_Z_Z0_ROOT_SEAL_V1.json\n", encoding="utf-8")

    print(json.dumps({"status": root_seal["status"], "panel_selected": len(panel["missing_tasks_without_replacement"]) and 36 or 40, "missing_tasks": panel["missing_tasks_without_replacement"], "root_sha256": root_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
