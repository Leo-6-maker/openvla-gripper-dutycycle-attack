#!/usr/bin/env python3
"""Create the append-only Stage Z Z0R1 structural/authority audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = "HOLD_STAGE_Z_Z0R1_THREE_MODEL_AUTHORITY_NOT_ESTABLISHED"
SNAPSHOT_UTC = "2026-08-22T16:03:47Z"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
V1_PANEL = "reports/STAGE_Z_Z0_SHARED_40_IDENTITY_PANEL_V1.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def digest_json(value: object) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def write_json(rel: str, value: dict) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def bind(rel: str) -> dict:
    path = ROOT / rel
    return {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    v1 = json.loads((ROOT / V1_PANEL).read_text(encoding="utf-8"))
    v1_hash = sha256_file(ROOT / V1_PANEL)
    rows = []
    selected = []
    missing = []
    for row in v1["rows"]:
        key = row["selected_parent_key"]
        if key:
            state = "FROZEN_SHARED_FRESH"
            selected.append(key)
        else:
            state = "STRUCTURAL_MISSING_NO_FRESH_IDENTITY"
            missing.append(f"{row['suite']}/task_{int(row['task_idx']):02d}")
        rows.append(
            {
                "suite": row["suite"],
                "task_idx": row["task_idx"],
                "state": state,
                "canonical_parent_key": key,
                "selected_rank_sha256": row["selected_rank_sha256"],
                "candidate_count_after_union_exclusion": row["candidate_count_after_union_exclusion"],
                "source_v1_row_digest": digest_json(
                    {
                        "suite": row["suite"],
                        "task_idx": row["task_idx"],
                        "selected_parent_key": key,
                        "selected_rank_sha256": row["selected_rank_sha256"],
                        "candidate_count_after_union_exclusion": row["candidate_count_after_union_exclusion"],
                    }
                ),
            }
        )
    assert len(rows) == 40 and len(selected) == 36
    assert missing == [
        "libero_goal/task_01",
        "libero_goal/task_04",
        "libero_goal/task_06",
        "libero_goal/task_09",
    ]

    m2_files = [
        ("assets/physical-intelligence/libero/norm_stats.json", 1914, "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"),
        ("params/_METADATA", 23493, "303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73"),
        ("params/_sharding", 17952, "63f4c57ba6ff10f4132a639b9942eabcc26942eb34081be1d93bf7ddab816501"),
        ("params/array_metadatas/process_0", 9062, "2b29474f08aa50922da11074deb4d7f35e30f0071a088a30f40df88edc1ebdb0"),
        ("params/d/98a77a52a8eb845ae4830eb7fe983979", 42307, "93b39327a1552b06b1b6ce5190d58131297bcca66c3f2a099bbb2882df7759af"),
        ("params/manifest.ocdbt", 120, "65246951e69bd2b5118e609646bc9e8c439229bccbc1325643833aeb74f77104"),
        ("params/ocdbt.process_0/d/0eaaecefaa9720d30a32cc56e65fd345", 2449323387, "0d4412552d4a69fba953f824a60d005ee7559822a9c1c16684906c7f0801b848"),
        ("params/ocdbt.process_0/d/155391c1cbf93a1be16266de052e5b48", 2307885530, "ad09cf4b2cacc0628c713afbb1ee9decfd36e295fdacae111c1e0c5eeeb6cc00"),
        ("params/ocdbt.process_0/d/2b6985f48e9da86f68627a7608c5bc25", 1077, "3b81b2e8afe1456a4584d19dcf600f2bb6f27beff9f061b336f06657a11d5cb5"),
        ("params/ocdbt.process_0/d/475fab3ee8821662585a8cde3eb32e22", 2150232827, "624a529b0b5f07bf535e07eba4f823e47fdde66a3d0b38295a35ac9d60cfcf0f"),
        ("params/ocdbt.process_0/d/6c54da5a6f62c20a09c0f3f8c3329e00", 1608436908, "1a5f80ef1aeaa82e70cca883c4d95da65bd410c8da19729d4814317533ba11f5"),
        ("params/ocdbt.process_0/d/896bf93c5cf2e8ddd274a6ea0a2feec0", 2240080987, "c92342da7593219debce6133f148f3c24ece184ff53f9f4bac800c09fcc6dc1a"),
        ("params/ocdbt.process_0/d/bc613ff288a162563e622d01bb60622a", 217, "1e99d5876e6b8db2e12c43c8079f6b98cffac2c26bbd84e59cc505e983d146d2"),
        ("params/ocdbt.process_0/d/bda87d9791f23df771cd2d15293780cc", 2926522, "9c717ff13084524b74f330803421b8a261bc3009a477c933e1e1a1477a796fc3"),
        ("params/ocdbt.process_0/d/efbb46173882cb35ed41ffe0c2db8a5e", 1680102856, "6e62d35e0689dc12779d0b2e65304823d6787fdd4ccd754c9ebd059fdc4b6765"),
        ("params/ocdbt.process_0/manifest.ocdbt", 322, "3bf70fbb0fac151675595b33aeb8203139e8809de17d606574ab606a758b3591"),
    ]
    m2_manifest = [{"path": p, "size": n, "sha256": h} for p, n, h in m2_files]
    m2_file_bytes = sum(x["size"] for x in m2_manifest)
    # `du -sb` on the checkpoint tree includes directory entries; the sealed
    # file manifest must use the sum of regular-file sizes instead.
    assert m2_file_bytes == 12439085481
    m2_tree_apparent_bytes = 12439122345
    assert m2_tree_apparent_bytes - m2_file_bytes == 36864
    m2_manifest_hash = digest_json(m2_manifest)

    m1 = {
        "source_commit": "e4287e94541f459edc4feabc4e181f537cd569a8",
        "source_tree": "0ae110ee28943b9e46feffad84429d2d6e026a32",
        "checkout": "/mnt/sdc/dty_user/openvla_attack/repos/openvla-oft-stage-z-e4287e9_20260823",
        "checkout_status_at_snapshot": "EXPECTED_CLEAN_FIXED_COMMIT",
        "runtime_files": {
            "experiments/robot/libero/run_libero_eval.py": "cdf27f56c5464808481d56af0898475541e10817",
            "experiments/robot/libero/libero_utils.py": "8e2c23458cbf48a9d78bf61e7cf0cbd5e196d635",
        },
        "checkpoints": {
            "libero_spatial": {"repo_id": "moojink/openvla-7b-oft-finetuned-libero-spatial", "revision": "6d0231af0e48c5985f1ff86908f4674b84bc049b", "files": 25, "bytes": 15939159216, "lfs_bytes": 15936583209, "manifest_sha256": "091cec71d0f5a6e31de9041d7b2ae26cde70454a1aeae88567b95aa9cd17befe", "materialized": False},
            "libero_object": {"repo_id": "moojink/openvla-7b-oft-finetuned-libero-object", "revision": "4c89574e1c538b6c102f43f0526d60a9d3650148", "files": 25, "bytes": 15939159270, "lfs_bytes": 15936583209, "manifest_sha256": "126ac5dc8607deafd70b08665e2070c328ec66a4d36b43427eb4a57eeff10f70", "materialized": False},
            "libero_goal": {"repo_id": "moojink/openvla-7b-oft-finetuned-libero-goal", "revision": "c2d0f9fbbd82674683b397ff923168a12f6a307b", "files": 25, "bytes": 15939159075, "lfs_bytes": 15936583117, "manifest_sha256": "7a5ff5fda6bd5a56ad84b133744195dce81a680df57aeb40b2513d0c91f52f01", "materialized": False},
            "libero_10": {"repo_id": "moojink/openvla-7b-oft-finetuned-libero-10", "revision": "95220f9a3421a7ff12d4218e73d09ade830fa9a3", "files": 25, "bytes": 15939159245, "lfs_bytes": 15936583209, "manifest_sha256": "30b2bf0b4b7c6e71847e45e572df7f6187c607439552d17696fd25be12fdba82", "materialized": False},
        },
    }

    authority = {
        "schema": "STAGE_Z_Z0R1_MODEL_AUTHORITY_MAP_V2",
        "status": STATUS,
        "git_binding": {"head_commit": head, "head_tree": tree},
        "z_m0": {
            "immutable_project_authority": bind("configs/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_V1.json"),
            "current_server_snapshot_utc": SNAPSHOT_UTC,
            "exact_reaudit": False,
            "reason": "current server byte/file inventories disagree with immutable authority; fail closed",
            "suites": {
                "libero_10": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10", "expected_file_count": 18, "expected_bytes": 15085093727, "expected_tree_sha256": "4a83f512232909d34ec2f835acf492713b4c174f0b016ac00cbb330ed5ff8dbd", "current_file_count": 18, "current_bytes": 15085097823, "exact_match": False},
                "libero_goal": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-goal", "expected_file_count": 19, "expected_bytes": 15085095390, "expected_tree_sha256": "5354cfe948abd56789ea3b50976fb3693d68a8b617771ca0db8fee368dfd542d", "current_file_count": 19, "current_bytes": 15085099486, "exact_match": False},
                "libero_object": {"path": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object", "expected_file_count": 19, "expected_bytes": 15085095882, "expected_tree_sha256": "f3e5c61db14bd2670e98ea742bfec6baace25533ce8ad2c11685d68e20957f6c", "current_file_count": 19, "current_bytes": 15085099978, "exact_match": False},
                "libero_spatial": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620", "expected_file_count": 19, "expected_bytes": 15085095735, "expected_tree_sha256": "b3faccff2e0c1b401973aca6e12e98ae23482441d85199ae9507251ac1dea1b5", "current_file_count": 22, "current_bytes": 15085134953, "exact_match": False},
            },
        },
        "z_m1": m1,
        "z_m2": {
            "source_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
            "source_tree": "a7f18af2745255b5fa98c86d6031f858bf73d1be",
            "checkout": "/mnt/sdc/dty_user/openvla_attack/repos/openpi-stage-z-15a9616a_20260822",
            "checkout_status_at_snapshot": "CLEAN_FIXED_COMMIT",
            "config": "pi05_libero",
            "checkpoint_uri": "gs://openpi-assets/checkpoints/pi05_libero",
            "checkpoint_path": "/llm_jzm/mt/models/openpi-assets/checkpoints/pi05_libero",
            "checkpoint_files": 16,
            "checkpoint_bytes": m2_tree_apparent_bytes,
            "checkpoint_bytes_semantics": "du -sb apparent tree bytes, including directory entries",
            "checkpoint_file_bytes": m2_file_bytes,
            "checkpoint_manifest_sha256": m2_manifest_hash,
            "checkpoint_file_manifest": m2_manifest,
            "modified_fork_excluded": {"path": "/mnt/sdc/dty_user/pi0_openpi", "head": "c23745b5ad24e98f66967ea795a07b2588ed6c79", "working_tree_modified": True},
        },
    }

    amendment = {
        "schema": "STAGE_Z_Z0R1_STRUCTURAL_MISSINGNESS_AMENDMENT_V1",
        "status": "STAGE_Z_Z0R1_STRUCTURAL_36_PANEL_AMENDMENT_SEALED",
        "git_binding": {"head_commit": head, "head_tree": tree},
        "supersedes_only": "the exact-40 denominator requirement for Stage Z, prospectively before any Stage-Z model outcome",
        "preserves": ["model panel", "doses", "physical endpoint", "mechanism criteria", "protected firewall", "no replacement after exposure"],
        "nominal_design": {"suite_count": 4, "tasks_per_suite": 10, "state_range": [0, 49], "nominal_task_cells": 40},
        "frozen_missing_cells": missing,
        "frozen_shared_identity_count": 36,
        "suite_denominators": {"libero_10": 10, "libero_goal": 6, "libero_object": 10, "libero_spatial": 10},
        "source_panel": {"path": V1_PANEL, "sha256": v1_hash, "selected_rows_are_reused_without_reranking": True},
        "forbidden_repairs": ["replacement", "top_up", "reusing exposed identities", "BRIDGE_V3", "protected/Eval160", "widening state range", "post-outcome selection"],
        "claim_safe_wording": "36 shared fresh task/state identities spanning all four LIBERO suites, from a 40-task design frame with four prospectively sealed structural-missing Goal cells",
    }

    shared36 = {
        "schema": "STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1",
        "status": "STAGE_Z_Z0R1_36_PANEL_SEALED",
        "git_binding": {"head_commit": head, "head_tree": tree},
        "source_v1_panel": {"path": V1_PANEL, "sha256": v1_hash, "selection_salt": v1["panel_contract"]["salt"], "selection_rule": v1["panel_contract"]["selection_order"]},
        "nominal_task_cells": 40,
        "shared_fresh_identities": 36,
        "structural_missing_cells": 4,
        "suite_counts": {"libero_10": 10, "libero_goal": 6, "libero_object": 10, "libero_spatial": 10},
        "rows": rows,
        "selected_parent_keys": selected,
        "missing_task_cells": missing,
        "denominator_policy": "36 frozen shared identities per model; four cells remain structural-missing and are not model failures, censoring, negative V_phys, or complete-case exclusions",
        "protected_boundary": {"model_inference": 0, "simulator": 0, "env_step": 0, "physical_intervention": 0, "v_phys": 0, "eval160": "UNREAD", "protected": "UNREAD"},
    }

    protocol = {
        "schema": "STAGE_Z_CROSS_MODEL_OPEN_DUTY_PROTOCOL_V2",
        "status": STATUS,
        "gate": "STAGE_Z_Z0R1_36_PANEL_AND_THREE_MODEL_AUTHORITY_CLOSURE",
        "git_binding": {"head_commit": head, "head_tree": tree},
        "population": amendment,
        "model_panel": {
            "Z-M0": "existing suite-matched OpenVLA authority; no checkpoint substitution",
            "Z-M1": m1,
            "Z-M2": {"source_commit": authority["z_m2"]["source_commit"], "source_tree": authority["z_m2"]["source_tree"], "config": "pi05_libero", "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero"},
        },
        "common_libero": {"suites": list(SUITES), "tasks_per_suite": 10, "action_dim": 7, "candidate_state_range": [0, 49], "horizons": {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}, "dummy_wait_steps": 10, "camera": {"name": "agentview", "height": 256, "width": 256}, "preprocess": {"rotate_180": True, "resize": 224}},
        "action_contract": {"arm_indices": [0, 1, 2, 3, 4, 5], "gripper_index": 6, "native_open": -1.0, "native_close": 1.0, "intervention": "replace only final gripper coordinate after authoritative model output", "arm_exact": True, "decode_reencode": False, "silent_fallback": False},
        "branch_contract": {"Z-M0": "fresh per-step decision", "Z-M1": {"fresh_boundary": "official OFT action queue boundary", "num_actions_chunk": 8, "num_open_loop_steps": 8}, "Z-M2": {"fresh_boundary": "official OpenPI replan boundary", "replan_steps": 5, "action_horizon": 10}, "residual_queue_at_branch": "forbidden", "state_atol": 1e-12, "state_rtol": 0.0, "action_atol": 1e-6},
        "future_selection_freezes": {"z1_canary_salt": "STAGE_Z_Z1_ENGINEERING_CANARY_V1_20260823", "critical_anchor_salt": "STAGE_Z_Z2_CRITICAL_ANCHOR_V1_20260823", "noncritical_control_salt": "STAGE_Z_Z2_NONCRITICAL_CONTROL_V1_20260823", "manual_audit_salt": "STAGE_Z_Z3_MANUAL_AUDIT_V1_20260823"},
        "doses": [3, 5, 10],
        "h_phys": 10,
        "arms": ["CLEAN_BRANCH_CRITICAL", "COMMAND_OPEN_T3_CRITICAL", "COMMAND_OPEN_T5_CRITICAL", "COMMAND_OPEN_T10_CRITICAL", "COMMAND_OPEN_T5_NONCRITICAL_CONTROL"],
        "endpoint": {"source": "X0 physical mechanism definitions", "v_phys": "unchanged", "sr": "auxiliary only", "unit": "model-parent pair"},
        "statistics": {"parent_bootstrap_replicates": 2000, "bootstrap_unit": "parent", "cross_model_pooled_p_value": False, "primary_denominator": 36},
        "decision_criteria_unchanged": True,
        "protected_firewall": {"eval160": "UNREAD", "protected": "UNREAD", "bridge_v3": "UNOPENED", "new_scientific_exposure_counters": 0},
        "execution_boundary": "Z0R1 PASS returns to PI; it does not automatically open Z1",
    }

    parity = {
        "schema": "STAGE_Z_Z0R1_ENVIRONMENT_ACTION_PARITY_V2",
        "status": STATUS,
        "git_binding": {"head_commit": head, "head_tree": tree},
        "server": {"hostname": "pm-364c0001", "user": "dty_user", "snapshot_utc": SNAPSHOT_UTC, "env": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800", "python": "3.10.16", "torch": "2.2.2+cu118", "numpy": "1.26.4", "transformers": "4.40.1", "mujoco": "3.9.0", "robosuite": "1.4.1"},
        "libero_runtime": {"config": "/home/dty_user/.libero/config.yaml", "configured_root": "/mnt/sdc/dty_user/pi0_openpi/third_party/libero/libero", "configured_source_is_modified_fork": True, "fork_head": "c23745b5ad24e98f66967ea795a07b2588ed6c79", "official_libero_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01", "official_libero_tree": "99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60", "official_checkout_path_exists": True, "official_checkout_materialized": False, "official_checkout_state": "PARTIAL_GIT_SKELETON_NO_WORKTREE", "common_bddl_init_asset_hashes_sealed": False},
        "source_action_bindings": {"OFT_run_libero_eval_blob": "cdf27f56c5464808481d56af0898475541e10817", "OpenPI_libero_main_blob": "dc015a61740f2d3174152bebb60176fac52f3f40", "OpenPI_libero_policy_blob": "fe5aab0add5795531913363d7c46d916c81d1f9b", "OpenPI_chunk_broker_blob": "8fa9d83d023b7c0c60a1d05531343af01e72d09"},
        "action_semantics": protocol["action_contract"],
        "branch_semantics": protocol["branch_contract"],
        "gpu_snapshot": {"contract": "free_memory_mib > 20480; max one project worker per physical GPU; foreign processes untouched", "admissible_by_memory": [4, 5, 6, 7], "project_workers": 0, "foreign_processes_touched": 0},
        "parity_checks": {"M0_exact": False, "M1_checkpoint_materialized": False, "M2_exact_source_checkpoint": True, "common_libero": False, "final_7d_action_authority": True, "fresh_branch_boundary_all_models": False},
    }

    durable_available = 26997472 * 1024 + 43588920 * 1024 + 906192 * 1024
    m1_required = sum(x["bytes"] for x in m1["checkpoints"].values())
    frozen_checkpoint_required = m1_required + 12439122345
    storage = {
        "schema": "STAGE_Z_Z0R1_STORAGE_PREFLIGHT_V1",
        "status": STATUS,
        "snapshot_utc": SNAPSHOT_UTC,
        "durable_mounts": {"/mnt/sdc": {"available_bytes": 26997472 * 1024, "capacity_pct": 100}, "/llm_jzm": {"available_bytes": 43588920 * 1024, "capacity_pct": 99}, "/": {"available_bytes": 906192 * 1024, "capacity_pct": 100}},
        "volatile_space_excluded": {"/dev/shm": {"reason": "tmpfs; not durable authority storage", "usable_for_stage_z": False}},
        "required_frozen_checkpoint_bytes": frozen_checkpoint_required,
        "required_oft_checkpoint_bytes": m1_required,
        "existing_openpi_checkpoint_bytes": 12439122345,
        "durable_available_bytes_sum": durable_available,
        "headroom_after_frozen_checkpoints_bytes": durable_available - frozen_checkpoint_required,
        "sufficient_for_all_checkpoints_and_future_stage_artifacts": False,
        "safe_archival_root_found": False,
        "destructive_cleanup_performed": False,
        "decision": "do not materialize OFT checkpoints until a new durable root or provenance-safe archival relocation is authorized and verified",
    }

    generated = [
        "configs/STAGE_Z_CROSS_MODEL_OPEN_DUTY_PROTOCOL_V2.json",
        "reports/STAGE_Z_Z0R1_STRUCTURAL_MISSINGNESS_AMENDMENT_V1.json",
        "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json",
        "reports/STAGE_Z_Z0R1_MODEL_AUTHORITY_MAP_V2.json",
        "reports/STAGE_Z_Z0R1_ENVIRONMENT_ACTION_PARITY_V2.json",
        "reports/STAGE_Z_Z0R1_STORAGE_PREFLIGHT_V1.json",
    ]
    write_json(generated[0], protocol)
    write_json(generated[1], amendment)
    write_json(generated[2], shared36)
    write_json(generated[3], authority)
    write_json(generated[4], parity)
    write_json(generated[5], storage)

    manifest_entries = [
        {"path": rel, "bytes": (ROOT / rel).stat().st_size, "sha256": sha256_file(ROOT / rel)} for rel in generated
    ]
    manifest_entries.append({"path": "scripts/stage_z/build_stage_z_z0r1.py", "bytes": (ROOT / "scripts/stage_z/build_stage_z_z0r1.py").stat().st_size, "sha256": sha256_file(ROOT / "scripts/stage_z/build_stage_z_z0r1.py")})
    manifest_rel = "reports/STAGE_Z_Z0R1_ARTIFACT_MANIFEST_V1.json"
    write_json(manifest_rel, {"schema": "STAGE_Z_Z0R1_ARTIFACT_MANIFEST_V1", "status": STATUS, "entries": manifest_entries, "root_and_sidecar_excluded": True})
    manifest_hash = sha256_file(ROOT / manifest_rel)
    root_seal = {
        "schema": "STAGE_Z_Z0R1_ROOT_SEAL_V1",
        "status": STATUS,
        "git_binding": {"head_commit": head, "head_tree": tree},
        "snapshot_utc": SNAPSHOT_UTC,
        "artifact_manifest": {"path": manifest_rel, "sha256": manifest_hash, "entries": manifest_entries},
        "population": {"nominal_task_cells": 40, "shared_fresh_identities": 36, "structural_missing_cells": missing},
        "blockers": ["M0 exact server authority mismatch", "OFT checkpoints not materialized because durable storage preflight is insufficient", "current LIBERO runtime points to modified fork and official common task hashes are not sealed", "durable root/headroom is insufficient for complete frozen panel authority and later artifacts"],
        "scientific_rollout_started": False,
        "counters": {"gpu_workers": 0, "model_inference": 0, "simulator": 0, "env_step": 0, "physical_intervention": 0, "v_phys": 0, "eval160": 0, "protected_reads": 0},
        "next_legal_action": "PI review of Z0R1 HOLD; obtain provenance-safe durable storage and resolve M0/common-LIBERO authority before any Z1",
    }
    root_rel = "reports/STAGE_Z_Z0R1_ROOT_SEAL_V1.json"
    write_json(root_rel, root_seal)
    root_hash = sha256_file(ROOT / root_rel)
    (ROOT / "reports/STAGE_Z_Z0R1_ROOT_SEAL_V1.sha256").write_text(root_hash + "  STAGE_Z_Z0R1_ROOT_SEAL_V1.json\n", encoding="utf-8")
    print(json.dumps({"status": STATUS, "head": head, "tree": tree, "shared_fresh": len(selected), "missing": missing, "durable_available_bytes": durable_available, "frozen_checkpoint_required_bytes": frozen_checkpoint_required, "root_sha256": root_hash}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
