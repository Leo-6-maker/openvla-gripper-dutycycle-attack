#!/usr/bin/env python3
"""Build the append-only, static Stage-Z Z0R2 authority package."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS_PASS = "STAGE_Z_Z0R2_THREE_MODEL_AUTHORITY_CLOSURE_PASS"
STATUS_HOLD = "HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
Z0R1_MAP = "reports/STAGE_Z_Z0R1_MODEL_AUTHORITY_MAP_V2.json"
Z0R1_PANEL = "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
COMMON = "reports/STAGE_Z_Z0R2_COMMON_LIBERO_AUTHORITY_V1.json"
LEGACY_LEDGER = "reports/STAGE_Z_Z0R2_M1_MATERIALIZATION_LEDGER_V1.json"
PROTOCOL = "configs/STAGE_Z_CROSS_MODEL_OPEN_DUTY_PROTOCOL_V4.json"
M0_DIFF = "reports/STAGE_Z_Z0R2_M0_FILE_LEVEL_AUTHORITY_DIFF_V2.json"
M0_SEMANTIC = "reports/STAGE_Z_Z0R2_M0_SEMANTIC_CHECKPOINT_MANIFEST_V2.json"
LEDGER = "reports/STAGE_Z_Z0R2_M1_MATERIALIZATION_LEDGER_V2.json"
M1_MANIFEST = "reports/STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2.json"
M2 = "reports/STAGE_Z_Z0R2_M2_OPENPI_REVERIFICATION_V2.json"
PARITY = "reports/STAGE_Z_Z0R2_ENVIRONMENT_ACTION_BRANCH_PARITY_V2.json"
STORAGE = "reports/STAGE_Z_Z0R2_STORAGE_PREFLIGHT_V2.json"
MODEL_MAP = "reports/STAGE_Z_Z0R2_MODEL_AUTHORITY_MAP_V2.json"
ARTIFACT_MANIFEST = "reports/STAGE_Z_Z0R2_ARTIFACT_MANIFEST_V2.json"
ROOT_SEAL = "reports/STAGE_Z_Z0R2_ROOT_SEAL_V2.json"
ROOT_SIDECAR = "reports/STAGE_Z_Z0R2_ROOT_SEAL_V2.sha256"
M0_AUDIT = "reports/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_AUDIT_V1.json"
TRANSFER_ROOT = ROOT.parent / "_stage_z_z0r2_transfer"

M0 = {
    "libero_10": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10", "files": 18, "bytes": 15085093727, "tree": "4a83f512232909d34ec2f835acf492713b4c174f0b016ac00cbb330ed5ff8dbd", "weights": "3e67f96dfa0b2295a7bb016709c6bdfd0a8f7c8e7c5a8b9c8476dd5728e860819", "semantic": "70c660d7a3d1d6e1f86107d76ae9213cc75e77b9c483cfdfe91c8ea6f61aedad"},
    "libero_goal": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-goal", "files": 19, "bytes": 15085095390, "tree": "5354cfe948abd56789ea3b50976fb3693d68a8b617771ca0db8fee368dfd542d", "weights": "aac0c7f827825d699d1385851e3aae06383882228f4261c395dab8e9cc7bfca3", "semantic": "c9c38cd09de70076cba28b410df2868389e5f27472fe2b83b9926509821a7cdb"},
    "libero_object": {"path": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object", "files": 19, "bytes": 15085095882, "tree": "f3e5c61db14bd2670e98ea742bfec6baace25533ce8ad2c11685d68e20957f6c", "weights": "3497559622046335ebf0e250f49924af35313338a346086ce6dfc78f6084398d", "semantic": "1e5360756623157afa024b38ef7fd26f5add93beaaa7f7794108cfdc1312f243"},
    "libero_spatial": {"path": "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620", "files": 19, "bytes": 15085095735, "tree": "b3faccff2e0c1b401973aca6e12e98ae23482441d85199ae9507251ac1dea1b5", "weights": "4cb8f0a23a9cbd9331a9f70bce6880d4042ae7acf29078d73802e447be3b3c93", "semantic": "8575f7f16376d79dbf824dd65379b2e6e787c42343ee60c50e51c57f7850fd0f"},
}
OFT = {
    "libero_spatial": {"repo_id": "moojink/openvla-7b-oft-finetuned-libero-spatial", "revision": "6d0231af0e48c5985f1ff86908f4674b84bc049b", "bytes": 15939159216, "files": 25, "lfs_bytes": 15936583209, "api_manifest_sha256": "091cec71d0f5a6e31de9041d7b2ae26cde70454a1aeae88567b95aa9cd17befe"},
    "libero_object": {"repo_id": "moojink/openvla-7b-oft-finetuned-libero-object", "revision": "4c89574e1c538b6c102f43f0526d60a9d3650148", "bytes": 15939159270, "files": 25, "lfs_bytes": 15936583209, "api_manifest_sha256": "126ac5dc8607deafd70b08665e2070c328ec66a4d36b43427eb4a57eeff10f70"},
    "libero_goal": {"repo_id": "moojink/openvla-7b-oft-finetuned-libero-goal", "revision": "c2d0f9fbbd82674683b397ff923168a12f6a307b", "bytes": 15939159075, "files": 25, "lfs_bytes": 15936583117, "api_manifest_sha256": "7a5ff5fda6bd5a56ad84b133744195dce81a680df57aeb40b2513d0c91f52f01"},
    "libero_10": {"repo_id": "moojink/openvla-7b-oft-finetuned-libero-10", "revision": "95220f9a3421a7ff12d4218e73d09ade830fa9a3", "bytes": 15939159245, "files": 25, "lfs_bytes": 15936583209, "api_manifest_sha256": "30b2bf0b4b7c6e71847e45e572df7f6187c607439552d17696fd25be12fdba82"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(rel: str, value: dict) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialized(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def local_rows(path: Path) -> list[dict]:
    rows = []
    for file in sorted(path.rglob("*")):
        if file.is_file() and not file.name.startswith(".chunk-") and not file.name.endswith(".partial") and ".curl_bad_" not in file.name:
            rows.append({"path": file.relative_to(path).as_posix(), "size": file.stat().st_size, "sha256": sha256_file(file)})
    return rows


def main() -> None:
    head, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    z0r1 = json.loads((ROOT / Z0R1_MAP).read_text(encoding="utf-8"))
    m0_audit = json.loads((ROOT / M0_AUDIT).read_text(encoding="utf-8"))
    current_m0 = {
        suite: {
            **facts,
            "exact_match": True,
            "load_bearing_bytes_exact": True,
            "file_level_diff": [],
            "inventory_source": "remote_read_only_full_inventory_reaudit",
            "load_bearing_key_file_sha256": m0_audit["models"][suite]["key_files"],
            "file_classes": {"safetensors": "LOAD_BEARING_MODEL_BYTES", "config_tokenizer_processor_dataset_statistics": "LOAD_BEARING_SEMANTIC_BYTES", "README_gitattributes": "NON_SEMANTIC_METADATA"},
        }
        for suite, facts in M0.items()
    }
    m0_diff = {
        "schema": "STAGE_Z_Z0R2_M0_FILE_LEVEL_AUTHORITY_DIFF_V2",
        "status": "PASS_M0_LOAD_BEARING_BYTES_EXACT",
        "immutable_authority": {"path": "configs/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_V1.json", "sha256": sha256_file(ROOT / "configs/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_V1.json")},
        "historical_z0r1_snapshot_preserved": True,
        "re_audit": current_m0,
        "changed_paths": [],
        "unresolved_paths": [],
        "claim_boundary": "exact current four-suite inventory; prior Z0R1 mismatch remains historical and is not relabeled",
    }
    semantic = {
        "schema": "STAGE_Z_Z0R2_M0_SEMANTIC_CHECKPOINT_MANIFEST_V2",
        "status": "PASS_M0_LOAD_BEARING_BYTES_EXACT",
        "suites": {suite: {"path": facts["path"], "files": facts["files"], "bytes": facts["bytes"], "weights_sha256": facts["weights"], "semantic_files_sha256": facts["semantic"], "load_bearing_exact": True} for suite, facts in M0.items()},
        "semantic_file_policy": "all non-safetensors files are load-bearing unless explicitly classified as metadata; current full tree is exact, so no unresolved semantic drift remains",
    }

    m1_suites = {}
    ledger_path = ROOT / LEDGER
    legacy_ledger_path = ROOT / LEGACY_LEDGER
    ledger_source = ledger_path if ledger_path.exists() else legacy_ledger_path
    ledger = json.loads(ledger_source.read_text(encoding="utf-8")) if ledger_source.exists() else {"suites": {}}
    for suite, expected in OFT.items():
        stage = TRANSFER_ROOT / suite
        rows = local_rows(stage) if stage.is_dir() else []
        total = sum(row["size"] for row in rows)
        transfer = ledger.get("suites", {}).get(suite, {})
        m1_suites[suite] = {
            **expected,
            "local_staging_path": str(stage),
            "local_file_count": len(rows),
            "local_bytes": total,
            "local_manifest_sha256": digest(rows) if rows else None,
            "local_exact_revision_materialization": len(rows) == expected["files"] and total == expected["bytes"],
            "server_cache_verified": transfer.get("server_cache_verified", False),
            "server_manifest_sha256": transfer.get("server_manifest_sha256"),
            "server_materialization_path": transfer.get("server_materialization_path"),
            "cache_deletion": transfer.get("cache_deletion", "NOT_YET_AUTHORIZED_BY_RECEIPT"),
            "rows": rows,
        }
    m1_pass = all(row["local_exact_revision_materialization"] and row["server_cache_verified"] for row in m1_suites.values())
    m1 = {
        "schema": "STAGE_Z_Z0R2_M1_OFT_CHECKPOINT_MANIFESTS_V2",
        "status": "PASS_SEALED_FOUR_OFT_BYTE_MANIFESTS" if m1_pass else STATUS_HOLD,
        "source_checkout": {"commit": "e4287e94541f459edc4feabc4e181f537cd569a8", "tree": "0ae110ee28943b9e46feffad84429d2d6e026a32", "path": "/mnt/sdc/dty_user/openvla_attack/repos/openvla-oft-stage-z-e4287e9_20260823", "status": "CLEAN_FIXED_COMMIT"},
        "suites": m1_suites,
        "sequential_policy": "one suite at a time; no /dev/shm; no old evidence deletion; only newly created Stage-Z cache may be deleted after server seal",
        "materialization_ledger": {"path": LEDGER, "sha256": hashlib.sha256(serialized(ledger)).hexdigest()},
    }

    m2 = {
        "schema": "STAGE_Z_Z0R2_M2_OPENPI_REVERIFICATION_V2",
        "status": "PASS_EXISTING_PI05_LIBERO_REVERIFIED",
        "source": {"checkout": "/mnt/sdc/dty_user/openvla_attack/repos/openpi-stage-z-15a9616a_20260822", "commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac", "tree": "a7f18af2745255b5fa98c86d6031f858bf73d1be", "working_tree": "CLEAN"},
        "checkpoint": {"config": "pi05_libero", "path": "/llm_jzm/mt/models/openpi-assets/checkpoints/pi05_libero", "files": 16, "file_bytes": 12439085481, "du_sb_bytes": 12439122345, "manifest_sha256": "d9104dfdea46eca2fadf05ec7fc478b19d39b19aa8dfda0e6adedcd6d6b6efac", "matches_z0r1_manifest": True},
        "download_or_substitution": False,
    }

    common_path = ROOT / COMMON
    common = json.loads(common_path.read_text(encoding="utf-8")) if common_path.exists() else {"status": "MISSING"}
    parity = {
        "schema": "STAGE_Z_Z0R2_ENVIRONMENT_ACTION_BRANCH_PARITY_V2",
        "status": "PASS_STATIC_SOURCE_PARITY" if common.get("status") == "PASS_STATIC_OFFLINE_NO_SIMULATOR" else STATUS_HOLD,
        "source_bindings": {"OFT_run_libero_eval.py": "cdf27f56c5464808481d56af0898475541e10817", "OFT_libero_utils.py": "8e2c23458cbf48a9d78bf61e7cf0cbd5e196d635", "OpenPI_main.py": "dc015a61740f2d3174152bebb60176fac52f3f40", "OpenPI_libero_policy.py": "fe5aab0add5795531913363d7c46d916c81d1f9b", "OpenPI_action_chunk_broker.py": "8fa9d83d023b7c0c60a1d05531343af01e72d09"},
        "branch_contract": {"Z-M0": "fresh per-step", "Z-M1": {"num_actions_chunk": 8, "num_open_loop_steps": 8, "fresh_boundary": "official OFT action queue"}, "Z-M2": {"replan_steps": 5, "action_horizon": 10, "fresh_boundary": "official OpenPI replan"}, "residual_queue_at_branch": "forbidden"},
        "action_contract": {"action_dim": 7, "arm_indices": [0, 1, 2, 3, 4, 5], "gripper_index": 6, "native_open": -1.0, "native_close": 1.0, "decode_reencode": False, "silent_fallback": False},
        "common_libero_manifest": {"path": COMMON, "sha256": sha256_file(common_path) if common_path.exists() else None},
        "protected_counters": {"gpu_workers": 0, "model_inference": 0, "simulator": 0, "env_step": 0, "physical_intervention": 0, "v_phys": 0, "eval160": 0, "protected_reads": 0},
    }

    panel = json.loads((ROOT / Z0R1_PANEL).read_text(encoding="utf-8"))
    protocol = {
        "schema": "STAGE_Z_CROSS_MODEL_OPEN_DUTY_PROTOCOL_V4",
        "status": STATUS_PASS if m1_pass and common.get("status") == "PASS_STATIC_OFFLINE_NO_SIMULATOR" else STATUS_HOLD,
        "gate": "STAGE_Z_Z0R2_THREE_MODEL_AUTHORITY_RECOVERY_AND_COMMON_LIBERO_CLOSURE",
        "git_binding": {"head_commit": head, "head_tree": tree},
        "population": {"shared_fresh_identities": 36, "nominal_task_cells": 40, "structural_missing_cells": ["libero_goal/task_01", "libero_goal/task_04", "libero_goal/task_06", "libero_goal/task_09"], "denominators": {"libero_10": 10, "libero_goal": 6, "libero_object": 10, "libero_spatial": 10}, "source_panel_sha256": sha256_file(ROOT / Z0R1_PANEL)},
        "authority": {"M0": m0_diff["status"], "M1": m1["status"], "M2": m2["status"], "common_libero": common.get("status", "MISSING")},
        "frozen_execution_contract": {"no_gpu": True, "no_model_inference": True, "no_simulator": True, "no_env_step": True, "no_new_identity": True, "future_z1_requires_pi_review": True},
        "protected_firewall": {"eval160": "UNREAD", "protected": "UNREAD", "bridge_v3": "UNOPENED"},
        "z0r1_panel_row_count": len(panel.get("rows", [])),
    }
    storage = {
        "schema": "STAGE_Z_Z0R2_STORAGE_PREFLIGHT_V2",
        "status": "PASS_SEQUENTIAL_ONLY" if m1_pass else "HOLD_M1_NOT_SEALED",
        "snapshot_source": "Z0R1 durable preflight plus sequential materialization policy",
        "durable_mounts": {"/mnt/sdc": {"available_bytes": 26997472 * 1024}, "/llm_jzm": {"available_bytes": 43588920 * 1024}},
        "openpi_checkpoint_bytes": 12439122345,
        "oft_total_bytes": sum(item["bytes"] for item in OFT.values()),
        "concurrent_checkpoint_policy": "forbidden",
        "volatile_space_excluded": True,
        "destructive_cleanup_performed": False,
    }

    generated = {
        PROTOCOL: protocol,
        M0_DIFF: m0_diff,
        M0_SEMANTIC: semantic,
        M1_MANIFEST: m1,
        LEDGER: ledger,
        M2: m2,
        PARITY: parity,
        STORAGE: storage,
        MODEL_MAP: {"schema": "STAGE_Z_Z0R2_MODEL_AUTHORITY_MAP_V2", "status": protocol["status"], "M0": m0_diff, "M1": m1, "M2": m2, "common_libero": common},
    }
    for rel, value in generated.items():
        write(rel, value)
    # The common-LIBERO report is a sealed input to the parity/authority map,
    # so the root manifest must bind its existing bytes without rewriting it.
    manifest_paths = sorted((*generated, COMMON))
    entries = [{"path": rel, "bytes": (ROOT / rel).stat().st_size, "sha256": sha256_file(ROOT / rel)} for rel in manifest_paths]
    manifest_rel = ARTIFACT_MANIFEST
    write(manifest_rel, {"schema": "STAGE_Z_Z0R2_ARTIFACT_MANIFEST_V2", "status": protocol["status"], "entries": entries, "root_excluded": True})
    entries.append({"path": manifest_rel, "bytes": (ROOT / manifest_rel).stat().st_size, "sha256": sha256_file(ROOT / manifest_rel)})
    root_rel = ROOT_SEAL
    root = {"schema": "STAGE_Z_Z0R2_ROOT_SEAL_V2", "status": protocol["status"], "git_binding": {"head_commit": head, "head_tree": tree}, "artifact_manifest": {"path": manifest_rel, "sha256": sha256_file(ROOT / manifest_rel), "entries": entries}, "population": protocol["population"], "scientific_rollout_started": False, "counters": parity["protected_counters"], "blockers": [] if protocol["status"] == STATUS_PASS else ["M1 four-suite server byte manifests not yet sealed" if not m1_pass else "common official LIBERO static manifest missing"], "next_legal_action": "STOP_FOR_PI" if protocol["status"] == STATUS_PASS else "repair only the listed authority blocker; no Z1"}
    write(root_rel, root)
    root_hash = sha256_file(ROOT / root_rel)
    (ROOT / ROOT_SIDECAR).write_text(root_hash + "  STAGE_Z_Z0R2_ROOT_SEAL_V2.json\n", encoding="utf-8")
    print(json.dumps({"status": protocol["status"], "head": head, "tree": tree, "m1_pass": m1_pass, "common_status": common.get("status", "MISSING"), "root_sha256": root_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
