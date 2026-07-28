"""[DeepSeek] FIT670 Schema Freezer — Gate F670-A.

Generates the four frozen schema files for FIT670 collection:
  FIT670_COLLECTION_PROTOCOL_V1.json
  FIT670_FEATURE_SCHEMA.json
  FIT670_GPU_SHARD_PLAN.json   (placeholder — filled by F670-D)

The identity allowlist is built separately by build_fit670_identity_allowlist.py.

Usage:
  python n5/phase2_labels/freeze_fit670_schemas.py --out /path/to/schema_root
"""
import argparse, json, os, time, uuid, shutil
from pathlib import Path

# Shared across all schema files
BUILD_TIME = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
SCHEMA_ROOT_DEFAULT = "FIT670_SCHEMAS_V1"


def seal_output(staging):
    """Minimal seal for schema output directory."""
    from fit_collection_core import sha256_file
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    sums = "\n".join(
        f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}"
        for p in payload) + "\n"
    (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return sums_sha


def write_protocol(staging):
    protocol = {
        "amendment": "PROTOCOL_AMENDMENT_V6_FIT670_ATOMIC",
        "schema": "FIT670_COLLECTION_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "parent_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
        "created_at": BUILD_TIME,
        "purpose": "Extends V5 fresh40 protocol to 670-identity multi-GPU atomic collection",
        "per_step_sequence": [
            "1. capture_forward: env.sim.forward() before reading body/site/geom poses",
            "2. source_stability: verify qpos/qvel/act/time unchanged by forward()",
            "3. telemetry_capture: record sim state, EEF, gripper, object_state, entities, contact_pairs",
            "4. student_input_capture: save RGB frame as PNG, compute frame SHA",
            "5. model_inference: predict_action_with_scores(image, task_language)",
            "6. action_validation: verify parity (raw==score), finiteness, shape",
            "7. action_postprocess: adapter.postprocess() → executed action",
            "8. env_step: env.step(executed)",
            "9. done_check: break if episode terminated"
        ],
        "per_episode_sequence": [
            "1. load_init_state: suite_obj.get_task_init_states(task_id)[state_id]",
            "2. verify_init_sha: pickle.dumps(canonical_state) matches initial_state_sha256",
            "3. verify_entity_identity: all C1 registry bindings match live MuJoCo model",
            "4. capture_model_geometry: snapshot body/site/geom census",
            "5. run step loop: per_step_sequence for up to HORIZONS[suite] steps",
            "6. validate_episode: shapes, finiteness, step_count > 0",
            "7. save_student_inputs: per-step PNG frames + SHA",
            "8. atomic_publish: seal staging → rename to episodes/{suite}/task_{id}/state_{id}/"
        ],
        "per_step_telemetry": {
            "required": [
                "sim_state.time", "sim_state.qpos", "sim_state.qvel", "sim_state.act",
                "robot0_eef_pos", "robot0_eef_quat",
                "robot0_gripper_qpos", "gripper_width",
                "object_state", "entities", "contact_pairs", "contact_count",
                "forward_before_capture", "protocol_amendment"
            ]
        },
        "per_episode_telemetry": {
            "model_geometry_snapshot": True,
            "c1_registry_binding": True,
            "bddl_sha256": True,
            "model_config_sha256": True
        },
        "student_input": {
            "per_step_rgb_png": True,
            "per_step_frame_sha256": True,
            "per_step_causal_proprio": True,
            "per_episode_task_language": True
        },
        "quaternion_convention": "wxyz",
        "atomic_publish": {
            "staging_pattern": ".<episode_id>.staging.<pid>.<uuid8>",
            "target_pattern": "episodes/{suite}/task_{task_id:02d}/state_{state_id:02d}/",
            "no_overwrite": True,
            "seal_before_publish": True
        },
        "sharding": {
            "algorithm": "COST_DESCENDING_GREEDY_BIN_PACK",
            "cost_metric": "HORIZONS[suite]",
            "secondary_sort": "episode_id",
            "balance_constraints": ["suite", "task_id"],
            "default_n_shards": 6,
            "max_n_shards": 8
        },
        "transition_schema": "FIT670_INFERENCE_TRANSITION_V1",
        "identity_pool": "D0-R2_DEV_POOL_670",
        "protected_union": 1330,
        "protected_overlap": 0,
        "training_authorized": False,
        "inference_authorized": True,
        "attack_authorized": False,
        "consumer_eligible": False,
        "cross_gpu_trajectory_parity_guaranteed": False,
        "sdpa_bf16_non_determinism_documented": True
    }
    (staging / "FIT670_COLLECTION_PROTOCOL_V1.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")


def write_feature_schema(staging):
    schema = {
        "schema": "FIT670_FEATURE_SCHEMA_V1",
        "created_at": BUILD_TIME,
        "inherits_from": "R5-F episode.json structure",
        "step_record": {
            "required": [
                "step", "suite", "task_idx", "state_id",
                "action_raw_7d", "score_action_7d", "action_env_7d",
                "generation_passes_per_step",
                "single_generation_parity_pass",
                "action_mutation_by_detector"
            ],
            "action_format": "7-D delta EEF pose (dx, dy, dz, droll, dpitch, dyaw) + gripper (open=1, close=-1)"
        },
        "telemetry_record": {
            "required": [
                "step", "suite", "task_idx", "state_id",
                "sim_state.time",
                "sim_state.qpos", "sim_state.qvel", "sim_state.act",
                "robot0_eef_pos", "robot0_eef_quat",
                "robot0_gripper_qpos", "gripper_width",
                "object_state",
                "entities", "contact_pairs", "contact_count",
                "forward_before_capture", "protocol_amendment"
            ],
            "entity_pose": {
                "entity_type": "body|site|geom",
                "entity_id": "int (MuJoCo index)",
                "entity_name": "str (MuJoCo name)",
                "parent_body_id": "int",
                "world_pose": {
                    "position": "[x, y, z] (float)",
                    "quaternion": "[w, x, y, z] (float, wxyz convention)"
                }
            },
            "contact_pair": {
                "geom1": "str", "geom2": "str",
                "body1": "str", "body2": "str",
                "dist": "float", "efc_address": "int"
            }
        },
        "episode_metadata": {
            "required": [
                "episode_id", "suite", "task_id", "state_id",
                "collection_seed", "pilot_identity_bound",
                "task_bddl_sha256", "registry_task_sha256",
                "step_count", "official_horizon",
                "generation_passes_per_step",
                "steps", "telemetry", "relations",
                "source_mode", "forward_before_capture",
                "protocol_amendment",
                "geometry_status", "placement_state",
                "model_inference", "attack_enabled",
                "detector_loaded", "teacher_labels_generated"
            ]
        },
        "model_geometry_snapshot": {
            "required": [
                "quaternion_convention", "nbody", "ngeom", "nsite",
                "bodies", "geoms", "sites"
            ],
            "body_fields": ["id", "name", "parent_id", "default_pos", "default_quat"],
            "geom_fields": ["id", "name", "type", "body_id", "pos", "quat", "size"],
            "site_fields": ["id", "name", "body_id", "pos", "quat", "size"]
        },
        "student_input": {
            "per_step": {
                "frame_rgb_png": "steps/step_{i:04d}.png",
                "frame_sha256": "embedded in step record"
            },
            "per_episode": {
                "task_language": "embedded in episode metadata"
            }
        }
    }
    (staging / "FIT670_FEATURE_SCHEMA.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


def write_shard_plan_placeholder(staging):
    plan = {
        "schema": "FIT670_GPU_SHARD_PLAN_V1",
        "status": "PLACEHOLDER",
        "created_at": BUILD_TIME,
        "n_shards": 6,
        "algorithm": "COST_DESCENDING_GREEDY_BIN_PACK",
        "cost_metric": "HORIZONS[suite]",
        "secondary_sort": "episode_id",
        "balance_constraints": ["suite", "task_id"],
        "shards": None,
        "note": "This placeholder is replaced by build_fit670_shard_plan.py in Gate F670-D"
    }
    (staging / "FIT670_GPU_SHARD_PLAN.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="Output directory for schema files")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")

    staging = out.parent / f".{out.name}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    published = False
    try:
        write_protocol(staging)
        write_feature_schema(staging)
        write_shard_plan_placeholder(staging)

        # Write manifest
        manifest = {
            "gate": "F670-A_SCHEMA_FREEZE",
            "created_at": BUILD_TIME,
            "files": [
                "FIT670_COLLECTION_PROTOCOL_V1.json",
                "FIT670_FEATURE_SCHEMA.json",
                "FIT670_GPU_SHARD_PLAN.json"
            ],
            "status": "FROZEN",
            "next_gate": "F670-B (identity allowlist build + core extraction)"
        }
        (staging / "SCHEMA_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        seal_output(staging)
        staging.rename(out)
        published = True
        print(f"F670-A schemas frozen: {out}")
        for f in manifest["files"]:
            print(f"  {f}")
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
