"""CLEAN1500 adapter: reads run_cross_suite_clean_v3 episode directories.

Source format characteristics:
- episode_summary.json: 15 fields, no detector/attack telemetry
- step_telemetry.csv: 50 columns, 13 f_* feature columns
- artifact_sha256.json: provenance hashes
- privileged_step_records.jsonl: per-step privileged state
- No invalid_feature_steps in summary — derived from telemetry
"""

import csv
import hashlib
import json
import os


def read_episode_summary(ep_dir):
    with open(os.path.join(ep_dir, "episode_summary.json")) as f:
        return json.load(f)


def read_artifact_hashes(ep_dir):
    p = os.path.join(ep_dir, "artifact_sha256.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def read_telemetry_header(ep_dir):
    p = os.path.join(ep_dir, "step_telemetry.csv")
    if not os.path.exists(p):
        return None, []
    with open(p) as f:
        reader = csv.reader(f)
        header = next(reader)
    return p, header


def analyze_telemetry(ep_dir):
    """Derive feature validity and step continuity from telemetry."""
    p = os.path.join(ep_dir, "step_telemetry.csv")
    if not os.path.exists(p):
        return {
            "n_telemetry_rows": 0,
            "first_valid_step": -1,
            "n_valid_steps": 0,
            "invalid_feature_steps": 0,
            "step_index_contiguous": False,
            "valid_steps_contiguous": False,
            "duplicate_step_count": 0,
            "missing_step_count": 0,
            "observed_initial_state_sha256": "",
        }

    steps = []
    first_valid = -1
    n_valid = 0
    n_invalid = 0

    # Collect initial state values for hashing
    init_vals = []
    valid_step_ids = []

    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_idx = int(row.get("step", -1))
            feat_valid = row.get("feat_valid", "").lower() == "true"

            steps.append(step_idx)
            if feat_valid:
                n_valid += 1
                valid_step_ids.append(step_idx)
                if first_valid < 0:
                    first_valid = step_idx
            else:
                n_invalid += 1

            # Capture step 0 state for initial_state_sha256
            if step_idx == 0:
                init_vals = _extract_initial_state(row)

    # Step continuity
    steps_sorted = sorted(set(steps))
    duplicates = len(steps) - len(steps_sorted)
    expected_max = steps_sorted[-1] if steps_sorted else 0
    missing = expected_max + 1 - len(steps_sorted) if steps_sorted else 0
    contiguous = (duplicates == 0 and missing == 0)

    # Valid steps contiguous — uses actual feat_valid=True step IDs
    valid_contiguous = _check_valid_contiguous(sorted(set(valid_step_ids)), first_valid)

    # Initial state hash
    obs_sha = _hash_initial_state(init_vals)

    return {
        "n_telemetry_rows": len(steps),
        "first_valid_step": first_valid,
        "n_valid_steps": n_valid,
        "invalid_feature_steps": n_invalid,
        "step_index_contiguous": contiguous,
        "valid_steps_contiguous": valid_contiguous,
        "duplicate_step_count": duplicates,
        "missing_step_count": missing,
        "observed_initial_state_sha256": obs_sha,
    }


def _extract_initial_state(row):
    """Extract state columns from step 0 for hashing."""
    keys = [
        "gripper_qpos", "gripper_width",
        "eef_x", "eef_y", "eef_z",
        "object_x", "object_y", "object_z",
        "target_x", "target_y", "target_z",
    ]
    vals = []
    for k in keys:
        v = row.get(k, "")
        if v:
            vals.append(v)
    return vals


def _hash_initial_state(vals):
    if not vals:
        return ""
    payload = "|".join(vals)
    return hashlib.sha256(payload.encode()).hexdigest()


def _check_valid_contiguous(valid_steps, first_valid):
    if not valid_steps:
        return False
    for i, s in enumerate(valid_steps):
        if s != first_valid + i:
            return False
    return True


def compute_file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_canonical(ep_dir, manifest_sha256=""):
    """Build a canonical episode row from a CLEAN1500 episode directory.

    Returns dict with all CANONICAL_FIELDS populated.
    Returns None if the episode is not CLEAN (attack directory).
    """
    ep = read_episode_summary(ep_dir)
    artifacts = read_artifact_hashes(ep_dir)
    telemetry = analyze_telemetry(ep_dir)

    # Gate: must be CLEAN
    condition = ep.get("condition", "")
    if condition != "CLEAN":
        return None

    suite = ep.get("suite", "")
    task_id = int(ep.get("task_idx", -1))
    state_id = int(ep.get("state_id", -1))
    eval_seed = int(ep.get("eval_seed", 0))

    # Build keys
    parent_key = "{}/task_{:02d}/state_{:03d}".format(suite, task_id, state_id)
    attempt_id = 1  # CLEAN1500 uses eval_seed=0 uniformly, single attempt
    episode_key = "{}/clean/attempt_{:02d}".format(parent_key, attempt_id)

    # Mechanism eligibility
    teacher_eligible = bool(ep.get("teacher_eligible", False))
    abstain_reason = ep.get("abstain_reason", "")
    mechanism_eligible = teacher_eligible and not abstain_reason

    # Feature schema: derived from telemetry columns
    p, header = read_telemetry_header(ep_dir)
    feature_cols = [c for c in header if c.startswith("f_")] if header else []
    feature_schema_id = "SC5_PROPRIO_NO_STEP_V1" if len(feature_cols) >= 13 else "UNKNOWN"
    feature_schema_sha = _derive_feature_schema_sha(feature_cols)

    # File SHAs
    complete_marker_path = os.path.join(ep_dir, "COMPLETE.json")
    episode_summary_path = os.path.join(ep_dir, "episode_summary.json")
    telemetry_path = os.path.join(ep_dir, "step_telemetry.csv")
    priv_path = os.path.join(ep_dir, "privileged_step_records.jsonl")

    row = {
        "episode_key": episode_key,
        "parent_key": parent_key,
        "suite": suite,
        "task_id": task_id,
        "task_name": ep.get("task_name", ""),
        "state_id": state_id,
        "eval_seed": eval_seed,
        "attempt_id": attempt_id,
        "source_format": "clean1500_v1",
        "source_root": ep_dir,

        "condition": condition,
        "task_success": bool(ep.get("task_success", False)),
        "gate_pass": bool(ep.get("gate_pass", False)),
        "complete": os.path.exists(complete_marker_path),
        "teacher_eligible": teacher_eligible,
        "mechanism_eligible": mechanism_eligible,
        "abstain_reason": abstain_reason,

        "n_steps": int(ep.get("n_steps", 0)),
        "n_telemetry_rows": telemetry["n_telemetry_rows"],
        "first_valid_step": telemetry["first_valid_step"],
        "n_valid_steps": telemetry["n_valid_steps"],
        "invalid_feature_steps": telemetry["invalid_feature_steps"],
        "step_index_contiguous": telemetry["step_index_contiguous"],
        "valid_steps_contiguous": telemetry["valid_steps_contiguous"],
        "duplicate_step_count": telemetry["duplicate_step_count"],
        "missing_step_count": telemetry["missing_step_count"],

        "feature_schema_id": feature_schema_id,
        "feature_schema_sha256": feature_schema_sha,

        "episode_summary_sha256": artifacts.get("episode_summary.json", compute_file_sha(episode_summary_path)),
        "step_telemetry_sha256": artifacts.get("step_telemetry.csv", compute_file_sha(telemetry_path) if os.path.exists(telemetry_path) else ""),
        "privileged_records_sha256": artifacts.get("privileged_step_records.jsonl", compute_file_sha(priv_path) if os.path.exists(priv_path) else ""),
        "complete_marker_sha256": artifacts.get("COMPLETE.json", compute_file_sha(complete_marker_path)),
        "artifact_inventory_sha256": _compute_artifact_inventory(ep_dir),
        "collector_code_sha256": artifacts.get("collector_sha256", ""),
        "source_manifest_sha256": manifest_sha256,
        "observed_initial_state_sha256": telemetry["observed_initial_state_sha256"],
    }

    return row


def _derive_feature_schema_sha(feature_cols):
    if not feature_cols:
        return ""
    payload = ",".join(sorted(feature_cols))
    return hashlib.sha256(payload.encode()).hexdigest()


def _compute_artifact_inventory(ep_dir):
    """Hash the set of filenames and their SHAs in the episode directory."""
    files = sorted(f for f in os.listdir(ep_dir) if os.path.isfile(os.path.join(ep_dir, f)))
    payload_lines = []
    for fn in files:
        fp = os.path.join(ep_dir, fn)
        sha = compute_file_sha(fp)
        payload_lines.append("{} {}".format(fn, sha))
    return hashlib.sha256("\n".join(payload_lines).encode()).hexdigest()


def list_episode_dirs(source_root):
    """Discover all episode directories under a CLEAN1500 source root.

    Expected structure:
      source_root/
        libero_spatial/task_XX/state_YY/
        libero_goal/task_XX/state_YY/
        libero_10/task_XX/state_YY/

    Returns list of absolute paths to episode directories.
    """
    episodes = []
    for suite in os.listdir(source_root):
        sp = os.path.join(source_root, suite)
        if not os.path.isdir(sp):
            continue
        if suite not in ("libero_spatial", "libero_goal", "libero_10"):
            continue
        for task_d in os.listdir(sp):
            tp = os.path.join(sp, task_d)
            if not os.path.isdir(tp):
                continue
            for state_d in os.listdir(tp):
                stp = os.path.join(tp, state_d)
                if not os.path.isdir(stp):
                    continue
                if "QUARANTINE" in state_d:
                    continue
                if os.path.exists(os.path.join(stp, "COMPLETE.json")):
                    episodes.append(stp)
    return episodes
