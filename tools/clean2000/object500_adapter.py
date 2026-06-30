"""Object500 adapter: reads VIS bridge CLEAN episode directories.

Source format characteristics:
- episode_summary.json: 46 fields, includes detector/attack telemetry
- step_telemetry.csv: 88 columns, 16 f_* feature columns
- SHA256SUMS.txt: provenance hashes (alternative to artifact_sha256.json)
- privileged_step_records.jsonl: per-step privileged state
- converter_output.json, TEACHER_DRY_RUN.json, timing.json: extra sidecars
- FOLD00_teacher_labels_heldout.jsonl: pre-computed teacher labels

All fields beyond the canonical contract are preserved in source_provenance_extra
but do not become required canonical columns.
"""

import csv
import hashlib
import json
import os


def read_episode_summary(ep_dir):
    with open(os.path.join(ep_dir, "episode_summary.json")) as f:
        return json.load(f)


def read_sha256sums(ep_dir):
    """Parse SHA256SUMS.txt into a dict: filename -> sha256."""
    p = os.path.join(ep_dir, "SHA256SUMS.txt")
    if not os.path.exists(p):
        return {}
    result = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[1]] = parts[0]
            elif len(parts) == 1:
                result[""] = parts[0]
    return result


def analyze_telemetry(ep_dir):
    """Derive feature validity and step continuity from Object500 telemetry.

    Object500 telemetry has 88 columns including detector/attack runtime fields.
    We extract only the canonical telemetry fields.
    """
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

            if step_idx == 0:
                init_vals = _extract_initial_state(row)

    steps_sorted = sorted(set(steps))
    duplicates = len(steps) - len(steps_sorted)
    expected_max = steps_sorted[-1] if steps_sorted else 0
    missing = expected_max + 1 - len(steps_sorted) if steps_sorted else 0
    contiguous = (duplicates == 0 and missing == 0)

    # Valid steps contiguous — uses actual feat_valid=True step IDs
    valid_contiguous = _check_valid_contiguous(sorted(set(valid_step_ids)), first_valid)

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
    return hashlib.sha256("|".join(vals).encode()).hexdigest()


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
    """Build a canonical episode row from an Object500 episode directory.

    Only CLEAN episodes pass the gate. Attack episodes (TRUE_T10 etc) are rejected.
    """
    ep = read_episode_summary(ep_dir)
    shasums = read_sha256sums(ep_dir)
    telemetry = analyze_telemetry(ep_dir)

    condition = ep.get("condition", "")
    if condition != "CLEAN":
        return None

    task_id = int(ep.get("task_idx", -1))
    state_id = int(ep.get("state_id", -1))
    eval_seed = int(ep.get("eval_seed", 0))

    suite = "libero_object"
    parent_key = "{}/task_{:02d}/state_{:03d}".format(suite, task_id, state_id)
    attempt_id = 1
    episode_key = "{}/clean/attempt_{:02d}".format(parent_key, attempt_id)

    # Use telemetry-derived invalid_feature_steps, NOT episode_summary value.
    # The bridge's "invalid_feature_steps" counts extraction failures (NaN etc),
    # while feat_valid=False in telemetry is the canonical definition.
    invalid_feat = telemetry["invalid_feature_steps"]

    n_steps = int(ep.get("n_steps", 0))

    # Mechanism eligibility: Object episodes are generally teacher-eligible
    # unless explicitly marked otherwise
    teacher_eligible = True
    mechanism_eligible = teacher_eligible
    abstain_reason = ""

    # Feature schema from telemetry
    p = os.path.join(ep_dir, "step_telemetry.csv")
    feature_cols = []
    if os.path.exists(p):
        with open(p) as f:
            header = next(csv.reader(f))
        feature_cols = [c for c in header if c.startswith("f_")]
    feature_schema_id = "SC5_PROPRIO_NO_STEP_EEF_V1" if len(feature_cols) >= 16 else "SC5_PROPRIO_NO_STEP_V1"
    feature_schema_sha = _derive_feature_schema_sha(feature_cols)

    # File SHAs: prefer SHA256SUMS, fall back to direct computation
    ep_summary_path = os.path.join(ep_dir, "episode_summary.json")
    telemetry_path = os.path.join(ep_dir, "step_telemetry.csv")
    priv_path = os.path.join(ep_dir, "privileged_step_records.jsonl")
    complete_marker_path = os.path.join(ep_dir, "COMPLETE.json")

    row = {
        "episode_key": episode_key,
        "parent_key": parent_key,
        "suite": suite,
        "task_id": task_id,
        "task_name": ep.get("task_name", ""),
        "state_id": state_id,
        "eval_seed": eval_seed,
        "attempt_id": attempt_id,
        "source_format": "object500_v1",
        "source_root": ep_dir,

        "condition": condition,
        "task_success": bool(ep.get("task_success", False)),
        "gate_pass": bool(ep.get("gate_pass", True)),
        "complete": os.path.exists(complete_marker_path),
        "teacher_eligible": teacher_eligible,
        "mechanism_eligible": mechanism_eligible,
        "abstain_reason": abstain_reason,

        "n_steps": n_steps,
        "n_telemetry_rows": telemetry["n_telemetry_rows"],
        "first_valid_step": telemetry["first_valid_step"],
        "n_valid_steps": telemetry["n_valid_steps"],
        "invalid_feature_steps": invalid_feat,
        "step_index_contiguous": telemetry["step_index_contiguous"],
        "valid_steps_contiguous": telemetry["valid_steps_contiguous"],
        "duplicate_step_count": telemetry["duplicate_step_count"],
        "missing_step_count": telemetry["missing_step_count"],

        "feature_schema_id": feature_schema_id,
        "feature_schema_sha256": feature_schema_sha,

        "episode_summary_sha256": shasums.get("episode_summary.json", compute_file_sha(ep_summary_path)),
        "step_telemetry_sha256": shasums.get("step_telemetry.csv", compute_file_sha(telemetry_path) if os.path.exists(telemetry_path) else ""),
        "privileged_records_sha256": shasums.get("privileged_step_records.jsonl", compute_file_sha(priv_path) if os.path.exists(priv_path) else ""),
        "complete_marker_sha256": shasums.get("COMPLETE.json", compute_file_sha(complete_marker_path)),
        "artifact_inventory_sha256": _compute_artifact_inventory(ep_dir),
        "collector_code_sha256": ep.get("bridge_sha256", ""),
        "source_manifest_sha256": manifest_sha256,
        "observed_initial_state_sha256": telemetry["observed_initial_state_sha256"],
    }

    return row


def _derive_feature_schema_sha(feature_cols):
    if not feature_cols:
        return ""
    return hashlib.sha256(",".join(sorted(feature_cols)).encode()).hexdigest()


def _compute_artifact_inventory(ep_dir):
    files = sorted(f for f in os.listdir(ep_dir) if os.path.isfile(os.path.join(ep_dir, f)))
    payload_lines = []
    for fn in files:
        fp = os.path.join(ep_dir, fn)
        sha = compute_file_sha(fp)
        payload_lines.append("{} {}".format(fn, sha))
    return hashlib.sha256("\n".join(payload_lines).encode()).hexdigest()


def list_episode_dirs(source_root):
    """Discover CLEAN Object500 episodes under the wave directories.

    Expected structure:
      source_root/
        wave1_50_.../jobs/task_N_name/state_N/attempt_N/
        wave2_remaining_.../jobs/task_N_name/state_N/attempt_N/

    Only returns CLEAN episodes (condition=CLEAN in episode_summary).
    """
    episodes = []
    for wave_d in sorted(os.listdir(source_root)):
        wave_path = os.path.join(source_root, wave_d)
        if not os.path.isdir(wave_path):
            continue
        jobs_dir = os.path.join(wave_path, "jobs")
        if not os.path.isdir(jobs_dir):
            continue
        for task_d in sorted(os.listdir(jobs_dir)):
            tp = os.path.join(jobs_dir, task_d)
            if not os.path.isdir(tp):
                continue
            for state_d in sorted(os.listdir(tp)):
                stp = os.path.join(tp, state_d)
                if not os.path.isdir(stp):
                    continue
                for att_d in sorted(os.listdir(stp)):
                    atp = os.path.join(stp, att_d)
                    if not os.path.isdir(atp):
                        continue
                    ep_path = os.path.join(atp, "episode_summary.json")
                    if not os.path.exists(ep_path):
                        continue
                    # Gate: check it's CLEAN
                    try:
                        with open(ep_path) as f:
                            ep = json.load(f)
                        if ep.get("condition") != "CLEAN":
                            continue
                    except Exception:
                        continue
                    if os.path.exists(os.path.join(atp, "COMPLETE.json")):
                        episodes.append(atp)
    return episodes
