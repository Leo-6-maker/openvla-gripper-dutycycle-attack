"""C3-S3: Non-Protected Episode Inventory.

Scans CS200 sidecar episodes and catalogs available fields per episode.
DEVELOPMENT_ONLY — consumable_as_formal_evidence = false.
"""
import json, os, sys, hashlib, time
from collections import defaultdict, Counter

CS200 = "/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean"
FOUR_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]

# Fields expected in a complete sidecar episode
EXPECTED_FIELDS = {
    "object_state", "robot0_eef_pos", "robot0_eef_quat",
    "robot0_gripper_qpos", "mujoco_contact_pairs",
    "step", "state_id",
}

OUT = os.environ.get("C3_S3_INVENTORY_OUT",
      "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_s3_inventory")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_episode(suite, task, state):
    """Inventory a single episode. Returns dict or None."""
    task_dir = os.path.join(CS200, suite, task)
    state_dir = os.path.join(task_dir, state)

    sidecar_path = os.path.join(state_dir, "privileged_teacher_sidecar.jsonl")
    summary_path = os.path.join(state_dir, "episode_summary.json")
    metadata_path = os.path.join(state_dir, "episode_metadata.json")
    steps_path = os.path.join(state_dir, "step_records.jsonl")

    ident = f"{suite}/{task}/{state}"
    record = {
        "episode_id": ident,
        "suite": suite,
        "task_idx": int(task.replace("task_", "")),
        "state_id": state,
        "status": "DEVELOPMENT_ONLY",
        "producer": "DeepSeek",
        "consumable_as_formal_evidence": False,
        "protected": False,
        "protected_rationale": "CS200 public development data, not in CAL/G10/T2R-D",
    }

    missing = []

    # Sidecar
    if not os.path.isfile(sidecar_path):
        missing.append("privileged_teacher_sidecar.jsonl")
        record["available"] = False
        record["missing_files"] = missing
        return record

    record["sidecar_sha256"] = sha256_file(sidecar_path)
    record["available"] = True

    # Parse first line for schema
    try:
        with open(sidecar_path) as f:
            first_line = f.readline().strip()
        if first_line:
            sidecar_schema = json.loads(first_line)
            record["sidecar_schema_version"] = sidecar_schema.get("schema", "unknown")
        else:
            record["sidecar_schema_version"] = "empty_first_line"
    except Exception:
        record["sidecar_schema_version"] = "parse_error"

    # Count steps
    try:
        with open(sidecar_path) as f:
            n_steps = sum(1 for _ in f)
        record["step_count"] = n_steps
    except Exception:
        n_steps = 0
        record["step_count"] = 0

    # Summary
    if os.path.isfile(summary_path):
        record["summary_sha256"] = sha256_file(summary_path)
        with open(summary_path) as f:
            ep_summary = json.load(f)
        record["official_success"] = ep_summary.get("success")
        record["termination_reason"] = ep_summary.get("termination_reason")
    else:
        missing.append("episode_summary.json")

    # Metadata (init state)
    if os.path.isfile(metadata_path):
        record["metadata_sha256"] = sha256_file(metadata_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        record["init_state_sha256"] = metadata.get("initial_state_sha256", "")[:16] + "..."
        record["bddl_path"] = metadata.get("bddl_path", "")
    else:
        missing.append("episode_metadata.json")

    # Step records (actions)
    if os.path.isfile(steps_path):
        record["step_records_sha256"] = sha256_file(steps_path)
        with open(steps_path) as f:
            n_actions = sum(1 for _ in f)
        record["action_count"] = n_actions
    else:
        missing.append("step_records.jsonl")
        record["action_count"] = 0

    # Field availability from first step
    available_fields = set()
    try:
        with open(sidecar_path) as f:
            # Skip first line (schema) and read first step
            f.readline()
            second = f.readline().strip()
        if second:
            step0 = json.loads(second)
            available_fields = set(step0.keys())
            # Check object_state length
            obj_state = step0.get("object_state", [])
            record["object_state_length"] = len(obj_state) if isinstance(obj_state, list) else 0
            record["has_contacts"] = "mujoco_contact_pairs" in step0
            record["has_eef_pos"] = "robot0_eef_pos" in step0
            record["has_eef_quat"] = "robot0_eef_quat" in step0
            record["has_gripper_qpos"] = "robot0_gripper_qpos" in step0
    except Exception:
        pass

    record["available_fields"] = sorted(available_fields)
    record["missing_expected_fields"] = sorted(EXPECTED_FIELDS - available_fields)
    record["missing_files"] = missing

    return record


def main():
    os.makedirs(OUT, exist_ok=True)

    print("=" * 60)
    print("C3-S3: Non-Protected Episode Inventory")
    print("=" * 60)

    all_records = []
    per_suite = defaultdict(list)
    field_counts = Counter()
    missing_field_counts = Counter()

    for suite in FOUR_SUITES:
        suite_dir = os.path.join(CS200, suite)
        if not os.path.isdir(suite_dir):
            continue
        for task in sorted(os.listdir(suite_dir)):
            task_dir = os.path.join(suite_dir, task)
            if not os.path.isdir(task_dir):
                continue
            for state in sorted(os.listdir(task_dir)):
                record = inventory_episode(suite, task, state)
                if record is None:
                    continue
                all_records.append(record)
                per_suite[suite].append(record["episode_id"])
                for f in record.get("available_fields", []):
                    field_counts[f] += 1
                for f in record.get("missing_expected_fields", []):
                    missing_field_counts[f] += 1
                status = "OK" if record.get("available") else "MISSING"
                print(f"  {record['episode_id']}: steps={record.get('step_count',0)} {status}")

    # Write inventory
    inv_path = os.path.join(OUT, "NONPROTECTED_EPISODE_INVENTORY.jsonl")
    with open(inv_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")

    # Field coverage summary
    n_total = len(all_records)
    field_summary = {
        "status": "DEVELOPMENT_ONLY",
        "producer": "DeepSeek",
        "consumable_as_formal_evidence": False,
        "total_episodes": n_total,
        "per_suite": {s: len(ids) for s, ids in per_suite.items()},
        "field_coverage": {f: {"count": c, "pct": round(c / max(1, n_total) * 100, 1)}
                          for f, c in field_counts.most_common()},
        "missing_expected_fields": {f: {"count": c, "pct": round(c / max(1, n_total) * 100, 1)}
                                     for f, c in missing_field_counts.most_common()},
    }
    fc_path = os.path.join(OUT, "FIELD_COVERAGE_SUMMARY.json")
    with open(fc_path, "w") as f:
        json.dump(field_summary, f, indent=2, ensure_ascii=False)

    # Task geometry coverage
    task_coverage = defaultdict(lambda: {"n_episodes": 0, "has_contacts": 0, "has_eef": 0, "has_gripper": 0})
    for r in all_records:
        tk = f"{r['suite']}/task_{r['task_idx']:02d}"
        task_coverage[tk]["n_episodes"] += 1
        if r.get("has_contacts"): task_coverage[tk]["has_contacts"] += 1
        if r.get("has_eef_pos"): task_coverage[tk]["has_eef"] += 1
        if r.get("has_gripper_qpos"): task_coverage[tk]["has_gripper"] += 1
    tc_path = os.path.join(OUT, "TASK_GEOMETRY_COVERAGE.json")
    with open(tc_path, "w") as f:
        json.dump({tk: dict(v) for tk, v in sorted(task_coverage.items())}, f, indent=2, ensure_ascii=False)

    # Missing field matrix
    mf_path = os.path.join(OUT, "MISSING_FIELD_MATRIX.csv")
    with open(mf_path, "w") as f:
        f.write("episode_id,missing_files,missing_fields\n")
        for r in all_records:
            f.write(f"{r['episode_id']},{';'.join(r.get('missing_files', []))},{';'.join(r.get('missing_expected_fields', []))}\n")

    # SHA256SUMS
    sums = {}
    for fn in ["NONPROTECTED_EPISODE_INVENTORY.jsonl", "FIELD_COVERAGE_SUMMARY.json",
               "TASK_GEOMETRY_COVERAGE.json", "MISSING_FIELD_MATRIX.csv"]:
        fp = os.path.join(OUT, fn)
        sums[fn] = sha256_file(fp)
    sums_path = os.path.join(OUT, "INPUT_SHA256SUMS")
    with open(sums_path, "w") as f:
        for fn, s in sums.items():
            f.write(f"{s}  {fn}\n")

    # README
    readme = os.path.join(OUT, "README_DEVELOPMENT_ONLY.md")
    with open(readme, "w") as f:
        f.write(f"""# C3-S3 Non-Protected Episode Inventory

**Status**: DEVELOPMENT_ONLY
**Producer**: DeepSeek
**consumable_as_formal_evidence**: false

## Scope
- {n_total} episodes across 4 suites
- CS200 public development data only
- No CAL/G10/T2R-D reads
- No GPU/model/rollout/attack

## Key Findings
- Episodes with contacts: {sum(1 for r in all_records if r.get('has_contacts'))}/{n_total}
- Episodes with EEF pose: {sum(1 for r in all_records if r.get('has_eef_pos'))}/{n_total}
- Episodes with gripper qpos: {sum(1 for r in all_records if r.get('has_gripper_qpos'))}/{n_total}
""")

    print(f"\nInventory: {OUT}")
    print(f"  Episodes: {n_total}")
    print(f"  Files: NONPROTECTED_EPISODE_INVENTORY.jsonl, FIELD_COVERAGE_SUMMARY.json, TASK_GEOMETRY_COVERAGE.json, MISSING_FIELD_MATRIX.csv, INPUT_SHA256SUMS, README_DEVELOPMENT_ONLY.md")


if __name__ == "__main__":
    main()
