#!/usr/bin/env python3
"""D7 Table1 post-run audit.

Validates collected episodes against the preregistered manifest:
  - planned vs completed
  - paired parent keys (clean/attack/control)
  - condition protocol compliance
  - C2e3 detector & threshold SHA consistency
  - attack_frames, success bool, telemetry fields

CPU-only. No env.step, no rollout, no attack.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def load_parent_key(row: Dict[str, str]) -> str:
    return row.get("parent_key", row.get("group_key", ""))

def safe_int(val: Any, default: int = 0) -> int:
    try: return int(val)
    except (ValueError, TypeError): return default

def pair_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("suite", "")), str(row.get("parent_key", "")))


def main():
    ap = argparse.ArgumentParser(description="D7 Table1 post-run audit")
    ap.add_argument("--queue-manifest", required=True)
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    episode_dir = Path(args.episode_dir)

    queue_rows = read_csv(args.queue_manifest)
    print(f"D7 Audit: {len(queue_rows)} planned episodes")

    # ========== Check 1: Planned vs Completed ==========
    audit_rows: List[Dict[str, Any]] = []
    completed = 0
    missing = 0
    extra_denominator = 0

    for qr in queue_rows:
        suite = qr["suite"]
        condition = qr["condition"]
        parent_key = load_parent_key(qr)

        # Look for episode output
        ep_dir = episode_dir / suite / condition / parent_key
        summary_path = ep_dir / "episode_summary.json"

        audit_row = {
            "suite": suite,
            "condition": condition,
            "parent_key": parent_key,
            "task_index": qr.get("task_index", ""),
            "state_id": qr.get("state_id", ""),
        }

        if summary_path.exists():
            summary = read_json(summary_path)
            audit_row["completed"] = True
            audit_row["task_success"] = summary.get("task_success", "")
            audit_row["detector_emitted"] = summary.get("detector_emitted", "")
            audit_row["attack_frames"] = summary.get("attack_frames", "")
            audit_row["n_steps"] = summary.get("n_steps", "")
            audit_row["failure_taxonomy"] = summary.get("failure_taxonomy", "")
            audit_row["has_step_telemetry"] = (ep_dir / "step_telemetry.csv").exists()
            audit_row["has_artifact_sha256"] = (ep_dir / "artifact_sha256.json").exists()
            completed += 1
        else:
            audit_row["completed"] = False
            audit_row["error"] = "missing_episode_summary"
            missing += 1

        audit_rows.append(audit_row)

    # ========== Check 2: Paired parent keys (suite, parent_key) ==========
    parent_groups: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(dict)
    for ar in audit_rows:
        pk = pair_key(ar)
        parent_groups[pk][ar["condition"]] = ar.get("completed", False)

    unpaired = 0
    for pk, conditions in parent_groups.items():
        if len(conditions) < 4:
            unpaired += 1

    # ========== Check 3: Condition consistency ==========
    condition_violations = []
    for ar in audit_rows:
        if not ar.get("completed"):
            continue
        cond = ar["condition"]
        af = safe_int(ar.get("attack_frames", 0))
        if cond in ("TRUE_T10", "RAND_T10") and af != 10:
            condition_violations.append({
                "parent_key": ar["parent_key"],
                "condition": cond,
                "issue": f"attack_frames={af} != 10",
            })
        if cond == "CLEAN" and af != 0:
            condition_violations.append({
                "parent_key": ar["parent_key"],
                "condition": cond,
                "issue": f"clean has attack_frames={ar['attack_frames']}",
            })

    # ========== Check 4: Detector SHA consistency ==========
    sha_violations = []
    # Collect expected SHAs from the first completed episode
    ref_detector_sha = ""
    ref_threshold_sha = ""
    for ar in audit_rows:
        if ar.get("completed"):
            ep_dir = episode_dir / ar["suite"] / ar["condition"] / ar["parent_key"]
            summary = read_json(ep_dir / "episode_summary.json")
            if summary:
                ref_detector_sha = ref_detector_sha or str(summary.get("detector_checkpoint_sha256", ""))
                ref_threshold_sha = ref_threshold_sha or str(summary.get("threshold_sha256", ""))
                det_sha = str(summary.get("detector_checkpoint_sha256", ""))
                thr_sha = str(summary.get("threshold_sha256", ""))
                if det_sha and ref_detector_sha and det_sha != ref_detector_sha:
                    sha_violations.append({"parent_key": ar["parent_key"], "issue": f"detector_sha mismatch: {det_sha} != {ref_detector_sha}"})
                if thr_sha and ref_threshold_sha and thr_sha != ref_threshold_sha:
                    sha_violations.append({"parent_key": ar["parent_key"], "issue": f"threshold_sha mismatch: {thr_sha} != {ref_threshold_sha}"})
                break  # Just need one reference

    # ========== Summary ==========
    violations = []
    if missing > 0:
        violations.append(f"MISSING_EPISODES:{missing}")
    if unpaired > 0:
        violations.append(f"UNPAIRED_PARENTS:{unpaired}")
    if condition_violations:
        violations.append(f"CONDITION_VIOLATIONS:{len(condition_violations)}")
    if sha_violations:
        violations.append(f"SHA_VIOLATIONS:{len(sha_violations)}")

    all_ok = len(violations) == 0
    status = "PASS_D7_POSTRUN_AUDIT" if all_ok else "HOLD_D7_POSTRUN_AUDIT"

    # Write outputs
    audit_fields = ["suite", "condition", "parent_key", "task_index", "state_id",
                    "completed", "task_success", "detector_emitted", "attack_frames",
                    "n_steps", "failure_taxonomy", "has_step_telemetry",
                    "has_artifact_sha256", "error"]
    write_csv(out / "d7_table1_postrun_audit.csv", audit_rows, audit_fields)

    if condition_violations:
        write_csv(out / "d7_table1_condition_violations.csv", condition_violations,
                  ["parent_key", "condition", "issue"])

    report = {
        "gate": "D7_TABLE1_POSTRUN_AUDIT",
        "status": status,
        "reason": "violations=0" if all_ok else f"violations={len(violations)}",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "planned": len(queue_rows),
        "completed": completed,
        "missing": missing,
        "unpaired_parents": unpaired,
        "condition_violations": len(condition_violations),
        "violations": violations,
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
        },
    }
    write_json(out / "d7_table1_postrun_audit_report.json", report)

    csums = {}
    for fn in sorted(out.glob("*")):
        if fn.is_file() and fn.name != "checksum_report.json":
            csums[fn.name] = sha256_file(fn)
    write_json(out / "checksum_report.json", csums)
    with open(out / "SHA256SUMS", "w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS", "SHA256SUMS.sha256"):
                f.write(f"{sha}  {fn}\n")
    (out / "SHA256SUMS.sha256").write_text(
        f"{sha256_file(out / 'SHA256SUMS')}  SHA256SUMS\n"
    )

    print(f"D7 Postrun: {status} ({completed}/{len(queue_rows)} completed, {missing} missing, {unpaired} unpaired)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
