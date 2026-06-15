#!/usr/bin/env python3
"""D4.3b: Independent panel auditor — reads raw episode artifacts.

Audits all 30 frozen panel states from raw CSV/manifest files.
Does NOT trust orchestrator summary (panel_result.json, slot_result.json).

Outputs honest classification:
  AUDITOR_PIPELINE: PASS/FAIL (did the audit run correctly)
  SCIENTIFIC_PANEL_GATE: PASS/FAIL (did all reference-shadow pairs match)
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED_EPISODE_FILES = [
    "ATTEMPT_STARTED.json", "MODEL_LOADED.json", "FIRST_ACTION_GENERATED.json",
    "step_trace.csv", "detector_candidates.csv", "detector_emission.json",
    "action_identity.csv", "latency.csv", "provenance.csv",
    "episode_manifest.json", "artifact_hashes.csv", "teacher_sidecar.json",
]

SAFE_TAG_RE = re.compile(
    r"^(?P<task>.+)_s(?P<state_id>\d+)_"
    r"(?P<mode>reference|shadow)_attempt(?P<attempt_id>[12])$"
)

FROZEN_10_TASKS = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce",
    "ketchup", "tomato_sauce", "butter", "milk",
    "chocolate_pudding", "orange_juice",
]


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def load_json(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


def load_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))


def parse_tag(tag, valid_keys):
    m = SAFE_TAG_RE.match(tag)
    if not m: return None
    task = m.group("task")
    sid = int(m.group("state_id"))
    if (task, sid) not in valid_keys: return None
    return {"task": task, "state_id": sid, "mode": m.group("mode"),
            "attempt_id": int(m.group("attempt_id"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-output-dir", required=True)
    ap.add_argument("--panel-manifest", required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    args = ap.parse_args()

    root = Path(args.panel_output_dir)
    msha = sha256_file(args.panel_manifest)
    assert msha == args.expected_manifest_sha256, f"Manifest SHA mismatch: {msha[:16]}..."
    print(f"Manifest SHA: {msha[:16]}... VERIFIED")

    # Load frozen panel states (30 states)
    manifest_rows = list(csv.DictReader(open(args.panel_manifest)))
    panel_states = [r for r in manifest_rows if r["subset"] == "panel"]
    assert len(panel_states) == 30, f"Expected 30 panel states, got {len(panel_states)}"
    valid_keys = {(r["task_key"], int(r["state_id"])) for r in panel_states}
    print(f"Frozen panel: {len(panel_states)} states")

    gates = []
    pipeline_ok = True
    scientific_ok = True

    # ── Discover episodes ──
    episodes = defaultdict(list)
    for d in root.iterdir():
        if not d.is_dir() or "attempt" not in d.name:
            continue
        info = parse_tag(d.name, valid_keys)
        if info is None:
            gates.append(("PARSE_FAIL", d.name, False))
            pipeline_ok = False; scientific_ok = False
            continue
        key = (info["task"], info["state_id"], info["mode"])
        episodes[key].append((d, info))

    print(f"Discovered {sum(len(v) for v in episodes.values())} episode dirs across {len(episodes)} keys")

    # ── Check all 60 expected keys exist ──
    for tk, sid in sorted(valid_keys):
        for mode in ["reference", "shadow"]:
            key = (tk, sid, mode)
            if key not in episodes:
                gates.append(("MISSING_KEY", f"{tk}_s{sid}_{mode}", False))
                pipeline_ok = False; scientific_ok = False

    # ── Per-key audit ──
    cohort_a_passed = 0  # vetted GPU: ref==sh on all gates
    cohort_b_divergent = []  # GPU 4,5: mismatch
    cohort_c_matched = []  # GPU 4,5: matched

    for key, ep_list in sorted(episodes.items()):
        tk, sid, mode = key
        tag = f"{tk}_s{sid}"
        ep_list.sort(key=lambda x: x[1]["attempt_id"])

        # Attempt legality
        ids = [e[1]["attempt_id"] for e in ep_list]
        if ids not in ([1], [1, 2]):
            gates.append(("ILLEGAL_ATTEMPTS", tag, False))
            pipeline_ok = False; scientific_ok = False
            continue
        if len(ep_list) == 2:
            a1_dir = ep_list[0][0]
            if (a1_dir / "FIRST_ACTION_GENERATED.json").exists():
                gates.append(("ILLEGAL_RETRY", tag, False))
                pipeline_ok = False; scientific_ok = False

        # Find successful attempt
        success_dir = None
        for d, info in ep_list:
            m = load_json(d / "episode_manifest.json")
            if m and not m.get("fatal") and m.get("infra_status") == "ok":
                success_dir = d
                break
        if success_dir is None:
            gates.append(("NO_SUCCESS", tag, False))
            pipeline_ok = False; scientific_ok = False
            continue

        # Check required files
        for fn in REQUIRED_EPISODE_FILES:
            if not (success_dir / fn).exists():
                gates.append(("MISSING_FILE", f"{tag}/{fn}", False))
                pipeline_ok = False

    # ── Paired reference/shadow comparison ──
    task_results = {}
    for tk, sid in sorted(valid_keys):
        tag = f"{tk}_s{sid}"
        ref_key = (tk, sid, "reference")
        sh_key = (tk, sid, "shadow")

        ref_dir = None; sh_dir = None
        for d, info in episodes.get(ref_key, []):
            m = load_json(d / "episode_manifest.json")
            if m and not m.get("fatal") and m.get("infra_status") == "ok":
                ref_dir = d; break
        for d, info in episodes.get(sh_key, []):
            m = load_json(d / "episode_manifest.json")
            if m and not m.get("fatal") and m.get("infra_status") == "ok":
                sh_dir = d; break

        if ref_dir is None or sh_dir is None:
            gates.append(("PAIR_MISSING", tag, False))
            pipeline_ok = False; scientific_ok = False
            continue

        ref_m = load_json(ref_dir / "episode_manifest.json")
        sh_m = load_json(sh_dir / "episode_manifest.json")

        # Sequence identity
        for sk in ["raw_action_sequence_sha256", "env_action_sequence_sha256",
                    "obs_sequence_sha256"]:
            rv = ref_m.get(sk, ""); sv = sh_m.get(sk, "")
            g = (rv == sv and rv != "")
            gates.append((f"SEQ_{sk}", tag, g))
            if not g: scientific_ok = False

        # Steps and success
        for sk in ["n_steps", "success_primary", "success_done_any",
                    "success_check_any", "success_step_primary", "done_step"]:
            g = ref_m.get(sk) == sh_m.get(sk)
            gates.append((sk, tag, g))
            if not g: scientific_ok = False

        # Action identity
        id_rows = load_csv(sh_dir / "action_identity.csv")
        id_fail = any(str(r.get("action_identical", "1")) == "0" for r in id_rows)
        gates.append(("ACTION_IDENTITY", tag, not id_fail))
        if id_fail: scientific_ok = False

        # Invalid fields
        st_rows = load_csv(sh_dir / "step_trace.csv")
        invalid_steps = 0
        flags = ["raw_valid", "env_valid", "qpos_valid", "eef_valid",
                 "convention_ok", "semantics_ok"]
        for row in st_rows:
            if any(str(row.get(f, "1")) == "0" for f in flags):
                invalid_steps += 1
        gates.append(("INVALID_FIELDS", tag, invalid_steps == 0))
        if invalid_steps > 0: scientific_ok = False

        # Detector exception
        gates.append(("DET_EXCEPTION", tag, not sh_m.get("detector_exception")))
        if sh_m.get("detector_exception"): scientific_ok = False

        # Reset state
        pre = sh_m.get("detector_pre_reset", {})
        for check, exp in [("next_expected_step", 0), ("emit_step", -1),
                            ("history_len", 0), ("candidate_count", 0)]:
            g = pre.get(check, -999) == exp
            gates.append((f"RESET_{check}", tag, g))
            if not g: scientific_ok = False

        # Emit candidate uniqueness
        emit_step = sh_m.get("detector_emit_step", -1)
        if isinstance(emit_step, int) and emit_step >= 0:
            cands = load_csv(sh_dir / "detector_candidates.csv")
            emit_cands = [c for c in cands if int(c.get("step", -1)) == emit_step]
            if len(emit_cands) != 1:
                gates.append(("EMIT_UNIQUE", tag, False))
                scientific_ok = False
            elif emit_cands[0].get("abstained") == "1" or emit_cands[0].get("abstain", ""):
                gates.append(("ABSTAIN_EMISSION", tag, False))
                scientific_ok = False

        # Row count
        n_steps = sh_m.get("n_steps", 0)
        for csv_name in ["step_trace.csv", "action_identity.csv", "latency.csv"]:
            actual = len(load_csv(sh_dir / csv_name))
            g = actual == n_steps
            gates.append((f"ROW_COUNT_{csv_name}", tag, g))
            if not g: pipeline_ok = False

        # Artifact hashes
        hash_rows = load_csv(sh_dir / "artifact_hashes.csv")
        for row in hash_rows:
            an = row["artifact"]; ap = sh_dir / an
            if ap.exists():
                actual = sha256_file(str(ap))
                if actual != row["sha256"]:
                    gates.append(("HASH_MISMATCH", f"{tag}/{an}", False))
                    pipeline_ok = False

        # Classify
        ref_sh_match = (ref_m.get("n_steps") == sh_m.get("n_steps") and
                        ref_m.get("success_primary") == sh_m.get("success_primary"))
        if ref_sh_match and not id_fail and invalid_steps == 0:
            cohort_a_passed += 1
        elif not ref_sh_match:
            cohort_b_divergent.append(tag)
        else:
            cohort_c_matched.append(tag)

        task_results[tag] = {
            "ref_steps": ref_m.get("n_steps"), "sh_steps": sh_m.get("n_steps"),
            "ref_success": ref_m.get("success_primary"),
            "sh_success": sh_m.get("success_primary"),
            "emit": sh_m.get("detector_emit_step"),
            "match": ref_sh_match,
        }

    # ── GPU snapshots ──
    for sn in ["gpu_processes_before.csv", "gpu_processes_after.csv"]:
        if not (root / sn).exists():
            gates.append(("GPU_SNAPSHOT_MISSING", sn, False))
            pipeline_ok = False

    # ── Summary ──
    print(f"\n{'='*60}")
    print("INDEPENDENT PANEL AUDIT")
    print(f"{'='*60}")
    n_fail = sum(1 for _, _, p in gates if not p)
    print(f"Gates: {len(gates)-n_fail} PASS, {n_fail} FAIL")

    print(f"\n=== Classification ===")
    print(f"Cohort A (vetted-GPU matched): {cohort_a_passed}/30")
    print(f"Cohort B (GPU 4,5 divergent): {len(cohort_b_divergent)} — {cohort_b_divergent}")
    print(f"Cohort C (GPU 4,5 matched): {len(cohort_c_matched)} — {cohort_c_matched}")

    auditor_pipeline = "PASS" if pipeline_ok else "FAIL"
    scientific_gate = "PASS" if scientific_ok else "FAIL"
    print(f"\nAUDITOR_PIPELINE: {auditor_pipeline}")
    print(f"SCIENTIFIC_PANEL_GATE: {scientific_gate}")

    # Write output
    with open(root / "panel_audit_result.json", "w") as f:
        json.dump({
            "auditor_pipeline": auditor_pipeline,
            "scientific_panel_gate": scientific_gate,
            "n_panel_states": 30,
            "cohort_a_matched": cohort_a_passed,
            "cohort_b_divergent": cohort_b_divergent,
            "cohort_c_gpu45_matched": cohort_c_matched,
            "gates": [{"name": n, "tag": t, "pass": p} for n, t, p in gates],
            "task_results": task_results,
        }, f, indent=2)

    print(f"\nOutput: {root / 'panel_audit_result.json'}")
    return 0 if pipeline_ok else 1


if __name__ == "__main__":
    sys.exit(main())
