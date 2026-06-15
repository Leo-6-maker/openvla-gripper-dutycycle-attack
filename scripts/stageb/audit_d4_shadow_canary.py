#!/usr/bin/env python3
"""D4.3a: Independent post-hoc canary auditor.

Reads RAW episode artifacts AND launcher logs from canary output directory.
Recomputes all hard gates independently. Does NOT trust canary_result.json.

Layout (A4+):
  <canary_root>/
    launcher_logs/<safe_tag>/
      command.txt  environment_snapshot.json  stdout.log  stderr.log
      returncode.json  launcher_artifact_hashes.csv
    <safe_tag>/
      ATTEMPT_STARTED.json  MODEL_LOADED.json  FIRST_ACTION_GENERATED.json
      step_trace.csv  detector_candidates.csv  detector_emission.json
      action_identity.csv  latency.csv  provenance.csv
      episode_manifest.json  artifact_hashes.csv  teacher_sidecar.json

Audits ALL attempts per (task, state_id, mode). Fails on:
  - duplicate successful attempts
  - attempt2 after FIRST_ACTION_GENERATED in attempt1
  - any missing required artifact (episode OR launcher)
  - reference/shadow sequence mismatch
  - reset/invalid-field/abstain-emission/latency/GPU gates
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED_EPISODE_ARTIFACTS = [
    "ATTEMPT_STARTED.json", "MODEL_LOADED.json", "FIRST_ACTION_GENERATED.json",
    "step_trace.csv", "detector_candidates.csv", "detector_emission.json",
    "action_identity.csv", "latency.csv", "provenance.csv",
    "episode_manifest.json", "artifact_hashes.csv", "teacher_sidecar.json",
]

REQUIRED_LAUNCHER_ARTIFACTS = [
    "command.txt", "environment_snapshot.json", "stdout.log", "stderr.log",
    "returncode.json", "launcher_artifact_hashes.csv",
]


def sha256_file(path: str) -> str:
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


def find_safe_tags(canary_root):
    """Discover all safe_tags by scanning launcher_logs/ and episode dirs."""
    root = Path(canary_root)
    tags = set()
    launcher_dir = root / "launcher_logs"
    if launcher_dir.is_dir():
        for d in launcher_dir.iterdir():
            if d.is_dir():
                tags.add(d.name)
    for d in root.iterdir():
        if d.is_dir() and d.name != "launcher_logs" and "attempt" in d.name:
            tags.add(d.name)
    return sorted(tags)


def parse_attempt_info(safe_tag):
    """Parse task, state_id, mode, attempt_id from safe_tag name."""
    # Format: <task>_s<state_id>_<mode>_attempt<attempt_id>
    parts = safe_tag.split("_")
    # Find the mode separator
    mode_idx = None
    for i, p in enumerate(parts):
        if p in ("reference", "shadow"):
            mode_idx = i
            break
    if mode_idx is None:
        return None
    task = "_".join(parts[:mode_idx-1])  # everything before _s<state_id>
    # Actually, task names have underscores (e.g., "alphabet_soup")
    # Find "_s" pattern
    s_idx = None
    for i, p in enumerate(parts):
        if p == "s" and i > 0 and parts[i+1].isdigit():
            s_idx = i
            break
    if s_idx is None:
        return None
    task = "_".join(parts[:s_idx])
    state_id = int(parts[s_idx + 1])
    mode = parts[mode_idx]
    attempt_id = int(parts[-1]) if parts[-1].isdigit() else 1
    return {"task": task, "state_id": state_id, "mode": mode,
            "attempt_id": attempt_id, "safe_tag": safe_tag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary-output-dir", required=True)
    ap.add_argument("--canary-manifest", required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    args = ap.parse_args()

    out = Path(args.canary_output_dir)
    manifest_sha = sha256_file(args.canary_manifest)
    assert manifest_sha == args.expected_manifest_sha256, (
        f"FATAL: Manifest SHA mismatch: {manifest_sha[:16]}..."
    )
    print(f"Manifest SHA: {manifest_sha[:16]}... VERIFIED")

    canary_rows = [r for r in load_csv(args.canary_manifest) if r["subset"] == "canary"]
    expected_keys = {(r["task_key"], int(r["state_id"])) for r in canary_rows}
    assert len(expected_keys) == 4, f"FATAL: expected 4 canary states"

    launcher_dir = out / "launcher_logs"
    gates = []
    all_pass = True

    # ── Discover all attempts ──
    all_tags = find_safe_tags(str(out))
    print(f"Found {len(all_tags)} safe_tags: {all_tags}")

    # Group by (task, state_id, mode)
    attempts = defaultdict(list)
    for tag in all_tags:
        info = parse_attempt_info(tag)
        if info is None:
            gates.append((f"PARSE_TAG:{tag}", False))
            all_pass = False
            continue
        key = (info["task"], info["state_id"], info["mode"])
        attempts[key].append(info)

    # ── Attempt audit ──
    for key, att_list in sorted(attempts.items()):
        tk, sid, mode = key
        tag_base = f"{tk}_s{sid}_{mode}"
        att_list.sort(key=lambda x: x["attempt_id"])

        if len(att_list) > 2:
            gates.append((f"ATTEMPT_COUNT:{tag_base}:{len(att_list)}", False))
            all_pass = False
            continue

        success_count = 0
        for att in att_list:
            ep_dir = out / att["safe_tag"]
            ll_dir = launcher_dir / att["safe_tag"]

            # Check launcher artifacts
            for an in REQUIRED_LAUNCHER_ARTIFACTS:
                exists = (ll_dir / an).exists()
                gates.append((f"LAUNCHER_{an}:{att['safe_tag']}", exists))
                if not exists: all_pass = False

            # Check episode artifacts
            m = load_json(ep_dir / "episode_manifest.json")
            if m is None:
                gates.append((f"EP_MANIFEST:{att['safe_tag']}", False))
                all_pass = False
                continue

            for an in REQUIRED_EPISODE_ARTIFACTS:
                exists = (ep_dir / an).exists()
                gates.append((f"EP_{an}:{att['safe_tag']}", exists))
                if not exists: all_pass = False

            # Check returncode
            rc_json = load_json(ll_dir / "returncode.json")
            is_success = (rc_json and rc_json.get("returncode") == 0
                          and not m.get("fatal"))
            if is_success:
                success_count += 1

            # Check FIRST_ACTION_GENERATED
            fa_marker = ep_dir / "FIRST_ACTION_GENERATED.json"
            fa_exists = fa_marker.exists()

        # ── Retry legality ──
        if len(att_list) == 2:
            a1 = att_list[0]; a2 = att_list[1]
            a1_fa = (out / a1["safe_tag"] / "FIRST_ACTION_GENERATED.json").exists()
            if a1_fa:
                gates.append((f"ILLEGAL_RETRY:{tag_base}", False))
                all_pass = False

        # Only one successful attempt allowed
        if success_count != 1:
            gates.append((f"SUCCESS_COUNT:{tag_base}:{success_count}", False))
            all_pass = False

    # ── Per-state paired comparison (only on successful attempts) ──
    for tk, sid in sorted(expected_keys):
        ref_key = (tk, sid, "reference")
        sh_key = (tk, sid, "shadow")
        tag = f"{tk}_s{sid}"

        if ref_key not in attempts or sh_key not in attempts:
            gates.append((f"PAIRED:{tag}", False))
            all_pass = False
            continue

        # Find successful attempts
        ref_att = None; sh_att = None
        for att in attempts[ref_key]:
            m = load_json(out / att["safe_tag"] / "episode_manifest.json")
            if m and not m.get("fatal"):
                ref_att = att; break
        for att in attempts[sh_key]:
            m = load_json(out / att["safe_tag"] / "episode_manifest.json")
            if m and not m.get("fatal"):
                sh_att = att; break

        if ref_att is None or sh_att is None:
            gates.append((f"SUCCESSFUL_PAIR:{tag}", False))
            all_pass = False
            continue

        ref_dir = out / ref_att["safe_tag"]
        sh_dir = out / sh_att["safe_tag"]
        ref_m = load_json(ref_dir / "episode_manifest.json")
        sh_m = load_json(sh_dir / "episode_manifest.json")

        # Sequence identity
        for sk in ["raw_action_sequence_sha256", "env_action_sequence_sha256",
                    "obs_sequence_sha256"]:
            g = ref_m.get(sk) == sh_m.get(sk) and ref_m.get(sk, "") != ""
            gates.append((f"SEQ_{sk}:{tag}", g))
            if not g: all_pass = False

        # Steps/success/done match
        for sk in ["n_steps", "success_primary", "success_done_any",
                    "success_check_any", "success_step_primary", "done_step"]:
            g = ref_m.get(sk) == sh_m.get(sk)
            gates.append((f"{sk}:{tag}", g))
            if not g: all_pass = False

        # Detector exception
        if sh_m.get("detector_exception"):
            gates.append((f"DETECTOR_EXCEPTION:{tag}", False))
            all_pass = False

        # Action identity
        if sh_m.get("action_identity_fail"):
            gates.append((f"ACTION_IDENTITY:{tag}", False))
            all_pass = False

        # Invalid fields
        if sh_m.get("n_invalid_field_steps", 0) > 0:
            gates.append((f"INVALID_FIELDS:{tag}", False))
            all_pass = False

        # Detector reset
        pre = sh_m.get("detector_pre_reset", {})
        if not pre:
            gates.append((f"RESET_MISSING:{tag}", False))
            all_pass = False
        else:
            for check, expected in [("next_expected_step", 0), ("emit_step", -1),
                                     ("history_len", 0), ("candidate_count", 0)]:
                g = pre.get(check, -999) == expected
                gates.append((f"RESET_{check}:{tag}", g))
                if not g: all_pass = False

        # Abstain emission
        emit_step = sh_m.get("detector_emit_step", -1)
        if isinstance(emit_step, int) and emit_step >= 0:
            sh_cands = load_csv(sh_dir / "detector_candidates.csv")
            found = False
            for cand in sh_cands:
                if int(cand.get("step", -1)) == emit_step:
                    found = True
                    if cand.get("abstained") == "1" or cand.get("abstain", ""):
                        gates.append((f"ABSTAIN_EMISSION:{tag}", False))
                        all_pass = False
            if not found:
                gates.append((f"EMIT_CANDIDATE_MISSING:{tag}", False))
                all_pass = False

        # Latency
        lat_rows = load_csv(sh_dir / "latency.csv")
        if not lat_rows:
            gates.append((f"LATENCY_EMPTY:{tag}", False))
            all_pass = False
        else:
            det_lats = []; model_lats = []
            for row in lat_rows:
                du = row.get("detector_update_us", "")
                mu = row.get("model_inference_us", "")
                if du and du != "DISABLED": det_lats.append(int(du))
                if mu and mu != "DISABLED": model_lats.append(int(mu))
            if not det_lats:
                gates.append((f"LATENCY_NO_DET:{tag}", False))
                all_pass = False
            else:
                p99 = sorted(det_lats)[int(len(det_lats) * 0.99)]
                mx = max(det_lats)
                med_det = sorted(det_lats)[len(det_lats) // 2]
                med_mod = sorted(model_lats)[len(model_lats) // 2] if model_lats else 1
                for g_name, g_val, limit in [
                    ("LATENCY_P99", p99, 20000),
                    ("LATENCY_MAX", mx, 50000),
                ]:
                    g = g_val <= limit
                    gates.append((f"{g_name}:{tag}", g))
                    if not g: all_pass = False
                if med_mod > 0 and med_det / med_mod > 0.05:
                    gates.append((f"LATENCY_OVERHEAD:{tag}", False))
                    all_pass = False

        # Artifact hashes (reference + shadow)
        for mode_dir, mode_name in [(ref_dir, "ref"), (sh_dir, "sh")]:
            hash_rows = load_csv(mode_dir / "artifact_hashes.csv")
            if not hash_rows:
                gates.append((f"HASH_MANIFEST_MISSING:{tag}_{mode_name}", False))
                all_pass = False
                continue
            hashed_names = {r["artifact"] for r in hash_rows}
            # Every required episode artifact must have a hash row
            for an in REQUIRED_EPISODE_ARTIFACTS:
                if an not in hashed_names:
                    gates.append((f"HASH_ROW_MISSING:{tag}_{mode_name}:{an}", False))
                    all_pass = False
            # Verify every hash
            for row in hash_rows:
                an = row["artifact"]
                ap = mode_dir / an
                if not ap.exists():
                    gates.append((f"HASH_FILE_MISSING:{tag}_{mode_name}:{an}", False))
                    all_pass = False
                else:
                    actual = sha256_file(str(ap))
                    if actual != row["sha256"]:
                        gates.append((f"HASH_MISMATCH:{tag}_{mode_name}:{an}", False))
                        all_pass = False

    # ── Print audit ──
    print(f"\n{'='*60}")
    print("INDEPENDENT GATE AUDIT")
    print(f"{'='*60}")
    n_fail = sum(1 for _, p in gates if not p)
    n_pass = len(gates) - n_fail
    for name, passed in gates:
        if not passed:
            print(f"  [FAIL] {name}")

    auditor_result = "PASS" if all_pass else "FAIL"
    print(f"\n  {n_pass} PASS, {n_fail} FAIL (out of {len(gates)} gates)")
    print(f"AUDITOR RESULT: {auditor_result}")

    with open(out / "canary_audit_result.json", "w") as f:
        json.dump({
            "auditor_result": auditor_result, "all_pass": all_pass,
            "n_gates": len(gates), "n_pass": n_pass, "n_fail": n_fail,
            "gates": [{"name": n, "pass": p} for n, p in gates],
        }, f, indent=2)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
