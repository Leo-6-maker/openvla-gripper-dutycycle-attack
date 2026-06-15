#!/usr/bin/env python3
"""D4.3a: Independent post-hoc canary auditor.

Reads RAW episode artifacts AND launcher logs. Recomputes all hard gates
independently. Does NOT trust canary_result.json.

Layout:
  <canary_root>/
    launcher_logs/<safe_tag>/
      command.txt  environment_snapshot.json  stdout.log  stderr.log
      returncode.json  launcher_artifact_hashes.csv
    <safe_tag>/
      ATTEMPT_STARTED.json  [MODEL_LOADED.json]  [FIRST_ACTION_GENERATED.json]
      [step_trace.csv  detector_candidates.csv  ...]

Two-tier artifact contract:
  Pre-action failed attempt:
    launcher logs complete, ATTEMPT_STARTED.json
    NO FIRST_ACTION_GENERATED.json
    returncode != 0

  Successful attempt:
    all phase markers, all episode artifacts, returncode == 0
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

# Files required in EVERY attempt (even pre-action failure)
REQUIRED_LAUNCHER_FILES = [
    "command.txt", "environment_snapshot.json", "stdout.log", "stderr.log",
    "returncode.json", "launcher_artifact_hashes.csv",
]

# Files required in a SUCCESSFUL attempt
REQUIRED_EPISODE_FILES = [
    "ATTEMPT_STARTED.json", "MODEL_LOADED.json", "FIRST_ACTION_GENERATED.json",
    "step_trace.csv", "detector_candidates.csv", "detector_emission.json",
    "action_identity.csv", "latency.csv", "provenance.csv",
    "episode_manifest.json", "artifact_hashes.csv", "teacher_sidecar.json",
]

# Files that must appear in artifact_hashes.csv (excludes the hash manifest itself)
HASHED_EPISODE_FILES = [
    "ATTEMPT_STARTED.json", "MODEL_LOADED.json", "FIRST_ACTION_GENERATED.json",
    "step_trace.csv", "detector_candidates.csv", "detector_emission.json",
    "action_identity.csv", "latency.csv", "provenance.csv",
    "episode_manifest.json", "teacher_sidecar.json",
]

# Launcher files that must appear in launcher_artifact_hashes.csv
HASHED_LAUNCHER_FILES = [
    "command.txt", "environment_snapshot.json", "stdout.log", "stderr.log",
    "returncode.json",
]

SAFE_TAG_RE = re.compile(
    r"^(?P<task>.+)_s(?P<state_id>\d+)_"
    r"(?P<mode>reference|shadow)_attempt(?P<attempt_id>[12])$"
)


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


def parse_safe_tag(tag, valid_keys):
    """Parse safe_tag name. Returns dict or None."""
    m = SAFE_TAG_RE.match(tag)
    if not m:
        return None
    task = m.group("task")
    state_id = int(m.group("state_id"))
    mode = m.group("mode")
    attempt_id = int(m.group("attempt_id"))

    # Must be in frozen canary manifest
    if (task, state_id) not in valid_keys:
        return None

    return {
        "task": task, "state_id": state_id, "mode": mode,
        "attempt_id": attempt_id, "safe_tag": tag,
    }


def is_successful_attempt(ep_dir, ll_dir):
    """Check if an attempt completed successfully."""
    rc_json = load_json(ll_dir / "returncode.json")
    ep_manifest = load_json(ep_dir / "episode_manifest.json")
    if rc_json is None:
        return False
    if rc_json.get("returncode", -1) != 0:
        return False
    if ep_manifest and ep_manifest.get("fatal"):
        return False
    # Must have FIRST_ACTION_GENERATED
    if not (ep_dir / "FIRST_ACTION_GENERATED.json").exists():
        return False
    return True


def has_first_action_generated(ep_dir):
    return (ep_dir / "FIRST_ACTION_GENERATED.json").exists()


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
    valid_keys = {(r["task_key"], int(r["state_id"])) for r in canary_rows}
    assert len(valid_keys) == 4, f"FATAL: expected 4 canary states, got {len(valid_keys)}"

    launcher_dir = out / "launcher_logs"
    gates = []
    all_pass = True

    # ── Discover all safe_tags ──
    all_tags = set()
    if launcher_dir.is_dir():
        for d in launcher_dir.iterdir():
            if d.is_dir():
                all_tags.add(d.name)
    for d in out.iterdir():
        if d.is_dir() and d.name != "launcher_logs" and "attempt" in d.name:
            all_tags.add(d.name)

    # ── Parse and validate all tags ──
    attempts = defaultdict(list)
    for tag in sorted(all_tags):
        info = parse_safe_tag(tag, valid_keys)
        if info is None:
            gates.append((f"PARSE_OR_UNKNOWN_TAG:{tag}", False))
            all_pass = False
            continue
        key = (info["task"], info["state_id"], info["mode"])
        attempts[key].append(info)

    print(f"Parsed {sum(len(v) for v in attempts.values())} attempts "
          f"across {len(attempts)} (task,state,mode) keys")

    # ── Verify all 8 expected keys have at least one attempt ──
    for tk, sid in sorted(valid_keys):
        for mode in ["reference", "shadow"]:
            key = (tk, sid, mode)
            if key not in attempts:
                gates.append((f"MISSING_KEY:{tk}_s{sid}_{mode}", False))
                all_pass = False

    # ── Per-key attempt audit ──
    for key, att_list in sorted(attempts.items()):
        tk, sid, mode = key
        tag_base = f"{tk}_s{sid}_{mode}"
        att_list.sort(key=lambda x: x["attempt_id"])

        # Max 2 attempts
        if len(att_list) > 2:
            gates.append((f"ATTEMPT_COUNT:{tag_base}:{len(att_list)}", False))
            all_pass = False
            continue

        # Check each attempt
        success_count = 0
        for att in att_list:
            ep_dir = out / att["safe_tag"]
            ll_dir = launcher_dir / att["safe_tag"]

            # ── Launcher artifacts (required for ALL attempts) ──
            for an in REQUIRED_LAUNCHER_FILES:
                fp = ll_dir / an
                exists = fp.exists()
                gates.append((f"LAUNCHER_EXISTS:{att['safe_tag']}:{an}", exists))
                if not exists: all_pass = False

            # ── Launcher hash verification ──
            lh_rows = load_csv(ll_dir / "launcher_artifact_hashes.csv")
            if not lh_rows:
                gates.append((f"LAUNCHER_HASH_MANIFEST_MISSING:{att['safe_tag']}", False))
                all_pass = False
            else:
                lh_names = {r["artifact"] for r in lh_rows}
                for an in HASHED_LAUNCHER_FILES:
                    if an not in lh_names:
                        gates.append((f"LAUNCHER_HASH_ROW_MISSING:{att['safe_tag']}:{an}", False))
                        all_pass = False
                for row in lh_rows:
                    an = row["artifact"]
                    ap = ll_dir / an
                    if not ap.exists():
                        gates.append((f"LAUNCHER_HASH_FILE_MISSING:{att['safe_tag']}:{an}", False))
                        all_pass = False
                    else:
                        actual = sha256_file(str(ap))
                        if actual != row["sha256"]:
                            gates.append((f"LAUNCHER_HASH_MISMATCH:{att['safe_tag']}:{an}", False))
                            all_pass = False

            # ── Determine attempt type ──
            is_success = is_successful_attempt(ep_dir, ll_dir)
            has_fa = has_first_action_generated(ep_dir)

            if is_success:
                success_count += 1

                # Successful attempt: ALL episode files required
                for an in REQUIRED_EPISODE_FILES:
                    fp = ep_dir / an
                    exists = fp.exists()
                    gates.append((f"EP_EXISTS:{att['safe_tag']}:{an}", exists))
                    if not exists: all_pass = False

                # Episode hash verification
                eh_rows = load_csv(ep_dir / "artifact_hashes.csv")
                if not eh_rows:
                    gates.append((f"EP_HASH_MANIFEST_MISSING:{att['safe_tag']}", False))
                    all_pass = False
                else:
                    eh_names = {r["artifact"] for r in eh_rows}
                    for an in HASHED_EPISODE_FILES:
                        if an not in eh_names:
                            gates.append((f"EP_HASH_ROW_MISSING:{att['safe_tag']}:{an}", False))
                            all_pass = False
                    for row in eh_rows:
                        an = row["artifact"]
                        ap = ep_dir / an
                        if not ap.exists():
                            gates.append((f"EP_HASH_FILE_MISSING:{att['safe_tag']}:{an}", False))
                            all_pass = False
                        else:
                            actual = sha256_file(str(ap))
                            if actual != row["sha256"]:
                                gates.append((f"EP_HASH_MISMATCH:{att['safe_tag']}:{an}", False))
                                all_pass = False

            else:
                # Failed attempt: must NOT have FIRST_ACTION_GENERATED
                if has_fa:
                    gates.append((f"POST_FA_RETRY:{att['safe_tag']}", False))
                    all_pass = False
                # Must have ATTEMPT_STARTED
                if not (ep_dir / "ATTEMPT_STARTED.json").exists():
                    gates.append((f"NO_STARTED:{att['safe_tag']}", False))
                    all_pass = False

        # ── Retry legality ──
        if len(att_list) == 2:
            a1 = att_list[0]; a2 = att_list[1]
            a1_dir = out / a1["safe_tag"]
            if has_first_action_generated(a1_dir):
                gates.append((f"ILLEGAL_RETRY_AFTER_FA:{tag_base}", False))
                all_pass = False

        # Exactly one successful attempt
        if success_count != 1:
            gates.append((f"SUCCESS_COUNT:{tag_base}:{success_count}", False))
            all_pass = False

    # ── Paired reference/shadow comparison (only on successful attempts) ──
    for tk, sid in sorted(valid_keys):
        tag = f"{tk}_s{sid}"
        ref_key = (tk, sid, "reference")
        sh_key = (tk, sid, "shadow")

        # Find successful attempt for each mode
        ref_dir = None; sh_dir = None
        for att in attempts.get(ref_key, []):
            ep_dir = out / att["safe_tag"]
            ll_dir = launcher_dir / att["safe_tag"]
            if is_successful_attempt(ep_dir, ll_dir):
                ref_dir = ep_dir; break
        for att in attempts.get(sh_key, []):
            ep_dir = out / att["safe_tag"]
            ll_dir = launcher_dir / att["safe_tag"]
            if is_successful_attempt(ep_dir, ll_dir):
                sh_dir = ep_dir; break

        if ref_dir is None or sh_dir is None:
            gates.append((f"SUCCESSFUL_PAIR:{tag}", False))
            all_pass = False
            continue

        ref_m = load_json(ref_dir / "episode_manifest.json")
        sh_m = load_json(sh_dir / "episode_manifest.json")

        # Sequence identity
        for sk in ["raw_action_sequence_sha256", "env_action_sequence_sha256",
                    "obs_sequence_sha256"]:
            rv = ref_m.get(sk, ""); sv = sh_m.get(sk, "")
            g = (rv == sv and rv != "")
            gates.append((f"SEQ_{sk}:{tag}", g))
            if not g: all_pass = False

        # Steps/success/done
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
