#!/usr/bin/env python3
"""D4.3a: Independent post-hoc canary auditor.

Reads RAW episode artifacts from canary output directory and recomputes
all hard gates independently. Does NOT trust canary_result.json.

Read-only. No GPU.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path


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


def find_episode_dirs(canary_root):
    """Find all episode directories in the canary output root."""
    root = Path(canary_root)
    dirs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and "attempt" in d.name:
            dirs.append(d)
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary-output-dir", required=True)
    ap.add_argument("--canary-manifest", required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    args = ap.parse_args()

    out = Path(args.canary_output_dir)
    result_path = out / "canary_result.json"
    assert result_path.exists(), f"FATAL: {result_path} not found"

    # Verify manifest SHA
    manifest_sha = sha256_file(args.canary_manifest)
    assert manifest_sha == args.expected_manifest_sha256, (
        f"FATAL: Manifest SHA mismatch: {manifest_sha[:16]}..."
    )
    print(f"Manifest SHA: {manifest_sha[:16]}... VERIFIED")

    # Load manifest for expected states
    canary_rows = [r for r in load_csv(args.canary_manifest) if r["subset"] == "canary"]
    expected_states = {(r["task_key"], int(r["state_id"])) for r in canary_rows}
    assert len(expected_states) == 4, f"FATAL: expected 4 canary states, got {len(expected_states)}"

    # Find episode directories
    ep_dirs = find_episode_dirs(out)
    print(f"Found {len(ep_dirs)} episode directories")

    # Organize by (task, state_id, mode)
    episodes = {}
    for d in ep_dirs:
        m = load_json(d / "episode_manifest.json")
        if m is None:
            print(f"  SKIP {d.name}: no manifest")
            continue
        key = (m["task"], m["state_id"], m["mode"])
        episodes[key] = d

    gates = []
    all_pass = True

    # ── Gate: All 8 episodes present (4 states x 2 modes) ──
    for tk, sid in sorted(expected_states):
        for mode in ["reference", "shadow"]:
            key = (tk, sid, mode)
            g = key in episodes
            gates.append((f"EPISODE_EXISTS:{tk}_s{sid}_{mode}", g))
            if not g: all_pass = False

    # ── Per-state gates ──
    for tk, sid in sorted(expected_states):
        ref_dir = episodes.get((tk, sid, "reference"))
        sh_dir = episodes.get((tk, sid, "shadow"))

        if ref_dir is None or sh_dir is None:
            gates.append((f"PAIRED:{tk}_s{sid}", False))
            all_pass = False
            continue

        ref_m = load_json(ref_dir / "episode_manifest.json")
        sh_m = load_json(sh_dir / "episode_manifest.json")
        if ref_m is None or sh_m is None:
            gates.append((f"MANIFEST:{tk}_s{sid}", False))
            all_pass = False
            continue

        tag = f"{tk}_s{sid}"

        # Phase markers
        for marker in ["ATTEMPT_STARTED", "MODEL_LOADED", "FIRST_ACTION_GENERATED"]:
            ref_has = (ref_dir / f"{marker}.json").exists()
            sh_has = (sh_dir / f"{marker}.json").exists()
            gates.append((f"MARKER_{marker}:{tag}_ref", ref_has))
            gates.append((f"MARKER_{marker}:{tag}_sh", sh_has))
            if not ref_has or not sh_has: all_pass = False

        # Subprocess logs exist
        for log_file in ["command.txt", "stdout.log", "stderr.log", "returncode.json",
                          "environment_snapshot.json"]:
            for mode_dir, mode_name in [(ref_dir, "ref"), (sh_dir, "sh")]:
                exists = (mode_dir / log_file).exists()
                gates.append((f"LOG_{log_file}:{tag}_{mode_name}", exists))
                if not exists: all_pass = False

        # Detector exception
        if sh_m.get("detector_exception"):
            gates.append((f"DETECTOR_EXCEPTION:{tag}", False))
            all_pass = False

        # Action identity (from manifest AND from CSV)
        if sh_m.get("action_identity_fail"):
            gates.append((f"ACTION_IDENTITY:{tag}", False))
            all_pass = False
        id_rows = load_csv(sh_dir / "action_identity.csv")
        for row in id_rows:
            if row.get("action_identical") == "0":
                gates.append((f"ACTION_IDENTITY_CSV:{tag}_step{row['step']}", False))
                all_pass = False

        # Sequence identity
        for sk in ["raw_action_sequence_sha256", "env_action_sequence_sha256",
                    "obs_sequence_sha256"]:
            rv = ref_m.get(sk, ""); sv = sh_m.get(sk, "")
            ok = (rv == sv and rv != "")
            gates.append((f"SEQ_{sk}:{tag}", ok))
            if not ok: all_pass = False

        # Length
        rn = ref_m.get("n_steps", -1); sn = sh_m.get("n_steps", -1)
        g = rn == sn
        gates.append((f"STEPS:{tag}", g))
        if not g: all_pass = False

        # Success/done
        for sk in ["success_primary", "success_done_any", "success_check_any",
                    "success_step_primary", "done_step"]:
            rv = ref_m.get(sk); sv = sh_m.get(sk)
            g = rv == sv
            gates.append((f"{sk}:{tag}", g))
            if not g: all_pass = False

        # Invalid fields (from step_trace CSV)
        if sh_m.get("n_invalid_field_steps", 0) > 0:
            gates.append((f"INVALID_FIELDS:{tag}", False))
            all_pass = False

        # Detector reset (from manifest)
        pre = sh_m.get("detector_pre_reset", {})
        if pre:
            g1 = pre.get("next_expected_step", -1) == 0
            g2 = pre.get("emit_step", -2) == -1
            gates.append((f"RESET_STEP_ZERO:{tag}", g1))
            gates.append((f"RESET_EMIT_NEGATIVE:{tag}", g2))
            if not g1 or not g2: all_pass = False

        # Abstain emission (from detector_candidates.csv)
        sh_cands = load_csv(sh_dir / "detector_candidates.csv")
        emit_step = sh_m.get("detector_emit_step", -1)
        if isinstance(emit_step, int) and emit_step >= 0 and sh_cands:
            for cand in sh_cands:
                if int(cand.get("step", -1)) == emit_step:
                    if cand.get("abstained") == "1" or cand.get("abstain", ""):
                        gates.append((
                            f"ABSTAIN_EMISSION:{tag}", False,
                        ))
                        all_pass = False

        # Latency
        lat_rows = load_csv(sh_dir / "latency.csv")
        if lat_rows:
            det_lats = []
            model_lats = []
            for row in lat_rows:
                du = row.get("detector_update_us", "")
                mu = row.get("model_inference_us", "")
                if du and du != "DISABLED": det_lats.append(int(du))
                if mu and mu != "DISABLED": model_lats.append(int(mu))
            if det_lats:
                p99 = sorted(det_lats)[int(len(det_lats) * 0.99)]
                mx = max(det_lats)
                med_det = sorted(det_lats)[len(det_lats) // 2]
                med_mod = sorted(model_lats)[len(model_lats) // 2] if model_lats else 1
                g_p99 = p99 <= 20000
                g_max = mx <= 50000
                g_overhead = (med_det / med_mod * 100) <= 5.0 if med_mod > 0 else True
                gates.append((f"LATENCY_P99:{tag}", g_p99))
                gates.append((f"LATENCY_MAX:{tag}", g_max))
                gates.append((f"LATENCY_OVERHEAD:{tag}", g_overhead))
                if not g_p99 or not g_max or not g_overhead: all_pass = False

        # Artifact hashes (verify each file's actual hash matches)
        hash_rows = load_csv(sh_dir / "artifact_hashes.csv")
        for row in hash_rows:
            an = row["artifact"]
            ap = sh_dir / an
            if ap.exists():
                actual = sha256_file(str(ap))
                expected = row["sha256"]
                g = (actual == expected)
                if not g:
                    gates.append((f"HASH_MISMATCH:{tag}:{an}", False))
                    all_pass = False

    # ── Print audit ──
    print(f"\n{'='*60}")
    print("INDEPENDENT GATE AUDIT")
    print(f"{'='*60}")
    n_fail = 0
    for name, passed in gates:
        if not passed:
            n_fail += 1
            print(f"  [FAIL] {name}")
    n_pass = len(gates) - n_fail
    print(f"\n  {n_pass} PASS, {n_fail} FAIL (out of {len(gates)} gates)")

    auditor_result = "PASS" if all_pass else "FAIL"
    print(f"\nAUDITOR RESULT: {auditor_result}")

    with open(out / "canary_audit_result.json", "w") as f:
        json.dump({
            "auditor_result": auditor_result,
            "all_pass": all_pass,
            "n_gates": len(gates),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "manifest_sha": manifest_sha,
            "gates": [{"name": n, "pass": p} for n, p in gates],
        }, f, indent=2)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
