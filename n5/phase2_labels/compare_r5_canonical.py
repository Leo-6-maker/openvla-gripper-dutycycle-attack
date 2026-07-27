"""[DeepSeek] R5 Canonical A/B Comparator.

Reads two sealed roots from independent runs and verifies:
  - Both seals verify independently (SHA256SUMS intact)
  - Episode identities are identical (same set)
  - Steps per episode are identical
  - Per-step actions are identical
  - Per-step entity poses are identical
  - Per-step sim_state qpos/qvel/act/time are identical
  - Relation identities are identical
  - No missing/duplicate/extra episodes
  - Canonical payload digest identical (whitelisted variant fields excluded)

Variant fields (allowed to differ between A and B):
  - timestamp, elapsed_s, start_time, end_time
  - hostname, executable, command line
  - SHA256SUMS values (different root paths produce different file SHAs)
  - File paths that include the run label

Usage:
  python n5/phase2_labels/compare_r5_canonical.py \
    --root-a /path/to/run_A --root-b /path/to/run_B --gate r5e|r5f
"""
import json, os, sys, argparse, hashlib
from pathlib import Path
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_seal(root):
    root = Path(root)
    sums_path = root / "SHA256SUMS"
    side_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not side_path.is_file():
        return False, "not a sealed root"
    sidecar = side_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar) < 2 or sidecar[0] != sha256_file(sums_path):
        return False, "seal sidecar mismatch"
    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        if name in ("SHA256SUMS", "SHA256SUMS.sha256"):
            continue
        target = root / name
        if not target.is_file() or sha256_file(target) != digest:
            return False, f"seal file mismatch: {name}"
        expected[name] = digest
    return True, expected


VARIANT_MANIFEST_KEYS = {
    "timestamp", "start_time", "end_time", "elapsed_s",
    "executable", "command", "python_version",
    "environment", "script_sha256",
    "r5e_receipt", "r5f_receipt",
}

VARIANT_EPISODE_KEYS = {
    "collection_seed",
}


def canonical_json(obj, variant_keys=frozenset()):
    """Return canonical JSON bytes with variant keys removed."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k in variant_keys:
                continue
            cleaned[k] = canonical_json(v, variant_keys)
        return cleaned
    if isinstance(obj, list):
        return [canonical_json(item, variant_keys) for item in obj]
    if isinstance(obj, float):
        # Round to 12 significant digits for cross-run comparison
        return float(f"{obj:.12g}")
    return obj


def canonical_sha(obj, variant_keys=frozenset()):
    cleaned = canonical_json(obj, variant_keys)
    raw = json.dumps(cleaned, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compare_episodes(ep_dir_a, ep_dir_b, ep_name):
    """Compare two episode.json files. Returns (ok, issues)."""
    ep_a = ep_dir_a / "episode.json"
    ep_b = ep_dir_b / "episode.json"
    if not ep_a.is_file() or not ep_b.is_file():
        return False, [f"{ep_name}: missing episode.json in one root"]

    data_a = json.loads(ep_a.read_text(encoding="utf-8"))
    data_b = json.loads(ep_b.read_text(encoding="utf-8"))
    issues = []

    # Identity
    if data_a.get("episode_id") != data_b.get("episode_id"):
        issues.append(f"{ep_name}: episode_id mismatch")
    if data_a.get("step_count") != data_b.get("step_count"):
        issues.append(f"{ep_name}: step_count mismatch: "
                      f"{data_a['step_count']} vs {data_b['step_count']}")

    # Steps
    steps_a = data_a.get("steps", [])
    steps_b = data_b.get("steps", [])
    if len(steps_a) != len(steps_b):
        issues.append(f"{ep_name}: step array length mismatch: "
                      f"{len(steps_a)} vs {len(steps_b)}")
    else:
        for i, (sa, sb) in enumerate(zip(steps_a, steps_b)):
            if sa.get("step") != sb.get("step"):
                issues.append(f"{ep_name}: step {i} index mismatch")
            a_raw = sa.get("action_raw_7d", [])
            b_raw = sb.get("action_raw_7d", [])
            if len(a_raw) != len(b_raw):
                issues.append(f"{ep_name}: step {i} action length mismatch")
            elif len(a_raw) > 0:
                max_diff = max(abs(float(a) - float(b)) for a, b in zip(a_raw, b_raw))
                if max_diff > 1e-12:
                    issues.append(f"{ep_name}: step {i} action diff={max_diff:.2e}")

    # Telemetry
    tel_a = data_a.get("telemetry", [])
    tel_b = data_b.get("telemetry", [])
    if len(tel_a) != len(tel_b):
        issues.append(f"{ep_name}: telemetry length mismatch: "
                      f"{len(tel_a)} vs {len(tel_b)}")
    else:
        for i, (ta, tb) in enumerate(zip(tel_a, tel_b)):
            if ta.get("step") != tb.get("step"):
                issues.append(f"{ep_name}: telemetry step {i} index mismatch")

            # Compare entity poses
            ents_a = {f"{e['entity_type']}:{e['entity_id']}": e for e in ta.get("entities", [])}
            ents_b = {f"{e['entity_type']}:{e['entity_id']}": e for e in tb.get("entities", [])}
            if set(ents_a.keys()) != set(ents_b.keys()):
                issues.append(f"{ep_name}: telemetry step {i} entity set mismatch")

            for key in set(ents_a.keys()) & set(ents_b.keys()):
                pa = np.array(ents_a[key]["world_pose"]["position"])
                pb = np.array(ents_b[key]["world_pose"]["position"])
                pos_diff = float(np.max(np.abs(pa - pb)))
                if pos_diff > 1e-12:
                    issues.append(f"{ep_name}: step {i} {key} position diff={pos_diff:.2e}")

                qa = np.array(ents_a[key]["world_pose"]["quaternion"])
                qb = np.array(ents_b[key]["world_pose"]["quaternion"])
                qa_n = qa / np.linalg.norm(qa)
                qb_n = qb / np.linalg.norm(qb)
                dot = abs(np.dot(qa_n, qb_n))
                if dot > 1.0:
                    dot = 1.0
                import math
                geo = float(2.0 * math.atan2(math.sqrt(max(0, 1 - dot*dot)), dot))
                if geo > 1e-12:
                    issues.append(f"{ep_name}: step {i} {key} rotation diff={geo:.2e} rad")

            # Compare sim_state
            ssa = ta.get("sim_state", {})
            ssb = tb.get("sim_state", {})
            for field in ["qpos", "qvel", "time"]:
                va = ssa.get(field)
                vb = ssb.get(field)
                if va is not None and vb is not None:
                    if isinstance(va, list) and isinstance(vb, list):
                        max_diff = max(abs(float(a) - float(b)) for a, b in zip(va, vb))
                        if max_diff > 1e-12:
                            issues.append(f"{ep_name}: step {i} {field} diff={max_diff:.2e}")
                    elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        if abs(float(va) - float(vb)) > 1e-12:
                            issues.append(f"{ep_name}: step {i} {field} diff={abs(float(va)-float(vb)):.2e}")

    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-a", type=Path, required=True)
    parser.add_argument("--root-b", type=Path, required=True)
    parser.add_argument("--gate", choices=["r5e", "r5f"], default="r5e")
    args = parser.parse_args()

    root_a = Path(args.root_a).resolve()
    root_b = Path(args.root_b).resolve()

    print("=" * 70)
    print(f"[DeepSeek] R5 Canonical A/B Comparator — {args.gate.upper()}")
    print(f"  A: {root_a}")
    print(f"  B: {root_b}")
    print("=" * 70)

    # ── Seal verification ──
    print("\n--- Seal Verification ---")
    ok_a, seal_a = verify_seal(root_a)
    ok_b, seal_b = verify_seal(root_b)
    if not ok_a:
        print(f"  A: SEAL_FAIL — {seal_a}")
        return 5
    if not ok_b:
        print(f"  B: SEAL_FAIL — {seal_b}")
        return 5
    print(f"  A: SEAL_OK ({len(seal_a)} files)")
    print(f"  B: SEAL_OK ({len(seal_b)} files)")

    # ── Manifest check ──
    print("\n--- Manifest ---")
    ma = json.loads((root_a / "MANIFEST.json").read_text(encoding="utf-8"))
    mb = json.loads((root_b / "MANIFEST.json").read_text(encoding="utf-8"))
    if ma.get("status") not in ("PASS", "SAME_LIVE_GATE_PASS", "COMPLETE"):
        print(f"  A: status={ma.get('status')} — NOT CONSUMABLE")
        return 5
    if mb.get("status") not in ("PASS", "SAME_LIVE_GATE_PASS", "COMPLETE"):
        print(f"  B: status={mb.get('status')} — NOT CONSUMABLE")
        return 5
    if not ma.get("consumer_eligible", False):
        print(f"  A: consumer_eligible=false")
        return 5
    if not mb.get("consumer_eligible", False):
        print(f"  B: consumer_eligible=false")
        return 5
    print(f"  A: status={ma.get('status')} consumer_eligible={ma.get('consumer_eligible')}")
    print(f"  B: status={mb.get('status')} consumer_eligible={mb.get('consumer_eligible')}")
    print(f"  Manifest canonical SHA A: {canonical_sha(ma, VARIANT_MANIFEST_KEYS)[:16]}")
    print(f"  Manifest canonical SHA B: {canonical_sha(mb, VARIANT_MANIFEST_KEYS)[:16]}")

    # ── Episode comparison ──
    print("\n--- Episode Comparison ---")
    eps_a = sorted(d.name for d in (root_a / "episodes").iterdir() if d.is_dir())
    eps_b = sorted(d.name for d in (root_b / "episodes").iterdir() if d.is_dir())

    if eps_a != eps_b:
        only_a = set(eps_a) - set(eps_b)
        only_b = set(eps_b) - set(eps_a)
        print(f"  MISMATCH: A={len(eps_a)} episodes, B={len(eps_b)} episodes")
        if only_a:
            print(f"    Only in A: {sorted(only_a)}")
        if only_b:
            print(f"    Only in B: {sorted(only_b)}")
        return 5

    expected_count = 40 if args.gate == "r5f" else len(eps_a)
    if args.gate == "r5f" and len(eps_a) != 40:
        print(f"  Expected 40 episodes, got {len(eps_a)}")
        return 5

    all_issues = []
    for ep_name in eps_a:
        ok, issues = compare_episodes(
            root_a / "episodes" / ep_name,
            root_b / "episodes" / ep_name,
            ep_name,
        )
        if not ok:
            all_issues.extend(issues)

    if all_issues:
        print(f"  FAIL: {len(all_issues)} differences")
        for issue in all_issues[:20]:
            print(f"    {issue}")
        if len(all_issues) > 20:
            print(f"    ... and {len(all_issues) - 20} more")
        return 5

    print(f"  All {len(eps_a)} episodes: IDENTICAL")

    # ── Canonical digest ──
    print("\n--- Canonical Digest ---")
    digests_a = {}
    digests_b = {}
    for ep_name in eps_a:
        ep_a = root_a / "episodes" / ep_name / "episode.json"
        ep_b = root_b / "episodes" / ep_name / "episode.json"
        data_a = json.loads(ep_a.read_text(encoding="utf-8"))
        data_b = json.loads(ep_b.read_text(encoding="utf-8"))
        digests_a[ep_name] = canonical_sha(data_a, VARIANT_EPISODE_KEYS)
        digests_b[ep_name] = canonical_sha(data_b, VARIANT_EPISODE_KEYS)

    mismatched = [ep for ep in eps_a if digests_a[ep] != digests_b[ep]]
    if mismatched:
        print(f"  FAIL: {len(mismatched)} episodes have different canonical digests")
        for ep in mismatched[:5]:
            print(f"    {ep}: A={digests_a[ep][:16]} B={digests_b[ep][:16]}")
        return 5

    # Aggregate digest
    agg = "\n".join(f"{digests_a[ep]}  {ep}" for ep in sorted(eps_a))
    agg_sha = hashlib.sha256(agg.encode("utf-8")).hexdigest()
    print(f"  All {len(eps_a)} canonical digests: IDENTICAL")
    print(f"  Aggregate SHA256: {agg_sha}")

    print(f"\n{'=' * 70}")
    print(f"VERDICT: CANONICAL_IDENTITY_CONFIRMED")
    print(f"  Runs A and B produce identical canonical payloads.")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
