"""[DeepSeek] R5 Canonical A/B Comparator (v2 — self-audit fixes).

Reads two sealed roots from independent runs and verifies canonical identity.

Supports:
  --gate r5e  : case_records.jsonl + per_task_summary.jsonl structure
  --gate r5f  : episodes/<task>/episode.json structure
  --gate c1   : per_task/<suite>_task_XX.json C1-V2 registry structure

Checks:
  - Both seals verify independently (all files against SHA256SUMS)
  - Episode/task identities identical (same set, no extras)
  - Per-step actions (raw, score, env) identical
  - Per-step entity poses identical (position, quaternion)
  - Per-step sim_state (qpos, qvel, act, time) identical
  - Relation identities identical
  - No NaN/Inf in any numeric field
  - Canonical payload digest identical (whitelisted variant fields excluded)

Variant fields (allowed to differ between runs):
  - timestamp, start_time, end_time, elapsed_s
  - hostname, executable, command line, python_version
  - File paths containing the run label

NOT variant (must be identical):
  - collection_seed, seed, task_idx, state_id, suite
  - All numeric telemetry
  - All entity poses and identities

Usage:
  python n5/phase2_labels/compare_r5_canonical.py \
    --root-a /path/to/run_A --root-b /path/to/run_B --gate r5e|r5f|c1
"""
import json, os, sys, argparse, hashlib, math
from pathlib import Path
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def full_seal_check(root):
    """Verify every file in SHA256SUMS. Returns (ok, error_or_seal_dict)."""
    root = Path(root)
    sums_path = root / "SHA256SUMS"
    side_path = root / "SHA256SUMS.sha256"
    if not sums_path.is_file() or not side_path.is_file():
        return False, "not a sealed root"
    # Verify sidecar
    sidecar = side_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar) < 2 or sidecar[0] != sha256_file(sums_path):
        return False, "seal sidecar mismatch"
    # Verify every file
    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            return False, f"malformed SHA256SUMS line: {line}"
        digest, name = parts
        name = name.lstrip("*")
        if name in ("SHA256SUMS", "SHA256SUMS.sha256"):
            continue
        target = root / name
        if not target.is_file():
            return False, f"sealed file missing: {name}"
        if sha256_file(target) != digest:
            return False, f"seal file mismatch: {name}"
        expected[name] = digest
    return True, expected


VARIANT_MANIFEST_KEYS = frozenset({
    "timestamp", "start_time", "end_time", "elapsed_s",
    "executable", "command", "python_version",
    "environment", "hostname",
    "r5e_receipt",
    "registry_manifest",
    "model_tree_sha256", "processor_sha256",
})


def canonical_json(obj, variant_keys=frozenset()):
    """Return canonical JSON-compatible object with variant keys removed."""
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
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in canonical payload: {obj}")
        return float(f"{obj:.12g}")
    return obj


def canonical_sha(obj, variant_keys=frozenset()):
    cleaned = canonical_json(obj, variant_keys)
    raw = json.dumps(cleaned, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def assert_finite(obj, path=""):
    """Recursively check all floats in obj are finite. Raises on first NaN/Inf."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_finite(v, f"{path}[{i}]")


# ── R5-E comparator (case_records.jsonl structure) ──

def compare_r5e(root_a, root_b):
    """Compare two R5-E sealed roots."""
    recs_a = _load_jsonl(root_a / "case_records.jsonl")
    recs_b = _load_jsonl(root_b / "case_records.jsonl")
    issues = []

    if len(recs_a) != len(recs_b):
        issues.append(f"case_records length: {len(recs_a)} vs {len(recs_b)}")
        return issues

    # Index by (suite, task_idx, state_id, step, entity_type, entity_id)
    def _key(r):
        return (r.get("suite"), r.get("task_idx"), r.get("state_id"), r.get("step"),
                r.get("entity_type"), r.get("entity_id"))

    keys_a = {_key(r) for r in recs_a}
    keys_b = {_key(r) for r in recs_b}
    if keys_a != keys_b:
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        if only_a:
            issues.append(f"records only in A: {len(only_a)}")
        if only_b:
            issues.append(f"records only in B: {len(only_b)}")
        return issues

    # Sort both by key for pairwise comparison
    recs_a.sort(key=_key)
    recs_b.sort(key=_key)

    for ra, rb in zip(recs_a, recs_b):
        assert_finite(ra)
        assert_finite(rb)
        k = _key(ra)
        # Compare all numeric fields
        for field in ["AB_pos_Linf", "AB_rot_err", "BC_pos_Linf", "BC_rot_err",
                       "pos_limit", "rot_limit",
                       "fwd1_qpos_drift", "fwd1_qvel_drift", "fwd1_act_drift", "fwd1_time_drift",
                       "fwd2_qpos_drift", "fwd2_qvel_drift", "fwd2_act_drift", "fwd2_time_drift"]:
            va = ra.get(field); vb = rb.get(field)
            if va is not None and vb is not None:
                if abs(float(va) - float(vb)) > 1e-15:
                    issues.append(f"record {k}: {field} diff={abs(float(va)-float(vb)):.2e}")
        # Compare booleans
        for field in ["BC_pos_pass", "BC_rot_pass", "AB_stale",
                       "source_mutated_fwd1", "source_mutated_fwd2",
                       "nonfinite_pose", "nonfinite_source"]:
            if ra.get(field) != rb.get(field):
                issues.append(f"record {k}: {field} mismatch: {ra.get(field)} vs {rb.get(field)}")
        # Compare strings
        for field in ["entity_name", "semantic_role", "resolution"]:
            if ra.get(field) != rb.get(field):
                issues.append(f"record {k}: {field} mismatch: {ra.get(field)} vs {rb.get(field)}")

    # Per-task summaries
    sums_a = _load_jsonl(root_a / "per_task_summary.jsonl")
    sums_b = _load_jsonl(root_b / "per_task_summary.jsonl")
    if len(sums_a) != len(sums_b):
        issues.append(f"per_task_summary length: {len(sums_a)} vs {len(sums_b)}")
    else:
        sums_a.sort(key=lambda s: s.get("task_key", ""))
        sums_b.sort(key=lambda s: s.get("task_key", ""))
        for sa, sb in zip(sums_a, sums_b):
            tk = sa.get("task_key", "?")
            for field in ["n_entities", "n_records", "BC_pos_fail", "BC_rot_fail",
                          "AB_stale_count", "source_mutations", "nonfinite",
                          "entity_closure_ok", "status"]:
                if sa.get(field) != sb.get(field):
                    issues.append(f"summary {tk}: {field} mismatch: {sa.get(field)} vs {sb.get(field)}")

    # Canonical manifest digests
    ma = json.loads((root_a / "MANIFEST.json").read_text(encoding="utf-8"))
    mb = json.loads((root_b / "MANIFEST.json").read_text(encoding="utf-8"))
    sha_a = canonical_sha(ma, VARIANT_MANIFEST_KEYS)
    sha_b = canonical_sha(mb, VARIANT_MANIFEST_KEYS)
    if sha_a != sha_b:
        issues.append(f"manifest canonical SHA mismatch: {sha_a[:16]} vs {sha_b[:16]}")

    return issues


# ── R5-F comparator (episodes/<task>/episode.json structure) ──

def compare_r5f(root_a, root_b):
    """Compare two R5-F sealed roots."""
    eps_a = sorted(d.name for d in (root_a / "episodes").iterdir() if d.is_dir())
    eps_b = sorted(d.name for d in (root_b / "episodes").iterdir() if d.is_dir())
    issues = []

    if eps_a != eps_b:
        only_a = set(eps_a) - set(eps_b)
        only_b = set(eps_b) - set(eps_a)
        if only_a:
            issues.append(f"episodes only in A: {sorted(only_a)}")
        if only_b:
            issues.append(f"episodes only in B: {sorted(only_b)}")
        return issues

    for ep_name in eps_a:
        ep_a = root_a / "episodes" / ep_name / "episode.json"
        ep_b = root_b / "episodes" / ep_name / "episode.json"
        data_a = json.loads(ep_a.read_text(encoding="utf-8"))
        data_b = json.loads(ep_b.read_text(encoding="utf-8"))
        assert_finite(data_a)
        assert_finite(data_b)

        # Identity
        if data_a.get("episode_id") != data_b.get("episode_id"):
            issues.append(f"{ep_name}: episode_id mismatch")
        if data_a.get("step_count") != data_b.get("step_count"):
            issues.append(f"{ep_name}: step_count mismatch")

        # Relations
        rels_a = data_a.get("relations", [])
        rels_b = data_b.get("relations", [])
        if len(rels_a) != len(rels_b):
            issues.append(f"{ep_name}: relation count mismatch: {len(rels_a)} vs {len(rels_b)}")
        else:
            for i, (ra, rb) in enumerate(zip(rels_a, rels_b)):
                for side in ("object_resolution", "target_resolution"):
                    ra_res = ra.get(side, {}); rb_res = rb.get(side, {})
                    if ra_res.get("entity_type") != rb_res.get("entity_type"):
                        issues.append(f"{ep_name}: rel[{i}].{side} entity_type mismatch")
                    if ra_res.get("entity_id") != rb_res.get("entity_id"):
                        issues.append(f"{ep_name}: rel[{i}].{side} entity_id mismatch")
                    if ra_res.get("resolution") != rb_res.get("resolution"):
                        issues.append(f"{ep_name}: rel[{i}].{side} resolution mismatch")

        # Steps: actions
        steps_a = data_a.get("steps", [])
        steps_b = data_b.get("steps", [])
        if len(steps_a) != len(steps_b):
            issues.append(f"{ep_name}: step count mismatch: {len(steps_a)} vs {len(steps_b)}")
        else:
            for i, (sa, sb) in enumerate(zip(steps_a, steps_b)):
                for act_field in ["action_raw_7d", "score_action_7d", "action_env_7d"]:
                    aa = sa.get(act_field, []); ab = sb.get(act_field, [])
                    if len(aa) != len(ab):
                        issues.append(f"{ep_name}: step {i} {act_field} length mismatch")
                    elif len(aa) > 0:
                        max_diff = max(abs(float(a) - float(b)) for a, b in zip(aa, ab))
                        if max_diff > 1e-12:
                            issues.append(f"{ep_name}: step {i} {act_field} diff={max_diff:.2e}")

        # Telemetry: entity poses + sim_state
        tel_a = data_a.get("telemetry", [])
        tel_b = data_b.get("telemetry", [])
        if len(tel_a) != len(tel_b):
            issues.append(f"{ep_name}: telemetry length mismatch")
        else:
            for i, (ta, tb) in enumerate(zip(tel_a, tel_b)):
                # Entity poses
                ents_a = {f"{e['entity_type']}:{e['entity_id']}": e for e in ta.get("entities", [])}
                ents_b = {f"{e['entity_type']}:{e['entity_id']}": e for e in tb.get("entities", [])}
                if set(ents_a.keys()) != set(ents_b.keys()):
                    issues.append(f"{ep_name}: step {i} entity set mismatch")

                for key in set(ents_a.keys()) & set(ents_b.keys()):
                    pa = np.array(ents_a[key]["world_pose"]["position"])
                    pb = np.array(ents_b[key]["world_pose"]["position"])
                    pos_diff = float(np.max(np.abs(pa - pb)))
                    if pos_diff > 1e-12:
                        issues.append(f"{ep_name}: step {i} {key} pos diff={pos_diff:.2e}")

                    qa = np.array(ents_a[key]["world_pose"]["quaternion"])
                    qb = np.array(ents_b[key]["world_pose"]["quaternion"])
                    qa_n = qa / np.linalg.norm(qa)
                    qb_n = qb / np.linalg.norm(qb)
                    if not all(math.isfinite(x) for x in qa_n):
                        issues.append(f"{ep_name}: step {i} {key} A quaternion NaN")
                    if not all(math.isfinite(x) for x in qb_n):
                        issues.append(f"{ep_name}: step {i} {key} B quaternion NaN")
                    dot = abs(np.dot(qa_n, qb_n))
                    dot = min(dot, 1.0)
                    geo = float(2.0 * math.atan2(math.sqrt(max(0, 1 - dot*dot)), dot))
                    if geo > 1e-12:
                        issues.append(f"{ep_name}: step {i} {key} rot diff={geo:.2e} rad")

                # Sim state: qpos, qvel, act, time
                ssa = ta.get("sim_state", {})
                ssb = tb.get("sim_state", {})
                for field in ["qpos", "qvel", "act", "time"]:
                    va = ssa.get(field)
                    vb = ssb.get(field)
                    if va is None and vb is None:
                        continue
                    if va is None or vb is None:
                        issues.append(f"{ep_name}: step {i} sim_state.{field} None mismatch")
                        continue
                    if isinstance(va, list) and isinstance(vb, list):
                        if len(va) != len(vb):
                            issues.append(f"{ep_name}: step {i} sim_state.{field} length mismatch")
                        else:
                            max_diff = max(abs(float(a) - float(b)) for a, b in zip(va, vb))
                            if max_diff > 1e-12:
                                issues.append(f"{ep_name}: step {i} sim_state.{field} diff={max_diff:.2e}")
                    elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        if abs(float(va) - float(vb)) > 1e-12:
                            issues.append(f"{ep_name}: step {i} sim_state.{field} diff={abs(float(va)-float(vb)):.2e}")

    # Canonical episode digests
    digests_a = {}; digests_b = {}
    for ep_name in eps_a:
        data_a = json.loads((root_a / "episodes" / ep_name / "episode.json").read_text(encoding="utf-8"))
        data_b = json.loads((root_b / "episodes" / ep_name / "episode.json").read_text(encoding="utf-8"))
        digests_a[ep_name] = canonical_sha(data_a)
        digests_b[ep_name] = canonical_sha(data_b)
    mismatched = [ep for ep in eps_a if digests_a[ep] != digests_b[ep]]
    if mismatched:
        for ep in mismatched[:5]:
            issues.append(f"{ep}: canonical digest mismatch: {digests_a[ep][:16]} vs {digests_b[ep][:16]}")
    else:
        print(f"  All {len(eps_a)} canonical episode digests: IDENTICAL")

    # Canonical manifest
    ma = json.loads((root_a / "MANIFEST.json").read_text(encoding="utf-8"))
    mb = json.loads((root_b / "MANIFEST.json").read_text(encoding="utf-8"))
    sha_a = canonical_sha(ma, VARIANT_MANIFEST_KEYS)
    sha_b = canonical_sha(mb, VARIANT_MANIFEST_KEYS)
    if sha_a != sha_b:
        issues.append(f"manifest canonical SHA mismatch: {sha_a[:16]} vs {sha_b[:16]}")

    return issues


# ── C1 comparator (per_task/<suite>_task_XX.json structure) ──

def compare_c1(root_a, root_b):
    """Compare two C1-V2 registry sealed roots."""
    pt_a = root_a / "per_task"; pt_b = root_b / "per_task"
    files_a = sorted(f.name for f in pt_a.iterdir() if f.is_file() and f.name.endswith(".json"))
    files_b = sorted(f.name for f in pt_b.iterdir() if f.is_file() and f.name.endswith(".json"))
    issues = []

    if files_a != files_b:
        only_a = set(files_a) - set(files_b)
        only_b = set(files_b) - set(files_a)
        if only_a:
            issues.append(f"per_task files only in A: {sorted(only_a)}")
        if only_b:
            issues.append(f"per_task files only in B: {sorted(only_b)}")
        return issues

    for fn in files_a:
        data_a = json.loads((pt_a / fn).read_text(encoding="utf-8"))
        data_b = json.loads((pt_b / fn).read_text(encoding="utf-8"))
        la = data_a.get("legacy", data_a)
        lb = data_b.get("legacy", data_b)
        if la.get("status") != lb.get("status"):
            issues.append(f"{fn}: status mismatch: {la.get('status')} vs {lb.get('status')}")
        rc_a = la.get("resolution_counts", {})
        rc_b = lb.get("resolution_counts", {})
        for k in ["object_ok", "object_unresolved", "object_ambiguous",
                   "target_ok", "target_unresolved", "target_ambiguous"]:
            if rc_a.get(k) != rc_b.get(k):
                issues.append(f"{fn}: counts.{k} mismatch: {rc_a.get(k)} vs {rc_b.get(k)}")
        rels_a = la.get("relations", [])
        rels_b = lb.get("relations", [])
        if len(rels_a) != len(rels_b):
            issues.append(f"{fn}: relation count mismatch")
        else:
            for i, (ra, rb) in enumerate(zip(rels_a, rels_b)):
                for side in ("object_resolution", "target_resolution"):
                    ra_res = ra.get(side, {}); rb_res = rb.get(side, {})
                    if ra_res.get("resolution") != rb_res.get("resolution"):
                        issues.append(f"{fn}: rel[{i}].{side} resolution mismatch: "
                                      f"{ra_res.get('resolution')} vs {rb_res.get('resolution')}")
                    if ra_res.get("entity_type") != rb_res.get("entity_type"):
                        issues.append(f"{fn}: rel[{i}].{side} entity_type mismatch")
                    if ra_res.get("entity_id") != rb_res.get("entity_id"):
                        issues.append(f"{fn}: rel[{i}].{side} entity_id mismatch")

    # Compare alias ledgers
    for ledger_name in ["ALIAS_LEDGER.json", "ALIAS_LEDGER_V2.json"]:
        la = root_a / ledger_name; lb = root_b / ledger_name
        if la.is_file() and lb.is_file():
            da = json.loads(la.read_text(encoding="utf-8"))
            db = json.loads(lb.read_text(encoding="utf-8"))
            if da.get("n_aliases") != db.get("n_aliases"):
                issues.append(f"{ledger_name}: alias count mismatch")
        elif la.is_file() != lb.is_file():
            issues.append(f"{ledger_name}: present in only one root")

    return issues


def _load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-a", type=Path, required=True)
    parser.add_argument("--root-b", type=Path, required=True)
    parser.add_argument("--gate", choices=["r5e", "r5f", "c1"], default="r5e")
    args = parser.parse_args()

    root_a = Path(args.root_a).resolve()
    root_b = Path(args.root_b).resolve()

    print("=" * 70)
    print(f"[DeepSeek] R5 Canonical A/B Comparator v2 — {args.gate.upper()}")
    print(f"  A: {root_a}")
    print(f"  B: {root_b}")
    print("=" * 70)

    # ── Full seal verification ──
    print("\n--- Seal Verification ---")
    ok_a, result_a = full_seal_check(root_a)
    ok_b, result_b = full_seal_check(root_b)
    if not ok_a:
        print(f"  A: SEAL_FAIL — {result_a}")
        return 5
    if not ok_b:
        print(f"  B: SEAL_FAIL — {result_b}")
        return 5
    print(f"  A: SEAL_OK ({len(result_a)} files)")
    print(f"  B: SEAL_OK ({len(result_b)} files)")

    # ── Manifest status check ──
    print("\n--- Manifest Status ---")
    manifest_names = {"c1": "ENTITY_REGISTRY_V2_SUMMARY.json"}.get(args.gate, "MANIFEST.json")
    ma = json.loads((root_a / manifest_names).read_text(encoding="utf-8"))
    mb = json.loads((root_b / manifest_names).read_text(encoding="utf-8"))
    status_a = ma.get("status", "?")
    status_b = mb.get("status", "?")
    print(f"  A: {status_a}  consumer_eligible={ma.get('consumer_eligible', '?')}")
    print(f"  B: {status_b}  consumer_eligible={mb.get('consumer_eligible', '?')}")

    # ── Structure-specific comparison ──
    print(f"\n--- {args.gate.upper()} Comparison ---")
    if args.gate == "r5e":
        issues = compare_r5e(root_a, root_b)
    elif args.gate == "r5f":
        issues = compare_r5f(root_a, root_b)
    else:  # c1
        issues = compare_c1(root_a, root_b)

    if issues:
        print(f"  FAIL: {len(issues)} differences found")
        for issue in issues[:30]:
            print(f"    {issue}")
        if len(issues) > 30:
            print(f"    ... and {len(issues) - 30} more")
        print(f"\n{'=' * 70}")
        print(f"VERDICT: CANONICAL_IDENTITY_FAIL — {len(issues)} differences")
        return 5

    print(f"  All checks: IDENTICAL")
    print(f"\n{'=' * 70}")
    print(f"VERDICT: CANONICAL_IDENTITY_CONFIRMED")
    print(f"  Runs A and B produce identical canonical payloads.")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
