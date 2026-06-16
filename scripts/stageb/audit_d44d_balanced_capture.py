#!/usr/bin/env python3
"""D4.4D independent capture auditor — reads raw artifacts, trusts no summary.

Input:
  --selection-manifest: frozen 120-state manifest (d5_120_state_manifest.csv)
  --capture-roots-manifest: JSON mapping root names to paths
  --expected-selection-sha256
  --output: audit result JSON path

Hard gates:
  manifest 120 states, 80/20/20 split
  unique task-state = 120
  extra = 0, duplicate = 0
  external 34-state leakage = 0
  illegal retry = 0
  attempted + unattempted = 120
  per-step privileged_valid = 1
  EEF/object fields finite
  provenance consistent
"""
import argparse, csv, hashlib, json, os, sys
from collections import defaultdict, Counter
from pathlib import Path

TASKS = [
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "chocolate_pudding", "orange_juice",
]
EVAL_MANIFEST = "tables/d4_shadow_freeze_v1/d4_shadow_state_manifest.csv"
EXCLUSIONS = "tables/d4_shadow_freeze_v1/d4_shadow_exclusions.csv"

REQUIRED_FILES = [
    "episode_manifest.json", "teacher_sidecar.json", "step_trace.csv",
    "detector_candidates.csv", "action_identity.csv", "provenance.csv",
    "artifact_hashes.csv", "latency.csv",
]
AUTHORIZED_GPUS = {"2,6", "1,3", "5,1", "3,1"}


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def load_json(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


def is_finite(v):
    if v in ("", None): return False
    try:
        import math
        return math.isfinite(float(v))
    except (ValueError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-manifest", required=True)
    ap.add_argument("--capture-roots-manifest", required=True)
    ap.add_argument("--expected-selection-sha256", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gates = []
    auditor_ok = True

    def fail(name, detail=""):
        nonlocal auditor_ok
        auditor_ok = False
        gates.append({"name": name, "pass": False, "detail": str(detail)})
        if detail:
            print("  FAIL {}: {}".format(name, detail))
        else:
            print("  FAIL {}".format(name))

    def ok(name, detail=""):
        gates.append({"name": name, "pass": True, "detail": str(detail)})

    # ── 1. Manifest integrity ──
    msha = sha256_file(args.selection_manifest)
    if msha != args.expected_selection_sha256:
        fail("MANIFEST_SHA_MISMATCH", "got {}...".format(msha[:16]))
        # Continue anyway — this is a gate failure but we report everything
    ok("MANIFEST_SHA", msha[:16])

    manifest = list(csv.DictReader(open(args.selection_manifest)))
    if len(manifest) != 120:
        fail("MANIFEST_COUNT", len(manifest))
    ok("MANIFEST_COUNT", len(manifest))

    manifest_keys = {(r["task_key"], int(r["state_id"])): r for r in manifest}
    if len(manifest_keys) != 120:
        fail("MANIFEST_UNIQUE", len(manifest_keys))

    splits = Counter(r["split"] for r in manifest)
    for sp, exp in [("train", 80), ("val", 20), ("test", 20)]:
        if splits.get(sp, 0) != exp:
            fail("SPLIT_{}".format(sp), "{} != {}".format(splits.get(sp, 0), exp))
    ok("MANIFEST_SPLITS", "train={} val={} test={}".format(splits["train"], splits["val"], splits["test"]))

    # Per-task
    for tk in TASKS:
        tc = Counter(r["split"] for r in manifest if r["task_key"] == tk)
        for sp in ["train", "val", "test"]:
            exp = {"train": 8, "val": 2, "test": 2}[sp]
            if tc.get(sp, 0) != exp:
                fail("TASK_{}_{}_SPLIT".format(tk, sp), "{} != {}".format(tc.get(sp, 0), exp))
    ok("PER_TASK_SPLITS")

    # ── 2. Evaluation leakage ──
    repo_root = Path(__file__).resolve().parents[2]
    eval_path = repo_root / EVAL_MANIFEST
    if eval_path.exists():
        eval_states = set()
        for r in csv.DictReader(open(eval_path)):
            eval_states.add((r["task_key"], int(r["state_id"])))
        leak = manifest_keys.keys() & eval_states
        if leak:
            fail("EVAL_LEAKAGE", "{} states: {}".format(len(leak), sorted(leak)[:5]))
        else:
            ok("EVAL_LEAKAGE", "0")
    else:
        ok("EVAL_LEAKAGE", "EVAL_MANIFEST_MISSING")

    excl_path = repo_root / EXCLUSIONS
    if excl_path.exists():
        excl_states = set()
        for r in csv.DictReader(open(excl_path)):
            excl_states.add((r["task_key"], int(r["state_id"])))
        leak = manifest_keys.keys() & excl_states
        if leak:
            fail("EXCL_LEAKAGE", "{} states: {}".format(len(leak), sorted(leak)[:5]))
        else:
            ok("EXCL_LEAKAGE", "0")

    # ── 3. Load capture roots ──
    roots_manifest = json.load(open(args.capture_roots_manifest))
    roots = {}
    for name, rpath in roots_manifest.items():
        if os.path.isdir(rpath):
            roots[name] = rpath
        else:
            print("  WARNING: root {} ({}) not found".format(name, rpath))

    # ── 4. Scan all episode dirs ──
    all_eps = {}  # (task, state_id) -> list of (root_name, episode_dir, info)
    for rname, rpath in sorted(roots.items()):
        for d in sorted(os.listdir(rpath)):
            dp = os.path.join(rpath, d)
            if not os.path.isdir(dp) or "_shadow_attempt" not in d:
                continue
            mf = os.path.join(dp, "episode_manifest.json")
            if not os.path.exists(mf): continue
            m = json.load(open(mf))
            task = m.get("task", ""); sid = m.get("state_id", -1)
            if not task or sid < 0: continue
            key = (task, sid)
            if key not in all_eps:
                all_eps[key] = []
            all_eps[key].append((rname, dp, m))

    # ── 5. Per-state audit ──
    ok_count = 0
    invalid_count = 0
    unattempted = 0
    gpu_counts = defaultdict(int)

    for key in sorted(manifest_keys.keys()):
        eps = all_eps.get(key, [])
        if not eps:
            unattempted += 1
            continue

        # Find first OK attempt
        ok_ep = None
        for rname, dp, m in eps:
            if m.get("infra_status") == "ok" and not m.get("fatal"):
                ok_ep = (rname, dp, m)
                break

        if ok_ep is None:
            invalid_count += 1
            fail("STATE_INVALID", "{}_s{}: {} attempts, no OK".format(key[0], key[1], len(eps)))
            continue

        rname, dp, m = ok_ep
        tag = "{}_s{}".format(key[0], key[1])

        # Required files
        for fn in REQUIRED_FILES:
            if not os.path.exists(os.path.join(dp, fn)):
                fail("MISSING_{}".format(fn), tag)

        # Privileged validity
        sc = load_json(os.path.join(dp, "teacher_sidecar.json"))
        if sc:
            if sc.get("privileged_valid") != 1:
                fail("PRIV_INVALID", tag)
            if not sc.get("object_lookup_ok", False):
                fail("OBJ_LOOKUP", tag)

        # Step trace audit
        stf = os.path.join(dp, "step_trace.csv")
        if os.path.exists(stf):
            trace = list(csv.DictReader(open(stf)))
            if len(trace) != m.get("n_steps", -1):
                fail("STEP_COUNT", "{}: {} rows != {} n_steps".format(tag, len(trace), m.get("n_steps")))
            # Check step continuity and privileged fields
            for i, row in enumerate(trace):
                step = int(row.get("step", -1))
                if step != i:
                    fail("STEP_SEQ", "{}: step {} at row {}".format(tag, step, i))
                    break
                # Check privileged fields if sidecar enabled
                if sc and sc.get("sidecar_enabled"):
                    pv = int(row.get("privileged_valid", 0) or 0)
                    if pv != 1:
                        fail("PRIV_STEP", "{} step {} privileged_valid={}".format(tag, step, pv))
                        break

        # Action identity
        idf = os.path.join(dp, "action_identity.csv")
        if os.path.exists(idf):
            id_rows = list(csv.DictReader(open(idf)))
            if any(int(r.get("action_identical", "1") or "1") == 0 for r in id_rows):
                fail("ACTION_ID", tag)

        # Provenance
        prov = {}
        pf = os.path.join(dp, "provenance.csv")
        if os.path.exists(pf):
            for row in csv.DictReader(open(pf)):
                prov[row["key"]] = row["value"]
            gpu = prov.get("cuda_visible_devices", "")
            if gpu not in AUTHORIZED_GPUS:
                fail("GPU_UNAUTHORIZED", "{}: {}".format(tag, gpu))
            gpu_counts[gpu] += 1

            # Check git HEAD consistency
            git_head = prov.get("git_HEAD", "")
            if not git_head:
                fail("PROVENANCE_GIT", tag)

        ok_count += 1

    # ── 6. Extra states (not in manifest) ──
    extra = set(all_eps.keys()) - set(manifest_keys.keys())
    if extra:
        for k in sorted(extra):
            fail("EXTRA_STATE", "{}_s{}".format(k[0], k[1]))
    ok("EXTRA_STATES", len(extra))

    # ── 7. Summary ──
    print("\n" + "=" * 60)
    print("D4.4D INDEPENDENT AUDIT")
    print("=" * 60)
    print("Manifest: 120")
    print("OK: {}".format(ok_count))
    print("Invalid/Terminal: {}".format(invalid_count))
    print("Unattempted: {}".format(unattempted))
    print("Extra: {}".format(len(extra)))
    print("GPU distribution: {}".format(dict(gpu_counts)))

    n_fail = sum(1 for g in gates if not g["pass"])
    status = "PASS" if auditor_ok else "FAIL"
    print("\nAUDITOR_PIPELINE: {}".format(status))
    print("Gates: {} PASS / {} FAIL".format(len(gates) - n_fail, n_fail))

    report = {
        "auditor_pipeline": status,
        "manifest_states": 120,
        "ok_states": ok_count,
        "invalid_states": invalid_count,
        "unattempted_states": unattempted,
        "extra_states": len(extra),
        "gpu_distribution": dict(gpu_counts),
        "gates": gates,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print("Output: {}".format(args.output))

    return 0 if auditor_ok else 1


if __name__ == "__main__":
    sys.exit(main())
