#!/usr/bin/env python3
"""Build d44d_accepted_episode_manifest.csv — unique attempt binding per state."""
import csv, hashlib, json, os, sys

MANIFEST_CSV = "/data/liuyu/outputs/d5_120_training_selection/d5_120_state_manifest.csv"
ROOTS = {
    "orig": "/data/liuyu/outputs/d5_120_privileged_capture",
    "gpu13": "/data/liuyu/outputs/d44d_balanced120_gpu13_r1",
    "gpu26": "/data/liuyu/outputs/d44d_balanced120_gpu26_r1",
    "gpu50": "/data/liuyu/outputs/d44d_balanced120_gpu50_r1",
}
OUT = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"

def sha256_file(path):
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {}
for r in csv.DictReader(open(MANIFEST_CSV)):
    manifest[(r["task_key"], int(r["state_id"]))] = r

rows = []
unbound = []

for key in sorted(manifest.keys()):
    task, sid = key
    sp = manifest[key]["split"]
    tag = "{}_s{}".format(task, sid)

    attempts = []
    for rname, rpath in ROOTS.items():
        for aname in [tag + "_shadow_attempt1", tag + "_shadow_attempt2"]:
            dp = os.path.join(rpath, aname)
            if not os.path.isdir(dp):
                continue
            mf = os.path.join(dp, "episode_manifest.json")
            if not os.path.exists(mf):
                continue
            m = json.load(open(mf))
            ok = not m.get("fatal") and m.get("infra_status") == "ok"
            fag = os.path.exists(os.path.join(dp, "FIRST_ACTION_GENERATED.json"))
            attempts.append({
                "root": rname, "dir": aname, "dp": dp, "ok": ok,
                "fag": fag, "n_steps": m.get("n_steps", -1),
                "succ": m.get("success_primary", -1),
            })

    ok_attempts = [a for a in attempts if a["ok"]]

    if not ok_attempts:
        unbound.append((key, "No OK attempt found among {} total".format(len(attempts))))
        continue

    # Deterministic tiebreak: first OK by (root name, attempt index)
    ok_attempts.sort(key=lambda a: (a["root"], a["dir"]))
    best = ok_attempts[0]
    if len(ok_attempts) > 1:
        unbound.append((key, "{} OK attempts, selected root={}".format(
            len(ok_attempts), best["root"])))
    dp = best["dp"]

    shas = {}
    for fn in ["episode_manifest", "step_trace", "teacher_sidecar",
               "detector_candidates", "action_identity", "provenance",
               "artifact_hashes"]:
        fp = os.path.join(dp, fn + (".json" if fn != "step_trace" and fn != "detector_candidates" and fn != "action_identity" and fn != "provenance" and fn != "artifact_hashes" else ".csv"))
        if fn == "episode_manifest":
            fp = os.path.join(dp, "episode_manifest.json")
        elif fn == "teacher_sidecar":
            fp = os.path.join(dp, "teacher_sidecar.json")
        elif fn == "step_trace":
            fp = os.path.join(dp, "step_trace.csv")
        elif fn == "detector_candidates":
            fp = os.path.join(dp, "detector_candidates.csv")
        elif fn == "action_identity":
            fp = os.path.join(dp, "action_identity.csv")
        elif fn == "provenance":
            fp = os.path.join(dp, "provenance.csv")
        elif fn == "artifact_hashes":
            fp = os.path.join(dp, "artifact_hashes.csv")
        shas[fn] = sha256_file(fp)

    gpu_indices = ""
    capture_commit = ""
    pf = os.path.join(dp, "provenance.csv")
    if os.path.exists(pf):
        for r in csv.DictReader(open(pf)):
            if r["key"] == "cuda_visible_devices":
                gpu_indices = r["value"]
            if r["key"] == "git_HEAD":
                capture_commit = r["value"]

    rows.append({
        "task": task, "state_id": sid, "split": sp,
        "accepted_root": best["root"],
        "accepted_episode_dir": best["dir"],
        "accepted_attempt_index": 1 if "attempt1" in best["dir"] else (2 if "attempt2" in best["dir"] else -1),
        "status": "BOUND", "reason": "",
        "n_steps": best["n_steps"], "success": best["succ"],
        "episode_manifest_sha256": shas.get("episode_manifest", ""),
        "step_trace_sha256": shas.get("step_trace", ""),
        "teacher_sidecar_sha256": shas.get("teacher_sidecar", ""),
        "detector_candidates_sha256": shas.get("detector_candidates", ""),
        "action_identity_sha256": shas.get("action_identity", ""),
        "provenance_sha256": shas.get("provenance", ""),
        "artifact_hashes_sha256": shas.get("artifact_hashes", ""),
        "capture_commit": capture_commit[:16] if capture_commit else "",
        "ordered_gpu_indices": gpu_indices,
        "attempt_count": len(attempts),
        "ok_attempt_count": len(ok_attempts),
    })

# Output
fieldnames = list(rows[0].keys()) if rows else []
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

bound = sum(1 for r in rows if r["status"] == "BOUND")
print("Bound: {}/120".format(bound))
for k, reason in unbound:
    print("  UNBOUND: {}_s{}: {}".format(k[0], k[1], reason))

if bound == 120:
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    print("ACCEPTED_MANIFEST_SHA256: {}".format(sha))
    print("ALL 120 BOUND")
else:
    print("MISSING: {} states unbound".format(120 - bound))
    sys.exit(1)
