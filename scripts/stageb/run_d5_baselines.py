#!/usr/bin/env python3
"""D5 baseline comparison: first-CLOSE, D1b detector, best total_score, D5 offline ranker."""
import csv, json, os, sys, math
from collections import defaultdict

ROOTS = {
    "orig": "/data/liuyu/outputs/d5_120_privileged_capture",
    "gpu13": "/data/liuyu/outputs/d44d_balanced120_gpu13_r1",
    "gpu26": "/data/liuyu/outputs/d44d_balanced120_gpu26_r1",
    "gpu50": "/data/liuyu/outputs/d44d_balanced120_gpu50_r1",
}
LABELS = "/data/liuyu/outputs/d5_label_generation/d5_teacher_p_labels_v2.csv"
ACCEPTED = "/data/liuyu/outputs/d5_label_generation/d44d_accepted_episode_manifest.csv"
CKPT = "/data/liuyu/outputs/d5_training/d5_candidate_best.pt"

sys.path.insert(0, "/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/scripts/stageb")
from train_d1b_detector import CandidateRanker, FEATURE_NAMES, normalize_features


def first_close_anchor(trace):
    streak = 0
    for i, r in enumerate(trace):
        env = float(r.get("env_gripper", 0) or 0)
        env_valid = int(r.get("env_valid", 1) or 1)
        sem_ok = int(r.get("semantics_ok", 1) or 1)
        ok = bool(env_valid) and bool(sem_ok)
        cc = 1 if (ok and env > 0.5) else 0
        co = 1 if (cc and streak == 0) else 0
        streak = streak + 1 if cc else 0
        if co:
            return i
    return -1


def get_detector_emit(edir):
    mf = os.path.join(edir, "episode_manifest.json")
    if os.path.exists(mf):
        m = json.load(open(mf))
        e = m.get("detector_emit_step", -1)
        if e == "DISABLED" or e == "" or e is None:
            return -1
        return int(e)
    return -1


def get_best_total_score(edir):
    ccf = os.path.join(edir, "detector_candidates.csv")
    if not os.path.exists(ccf):
        return -1
    best_step = -1
    best_score = -999
    for c in csv.DictReader(open(ccf)):
        s = float(c.get("feat_total_score", c.get("total_score", 0)) or 0)
        if s > best_score:
            best_score = s
            best_step = int(c["step"])
    return best_step


def main():
    labels = {}
    for r in csv.DictReader(open(LABELS)):
        labels[(r["task"], int(r["state_id"]))] = r

    accepted = {}
    for r in csv.DictReader(open(ACCEPTED)):
        if r.get("status") == "BOUND":
            accepted[(r["task"], int(r["state_id"]))] = r

    # Load D5 model
    import torch
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    means = ckpt["means"]; stdevs = ckpt["stdevs"]; impute = ckpt["impute"]
    model = CandidateRanker(n_features=16)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def d5_top_candidate(edir):
        ccf = os.path.join(edir, "detector_candidates.csv")
        if not os.path.exists(ccf):
            return -1
        cands = list(csv.DictReader(open(ccf)))
        if not cands:
            return -1
        rows = []
        valid_indices = []
        for i, c in enumerate(cands):
            abstained = int(c.get("abstained", 0) or 0) == 1
            row = {}
            for fn in FEATURE_NAMES:
                row[fn] = c.get("feat_" + fn, c.get(fn, ""))
            rows.append(row)
            if not abstained:
                valid_indices.append(i)
        if not valid_indices:
            return -1
        X = normalize_features(rows, means, stdevs, impute)
        with torch.no_grad():
            scores = model(X)
            best = max(valid_indices, key=lambda i: scores[i].item())
        return int(cands[best]["step"])

    results = []
    for (task, sid), acc in sorted(accepted.items()):
        lp = labels.get((task, sid), {})
        if lp.get("status") != "VALID_LABELED":
            continue
        p_anchor = int(lp["anchor"]); p_ws = int(lp["ws"]); p_we = int(lp["we"])
        rname = acc["accepted_root"]; edir_name = acc["accepted_episode_dir"]
        rpath = ROOTS.get(rname, "")
        edir = os.path.join(rpath, edir_name) if rpath else ""
        if not os.path.isdir(edir):
            continue

        sp = acc["split"]
        stf = os.path.join(edir, "step_trace.csv")
        trace = list(csv.DictReader(open(stf))) if os.path.exists(stf) else []
        fc = first_close_anchor(trace)
        det = get_detector_emit(edir)
        ts = get_best_total_score(edir)
        d5 = d5_top_candidate(edir)

        def in_window(step):
            return step >= p_ws and step < p_we if step >= 0 else False

        results.append({
            "task": task, "sid": sid, "split": sp, "anchor": p_anchor,
            "ws": p_ws, "we": p_we,
            "fc": fc, "det": det, "ts": ts, "d5": d5,
            "fc_in_win": in_window(fc), "det_in_win": in_window(det),
            "ts_in_win": in_window(ts), "d5_in_win": in_window(d5),
            "fc_emit": fc >= 0, "det_emit": det >= 0,
            "ts_emit": ts >= 0, "d5_emit": d5 >= 0,
            "fc_exact": fc == p_anchor, "det_exact": det == p_anchor,
            "ts_exact": ts == p_anchor, "d5_exact": d5 == p_anchor,
        })

    n = len(results)
    print("Labeled traces: {}".format(n))
    print("")
    print("Method           | Emit % | In-Win % | Exact % | Med Offset")
    print("-" * 65)
    for name, emit_k, win_k, exact_k, offset_k in [
        ("1. First-CLOSE   ", "fc_emit", "fc_in_win", "fc_exact", "fc"),
        ("2. D1b detector  ", "det_emit", "det_in_win", "det_exact", "det"),
        ("3. Best total_sc ", "ts_emit", "ts_in_win", "ts_exact", "ts"),
        ("4. D5 offline    ", "d5_emit", "d5_in_win", "d5_exact", "d5"),
    ]:
        emit_n = sum(1 for r in results if r[emit_k])
        emit_pct = 100 * emit_n / n
        win_n = sum(1 for r in results if r[win_k])
        win_pct = 100 * win_n / n
        exact_n = sum(1 for r in results if r[exact_k])
        exact_pct = 100 * exact_n / n
        offsets = [r[offset_k] - r["anchor"] for r in results if r[emit_k]]
        med = sorted(offsets)[len(offsets) // 2] if offsets else float("nan")
        print("{} | {:5.0f}% | {:7.0f}% | {:6.0f}% | {:+.0f}".format(
            name, emit_pct, win_pct, exact_pct, med))

    print("")
    for sp in ["train", "val", "test"]:
        sp_res = [r for r in results if r["split"] == sp]
        if not sp_res:
            continue
        print("{} (n={}): D1b in-win={:.0f}%  BestTS in-win={:.0f}%  D5 in-win={:.0f}%".format(
            sp, len(sp_res),
            100 * sum(1 for r in sp_res if r["det_in_win"]) / len(sp_res),
            100 * sum(1 for r in sp_res if r["ts_in_win"]) / len(sp_res),
            100 * sum(1 for r in sp_res if r["d5_in_win"]) / len(sp_res)))

    # Per-task
    print("")
    print("Per-task D5 in-window:")
    from collections import defaultdict
    task_res = defaultdict(list)
    for r in results:
        task_res[r["task"]].append(r)
    for tk in sorted(task_res):
        tr = task_res[tk]
        print("  {}: {}/{} ({:.0f}%)".format(
            tk, sum(1 for r in tr if r["d5_in_win"]), len(tr),
            100 * sum(1 for r in tr if r["d5_in_win"]) / len(tr)))


if __name__ == "__main__":
    main()
