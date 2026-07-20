#!/usr/bin/env python3
"""Gate F3: Independent sealed geometry audit for Factorized Teacher.

Must produce exactly: PASS_FINAL_STUDENT_TRAINING or HOLD.
No threshold tuning, no reading FIT-DEV/CAL/CHECK/attack results.
"""

import argparse, csv, hashlib, json, math, os, sys, uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
SUPPORTED_ROUTES = {"single_object_pick_place", "multi_object_transfer"}


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def verify_seal(root: Path) -> str:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise SystemExit(f"SEAL MISSING: {root}")
    expected = sidecar.read_text().strip()
    actual = f"{sha256_file(sums)}  SHA256SUMS"
    if expected != actual:
        raise SystemExit(f"SEAL MISMATCH: {root}")
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        target = root / name
        if not target.is_file() or sha256_file(target) != digest:
            raise SystemExit(f"FILE MISMATCH: {root}/{name}")
    return sha256_file(sums)


def jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_seal(root: Path) -> str:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in excluded),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    content = "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files)
    _atomic_text(root / "SHA256SUMS", content)
    _atomic_text(root / "SHA256SUMS.sha256", f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n")
    return sha256_file(root / "SHA256SUMS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.teacher_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"output exists: {output}")

    teacher_seal = verify_seal(root)
    manifest = json.loads((root / "factorized_teacher_v1_manifest.json").read_text())
    labels_root = root / "labels"

    errors: list[str] = []
    hold = False

    # ── 1. Identity/step closure ────────────────────────────────────────
    n_ids = 0
    n_steps = 0
    missing = 0
    for suite in SUITES:
        for task in range(10):
            for state in range(20):
                p = labels_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "factorized_teacher_v1.jsonl"
                if not p.is_file():
                    missing += 1; continue
                n_ids += 1
                rows = jsonl(p)
                n_steps += len(rows)
                for i, r in enumerate(rows):
                    if r.get("step") != i:
                        errors.append(f"STEP_INDEX: {suite}/task_{task:02d}/state_{state:02d} step {i}")
                        hold = True

    if n_ids != 800:
        errors.append(f"IDENTITY_COUNT: {n_ids} != 800"); hold = True
    if n_steps != 176336:
        errors.append(f"STEP_COUNT: {n_steps} != 176336"); hold = True
    if missing > 0:
        errors.append(f"MISSING: {missing} identities"); hold = True

    # ── 2. Head geometry ────────────────────────────────────────────────
    grasp_pos = grasp_neg = grasp_unk = 0
    manip_pos = manip_neg = manip_unk = 0
    release_pos = release_neg = release_unk = 0
    logical_violations = 0
    unsupported_known = 0
    event_errors = 0
    event_obj_conflicts = 0

    grasp_segments: list[int] = []
    cur_seg = 0
    release_eps: set[str] = set()

    per_route: dict[str, dict] = defaultdict(lambda: {"eps": 0, "events": 0, "grasp_pos": 0, "manip_pos": 0, "has_pos": 0, "tasks": set()})
    per_task = defaultdict(lambda: Counter())
    per_suite = defaultdict(lambda: Counter())

    for suite in SUITES:
        for task in range(10):
            for state in range(20):
                identity = f"{suite}/task_{task:02d}/state_{state:02d}"
                tk = (suite, f"task_{task:02d}")
                p = labels_root / suite / f"task_{task:02d}" / f"state_{state:02d}" / "factorized_teacher_v1.jsonl"
                if not p.is_file():
                    continue
                rows = jsonl(p)
                route = rows[0].get("mechanism_type", "?")
                supported = route in SUPPORTED_ROUTES

                ep_grasp = ep_manip = ep_release = 0
                ep_events: set[int] = set()
                prev_grasp = False
                cur_seg = 0

                for r in rows:
                    gv, gk = r["grasp_established"], r["grasp_established_known_mask"]
                    mv, mk = r["manipulation_active"], r["manipulation_active_known_mask"]
                    rv, rk = r["release_or_instability"], r["release_or_instability_known_mask"]

                    # Head counts
                    if gk:
                        if gv: grasp_pos += 1; ep_grasp += 1
                        else: grasp_neg += 1
                    else: grasp_unk += 1

                    if mk:
                        if mv: manip_pos += 1; ep_manip += 1
                        else: manip_neg += 1
                    else: manip_unk += 1

                    if rk:
                        if rv: release_pos += 1; ep_release += 1
                        else: release_neg += 1
                    else: release_unk += 1

                    # Logical violations
                    if mv and not gv:
                        logical_violations += 1
                    if mk and not gv:
                        logical_violations += 1
                    if supported and not gk:
                        unsupported_known += 1

                    # Grasp segment tracking
                    if gv and not prev_grasp:
                        if cur_seg > 0:
                            grasp_segments.append(cur_seg)
                        cur_seg = 1
                    elif gv:
                        cur_seg += 1
                    elif cur_seg > 0:
                        grasp_segments.append(cur_seg)
                        cur_seg = 0
                    prev_grasp = gv

                    # Event tracking
                    eid = r.get("event_id", -1)
                    if eid >= 0:
                        ep_events.add(eid)
                    if r.get("event_attribution_conflict"):
                        event_obj_conflicts += 1

                if cur_seg > 0:
                    grasp_segments.append(cur_seg)
                if ep_release > 0:
                    release_eps.add(identity)

                per_task[tk]["eps"] += 1
                per_task[tk]["grasp_pos"] += ep_grasp
                per_task[tk]["manip_pos"] += ep_manip
                per_task[tk]["release_pos"] += ep_release
                per_task[tk]["events"] += len(ep_events)
                per_task[tk]["has_positive"] += 1 if (ep_grasp > 0 and ep_manip > 0) else 0
                per_route[route]["eps"] += 1
                per_route[route]["events"] += len(ep_events)
                per_route[route]["grasp_pos"] += ep_grasp
                per_route[route]["manip_pos"] += ep_manip
                per_route[route]["tasks"].add(tk)
                per_route[route]["has_pos"] += 1 if (ep_grasp > 0 and ep_manip > 0) else 0

            # Per-suite
            per_suite[suite]["eps"] += 1

    # ── Gate checks ──────────────────────────────────────────────────────

    # G1: Logical violations
    if logical_violations > 0:
        errors.append(f"LOGICAL_VIOLATION: {logical_violations} (manipulation without grasp)"); hold = True

    # G2: Unsupported known labels
    if unsupported_known > 0:
        errors.append(f"UNSUPPORTED_KNOWN: {unsupported_known}"); hold = True

    # G3: Event object conflicts
    if event_obj_conflicts > 0:
        errors.append(f"EVENT_OBJ_CONFLICT: {event_obj_conflicts}"); hold = True

    # G4: Three heads all have positive + negative
    for name, pos, neg in [("grasp", grasp_pos, grasp_neg), ("manipulation", manip_pos, manip_neg), ("release", release_pos, release_neg)]:
        if pos == 0 or neg == 0:
            errors.append(f"HEAD_DEGENERATE: {name} pos={pos} neg={neg}"); hold = True

    # G5: grasp positive segment median >= 10
    if grasp_segments:
        seg_sorted = sorted(grasp_segments)
        median = seg_sorted[len(seg_sorted) // 2]
        if median < 10:
            errors.append(f"GRASP_SEGMENT_MEDIAN: {median} < 10"); hold = True
    else:
        errors.append("GRASP_SEGMENT_MEDIAN: no segments"); hold = True

    # G6: release positive episodes >= 20
    if len(release_eps) < 20:
        errors.append(f"RELEASE_EPS: {len(release_eps)} < 20"); hold = True

    # G7: Per-route checks
    for route in SUPPORTED_ROUTES:
        rd = per_route[route]
        n_tasks = len(rd["tasks"])
        n_events = rd["events"]
        n_pos_eps = rd["has_pos"]
        if n_tasks < 3:
            errors.append(f"ROUTE_TASKS: {route} has {n_tasks} tasks < 3"); hold = True
        if n_events < 30:
            errors.append(f"ROUTE_EVENTS: {route} has {n_events} events < 30"); hold = True

    # G8: Single task ≤ 35% of route positive
    for route in SUPPORTED_ROUTES:
        total_pos = per_route[route]["manip_pos"] + per_route[route]["grasp_pos"]
        for tk in per_route[route]["tasks"]:
            task_pos = per_task[tk]["grasp_pos"] + per_task[tk]["manip_pos"]
            if total_pos > 0 and task_pos / total_pos > 0.35:
                errors.append(f"TASK_OVERFIT: {tk} has {100*task_pos/total_pos:.1f}% of {route} positive > 35%"); hold = True

    # ── Write sealed output ──────────────────────────────────────────────
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    status = "HOLD" if hold else "PASS_FINAL_STUDENT_TRAINING"

    audit = {
        "schema": "DETECTOR_V5_F3_GEOMETRY_AUDIT_V1",
        "status": status,
        "teacher_root": str(root),
        "teacher_root_seal": teacher_seal,
        "teacher_manifest_schema": manifest.get("schema"),
        "teacher_source_commit": manifest.get("source_git_commit"),
        "identity_count": n_ids,
        "step_count": n_steps,
        "logical_violations": logical_violations,
        "unsupported_known_labels": unsupported_known,
        "event_object_conflicts": event_obj_conflicts,
        "head_geometry": {
            "grasp": {"positive": grasp_pos, "negative": grasp_neg, "unknown": grasp_unk},
            "manipulation": {"positive": manip_pos, "negative": manip_neg, "unknown": manip_unk},
            "release": {"positive": release_pos, "negative": release_neg, "unknown": release_unk},
        },
        "grasp_segment_count": len(grasp_segments),
        "grasp_segment_median": sorted(grasp_segments)[len(grasp_segments)//2] if grasp_segments else 0,
        "release_positive_episodes": len(release_eps),
        "per_route": {r: {"tasks": len(d["tasks"]), "events": d["events"], "positive_eps": d["has_pos"],
                           "grasp_pos": d["grasp_pos"], "manip_pos": d["manip_pos"]}
                      for r, d in per_route.items()},
        "errors": errors,
        "formal_training_authorized": not hold,
        "formal_attack_authorized": False,
    }
    _atomic_text(staging / "geometry_audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")

    # Per-task CSV
    with (staging / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite","task","eps","grasp_pos","manip_pos","release_pos","events","has_pos"])
        w.writeheader()
        for (suite, tn), s in sorted(per_task.items()):
            w.writerow({"suite": suite, "task": tn, **{k: s[k] for k in ["eps","grasp_pos","manip_pos","release_pos","events","has_pos"]}})

    # Source bindings
    _atomic_text(staging / "source_bindings.json", json.dumps({
        "schema": "DETECTOR_V5_F3_SOURCE_BINDINGS_V1",
        "teacher_root": str(root),
        "teacher_root_seal": teacher_seal,
    }, indent=2, sort_keys=True) + "\n")

    seal_sha = write_seal(staging)
    os.replace(staging, output)

    print(f"F3 STATUS: {status}")
    print(f"  identities: {n_ids}  steps: {n_steps}")
    print(f"  grasp: +{grasp_pos} -{grasp_neg} ?{grasp_unk}")
    print(f"  manipulation: +{manip_pos} -{manip_neg} ?{manip_unk}")
    print(f"  release: +{release_pos} -{release_neg} ?{release_unk}")
    print(f"  grasp segments: {len(grasp_segments)} median={audit['grasp_segment_median']}")
    print(f"  release episodes: {len(release_eps)}")
    print(f"  logical violations: {logical_violations}")
    print(f"  event obj conflicts: {event_obj_conflicts}")
    print(f"  output: {output}")
    print(f"  seal: {seal_sha}")
    if errors:
        print(f"  ERRORS:")
        for e in errors:
            print(f"    {e}")
    return 1 if hold else 0


if __name__ == "__main__":
    raise SystemExit(main())
