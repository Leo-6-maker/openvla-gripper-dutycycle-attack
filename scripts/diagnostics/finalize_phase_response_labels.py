#!/usr/bin/env python3
"""Build vulnerability-ready labels from multi-batch VIS summary CSVs.

This is a CPU-only label builder. It does not run rollouts, VIS, GPU work, or
detector training. Detector training should run only after this builder and the
schema audit both pass.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict


ALLOWED_STATUS = {"positive", "negative", "ignore", "manual_review"}
TRAIN_STATUS = {"positive", "negative"}
CONTROL_ROLES = {
    "stable_post_lock": "stable_post_lock_control",
    "stable_post_lock_control": "stable_post_lock_control",
    "far_too_early": "far_too_early_control",
    "far_too_early_control": "far_too_early_control",
    "pre_lock": "pre_lock_control",
    "pre_lock_control": "pre_lock_control",
}
BLOCKED_TOKENS = {
    "polluted",
    "random_failed",
    "denominator_failed",
    "infra_failed",
    "xid",
    "oom",
    "missing_trace",
    "provenance_failed",
    "schema_incomplete",
    "ambiguous_merge",
}
LABEL_FIELDS = [
    "source_batch",
    "task_key",
    "state_id",
    "window_start",
    "window_end",
    "candidate_role",
    "control_type",
    "phase_bin_proxy",
    "denominator_type",
    "provenance_status",
    "provenance_note",
    "reason_selected",
    "VIS_OPEN",
    "vis_open_count",
    "qpos_opening_delta",
    "qpos_label",
    "done",
    "taxonomy",
    "denominator_clean",
    "claim_usable",
    "action_bridge_confounded",
    "label_action_bridge",
    "label_physical_response",
    "label_task_failure",
    "label_vulnerability_ready",
    "label_status",
    "label_use",
    "exclusion_or_uncertain_reason",
]
CANDIDATE_FILL_FIELDS = [
    "candidate_role",
    "control_type",
    "phase_bin_proxy",
    "denominator_type",
    "action_bridge_confounded",
    "reason_selected",
]
CONFLICT_FIELDS = [
    "task_key",
    "state_id",
    "window_start",
    "window_end",
    "sources",
    "labels",
    "statuses",
    "roles",
    "reason",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch1-merged", default="tables/object_teacher_delay50_vis_smoke_merged_summary.csv")
    ap.add_argument("--batch2b-vis", default="tables/object_phase_response_batch2b_vis_summary.csv")
    ap.add_argument("--batch2b-provenance", default="tables/object_phase_response_batch2b_vis_provenance.csv")
    ap.add_argument("--batch3-vis", default="tables/object_phase_response_batch3_vis_summary.csv")
    ap.add_argument("--batch3b-vis", default="")
    ap.add_argument("--batch3c-vis", default="")
    ap.add_argument("--batch2b-candidates", default="")
    ap.add_argument("--batch3-candidates", default="")
    ap.add_argument("--batch3b-candidates", default="")
    ap.add_argument("--batch3c-candidates", default="")
    ap.add_argument("--descriptors", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--output-labels", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--output-readiness", default="reports/OBJECT_PHASE_RESPONSE_LABEL_READINESS_V2.md")
    ap.add_argument("--output-conflicts", default="tables/object_phase_response_label_conflicts_v2.csv")
    ap.add_argument("--output-metrics", default="tables/vulnerability_ready_smoke_metrics_v1.csv",
                    help="Kept for CLI compatibility; this builder does not train detector models.")
    ap.add_argument("--output-predictions", default="tables/vulnerability_ready_smoke_predictions_v1.csv",
                    help="Kept for CLI compatibility; this builder does not train detector models.")
    ap.add_argument("--output-report", default="reports/VULNERABILITY_READY_SMOKE_DETECTOR_V1.md",
                    help="Kept for CLI compatibility; this builder does not train detector models.")
    ap.add_argument("--use-frozen-batch2b", action="store_true",
                    help="Use verified 9-outcome hardcoded set for legacy reproduction.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def parse_bool(value, default=False):
    v = lower(value)
    if v in {"true", "1", "yes", "y", "clean"}:
        return True
    if v in {"false", "0", "no", "n", "polluted", "polluted_or_incomplete", "failed"}:
        return False
    return default


def parse_float(value, default=0.0):
    try:
        v = norm(value)
        if v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_open(value, default=0):
    v = norm(value)
    if not v:
        return default
    if "/" in v:
        v = v.split("/", 1)[0]
    return int(parse_float(v, default))


def clean_role(value):
    role = lower(value)
    return CONTROL_ROLES.get(role, role)


def normalize_candidate_field(field, value):
    if field in {"candidate_role", "control_type"}:
        return clean_role(value)
    if field == "action_bridge_confounded":
        return str(parse_bool(value, False))
    return norm(value)


def first(row, *fields):
    for field in fields:
        if field in row and norm(row.get(field)) != "":
            return row.get(field)
    return ""


def join_key(row):
    return (
        norm(first(row, "task_key", "task")),
        norm(first(row, "state_id", "state")),
        norm(row.get("window_start")),
        norm(row.get("window_end")),
    )


def load_candidate_map(path):
    candidate_map = {}
    conflicts = []
    if not path or not os.path.exists(path):
        return candidate_map, conflicts
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = join_key(row)
            if not key[0] or key[2] == "" or key[3] == "":
                continue
            meta = {}
            for field in CANDIDATE_FILL_FIELDS:
                value = normalize_candidate_field(field, row.get(field))
                if value != "":
                    meta[field] = value
            if key in candidate_map:
                old = candidate_map[key]
                changed = {
                    field
                    for field in CANDIDATE_FILL_FIELDS
                    if old.get(field, "") and meta.get(field, "") and old.get(field) != meta.get(field)
                }
                if changed:
                    conflicts.append(metadata_conflict(key, "candidate_duplicate_metadata_conflict", old, meta))
                    continue
                old.update({k: v for k, v in meta.items() if v != ""})
            else:
                candidate_map[key] = meta
    return candidate_map, conflicts


def metadata_conflict(key, reason, summary_meta, candidate_meta):
    return {
        "task_key": key[0],
        "state_id": key[1],
        "window_start": key[2],
        "window_end": key[3],
        "sources": "summary|candidate",
        "labels": "",
        "statuses": "",
        "roles": "%s|%s" % (
            norm(summary_meta.get("candidate_role")),
            norm(candidate_meta.get("candidate_role")),
        ),
        "reason": reason,
    }


def merge_candidate_metadata(source, row, candidate_meta):
    merged = dict(row)
    conflicts = []
    key = join_key(row)
    summary_role = clean_role(first(row, "candidate_role", "control_type"))
    candidate_role = clean_role(candidate_meta.get("candidate_role")) if candidate_meta else ""
    if summary_role and candidate_role and summary_role != candidate_role:
        conflicts.append(metadata_conflict(key, "summary_candidate_role_conflict", {"candidate_role": summary_role}, candidate_meta))
    for field in CANDIDATE_FILL_FIELDS:
        if norm(merged.get(field)) == "" and candidate_meta.get(field, "") != "":
            merged[field] = candidate_meta[field]
    if source == "batch3c" and clean_role(merged.get("candidate_role")) == "":
        merged["_missing_candidate_role"] = "true"
    return merged, conflicts


def load_phase_map(path):
    phase_map = {}
    if not path or not os.path.exists(path):
        return phase_map
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (norm(r.get("task_key")), norm(r.get("state_id")), norm(r.get("window_start")), norm(r.get("window_end")))
            phase_map[key] = r
    return phase_map


def infer_done(row, taxonomy):
    if norm(row.get("done")) != "":
        return parse_bool(row.get("done"))
    if norm(row.get("VIS_done")) != "":
        return parse_bool(row.get("VIS_done"))
    if norm(row.get("vis_done_all_false")) != "":
        return not parse_bool(row.get("vis_done_all_false"))
    tax = lower(taxonomy)
    if "task_negative" in tax or "no_action" in tax:
        return True
    if "task_positive" in tax or "task_failure" in tax:
        return False
    return False


def denominator_clean(row):
    if norm(row.get("denominator_clean")) != "":
        return parse_bool(row.get("denominator_clean"))
    if norm(row.get("denominator_status")) != "":
        return lower(row.get("denominator_status")) == "clean"
    if norm(row.get("random_all_clean")) != "":
        return parse_bool(row.get("random_all_clean"))
    return True


def provenance_status(row):
    status = first(row, "provenance_status", "provenance_note", "trace_status", "validity_status")
    return norm(status) if norm(status) else "unknown"


def blocked_reason(row, provenance):
    blob = " ".join(
        lower(row.get(f))
        for f in [
            "taxonomy",
            "taxonomy_label",
            "denominator_status",
            "provenance_status",
            "provenance_note",
            "validity_status",
            "stop_reason",
            "failure_reason",
            "exclusion_reason",
        ]
    )
    blob = (blob + " " + lower(provenance)).strip()
    hits = sorted(token for token in BLOCKED_TOKENS if token in blob)
    if hits:
        return "|".join(hits)
    return ""


def normalize_row(source, row, phase_map):
    task = norm(first(row, "task_key", "task"))
    state = norm(first(row, "state_id", "state"))
    ws = norm(row.get("window_start"))
    we = norm(row.get("window_end"))
    key = (task, state, ws, we)
    phase = phase_map.get(key, {})
    taxonomy = norm(first(row, "taxonomy_label", "taxonomy"))
    role = clean_role(first(row, "candidate_role", "control_type", "merge_type"))
    control_type = clean_role(first(row, "control_type", "candidate_role"))
    phase_bin = norm(first(row, "phase_bin_proxy", "phase_bin", "phase")) or norm(phase.get("phase_bin_proxy"))
    qpos = parse_float(first(row, "qpos_opening_delta", "qpos_delta", "vis_qpos_opening_delta_mean", "qpos"), 0.0)
    vis_open_raw = first(row, "VIS_OPEN", "vis_open", "vis_OPEN_mean", "vis_OPEN_min", "vis_open_count")
    vis_open_count = parse_open(vis_open_raw, 0)
    denom = denominator_clean(row)
    claim = parse_bool(row.get("claim_usable"), False)
    done = infer_done(row, taxonomy)
    prov = provenance_status(row)
    denom_type = norm(first(row, "denominator_type", "denominator_kind"))
    if not denom_type and role:
        denom_type = get_denominator_type_safe(role)
    if not denom_type:
        denom_type = "standard"
    return {
        "source_batch": source,
        "task_key": task,
        "state_id": state,
        "window_start": ws,
        "window_end": we,
        "candidate_role": role,
        "control_type": control_type,
        "phase_bin_proxy": phase_bin,
        "denominator_type": denom_type,
        "provenance_status": prov,
        "provenance_note": norm(first(row, "provenance_note", "notes")),
        "reason_selected": norm(row.get("reason_selected")),
        "VIS_OPEN": norm(vis_open_raw),
        "vis_open_count": vis_open_count,
        "qpos_opening_delta": qpos,
        "done": done,
        "taxonomy": taxonomy,
        "denominator_clean": denom,
        "claim_usable": claim,
        "action_bridge_confounded_source": parse_bool(row.get("action_bridge_confounded"), False),
        "missing_candidate_role": parse_bool(row.get("_missing_candidate_role"), False),
        "_raw": row,
    }


def get_denominator_type_safe(role):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from role_specific_gates import get_denominator_type
        return get_denominator_type(role)
    except Exception:
        if role == "stable_post_lock_control":
            return "late_open_control"
        if role in {"far_too_early_control", "pre_lock_control"}:
            return "closed_window_control"
        return "standard"


def classify_role_specific(o):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from role_specific_gates import classify_vis_outcome
        return classify_vis_outcome(
            o["candidate_role"],
            int(o["vis_open_count"]),
            float(o["qpos_opening_delta"]),
            bool(o["done"]),
            bool(o["denominator_clean"]),
        )
    except Exception:
        return "", "manual_review", "role_specific_gate_error", False


def classify_standard(o):
    blocked = blocked_reason(o["_raw"], o["provenance_status"])
    if blocked:
        return "", "ignore", blocked, False
    if not o["denominator_clean"]:
        return "", "ignore", "denominator_not_clean", False
    action_pos = int(o["vis_open_count"]) >= 16 or "action_positive" in lower(o["taxonomy"])
    phys_strong = float(o["qpos_opening_delta"]) >= 0.03
    phys_weak = 0.01 <= float(o["qpos_opening_delta"]) < 0.03
    task_fail = not bool(o["done"])
    if o["claim_usable"]:
        if action_pos and phys_strong and task_fail:
            return 1, "positive", o["taxonomy"] or "claim_usable_positive", False
        return "", "manual_review", "claim_gate_inconsistent", False
    if action_pos and phys_strong and not task_fail:
        return 0, "negative", o["taxonomy"] or "physical_strong_task_negative", False
    if phys_strong and not task_fail:
        return 0, "negative", o["taxonomy"] or "physical_strong_task_negative", False
    if action_pos and phys_weak:
        return "", "ignore", "weak_physical_uncertain", False
    if not action_pos:
        return 0, "negative", o["taxonomy"] or "action_only", False
    return "", "manual_review", "unclassified_action_positive", False


def classify_outcome(o):
    if o.get("missing_candidate_role"):
        return "", "manual_review", "missing_candidate_role_for_batch3c_control", False
    if o["candidate_role"] in CONTROL_ROLES.values():
        return classify_role_specific(o)
    return classify_standard(o)


def load_csv_sources(args, phase_map):
    if args.use_frozen_batch2b:
        return frozen_batch2b_outcomes(), []
    sources = [
        ("batch1", args.batch1_merged, ""),
        ("batch2b", args.batch2b_vis, args.batch2b_candidates),
        ("batch3", args.batch3_vis, args.batch3_candidates),
        ("batch3b", args.batch3b_vis, args.batch3b_candidates),
        ("batch3c", args.batch3c_vis, args.batch3c_candidates),
    ]
    outcomes = []
    conflicts = []
    for source, path, candidate_path in sources:
        if not path or not os.path.exists(path):
            continue
        candidate_map, candidate_conflicts = load_candidate_map(candidate_path)
        conflicts.extend(candidate_conflicts)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merged, row_conflicts = merge_candidate_metadata(source, row, candidate_map.get(join_key(row), {}))
                conflicts.extend(row_conflicts)
                item = normalize_row(source, merged, phase_map)
                if item["task_key"] and item["window_start"] != "" and item["window_end"] != "":
                    outcomes.append(item)
    return outcomes, conflicts


def frozen_batch2b_outcomes():
    rows = [
        ("batch1", "alphabet_soup", "0", "3", "20", 18, 0.027619, False, False, True, "weak_physical_uncertain"),
        ("batch2b", "alphabet_soup", "2", "11", "28", 18, 0.037643, True, False, True, "action_physical_strong_task_positive"),
        ("batch2b", "bbq_sauce", "0", "25", "42", 18, 0.038055, False, True, True, "physical_strong_task_negative"),
        ("batch2b", "bbq_sauce", "4", "14", "31", 18, 0.037853, False, True, True, "physical_strong_task_negative"),
        ("batch1", "butter", "0", "29", "46", 18, 0.037905, True, False, True, "action_physical_strong_task_positive"),
        ("batch2b", "butter", "0", "32", "49", 18, 0.037934, False, True, True, "physical_strong_task_negative"),
        ("batch2b", "butter", "2", "23", "40", 18, 0.037462, True, False, True, "action_physical_strong_task_positive"),
        ("batch1", "ketchup", "0", "16", "33", 18, 0.038042, True, False, True, "action_physical_strong_task_positive"),
        ("batch2b", "ketchup", "1", "28", "45", 18, 0.037948, False, True, True, "physical_strong_task_negative"),
    ]
    outcomes = []
    for src, task, state, ws, we, vis_open, qpos, claim, done, denom, tax in rows:
        outcomes.append({
            "source_batch": src,
            "task_key": task,
            "state_id": state,
            "window_start": ws,
            "window_end": we,
            "candidate_role": "",
            "control_type": "",
            "phase_bin_proxy": "",
            "denominator_type": "standard",
            "provenance_status": "frozen_batch2b",
            "provenance_note": "",
            "reason_selected": "",
            "VIS_OPEN": str(vis_open),
            "vis_open_count": vis_open,
            "qpos_opening_delta": qpos,
            "done": done,
            "taxonomy": tax,
            "denominator_clean": denom,
            "claim_usable": claim,
            "action_bridge_confounded_source": False,
            "missing_candidate_role": False,
            "_raw": {},
        })
    return outcomes


def build_labels(outcomes):
    labels = []
    for o in outcomes:
        label, status, taxonomy, action_confounded = classify_outcome(o)
        if status not in ALLOWED_STATUS:
            status = "manual_review"
        if status not in TRAIN_STATUS:
            label = ""
        qpos = float(o["qpos_opening_delta"])
        labels.append({
            "source_batch": o["source_batch"],
            "task_key": o["task_key"],
            "state_id": o["state_id"],
            "window_start": o["window_start"],
            "window_end": o["window_end"],
            "candidate_role": o["candidate_role"],
            "control_type": o["control_type"],
            "phase_bin_proxy": o["phase_bin_proxy"],
            "denominator_type": o["denominator_type"],
            "provenance_status": o["provenance_status"],
            "provenance_note": o["provenance_note"],
            "reason_selected": o["reason_selected"],
            "VIS_OPEN": o["VIS_OPEN"],
            "vis_open_count": o["vis_open_count"],
            "qpos_opening_delta": round(qpos, 6),
            "qpos_label": "strong" if qpos >= 0.03 else ("weak" if qpos >= 0.01 else "none"),
            "done": o["done"],
            "taxonomy": taxonomy,
            "denominator_clean": o["denominator_clean"],
            "claim_usable": o["claim_usable"],
            "action_bridge_confounded": bool(action_confounded) or bool(o.get("action_bridge_confounded_source")),
            "label_action_bridge": 1 if int(o["vis_open_count"]) >= 16 else 0,
            "label_physical_response": 1 if qpos >= 0.03 else (0.5 if qpos >= 0.01 else 0),
            "label_task_failure": 0 if o["done"] else 1,
            "label_vulnerability_ready": label,
            "label_status": status,
            "label_use": "train" if status in TRAIN_STATUS else status,
            "exclusion_or_uncertain_reason": "" if status in TRAIN_STATUS else taxonomy,
        })
    return labels


def find_conflicts(labels):
    by_key = defaultdict(list)
    for row in labels:
        key = (row["task_key"], row["state_id"], row["window_start"], row["window_end"])
        by_key[key].append(row)
    conflicts = []
    for key, rows in by_key.items():
        states = {(norm(r["label_status"]), norm(r["label_vulnerability_ready"])) for r in rows}
        if len(states) <= 1:
            continue
        conflicts.append({
            "task_key": key[0],
            "state_id": key[1],
            "window_start": key[2],
            "window_end": key[3],
            "sources": "|".join(sorted(set(r["source_batch"] for r in rows))),
            "labels": "|".join(sorted(set(norm(r["label_vulnerability_ready"]) for r in rows))),
            "statuses": "|".join(sorted(set(norm(r["label_status"]) for r in rows))),
            "roles": "|".join(sorted(set(norm(r["candidate_role"]) for r in rows))),
            "reason": "duplicate_label_conflict",
        })
    return conflicts


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_readiness(path, labels, conflicts, output_labels, output_conflicts):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    status_counts = Counter(r["label_status"] for r in labels)
    source_counts = Counter(r["source_batch"] for r in labels)
    role_counts = Counter(r["candidate_role"] or "standard" for r in labels)
    train = [r for r in labels if r["label_status"] in TRAIN_STATUS]
    train_manual = [r for r in train if r["label_status"] == "manual_review"]
    blockers = []
    if conflicts:
        blockers.append("duplicate label conflicts present")
    if train_manual:
        blockers.append("manual_review rows entered train")
    if not labels:
        blockers.append("no labels built")
    verdict = "PASS" if not blockers else "FAIL"
    lines = [
        "# Object Phase Response Label Readiness V2",
        "",
        f"**Labels CSV**: `{output_labels}`",
        f"**Conflict CSV**: `{output_conflicts}`",
        f"**Rows**: {len(labels)}",
        f"**Train rows**: {len(train)}",
        f"**Verdict**: **{verdict}**",
        "",
        "## Blocking Issues",
        "",
    ]
    if blockers:
        for b in blockers:
            lines.append("- " + b)
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Label Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for k in sorted(status_counts):
        lines.append(f"| {k} | {status_counts[k]} |")
    lines += ["", "## Source Counts", "", "| Source | Count |", "|---|---:|"]
    for k in sorted(source_counts):
        lines.append(f"| {k} | {source_counts[k]} |")
    lines += ["", "## Role Counts", "", "| Role | Count |", "|---|---:|"]
    for k in sorted(role_counts):
        lines.append(f"| {k} | {role_counts[k]} |")
    lines += [
        "",
        "## Boundaries",
        "",
        "- Only positive/negative rows are train-eligible.",
        "- manual_review, ignore, polluted, random-failed, denominator-failed, infra-failed, Xid/OOM, missing-trace, provenance-failed, schema-incomplete, and ambiguous rows must not enter train.",
        "- This builder does not train detector v2.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    phase_map = load_phase_map(args.descriptors)
    outcomes, metadata_conflicts = load_csv_sources(args, phase_map)
    labels = build_labels(outcomes)
    conflicts = metadata_conflicts + find_conflicts(labels)
    write_csv(args.output_labels, labels, LABEL_FIELDS)
    write_csv(args.output_conflicts, conflicts, CONFLICT_FIELDS)
    write_readiness(args.output_readiness, labels, conflicts, args.output_labels, args.output_conflicts)
    print("Built %d labels -> %s" % (len(labels), args.output_labels))
    print("Conflicts: %d -> %s" % (len(conflicts), args.output_conflicts))
    print("Readiness: %s" % args.output_readiness)
    if conflicts:
        print("ERROR: duplicate label conflicts present")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
