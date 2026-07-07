#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

GATE = "D1C3_CLEAN2000_TEACHER_LABEL_V2_COMPLETE_MANIFEST_MATERIALIZATION"
PASS = "PASS_CLEAN2000_TEACHER_LABEL_V2_COMPLETE_MANIFEST_MATERIALIZED"
ACCEPTED_LIBERO10_QUALITY = {"PASS_LIBERO10_SEGMENT_CANDIDATE_QUALITY_AUDITED", "PASS_LIBERO10_SEGMENT_CANDIDATE_QUALITY_PADDING_AWARE_AUDITED"}
ACCEPTED_SINGLE_EVENT = {"PASS_CLEAN2000_SINGLE_EVENT_ANCHORS_RESOLVED_FROM_BINDINGS"}
OUT_FILES = ["clean2000_teacher_labels_v2_complete_manifest_report.json", "clean2000_teacher_labels_v2_complete_manifest.csv", "clean2000_teacher_labels_v2_complete_summary_by_suite.csv", "clean2000_teacher_labels_v2_complete_coverage_by_record.csv", "checksum_report.json"]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
ID_FIELDS = ["parent_id", "episode_key", "run_id", "record_id", "id"]
TEXT_FIELDS = ["suite", "suite_name", "benchmark", "libero_suite", "task_id", "task_name", "instruction", "language_instruction", "path", "output_root", "state_id"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows=[]
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row=dict(row); row["__source_line"]=line_no; rows.append(row)
        return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def first(row: Dict[str, Any], keys: List[str], default: str="") -> str:
    for k in keys:
        if row.get(k) not in (None, ""): return str(row[k])
    return default


def rid(row: Dict[str, Any]) -> str:
    return first(row, ID_FIELDS, f"row_{row.get('__source_line','')}")


def infer_suite(row: Dict[str, Any]) -> str:
    direct = first(row, ["suite", "suite_name", "benchmark", "libero_suite"], "").strip()
    if direct in SUITES: return direct
    txt = " ".join(str(row.get(k,"") or "") for k in TEXT_FIELDS+ID_FIELDS).lower()
    if "libero_10" in txt or "libero-10" in txt or "libero10" in txt or "moka" in txt: return "libero_10"
    if "libero_spatial" in txt or "spatial" in txt or "black_bowl" in txt: return "libero_spatial"
    if "libero_goal" in txt or "goal" in txt or "drawer" in txt: return "libero_goal"
    if "libero_object" in txt or "object" in txt or "alphabet_soup" in txt: return "libero_object"
    return "UNKNOWN"


def write_checksums(out: Path) -> None:
    reported = {name: sha256_file(out/name) for name in OUT_FILES[:-1] if (out/name).exists()}
    write_json(out/"checksum_report.json", {"algorithm":"sha256", "reported_files":reported, "self_referential_checksum_fields":"ABSENT_BY_DESIGN"})
    present=[n for n in OUT_FILES if (out/n).exists()]
    sums=out/"SHA256SUMS"; sums.write_text("".join(f"{sha256_file(out/n)}  {n}\n" for n in present), encoding="utf-8")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def normalize_single(row: Dict[str, Any]) -> Dict[str, Any]:
    out=dict(row); out["teacher_label_v2_source"]="single_event_anchor_resolver"; return out


def normalize_libero10(row: Dict[str, Any]) -> Dict[str, Any]:
    out={
        "record_id":row.get("record_id",""), "suite":row.get("suite","libero_10"), "task_id":row.get("task_id",""), "task_name":row.get("task_name",""),
        "event_id":row.get("event_id",""), "segment_id":row.get("segment_id",""), "segment_index":row.get("segment_index",""),
        "segment_start_step":row.get("segment_start_step",""), "segment_end_step":row.get("segment_end_step",""), "teacher_anchor_step":row.get("teacher_anchor_step",""),
        "teacher_window_start":row.get("teacher_window_start",""), "teacher_window_end":row.get("teacher_window_end",""), "segment_role":row.get("segment_role",""),
        "subtask_index":row.get("subtask_index",""), "event_role":row.get("event_role",""), "event_status":row.get("event_status",""),
        "phase_label":row.get("phase_label",""), "corridor_label":row.get("corridor_label",""), "release_safe_label":row.get("release_safe_label",""),
        "label_confidence":row.get("label_confidence",""), "label_source":"d1a_libero10_segment_candidate_resolver_d1b2_quality_audited", "source_status":"SEGMENT_RESOLVED", "resolver_method":row.get("resolver_method",""), "resolver_signal_field":row.get("resolver_signal_field",""), "artifact_dir":row.get("artifact_dir",""), "online_input_allowed":"NO_LABEL_ONLY", "teacher_label_v2_source":"libero10_segment_resolver"
    }
    return out


def run(args: argparse.Namespace) -> int:
    out=Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    source=read_csv(Path(args.clean2000_records)); libero=read_csv(Path(args.libero10_segments)); single=read_csv(Path(args.single_event_labels))
    q=json.loads(Path(args.libero10_quality_report).read_text())
    srep=json.loads(Path(args.single_event_report).read_text())
    qstatus=str(q.get("status","")); sstatus=str(srep.get("status",""))
    if qstatus not in ACCEPTED_LIBERO10_QUALITY and not args.allow_unverified_inputs: raise SystemExit(f"bad LIBERO-10 quality status: {qstatus}")
    if sstatus not in ACCEPTED_SINGLE_EVENT and not args.allow_unverified_inputs: raise SystemExit(f"bad single-event status: {sstatus}")
    labels=[normalize_libero10(r) for r in libero] + [normalize_single(r) for r in single]
    source_ids={rid(r): infer_suite(r) for r in source}; covered=defaultdict(int)
    for row in labels: covered[row.get("record_id","")] += 1
    coverage=[]; missing=[]
    for r,suite in sorted(source_ids.items()):
        n=covered.get(r,0); coverage.append({"record_id":r,"suite":suite,"label_rows":n,"covered":int(n>0)})
        if n<=0: missing.append(r)
    by_suite=defaultdict(Counter)
    for row in labels:
        suite=row.get("suite",""); by_suite[suite]["label_rows"] += 1; by_suite[suite][row.get("event_status","")] += 1; by_suite[suite][row.get("event_role","")] += 1
    for r,suite in source_ids.items(): by_suite[suite]["source_records"] += 1
    for r in missing: by_suite[source_ids.get(r,"UNKNOWN")]["missing_records"] += 1
    summary=[]
    for suite in SUITES:
        c=by_suite.get(suite, Counter()); summary.append({"suite":suite,"source_records":c.get("source_records",0),"label_rows":c.get("label_rows",0),"valid_primary":c.get("VALID_PRIMARY_CANDIDATE",0),"auxiliary":c.get("auxiliary_manipulation",0),"no_event":c.get("NO_EVENT",0),"missing_records":c.get("missing_records",0)})
    status=PASS if len(source)==args.expected_total and len(missing)==0 else "HOLD_TEACHER_LABEL_V2_COMPLETE_COVERAGE_MISMATCH"
    reason="" if status==PASS else f"source={len(source)} expected={args.expected_total} missing={len(missing)}"
    fields=["record_id","suite","task_id","task_name","event_id","segment_id","segment_index","segment_start_step","segment_end_step","teacher_anchor_step","teacher_window_start","teacher_window_end","segment_role","subtask_index","event_role","event_status","phase_label","corridor_label","release_safe_label","label_confidence","label_source","source_status","resolver_method","resolver_signal_field","artifact_dir","online_input_allowed","teacher_label_v2_source"]
    write_csv(out/"clean2000_teacher_labels_v2_complete_manifest.csv", labels, fields)
    write_csv(out/"clean2000_teacher_labels_v2_complete_summary_by_suite.csv", summary, ["suite","source_records","label_rows","valid_primary","auxiliary","no_event","missing_records"])
    write_csv(out/"clean2000_teacher_labels_v2_complete_coverage_by_record.csv", coverage, ["record_id","suite","label_rows","covered"])
    report={"gate":GATE,"status":status,"reason":reason,"clean2000_records":args.clean2000_records,"clean2000_records_sha256":sha256_file(Path(args.clean2000_records)),"libero10_segments":args.libero10_segments,"libero10_segments_sha256":sha256_file(Path(args.libero10_segments)),"libero10_quality_report":args.libero10_quality_report,"libero10_quality_status":qstatus,"single_event_labels":args.single_event_labels,"single_event_labels_sha256":sha256_file(Path(args.single_event_labels)),"single_event_report":args.single_event_report,"single_event_status":sstatus,"expected_total":args.expected_total,"source_record_count":len(source),"label_row_count":len(labels),"covered_record_count":sum(1 for x in coverage if x["covered"]),"missing_record_count":len(missing),"summary_by_suite":summary,"created_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"interpretation":"CPU-only complete teacher-label-v2 manifest combining audited LIBERO-10 segment labels and resolved single-event suite labels. PASS is a label-manifest gate only, not a detector dataset or training gate.","boundaries":{"CUDA_required":"NOT_REQUIRED","OpenVLA_model":"NOT_LOADED","model_inference":"NOT_PERFORMED","LIBERO_runtime":"NOT_PERFORMED","env_step":"NOT_PERFORMED","rollout":"NOT_PERFORMED","intervention":"NOT_PERFORMED","detector_training":"NOT_PERFORMED"},"git_commit":args.git_commit,"files_changed":args.files_changed,"tests":args.tests}
    write_json(out/"clean2000_teacher_labels_v2_complete_manifest_report.json", report); write_checksums(out); print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--clean2000-records", required=True); p.add_argument("--libero10-segments", required=True); p.add_argument("--libero10-quality-report", required=True); p.add_argument("--single-event-labels", required=True); p.add_argument("--single-event-report", required=True); p.add_argument("--expected-total", type=int, default=2000); p.add_argument("--allow-unverified-inputs", action="store_true"); p.add_argument("--output-root", required=True); p.add_argument("--git-commit", required=True); p.add_argument("--files-changed", action="append", default=[]); p.add_argument("--tests", action="append", default=[]); return run(p.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
