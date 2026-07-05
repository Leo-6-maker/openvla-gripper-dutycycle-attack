#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import py_compile
from pathlib import Path

import yaml

GATE = "C6_1R0_PRE_GPU_SINGLE_CLEAN_DECODE_BOUNDARY_AUDIT"
ACCEPTED_C6_1Q = {
    "PASS_MULTISUITE_RESET_OBSERVATION_BINDING_AUDITED",
    "PASS_RESET_STATE_AND_PREPROCESS_STABLE_RAW_CAMERA_NONBITWISE_AUDITED",
    "PASS_RESET_STATE_STABLE_PREPROCESS_OBSERVATION_NONBITWISE_AUDITED",
}
PASS = "PASS_PRE_GPU_SINGLE_CLEAN_DECODE_BOUNDARY_READY"
OUT_FILES = ["pre_gpu_single_clean_decode_boundary_audit.json", "task_model_path_audit.csv", "source_boundary_audit.csv", "checksum_report.json"]
REQUIRED_SOURCE_TERMS = [
    "load_model",
    "decode_with_scores",
    "resolve_unnorm_key",
    "postprocess_openvla_action_for_libero",
    "env.set_init_state",
    "env.step",
    "rollout",
    "attack_condition",
    "SINGLE_CLEAN_DECODE_ONLY",
]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_path(text):
    return os.path.expandvars(os.path.expanduser(str(text or "")))


def load_tasks(tasks_config, task_ids_text):
    cfg = load_yaml(tasks_config)
    rows = [dict(x) for x in list((cfg or {}).get("tasks", []))]
    selected = [x.strip() for x in str(task_ids_text or "").split(",") if x.strip()]
    if selected:
        order = {tid: i for i, tid in enumerate(selected)}
        rows = [r for r in rows if str(r.get("task_id")) in order]
        rows.sort(key=lambda r: order[str(r.get("task_id"))])
        missing = [tid for tid in selected if tid not in {str(r.get("task_id")) for r in rows}]
        if missing:
            raise KeyError(f"task ids not found in tasks config: {missing}")
    return rows


def compile_ok(path):
    try:
        py_compile.compile(str(path), doraise=True)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def validate_c6_1q(path, expected_sha):
    observed = sha256_file(path)
    if observed != expected_sha:
        return observed, "HOLD_C6_1Q_HASH_MISMATCH", observed
    obj = read_json(path)
    status = str(obj.get("status", ""))
    if status not in ACCEPTED_C6_1Q:
        return observed, "HOLD_C6_1Q_STATUS_NOT_ACCEPTED", status
    boundaries = dict(obj.get("boundaries") or {})
    if str(boundaries.get("OpenVLA_model", "")) != "NOT_LOADED":
        return observed, "HOLD_C6_1Q_BOUNDARY_MODEL_NOT_CLEAN", json.dumps(boundaries, sort_keys=True)
    if str(boundaries.get("env_step", "")) != "NOT_PERFORMED":
        return observed, "HOLD_C6_1Q_BOUNDARY_STEP_NOT_CLEAN", json.dumps(boundaries, sort_keys=True)
    return observed, "", ""


def scan_source(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = []
    for i, line in enumerate(text.splitlines(), start=1):
        hits = [term for term in REQUIRED_SOURCE_TERMS if term in line]
        if hits:
            rows.append({"path": str(path), "line": i, "matched_terms": ";".join(hits), "text": line.strip()[:500]})
    return rows


def source_boundary_ok(source_path):
    text = Path(source_path).read_text(encoding="utf-8", errors="replace")
    required_present = ["load_model", "decode_with_scores", "resolve_unnorm_key", "env.set_init_state", "SINGLE_CLEAN_DECODE_ONLY"]
    missing = [term for term in required_present if term not in text]
    forbidden_runtime = ["env.step(", "attacker.attack(", "OpenVLAVisualAttacker("]
    present_forbidden = [term for term in forbidden_runtime if term in text]
    return missing, present_forbidden


def model_path_rows(tasks, attack_config, explicit_model_path, require_exists):
    cfg = load_yaml(attack_config)
    paths = dict((cfg or {}).get("model_paths", {}) or {})
    rows = []
    for task in tasks:
        suite = str(task.get("suite", ""))
        raw = str(explicit_model_path or "").strip() or str(paths.get(suite) or paths.get("base") or paths.get("libero_goal") or "")
        expanded = expand_path(raw)
        rows.append({
            "task_id": str(task.get("task_id", "")),
            "suite": suite,
            "task_name": str(task.get("task_name", "")),
            "default_unnorm_key": str(task.get("default_unnorm_key", "")),
            "raw_model_path": raw,
            "expanded_model_path": expanded,
            "model_path_exists": Path(expanded).exists(),
            "require_exists": bool(require_exists),
        })
    return rows


def write_checksums(out):
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args):
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo_root)
    status = PASS
    reason = ""
    q_sha = ""

    if status == PASS:
        try:
            q_sha, hold, hold_reason = validate_c6_1q(args.input_c6_1q_json, args.expected_c6_1q_sha256)
            if hold:
                status = hold
                reason = hold_reason
        except Exception as exc:
            status = "HOLD_C6_1Q_VALIDATION_FAILED"
            reason = f"{type(exc).__name__}: {exc}"

    decode_tool = repo / "tools/multisuite_detector/prove_c6_multisuite_single_clean_decode_no_step_v1.py"
    compile_targets = [
        repo / "src/gripper_attack/openvla_preprocess.py",
        repo / "scripts/v4_run_eval_openvla.py",
        decode_tool,
    ]
    compile_rows = []
    if status == PASS:
        for target in compile_targets:
            ok, err = compile_ok(target)
            compile_rows.append({"path": str(target), "py_compile_ok": ok, "error": err})
            if not ok:
                status = "HOLD_PY_COMPILE_FAILED"
                reason = json.dumps(compile_rows, sort_keys=True)
                break

    source_rows = scan_source(decode_tool) if decode_tool.exists() else []
    if status == PASS:
        missing, forbidden = source_boundary_ok(decode_tool)
        if missing:
            status = "HOLD_DECODE_TOOL_SOURCE_REQUIRED_TERMS_MISSING"
            reason = json.dumps(missing)
        elif forbidden:
            status = "HOLD_DECODE_TOOL_SOURCE_FORBIDDEN_RUNTIME_TERMS_PRESENT"
            reason = json.dumps(forbidden)

    tasks = []
    model_rows = []
    if status == PASS:
        try:
            tasks = load_tasks(repo / args.tasks_config, args.task_ids)
            model_rows = model_path_rows(tasks, repo / args.attack_config, args.model_path, args.require_model_paths_exist)
            if args.require_model_paths_exist and any(not row["model_path_exists"] for row in model_rows):
                status = "HOLD_MODEL_PATHS_MISSING"
                reason = json.dumps(model_rows, sort_keys=True)
        except Exception as exc:
            status = "HOLD_CONFIG_OR_MODEL_PATH_AUDIT_FAILED"
            reason = f"{type(exc).__name__}: {exc}"

    write_csv(out / "source_boundary_audit.csv", source_rows, ["path", "line", "matched_terms", "text"])
    write_csv(out / "task_model_path_audit.csv", model_rows, ["task_id", "suite", "task_name", "default_unnorm_key", "raw_model_path", "expanded_model_path", "model_path_exists", "require_exists"])

    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "input_c6_1q_json_sha256": q_sha,
        "expected_c6_1q_json_sha256": args.expected_c6_1q_sha256,
        "repo_root": str(repo),
        "task_ids": [str(t.get("task_id")) for t in tasks],
        "model_path_rows": model_rows,
        "compile_rows": compile_rows,
        "source_audit_match_count": len(source_rows),
        "next_gate": "C6_1R_MULTISUITE_SINGLE_CLEAN_DECODE_NO_STEP",
        "interpretation": "Pre-GPU boundary only. This validates C6_1R source/config/model-path readiness before loading OpenVLA or using LIBERO/model runtime.",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED_FOR_THIS_AUDIT",
            "OpenVLA_model": "NOT_LOADED",
            "model_inference": "NOT_PERFORMED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_reset": "NOT_PERFORMED",
            "env_set_init_state": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
        },
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "pre_gpu_single_clean_decode_boundary_audit.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1q-json", required=True)
    p.add_argument("--expected-c6-1q-sha256", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--tasks-config", default="configs/v4_tasks_libero.yaml")
    p.add_argument("--attack-config", default="configs/v4_attack.yaml")
    p.add_argument("--task-ids", default="libero_spatial_black_bowl,libero_object_alphabet_soup,libero_goal_open_middle_drawer,libero10_moka_pots")
    p.add_argument("--model-path", default="")
    p.add_argument("--require-model-paths-exist", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
