#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import yaml

GATE = "C6_1R0_SCATTERED_MODEL_PATH_CONFIG_MATERIALIZATION"
PASS = "PASS_SCATTERED_MODEL_PATH_CONFIG_MATERIALIZED"
OUT_FILES = ["v4_attack_scattered_model_paths.yaml", "scattered_model_path_config_report.json", "scattered_model_path_audit.csv", "checksum_report.json"]
ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
REQUIRED_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


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


def expand_path(text):
    return os.path.expandvars(os.path.expanduser(str(text or "")))


def unresolved_env_vars(text):
    return sorted({a or b for a, b in ENV_REF_RE.findall(str(text or ""))})


def parse_suite_path(items):
    out = {}
    for item in list(items or []):
        if "=" not in str(item):
            raise ValueError(f"expected SUITE=PATH, got {item!r}")
        suite, path = str(item).split("=", 1)
        suite = suite.strip()
        path = path.strip()
        if suite not in REQUIRED_SUITES:
            raise ValueError(f"unknown suite {suite!r}; expected one of {REQUIRED_SUITES}")
        if not path:
            raise ValueError(f"empty path for suite {suite!r}")
        out[suite] = path
    return out


def build_rows(model_paths, require_exists):
    rows = []
    for suite in REQUIRED_SUITES:
        raw = str(model_paths.get(suite, ""))
        expanded = expand_path(raw)
        unresolved = unresolved_env_vars(expanded)
        exists = False if unresolved else Path(expanded).exists()
        rows.append({
            "suite": suite,
            "raw_model_path": raw,
            "expanded_model_path": expanded,
            "unresolved_env_vars": ";".join(unresolved),
            "env_unresolved": bool(unresolved),
            "model_path_exists": exists,
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
    status = PASS
    reason = ""
    base_cfg = {}
    rows = []
    output_config = out / "v4_attack_scattered_model_paths.yaml"
    try:
        base_cfg = load_yaml(args.base_attack_config)
        if not isinstance(base_cfg, dict):
            raise TypeError("base attack config must be a YAML mapping")
        overrides = parse_suite_path(args.suite_model_path)
        missing = [suite for suite in REQUIRED_SUITES if suite not in overrides]
        if missing:
            raise KeyError(f"missing required suite model path overrides: {missing}")
        cfg = dict(base_cfg)
        model_paths = dict(cfg.get("model_paths", {}) or {})
        for suite, path in overrides.items():
            model_paths[suite] = expand_path(path)
        cfg["model_paths"] = model_paths
        rows = build_rows(model_paths, args.require_exists)
        if any(row["env_unresolved"] for row in rows):
            status = "HOLD_SCATTERED_MODEL_PATH_ENV_UNRESOLVED"
            reason = json.dumps(rows, sort_keys=True)
        elif args.require_exists and any(not row["model_path_exists"] for row in rows):
            status = "HOLD_SCATTERED_MODEL_PATHS_MISSING"
            reason = json.dumps(rows, sort_keys=True)
        write_yaml(output_config, cfg)
    except Exception as exc:
        status = "HOLD_SCATTERED_MODEL_PATH_CONFIG_FAILED"
        reason = f"{type(exc).__name__}: {exc}"

    write_csv(out / "scattered_model_path_audit.csv", rows, ["suite", "raw_model_path", "expanded_model_path", "unresolved_env_vars", "env_unresolved", "model_path_exists", "require_exists"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "base_attack_config": str(args.base_attack_config),
        "output_attack_config": str(output_config),
        "model_path_rows": rows,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
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
    write_json(out / "scattered_model_path_config_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-attack-config", default="configs/v4_attack.yaml")
    p.add_argument("--suite-model-path", action="append", default=[], help="Repeat as SUITE=PATH; suites: libero_spatial, libero_object, libero_goal, libero_10")
    p.add_argument("--require-exists", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
