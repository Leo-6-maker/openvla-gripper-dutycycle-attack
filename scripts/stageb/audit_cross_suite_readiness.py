#!/usr/bin/env python3
"""CPU-only readiness checks for the SC5 cross-suite protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REQUIRED_FALSE_FLAGS = [
    "uses_target_suite_training_data",
    "uses_target_suite_normalization",
    "uses_online_privileged_state",
    "uses_attack_outcome_for_selection",
    "uses_manual_anchor_for_trigger",
]

REQUIRED_FEATURE_COUNT = 25


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        data = parse_simple_yaml(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def parse_simple_yaml(text: str) -> dict[str, Any]:
    def parse_value(raw: str):
        raw = raw.strip()
        if raw in {"true", "false"}:
            return raw == "true"
        if raw.startswith("[") and raw.endswith("]"):
            body = raw[1:-1].strip()
            if not body:
                return []
            return [parse_value(part.strip()) for part in body.split(",")]
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw

    raw_lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    root: dict[str, Any] = {}
    stack: list[tuple[int, object]] = [(-1, root)]
    for idx, line in enumerate(raw_lines):
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"list item without list parent: {line}")
            parent.append(parse_value(content[2:]))
            continue
        key, sep, rest = content.partition(":")
        if not sep:
            raise ValueError(f"unsupported YAML line: {line}")
        key = key.strip()
        rest = rest.strip()
        if rest:
            value = parse_value(rest)
        else:
            next_content = ""
            for future in raw_lines[idx + 1:]:
                next_indent = len(future) - len(future.lstrip(" "))
                if next_indent > indent:
                    next_content = future.strip()
                    break
            value = [] if next_content.startswith("- ") else {}
        if not isinstance(parent, dict):
            raise ValueError(f"mapping entry without mapping parent: {line}")
        parent[key] = value
        if isinstance(value, (dict, list)):
            stack.append((indent, value))
    return root


def audit_protocol(protocol: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    detector = protocol.get("detector", {})
    features = list(detector.get("feature_order", []))
    rows.append(check("feature_count_25", len(features) == REQUIRED_FEATURE_COUNT, f"count={len(features)}"))
    rows.append(check("checkpoint_sha_present", bool(detector.get("checkpoint_sha256")), "detector checkpoint sha"))
    rows.append(check("dataset_sha_present", bool(detector.get("dataset_sha256")), "detector dataset sha"))
    rows.append(check("normalization_source_object_train", detector.get("normalization_source") == "object_train_only_checkpoint_mean_std", str(detector.get("normalization_source"))))

    for flag in REQUIRED_FALSE_FLAGS:
        rows.append(check(flag, detector.get(flag) is False, f"value={detector.get(flag)!r}"))

    runtime = protocol.get("runtime_freeze", {})
    rows.append(check("k_equals_10", runtime.get("k_attack_frames") == 10, f"K={runtime.get('k_attack_frames')}"))
    rows.append(check("target_token_31744", runtime.get("target_token_id") == 31744, f"target={runtime.get('target_token_id')}"))
    rows.append(check("pgd_steps_20", runtime.get("pgd_steps") == 20, f"steps={runtime.get('pgd_steps')}"))
    rows.append(check("epsilon_6_over_255", abs(float(runtime.get("vis_epsilon", -1)) - (6 / 255)) < 1e-12, f"epsilon={runtime.get('vis_epsilon')}"))

    policies = protocol.get("victim_policies", {})
    for suite in ["libero_spatial", "libero_goal", "libero_10"]:
        item = policies.get(suite, {})
        rows.append(check(f"{suite}_checkpoint_declared", bool(item.get("checkpoint_path")), str(item.get("checkpoint_path", ""))))
        rows.append(check(f"{suite}_unnorm_key_declared", item.get("unnorm_key") == suite, str(item.get("unnorm_key", ""))))

    return rows


def check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/sc5_cross_suite_protocol_v1.yaml")
    ap.add_argument("--output_json", default="")
    ap.add_argument("--output_csv", default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--manifest_only", action="store_true")
    ap.add_argument("--no_gpu", action="store_true")
    args = ap.parse_args(argv)
    if not args.no_gpu:
        raise SystemExit("--no_gpu is required for readiness audit")

    rows = audit_protocol(load_yaml(Path(args.protocol)))
    status = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    result = {"status": status, "checks": rows}
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
