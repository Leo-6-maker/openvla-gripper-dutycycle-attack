#!/usr/bin/env python3
"""Bind frozen Label V2 rows to a per-step SC5 25D feature CSV.

CPU-only metadata layer. It does not build detector rows, train, infer, or run
rollouts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.load_label_v2_artifact import load_label_v2_artifact  # noqa: E402

SCHEMA_VERSION = "feature_binding_manifest_v1"
FEATURE_CSV_SCHEMA = "clean2000_sc5_25d_all_steps_v1"
META_FIELDS = ["episode_key", "parent_key", "suite", "task_id", "trace_length"]
OUTPUTS = {
    "binding_manifest": "binding_manifest.json",
    "dataset_manifest": "dataset_manifest.json",
    "dataset_statistics": "dataset_statistics.json",
    "population_summary": "population_summary.csv",
    "feature_summary": "feature_summary.csv",
}


class FeatureBindingError(ValueError):
    pass


def fail(message: str) -> None:
    raise FeatureBindingError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_sc5_features() -> list[str]:
    source = REPO / "src" / "gripper_attack" / "sc5mlp_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SC5_FEATURES":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list) or len(value) != 25 or len(set(value)) != 25:
                        fail("SC5_FEATURES must be 25 unique names")
                    return value
    fail("SC5_FEATURES not found")


SC5_FEATURES = load_sc5_features()
FEATURE_SCHEMA_SHA256 = json_sha({"feature_names": SC5_FEATURES, "feature_count": 25})


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows: list[dict[str, str]] = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: extra cells")
            if any(v is None for v in row.values()):
                fail(f"{path.name}:{line_no}: missing cells")
            rows.append(row)
    return header, rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: str, field: str, episode: str) -> int:
    try:
        out = int(value)
    except ValueError:
        fail(f"{episode}: {field} must be int")
    if str(out) != value:
        fail(f"{episode}: {field} must be canonical int")
    return out


def parse_float(value: str, field: str, episode: str) -> float:
    try:
        out = float(value)
    except ValueError:
        fail(f"{episode}: {field} must be finite float")
    if not math.isfinite(out):
        fail(f"{episode}: {field} must be finite float")
    return out


def ensure_feature_order(header: list[str]) -> None:
    positions = []
    for feature in SC5_FEATURES:
        if feature not in header:
            fail(f"missing feature column: {feature}")
        positions.append(header.index(feature))
    if positions != sorted(positions):
        fail("SC5 feature ordering mismatch")


def label_population(row: dict[str, str]) -> str:
    if row["mechanism_eligible"] == "false":
        return "DETECTOR_SAFETY"
    return "DETECTOR_ELIGIBLE"


def load_feature_rows(feature_csv: str | Path) -> dict[str, object]:
    path = Path(feature_csv)
    header, rows = read_csv(path)
    if "episode_key" not in header or "step" not in header:
        fail("feature CSV must include episode_key and step")
    ensure_feature_order(header)
    by_episode: dict[str, dict[str, object]] = {}
    feature_stats = {name: {"missing": 0, "nan": 0, "inf": 0, "count": 0} for name in SC5_FEATURES}
    seen_steps: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        episode = row["episode_key"]
        if not episode:
            fail("feature row missing episode_key")
        step = parse_int(row["step"], "step", episode)
        if step < 0:
            fail(f"{episode}: negative step")
        if step in seen_steps[episode]:
            fail(f"{episode}: duplicate step")
        seen_steps[episode].add(step)
        record = by_episode.setdefault(episode, {"rows": [], "metadata": {}})
        for field in META_FIELDS[1:]:
            if field in row and row[field]:
                meta = record["metadata"]
                if field in meta and meta[field] != row[field]:
                    fail(f"{episode}: inconsistent {field} in feature CSV")
                meta[field] = row[field]
        for name in SC5_FEATURES:
            raw = row[name]
            if raw == "":
                feature_stats[name]["missing"] += 1
                fail(f"{episode}: missing {name}")
            value = parse_float(raw, name, episode)
            feature_stats[name]["count"] += 1
            if math.isnan(value):
                feature_stats[name]["nan"] += 1
            if math.isinf(value):
                feature_stats[name]["inf"] += 1
        record["rows"].append(row)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "header": header,
        "rows": rows,
        "episodes": by_episode,
        "row_count": len(rows),
        "feature_stats": feature_stats,
    }


def load_label(label_root: str | Path, expected_label_mode: str) -> dict[str, object]:
    label = load_label_v2_artifact(label_root, expected_mode=expected_label_mode)
    rows = label["label_rows"]
    seen = set()
    by_episode = {}
    for row in rows:
        ep = row["episode_key"]
        if ep in seen:
            fail(f"duplicate label episode_key: {ep}")
        seen.add(ep)
        by_episode[ep] = row
    return {**label, "by_episode": by_episode}


def build_binding(label_root: str | Path, feature_csv: str | Path, *, expected_label_mode: str = "formal-ledger-build") -> dict[str, object]:
    label = load_label(label_root, expected_label_mode)
    features = load_feature_rows(feature_csv)
    label_rows = label["by_episode"]
    feature_eps = set(features["episodes"])
    label_eps = set(label_rows)
    missing = sorted(label_eps - feature_eps)
    orphan = sorted(feature_eps - label_eps)
    if missing:
        fail(f"missing feature episodes: {missing[:5]}")
    if orphan:
        fail(f"orphan feature episodes: {orphan[:5]}")

    episodes = []
    suite_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    population_counts: Counter[str] = Counter()
    trace_hist: Counter[str] = Counter()
    for ep in sorted(label_eps):
        label_row = label_rows[ep]
        feature_record = features["episodes"][ep]
        trace = parse_int(label_row["trace_length"], "trace_length", ep)
        meta = feature_record["metadata"]
        for field in ["parent_key", "suite", "task_id"]:
            if field in meta and meta[field] != label_row[field]:
                fail(f"{ep}: {field} mismatch")
        if "trace_length" in meta and parse_int(meta["trace_length"], "trace_length", ep) != trace:
            fail(f"{ep}: trace_length mismatch")
        steps = sorted(parse_int(row["step"], "step", ep) for row in feature_record["rows"])
        if steps != list(range(trace)):
            fail(f"{ep}: steps must cover 0..trace_length-1")
        population = label_population(label_row)
        suite_counts[label_row["suite"]] += 1
        task_counts[f"{label_row['suite']}/{label_row['task_id']}"] += 1
        population_counts[population] += 1
        trace_hist[str(trace)] += 1
        episodes.append({
            "episode_key": ep,
            "parent_key": label_row["parent_key"],
            "suite": label_row["suite"],
            "task_id": label_row["task_id"],
            "trace_length": trace,
            "step_count": len(steps),
            "population_id": population,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "label_v2_artifact_root": str(label_root),
        "label_v2_artifact_sha256": sha256_file(Path(label_root) / "SHA256SUMS"),
        "label_v2_build_manifest_sha256": sha256_file(Path(label_root) / "build_manifest.json"),
        "label_v2_manual_audit_sha256": sha256_file(Path(label_root) / "manual_audit_sample_manifest.csv"),
        "label_report": label["report"],
        "feature_csv": str(feature_csv),
        "feature_csv_sha256": features["sha256"],
        "feature_csv_schema": FEATURE_CSV_SCHEMA,
        "feature_header": features["header"],
        "feature_names": list(SC5_FEATURES),
        "feature_count": len(SC5_FEATURES),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "episode_count": len(episodes),
        "step_count": features["row_count"],
        "population_counts": dict(population_counts),
        "suite_counts": dict(suite_counts),
        "task_counts": dict(task_counts),
        "trace_length_histogram": dict(trace_hist),
        "feature_stats": features["feature_stats"],
        "episodes": episodes,
        "formal_detector_dataset_build": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "gpu": "NOT_PERFORMED",
    }


def write_binding_manifest(label_root: str | Path, feature_csv: str | Path, output_json: str | Path, *, expected_label_mode: str = "formal-ledger-build") -> dict[str, object]:
    manifest = build_binding(label_root, feature_csv, expected_label_mode=expected_label_mode)
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_binding_manifest(manifest_json: str | Path, *, expected_label_mode: str = "formal-ledger-build") -> dict[str, object]:
    path = Path(manifest_json)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail("binding manifest schema_version mismatch")
    rebuilt = build_binding(manifest["label_v2_artifact_root"], manifest["feature_csv"], expected_label_mode=expected_label_mode)
    checks = [
        "label_v2_artifact_sha256", "label_v2_build_manifest_sha256", "label_v2_manual_audit_sha256",
        "feature_csv_sha256", "feature_schema_sha256", "feature_names", "feature_count",
        "episode_count", "step_count", "population_counts", "suite_counts", "task_counts",
        "trace_length_histogram",
    ]
    for key in checks:
        if manifest.get(key) != rebuilt.get(key):
            fail(f"binding manifest {key} mismatch")
    if manifest.get("episodes") != rebuilt.get("episodes"):
        fail("binding manifest episode rows mismatch")
    return {"status": "PASS", "binding_manifest": str(path), "episode_count": rebuilt["episode_count"], "step_count": rebuilt["step_count"]}


def write_metadata_outputs(binding_manifest_json: str | Path, output_root: str | Path) -> dict[str, object]:
    manifest = json.loads(Path(binding_manifest_json).read_text(encoding="utf-8"))
    validate_binding_manifest(binding_manifest_json, expected_label_mode=manifest["label_report"]["mode"])
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    dataset_manifest = {
        "schema_version": "detector_dataset_metadata_manifest_v1",
        "binding_manifest": str(binding_manifest_json),
        "binding_manifest_sha256": sha256_file(Path(binding_manifest_json)),
        "episode_count": manifest["episode_count"],
        "step_count": manifest["step_count"],
        "population_counts": manifest["population_counts"],
        "formal_detector_dataset_build": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "gpu": "NOT_PERFORMED",
    }
    (root / OUTPUTS["dataset_manifest"]).write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stats = {
        "schema_version": "detector_dataset_statistics_v1",
        "suite_counts": manifest["suite_counts"],
        "task_counts": manifest["task_counts"],
        "trace_length_histogram": manifest["trace_length_histogram"],
        "feature_summary_rows": len(manifest["feature_stats"]),
    }
    (root / OUTPUTS["dataset_statistics"]).write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(root / OUTPUTS["population_summary"], ["population_id", "episode_count"], [
        {"population_id": k, "episode_count": v} for k, v in sorted(manifest["population_counts"].items())
    ])
    write_csv(root / OUTPUTS["feature_summary"], ["feature_name", "count", "missing", "nan", "inf"], [
        {"feature_name": k, **v} for k, v in manifest["feature_stats"].items()
    ])
    return dataset_manifest


def summary_report(binding_manifest_json: str | Path) -> dict[str, object]:
    manifest = json.loads(Path(binding_manifest_json).read_text(encoding="utf-8"))
    validate_binding_manifest(binding_manifest_json, expected_label_mode=manifest["label_report"]["mode"])
    duplicates = 0
    missing = sum(v["missing"] for v in manifest["feature_stats"].values())
    nans = sum(v["nan"] for v in manifest["feature_stats"].values())
    infs = sum(v["inf"] for v in manifest["feature_stats"].values())
    return {
        "status": "PASS",
        "episodes": manifest["episode_count"],
        "steps": manifest["step_count"],
        "duplicates": duplicates,
        "orphans": 0,
        "coverage": "PASS",
        "suite_counts": manifest["suite_counts"],
        "task_counts": manifest["task_counts"],
        "feature_completeness": "PASS" if missing == 0 else "HOLD",
        "missing_values": missing,
        "nan": nans,
        "inf": infs,
        "step_continuity": "PASS",
        "trace_length_histogram": manifest["trace_length_histogram"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--label-artifact-root", required=True)
    b.add_argument("--feature-csv", required=True)
    b.add_argument("--output-json", required=True)
    b.add_argument("--expected-label-mode", default="formal-ledger-build", choices=["synthetic-dry-run", "formal-ledger-build"])
    v = sub.add_parser("validate")
    v.add_argument("--binding-manifest", required=True)
    v.add_argument("--expected-label-mode", default="formal-ledger-build", choices=["synthetic-dry-run", "formal-ledger-build"])
    args = parser.parse_args(argv)
    try:
        if args.cmd == "build":
            report = write_binding_manifest(args.label_artifact_root, args.feature_csv, args.output_json, expected_label_mode=args.expected_label_mode)
        else:
            report = validate_binding_manifest(args.binding_manifest, expected_label_mode=args.expected_label_mode)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
