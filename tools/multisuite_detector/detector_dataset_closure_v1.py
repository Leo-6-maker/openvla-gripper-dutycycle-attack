#!/usr/bin/env python3
"""Synthetic-only C2 detector dataset closure contracts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.load_label_v2_artifact import load_label_v2_artifact


class DetectorDatasetClosureError(ValueError):
    pass


SC5_SOURCE = REPO / "src" / "gripper_attack" / "sc5mlp_v1.py"
META_COLUMNS = ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "step"]
SPLITS = {"train", "val", "test"}
POPULATIONS = {"DETECTOR_ELIGIBLE", "DETECTOR_SAFETY"}
STATE_HASH_RE = "0123456789abcdef"


def fail(message: str) -> None:
    raise DetectorDatasetClosureError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha(obj: object) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def sc5_features() -> list[str]:
    tree = ast.parse(SC5_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SC5_FEATURES":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list) or len(value) != 25:
                        fail("SC5_FEATURES must be a 25-item list")
                    if len(set(value)) != len(value):
                        fail("SC5_FEATURES contains duplicates")
                    return value
    fail("SC5_FEATURES not found")


SC5_FEATURES = sc5_features()
FEATURE_CONTRACT = {
    "schema_version": "sc5_feature_artifact_v1",
    "feature_names": SC5_FEATURES,
    "feature_count": len(SC5_FEATURES),
}
FEATURE_SCHEMA_SHA256 = json_sha(FEATURE_CONTRACT)


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            fail(f"{path.name}: expected exact header")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: extra cells")
            if any(v is None or v == "" for v in row.values()):
                fail(f"{path.name}:{line_no}: empty field")
            rows.append(row)
        return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value: str, field: str, episode: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        fail(f"{episode}: {field} must be int")
    if str(parsed) != value:
        fail(f"{episode}: {field} must be canonical int")
    return parsed


def parse_float(value: str, field: str, episode: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        fail(f"{episode}: {field} must be finite float")
    if not math.isfinite(parsed):
        fail(f"{episode}: {field} must be finite float")
    return parsed


def validate_state_hash(value: str, episode: str) -> None:
    if len(value) != 64 or any(ch not in STATE_HASH_RE for ch in value):
        fail(f"{episode}: initial_state_hash must be 64 lowercase hex")


def load_feature_artifact(path: str | Path) -> dict[str, object]:
    path = Path(path)
    rows = read_csv(path, META_COLUMNS + SC5_FEATURES)
    by_episode: dict[str, dict[str, object]] = {}
    seen_steps: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        episode = row["episode_key"]
        validate_state_hash(row["initial_state_hash"], episode)
        trace_length = parse_int(row["trace_length"], "trace_length", episode)
        step = parse_int(row["step"], "step", episode)
        if trace_length <= 0 or step < 0 or step >= trace_length:
            fail(f"{episode}: invalid trace_length/step")
        if step in seen_steps[episode]:
            fail(f"{episode}: duplicate feature step")
        seen_steps[episode].add(step)
        values = [parse_float(row[name], name, episode) for name in SC5_FEATURES]
        meta = {key: row[key] for key in META_COLUMNS[:-1]}
        existing = by_episode.setdefault(episode, {**meta, "steps": {}})
        for key in META_COLUMNS[:-1]:
            if existing[key] != meta[key]:
                fail(f"{episode}: inconsistent feature metadata")
        existing["steps"][step] = values
    for episode, record in by_episode.items():
        trace_length = int(record["trace_length"])
        if set(record["steps"]) != set(range(trace_length)):
            fail(f"{episode}: feature steps must cover 0..trace_length-1")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "feature_names": list(SC5_FEATURES),
        "feature_count": len(SC5_FEATURES),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "episodes": by_episode,
    }


def row_population(row: dict[str, str]) -> str:
    if row["mechanism_eligible"] == "false":
        return "DETECTOR_SAFETY"
    return "DETECTOR_ELIGIBLE"


def build_dataset_manifest(label_artifact_root: str | Path, feature_csv: str | Path, output: str | Path) -> dict[str, object]:
    label = load_label_v2_artifact(label_artifact_root, expected_mode="synthetic-dry-run")
    features = load_feature_artifact(feature_csv)
    label_rows = {row["episode_key"]: row for row in label["label_rows"]}
    feature_rows = features["episodes"]
    if set(label_rows) != set(feature_rows):
        fail("Label V2 and feature artifact episode sets differ")
    rows = []
    population_counts = {"DETECTOR_ELIGIBLE": 0, "DETECTOR_SAFETY": 0}
    for episode in sorted(label_rows):
        label_row = label_rows[episode]
        feature_row = feature_rows[episode]
        for field in ["parent_key", "suite", "task_id"]:
            if label_row[field] != feature_row[field]:
                fail(f"{episode}: {field} mismatch")
        if int(label_row["trace_length"]) != int(feature_row["trace_length"]):
            fail(f"{episode}: trace_length mismatch")
        population = row_population(label_row)
        population_counts[population] += 1
        rows.append({
            "episode_key": episode,
            "parent_key": label_row["parent_key"],
            "suite": label_row["suite"],
            "task_id": label_row["task_id"],
            "initial_state_hash": feature_row["initial_state_hash"],
            "trace_length": label_row["trace_length"],
            "population_id": population,
        })
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "population_id"], rows)
    manifest = {
        "schema_version": "detector_dataset_manifest_v1",
        "label_artifact_root": str(label_artifact_root),
        "label_report": label["report"],
        "feature_artifact_path": features["path"],
        "feature_artifact_sha256": features["sha256"],
        "feature_names": features["feature_names"],
        "feature_count": features["feature_count"],
        "feature_schema_sha256": features["feature_schema_sha256"],
        "row_count": len(rows),
        "population_counts": {
            **population_counts,
            "DETECTOR_MULTI_EVENT": "UNAVAILABLE_SEPARATE_ARTIFACT_REQUIRED",
        },
        "population_by_episode": {row["episode_key"]: row["population_id"] for row in rows},
        "dataset_manifest_path": str(out),
        "dataset_manifest_sha256": sha256_file(out),
        "synthetic_contract_validation": "PASS",
        "real_artifact_validation": "NOT_PERFORMED",
        "formal_detector_dataset_build": "NOT_PERFORMED",
        "server_execution": "NOT_PERFORMED",
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_dataset_manifest(path: str | Path) -> list[dict[str, str]]:
    rows = read_csv(Path(path), ["episode_key", "parent_key", "suite", "task_id", "initial_state_hash", "trace_length", "population_id"])
    seen = set()
    for row in rows:
        episode = row["episode_key"]
        if episode in seen:
            fail(f"duplicate dataset episode_key: {episode}")
        seen.add(episode)
        validate_state_hash(row["initial_state_hash"], episode)
        if row["population_id"] not in POPULATIONS:
            fail(f"{episode}: unknown population_id")
        if parse_int(row["trace_length"], "trace_length", episode) <= 0:
            fail(f"{episode}: trace_length must be positive")
    return rows


def validate_dataset_rows(rows: list[dict[str, str]], features: dict[str, object], expected_populations: dict[str, str] | None = None) -> None:
    feature_rows = features["episodes"]
    if set(r["episode_key"] for r in rows) != set(feature_rows):
        fail("dataset/feature episode set mismatch")
    for row in rows:
        episode = row["episode_key"]
        feature = feature_rows[episode]
        for field in ["parent_key", "suite", "task_id", "initial_state_hash"]:
            if row[field] != feature[field]:
                fail(f"{episode}: dataset {field} mismatch")
        if int(row["trace_length"]) != int(feature["trace_length"]):
            fail(f"{episode}: dataset trace_length mismatch")
        if expected_populations is not None and expected_populations.get(episode) != row["population_id"]:
            fail(f"{episode}: population_id mismatch")


def connected_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    parent_to_eps: dict[str, set[str]] = defaultdict(set)
    state_to_eps: dict[str, set[str]] = defaultdict(set)
    by_episode = {row["episode_key"]: row for row in rows}
    for row in rows:
        parent_to_eps[row["parent_key"]].add(row["episode_key"])
        state_to_eps[row["initial_state_hash"]].add(row["episode_key"])
    unseen = set(by_episode)
    groups = []
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        eps = {start}
        while queue:
            ep = queue.popleft()
            row = by_episode[ep]
            neighbors = parent_to_eps[row["parent_key"]] | state_to_eps[row["initial_state_hash"]]
            for nxt in neighbors - eps:
                eps.add(nxt)
                if nxt in unseen:
                    unseen.remove(nxt)
                queue.append(nxt)
        group_rows = [by_episode[ep] for ep in sorted(eps)]
        groups.append({
            "group_id": sha256_bytes("|".join(sorted(eps)).encode())[:16],
            "episodes": sorted(eps),
            "parents": sorted({r["parent_key"] for r in group_rows}),
            "states": sorted({r["initial_state_hash"] for r in group_rows}),
            "suites": sorted({r["suite"] for r in group_rows}),
            "tasks": sorted({r["task_id"] for r in group_rows}),
        })
    return sorted(groups, key=lambda g: g["group_id"])


def split_rows_from_groups(groups: list[dict[str, object]], assignments: dict[str, str], split_type: str, fold_id: str) -> list[dict[str, object]]:
    out = []
    for group in groups:
        split = assignments[group["group_id"]]
        for episode in group["episodes"]:
            out.append({"split_type": split_type, "fold_id": fold_id, "group_id": group["group_id"], "episode_key": episode, "split": split})
    return sorted(out, key=lambda row: (row["fold_id"], row["split"], row["episode_key"]))


def build_parent_random_split(dataset_csv: str | Path, output: str | Path, *, seed: int, train_ratio: float, val_ratio: float) -> dict[str, object]:
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        fail("train/val ratios must be explicit and sum below 1")
    rows = load_dataset_manifest(dataset_csv)
    groups = connected_groups(rows)
    ids = [g["group_id"] for g in groups]
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if min(n_train, n_val, n - n_train - n_val) <= 0:
        fail("ratios produce an empty split")
    assignments = {gid: "test" for gid in ids}
    for gid in ids[:n_train]:
        assignments[gid] = "train"
    for gid in ids[n_train:n_train + n_val]:
        assignments[gid] = "val"
    return write_split(output, split_rows_from_groups(groups, assignments, "parent_random_split_v1", "parent_random"), dataset_csv, {"seed": seed, "train_ratio": train_ratio, "val_ratio": val_ratio})


def build_object_loto_split(dataset_csv: str | Path, output: str | Path) -> dict[str, object]:
    rows = load_dataset_manifest(dataset_csv)
    groups = connected_groups(rows)
    object_tasks = sorted({r["task_id"] for r in rows if r["suite"] == "Object"})
    out = []
    for task in object_tasks:
        assignments = {}
        for group in groups:
            assignments[group["group_id"]] = "test" if "Object" in group["suites"] and task in group["tasks"] else "train"
        out.extend(split_rows_from_groups(groups, assignments, "object_leave_task_out_v1", f"object_loto_{task}"))
    return write_split(output, out, dataset_csv, {"held_out_tasks": object_tasks})


def build_suite_loso_split(dataset_csv: str | Path, output: str | Path) -> dict[str, object]:
    rows = load_dataset_manifest(dataset_csv)
    groups = connected_groups(rows)
    suites = sorted({r["suite"] for r in rows})
    out = []
    for suite in suites:
        assignments = {g["group_id"]: ("test" if suite in g["suites"] else "train") for g in groups}
        out.extend(split_rows_from_groups(groups, assignments, "suite_loso_split_v1", f"loso_{suite}"))
    return write_split(output, out, dataset_csv, {"held_out_suites": suites})


def write_split(output: str | Path, rows: list[dict[str, object]], dataset_csv: str | Path, extra: dict[str, object]) -> dict[str, object]:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["split_type", "fold_id", "group_id", "episode_key", "split"]
    write_csv(out, columns, rows)
    report = {
        "schema_version": rows[0]["split_type"] if rows else "empty_split",
        "source_dataset_manifest_sha256": sha256_file(Path(dataset_csv)),
        "split_manifest_path": str(out),
        "split_manifest_sha256": sha256_file(out),
        "row_count": len(rows),
        **extra,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_split(dataset_csv: str | Path, split_csv: str | Path) -> dict[str, object]:
    rows = load_dataset_manifest(dataset_csv)
    split_rows = read_csv(Path(split_csv), ["split_type", "fold_id", "group_id", "episode_key", "split"])
    by_episode = {r["episode_key"]: r for r in rows}
    groups_list = connected_groups(rows)
    groups = {g["group_id"]: set(g["episodes"]) for g in groups_list}
    expected_group_by_episode = {
        episode: group["group_id"]
        for group in groups_list
        for episode in group["episodes"]
    }
    by_fold: dict[str, dict[str, str]] = defaultdict(dict)
    for row in split_rows:
        if row["episode_key"] not in by_episode:
            fail(f"split references unknown episode: {row['episode_key']}")
        if row["group_id"] != expected_group_by_episode[row["episode_key"]]:
            fail(f"{row['episode_key']}: split group_id mismatch")
        if row["group_id"] not in groups:
            fail(f"unknown split group_id: {row['group_id']}")
        if row["split"] not in SPLITS:
            fail(f"unknown split: {row['split']}")
        fold = by_fold[row["fold_id"]]
        if row["episode_key"] in fold:
            fail(f"duplicate split assignment: {row['episode_key']}")
        fold[row["episode_key"]] = row["split"]
    all_eps = set(by_episode)
    for fold_id, assigned in by_fold.items():
        if set(assigned) != all_eps:
            fail(f"{fold_id}: split coverage mismatch")
        for group_id, episodes in groups.items():
            splits = {assigned[ep] for ep in episodes}
            if len(splits) != 1:
                fail(f"{fold_id}: parent/state group leakage: {group_id}")
        split_type = next(r["split_type"] for r in split_rows if r["fold_id"] == fold_id)
        if split_type == "object_leave_task_out_v1":
            held = fold_id.replace("object_loto_", "")
            if any(by_episode[ep]["suite"] == "Object" and by_episode[ep]["task_id"] == held and split != "test" for ep, split in assigned.items()):
                fail(f"{fold_id}: held-out Object task leakage")
        if split_type == "suite_loso_split_v1":
            held_suite = fold_id.replace("loso_", "")
            if any(by_episode[ep]["suite"] == held_suite and split != "test" for ep, split in assigned.items()):
                fail(f"{fold_id}: held-out suite leakage")
    return {"status": "PASS", "fold_count": len(by_fold), "row_count": len(split_rows)}


def build_normalization(feature_csv: str | Path, dataset_csv: str | Path, split_csv: str | Path, output: str | Path, *, population_id: str, fold_id: str) -> dict[str, object]:
    if population_id not in POPULATIONS:
        fail("normalization population must be DETECTOR_ELIGIBLE or DETECTOR_SAFETY")
    features = load_feature_artifact(feature_csv)
    dataset = {r["episode_key"]: r for r in load_dataset_manifest(dataset_csv)}
    split_rows = read_csv(Path(split_csv), ["split_type", "fold_id", "group_id", "episode_key", "split"])
    train_eps = [r["episode_key"] for r in split_rows if r["fold_id"] == fold_id and r["split"] == "train" and dataset[r["episode_key"]]["population_id"] == population_id]
    if not train_eps:
        fail("normalization has no training episodes")
    values = [[] for _ in SC5_FEATURES]
    for ep in train_eps:
        for step_values in features["episodes"][ep]["steps"].values():
            for i, value in enumerate(step_values):
                values[i].append(value)
    means, stds = [], []
    for name, vals in zip(SC5_FEATURES, values):
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        if std == 0 or not math.isfinite(std):
            fail(f"zero or non-finite variance for {name}")
        means.append(mean)
        stds.append(std)
    report = {
        "schema_version": "detector_normalization_v1",
        "feature_names": SC5_FEATURES,
        "count_per_feature": [len(v) for v in values],
        "mean": means,
        "std": stds,
        "population_id": population_id,
        "fold_id": fold_id,
        "source_dataset_manifest_sha256": sha256_file(Path(dataset_csv)),
        "source_split_manifest_sha256": sha256_file(Path(split_csv)),
        "source_feature_artifact_sha256": features["sha256"],
        "normalization_source": "train_only",
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["normalization_sha256"] = sha256_file(out)
    return report


def validate_normalization(norm: dict[str, object], dataset_csv: str | Path, feature_csv: str | Path, split_csv: str | Path) -> None:
    if norm.get("schema_version") != "detector_normalization_v1":
        fail("normalization schema_version mismatch")
    if norm.get("feature_names") != SC5_FEATURES:
        fail("normalization feature order mismatch")
    if norm.get("normalization_source") != "train_only":
        fail("normalization_source must be train_only")
    if norm.get("source_dataset_manifest_sha256") != sha256_file(Path(dataset_csv)):
        fail("normalization dataset SHA mismatch")
    if norm.get("source_split_manifest_sha256") != sha256_file(Path(split_csv)):
        fail("normalization split SHA mismatch")
    if norm.get("source_feature_artifact_sha256") != sha256_file(Path(feature_csv)):
        fail("normalization feature artifact SHA mismatch")
    if norm.get("population_id") not in POPULATIONS:
        fail("normalization population_id mismatch")
    if not norm.get("fold_id"):
        fail("normalization fold_id missing")
    for field in ["count_per_feature", "mean", "std"]:
        values = norm.get(field)
        if not isinstance(values, list) or len(values) != len(SC5_FEATURES):
            fail(f"normalization {field} length mismatch")
    for count in norm["count_per_feature"]:
        if not isinstance(count, int) or count <= 0:
            fail("normalization count_per_feature must be positive integers")
    for value in norm["mean"]:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            fail("normalization mean must be finite")
    for value in norm["std"]:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            fail("normalization std must be finite positive")


def validate_dataset_closure(dataset_csv: str | Path, feature_csv: str | Path, split_csv: str | Path | None = None, normalization_json: str | Path | None = None) -> dict[str, object]:
    dataset_rows = load_dataset_manifest(dataset_csv)
    features = load_feature_artifact(feature_csv)
    sidecar = Path(dataset_csv).with_suffix(".json")
    if not sidecar.is_file():
        fail("dataset sidecar manifest missing")
    dataset_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_populations = dataset_manifest.get("population_by_episode")
    if not isinstance(expected_populations, dict):
        fail("dataset sidecar population_by_episode missing")
    validate_dataset_rows(dataset_rows, features, expected_populations)
    report = {
        "status": "PASS",
        "synthetic_contract_validation": "PASS",
        "real_artifact_validation": "NOT_PERFORMED",
        "formal_detector_dataset_build": "NOT_PERFORMED",
        "server_execution": "NOT_PERFORMED",
        "row_count": len(dataset_rows),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
    }
    if split_csv:
        report["split_validation"] = validate_split(dataset_csv, split_csv)
    if normalization_json:
        if not split_csv:
            fail("normalization validation requires split manifest")
        norm = json.loads(Path(normalization_json).read_text(encoding="utf-8"))
        validate_normalization(norm, dataset_csv, feature_csv, split_csv)
        report["normalization_validation"] = "PASS"
    return report


def _json_print(obj: object) -> int:
    print(json.dumps(obj, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("feature-contract")
    p.add_argument("--feature-csv", required=True)
    p = sub.add_parser("build-dataset")
    p.add_argument("--label-artifact-root", required=True)
    p.add_argument("--feature-csv", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("split")
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", required=True, choices=["parent-random", "object-loto", "suite-loso"])
    p.add_argument("--seed", type=int)
    p.add_argument("--train-ratio", type=float)
    p.add_argument("--val-ratio", type=float)
    p = sub.add_parser("normalization")
    p.add_argument("--feature-csv", required=True)
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--split-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--population-id", required=True)
    p.add_argument("--fold-id", required=True)
    p = sub.add_parser("validate")
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--feature-csv", required=True)
    p.add_argument("--split-csv")
    p.add_argument("--normalization-json")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "feature-contract":
            return _json_print({k: v for k, v in load_feature_artifact(args.feature_csv).items() if k != "episodes"})
        if args.cmd == "build-dataset":
            return _json_print(build_dataset_manifest(args.label_artifact_root, args.feature_csv, args.output))
        if args.cmd == "split":
            if args.kind == "parent-random":
                if args.seed is None or args.train_ratio is None or args.val_ratio is None:
                    fail("parent-random requires --seed, --train-ratio, and --val-ratio")
                return _json_print(build_parent_random_split(args.dataset_csv, args.output, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio))
            if args.kind == "object-loto":
                return _json_print(build_object_loto_split(args.dataset_csv, args.output))
            return _json_print(build_suite_loso_split(args.dataset_csv, args.output))
        if args.cmd == "normalization":
            return _json_print(build_normalization(args.feature_csv, args.dataset_csv, args.split_csv, args.output, population_id=args.population_id, fold_id=args.fold_id))
        if args.cmd == "validate":
            return _json_print(validate_dataset_closure(args.dataset_csv, args.feature_csv, args.split_csv, args.normalization_json))
    except (OSError, json.JSONDecodeError, csv.Error, DetectorDatasetClosureError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
