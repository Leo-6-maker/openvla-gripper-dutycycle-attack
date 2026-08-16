"""Read-only Stage VIII within-parent timing identifiability audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SEAL_FILES = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
COUNTERS = {
    "protected_reads": 0,
    "eval160_reads": 0,
    "attack_rollouts": 0,
    "vis_pgd_attack_rollouts": 0,
}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


class AuditError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"JSON_OBJECT_REQUIRED:{path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require_zero_counters(value: Any, label: str) -> None:
    if not isinstance(value, dict) or any(item != 0 for item in value.values()):
        raise AuditError(f"PROTECTED_COUNTER_NONZERO:{label}:{value}")


def verify_file_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise AuditError(f"INPUT_FILE_MISSING:{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AuditError(f"INPUT_SHA256_MISMATCH:{path}:{actual}:{expected}")


def verify_sealed_root(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise AuditError(f"ROOT_MISSING:{root}")
    sums = root / "SHA256SUMS"
    sums_sidecar = root / "SHA256SUMS.sha256"
    seal = root / "ROOT_SEAL.json"
    seal_sidecar = root / "ROOT_SEAL.sha256"
    for path in (sums, sums_sidecar, seal, seal_sidecar):
        if not path.is_file():
            raise AuditError(f"SEAL_FILE_MISSING:{path}")
    entries: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise AuditError(f"BAD_SHA256SUMS_LINE:{root}:{line}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise AuditError(f"UNSAFE_SHA256SUMS_PATH:{root}:{relative}")
        target = root / relative_path
        if not target.is_file() or sha256_file(target) != digest:
            raise AuditError(f"SEALED_FILE_MISMATCH:{root}:{relative}")
        entries[relative_path.as_posix()] = digest
    if sums.read_text(encoding="utf-8").splitlines() and sums_sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(sums):
        raise AuditError(f"SHA256SUMS_SIDECAR_MISMATCH:{root}")
    if seal_sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(seal):
        raise AuditError(f"ROOT_SEAL_SIDECAR_MISMATCH:{root}")
    seal_value = read_json(seal)
    require_zero_counters(seal_value.get("protected_counters"), f"root:{root}")
    if seal_value.get("eval160", seal_value.get("eval160_status")) not in ("UNREAD", None):
        raise AuditError(f"EVAL160_NOT_UNREAD:{root}")
    return {
        "root": str(root),
        "root_seal_sha256": sha256_file(seal),
        "sha256sums_sha256": sha256_file(sums),
        "listed_files": len(entries),
        "root_seal": seal_value,
    }


def label_key(stage: str, row: dict[str, Any], dose: str) -> tuple[str, str, str, int, str]:
    try:
        return (
            stage,
            str(row["canonical_parent_key"]),
            str(row["probe_id"]),
            int(row["probe_step"]),
            str(row["dose"] if "dose" in row else dose),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"BAD_ROW_IDENTITY:{stage}:{row}") from exc


def load_labels(protocol: dict[str, Any]) -> tuple[dict[tuple[str, str, str, int, str], dict[str, Any]], dict[str, Any]]:
    dose = protocol["primary_population"]["dose"]
    label_map: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for stage, spec in protocol["primary_population"]["label_files"].items():
        path = Path(spec["path"])
        verify_file_hash(path, spec["sha256"])
        rows = read_jsonl(path)
        stage_count = 0
        stage_abstain = 0
        for row in rows:
            if row.get("dose") != dose:
                continue
            stage_count += 1
            require_zero_counters(row.get("protected_counters"), f"label:{stage}:{row.get('label_id')}")
            key = label_key(stage, row, dose)
            if key in label_map:
                raise AuditError(f"DUPLICATE_LABEL_KEY:{key}")
            binary = bool(row.get("binary_label_consumable"))
            label_class = row.get("label_class")
            if not binary:
                stage_abstain += 1
            elif label_class not in protocol["primary_population"]["binary_label_classes"]:
                raise AuditError(f"BINARY_CLASS_INVALID:{key}:{label_class}")
            label_map[key] = {
                "stage": stage,
                "suite": row.get("suite") or str(row["canonical_parent_key"]).split("/", 1)[0],
                "canonical_parent_key": row["canonical_parent_key"],
                "probe_id": row["probe_id"],
                "probe_step": int(row["probe_step"]),
                "dose": dose,
                "consumable": binary,
                "y": int(label_class == "V_PHYS") if binary else None,
                "label_class": label_class,
                "label_id": row.get("label_id"),
            }
        bindings[stage] = {
            "path": str(path),
            "sha256": spec["sha256"],
            "total_t5_rows": stage_count,
            "abstain_t5_rows": stage_abstain,
            "consumable_t5_rows": stage_count - stage_abstain,
        }
    return label_map, bindings


def score_key(stage: str, row: dict[str, Any], dose: str) -> tuple[str, str, str, int, str]:
    return label_key(stage, row, dose)


def load_source(name: str, spec: dict[str, Any], labels: dict[tuple[str, str, str, int, str], dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    root = Path(spec["root"])
    root_binding = verify_sealed_root(root)
    prediction_path = Path(spec["predictions"])
    verify_file_hash(prediction_path, spec["predictions_sha256"])
    dose = protocol["primary_population"]["dose"]
    rows = read_jsonl(prediction_path)
    scores: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    expected_stage = spec.get("requires_stage_join")
    for row in rows:
        if row.get("dose") != dose:
            continue
        stage = str(row.get("stage") or expected_stage)
        if stage not in ("STAGE_V", "STAGE_VI_B2"):
            raise AuditError(f"BAD_SCORE_STAGE:{name}:{stage}")
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"BAD_SCORE:{name}:{row}") from exc
        if not math.isfinite(score):
            raise AuditError(f"NONFINITE_SCORE:{name}:{row}")
        key = score_key(stage, row, dose)
        if key in scores:
            raise AuditError(f"DUPLICATE_SCORE_KEY:{name}:{key}")
        label = labels.get(key)
        if label is None:
            raise AuditError(f"SCORE_LABEL_MISSING:{name}:{key}")
        if row.get("suite") is not None and row["suite"] != label["suite"]:
            raise AuditError(f"SCORE_SUITE_MISMATCH:{name}:{key}")
        if row.get("y") is not None and label["consumable"] and int(row["y"]) != label["y"]:
            raise AuditError(f"SCORE_Y_MISMATCH:{name}:{key}")
        if row.get("consumable") is not None and bool(row["consumable"]) != label["consumable"]:
            raise AuditError(f"SCORE_CONSUMABLE_MISMATCH:{name}:{key}")
        scores[key] = {
            **label,
            "score": score,
            "source": name,
        }
    eligible = [row for key, row in scores.items() if labels[key]["consumable"]]
    scope_stages = ["STAGE_VI_B2"] if name == "B2-C" else ["STAGE_V", "STAGE_VI_B2"]
    scoped_labels = [row for key, row in labels.items() if key[0] in scope_stages and row["consumable"]]
    coverage = len(eligible) / len(scoped_labels) if scoped_labels else None
    return {
        "name": name,
        "status": "IDENTIFIABLE_EXACT" if eligible else "NOT_IDENTIFIABLE_SCORE_SOURCE",
        "scope": spec["scope"],
        "root": root_binding,
        "predictions": {
            "path": str(prediction_path),
            "sha256": spec["predictions_sha256"],
            "all_rows": len(rows),
            "t5_rows": len(scores),
            "exact_consumable_rows": len(eligible),
        },
        "coverage": {
            "scoped_consumable_label_rows": len(scoped_labels),
            "exact_consumable_score_rows": len(eligible),
            "fraction": coverage,
        },
        "rows": eligible,
    }


def auc(y: list[int], scores: list[float]) -> float | None:
    positives = sum(y)
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, y), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(item[1] for item in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def bootstrap(values: list[float], protocol: dict[str, Any]) -> dict[str, Any] | None:
    if not values:
        return None
    config = protocol["metric_definitions"]["bootstrap"]
    rng = random.Random(int(config["seed"]))
    samples = []
    for _ in range(int(config["replicates"])):
        samples.append(mean(rng.choice(values) for _ in values))
    samples.sort()
    lo = samples[max(0, int(0.025 * len(samples)) - 1)]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return {"replicates": len(samples), "seed": config["seed"], "ci_95": [lo, hi]}


def parent_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_parent_key"]].append(row)
    return dict(grouped)


def parent_metric(parent: str, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any] | None:
    positives = [row for row in rows if row["y"] == 1]
    negatives = [row for row in rows if row["y"] == 0]
    if not positives or not negatives:
        return None
    wins = ties = 0
    for positive in positives:
        for negative in negatives:
            if positive["score"] > negative["score"]:
                wins += 1
            elif positive["score"] == negative["score"]:
                ties += 1
    pair_count = len(positives) * len(negatives)
    ordered = sorted(rows, key=lambda row: (-row["score"], str(row["probe_id"]), int(row["probe_step"])))
    result: dict[str, Any] = {
        "canonical_parent_key": parent,
        "suite": rows[0]["suite"],
        "row_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "pair_count": pair_count,
        "within_parent_auc": (wins + 0.5 * ties) / pair_count,
        "wins": wins,
        "ties": ties,
        "random_prevalence": len(positives) / len(rows),
        "argmax_y": ordered[0]["y"],
        "argmax_probe_id": ordered[0]["probe_id"],
        "argmax_probe_step": ordered[0]["probe_step"],
    }
    for k in (1, 3, 5):
        if len(ordered) >= k:
            result[f"top_{k}_positive_rate"] = sum(row["y"] for row in ordered[:k]) / k
        else:
            result[f"top_{k}_positive_rate"] = None
    return result


def aggregate_parent_metrics(metrics: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {"status": "NON_IDENTIFIABLE", "eligible_parent_count": 0}
    macro_auc = mean(item["within_parent_auc"] for item in metrics)
    total_pairs = sum(item["pair_count"] for item in metrics)
    pooled_auc = sum(item["within_parent_auc"] * item["pair_count"] for item in metrics) / total_pairs
    random_baseline = mean(item["random_prevalence"] for item in metrics)
    result: dict[str, Any] = {
        "status": "IDENTIFIABLE",
        "eligible_parent_count": len(metrics),
        "positive_probe_count": sum(item["positive_count"] for item in metrics),
        "negative_probe_count": sum(item["negative_count"] for item in metrics),
        "pair_count": total_pairs,
        "parent_macro_within_parent_auc": macro_auc,
        "pooled_pair_weighted_auc": pooled_auc,
        "median_parent_auc": median(item["within_parent_auc"] for item in metrics),
        "parent_bootstrap_ci": bootstrap([item["within_parent_auc"] for item in metrics], protocol),
        "random_baseline_prevalence": random_baseline,
    }
    for k in (1, 3, 5):
        values = [item[f"top_{k}_positive_rate"] for item in metrics if item[f"top_{k}_positive_rate"] is not None]
        selected = mean(values) if values else None
        result[f"top_{k}"] = {
            "parent_count": len(values),
            "selected_positive_rate": selected,
            "random_expected_rate": random_baseline if len(values) == len(metrics) else mean(item["random_prevalence"] for item in metrics if item[f"top_{k}_positive_rate"] is not None) if values else None,
            "lift": selected / (random_baseline if len(values) == len(metrics) else mean(item["random_prevalence"] for item in metrics if item[f"top_{k}_positive_rate"] is not None)) if selected is not None and (random_baseline if len(values) == len(metrics) else mean(item["random_prevalence"] for item in metrics if item[f"top_{k}_positive_rate"] is not None)) else None,
        }
    zero_regret = mean(item["argmax_y"] for item in metrics)
    result["zero_regret_parent_rate"] = zero_regret
    result["zero_regret_margin_over_random"] = zero_regret - random_baseline
    result["top_1_hit_rate"] = result["top_1"]["selected_positive_rate"]
    result["top_1_lift"] = result["top_1"]["lift"]
    result["top_3_lift"] = result["top_3"]["lift"]
    return result


def source_metrics(source: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    rows = source.pop("rows")
    groups = parent_rows(rows)
    parents = []
    for parent, parent_data in sorted(groups.items()):
        metric = parent_metric(parent, parent_data, protocol)
        if metric is not None:
            parents.append(metric)
    overall = aggregate_parent_metrics(parents, protocol)
    suite_metrics = {}
    for suite in SUITES:
        suite_parents = [item for item in parents if item["suite"] == suite]
        suite_metrics[suite] = aggregate_parent_metrics(suite_parents, protocol)
    global_rows = rows
    global_value = auc([row["y"] for row in global_rows], [row["score"] for row in global_rows])
    global_by_suite = {}
    for suite in SUITES:
        suite_rows = [row for row in rows if row["suite"] == suite]
        global_by_suite[suite] = {
            "row_count": len(suite_rows),
            "auroc": auc([row["y"] for row in suite_rows], [row["score"] for row in suite_rows]),
        }
    gate = protocol["promotion_gate"]
    identifiable_suites = [value for value in suite_metrics.values() if value.get("status") == "IDENTIFIABLE"]
    suite_pass = bool(identifiable_suites) and all(value["parent_macro_within_parent_auc"] >= gate["every_identifiable_suite_within_parent_auc_min"] for value in identifiable_suites)
    overall_gate = {
        "parent_macro_auc": overall.get("parent_macro_within_parent_auc", 0.0) >= gate["parent_macro_within_parent_auc_min"],
        "every_identifiable_suite_auc": suite_pass,
        "top_1_lift": (overall.get("top_1_lift") or 0.0) >= gate["top_1_lift_min"],
        "top_3_lift": (overall.get("top_3_lift") or 0.0) >= gate["top_3_lift_min"],
        "zero_regret_margin": (overall.get("zero_regret_margin_over_random") or 0.0) >= gate["zero_regret_parent_rate_margin_over_random_min"],
    }
    relative_reference = protocol["metric_definitions"]["global_vs_relative_reference"]
    relative_signal = (
        overall.get("parent_macro_within_parent_auc", 0.0) >= relative_reference["relative_signal_parent_macro_auc"]
        and bool(identifiable_suites)
        and all(value["parent_macro_within_parent_auc"] >= relative_reference["relative_signal_suite_auc"] for value in identifiable_suites)
    )
    global_signal = (
        global_value is not None
        and global_value >= relative_reference["global_signal_auroc"]
        and all(value["auroc"] is not None and value["auroc"] >= relative_reference["global_suite_auroc"] for value in global_by_suite.values() if value["row_count"])
    )
    if relative_signal and global_signal:
        decomposition = "ABSOLUTE_AND_RELATIVE_SIGNAL"
    elif relative_signal:
        decomposition = "RELATIVE_ONLY_SIGNAL"
    elif global_signal:
        decomposition = "GLOBAL_ONLY_SIGNAL"
    else:
        decomposition = "NO_STABLE_SIGNAL"
    source["rows"] = None
    source.update({
        "metrics": {
            "overall": overall,
            "per_suite": suite_metrics,
            "global_auroc": global_value,
            "global_per_suite": global_by_suite,
        },
        "promotion_gate": {"checks": overall_gate, "pass": all(overall_gate.values())},
        "relative_signal_reference": relative_signal,
        "global_signal_reference": global_signal,
        "global_vs_relative_classification": decomposition,
        "parent_metrics_count": len(parents),
    })
    return source, parents


def seal(root: Path, summary: dict[str, Any]) -> None:
    summary_path = root / "STAGE_VIII_R0_RELATIVE_TIMING_IDENTIFIABILITY.json"
    write_json(summary_path, summary)
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in SEAL_FILES):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    write_json(root / "ROOT_SEAL.json", {
        "schema": "STAGE_VIII_R0_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": sha256_file(summary_path),
        "sha256sums_sha256": sums_sha,
        "new_training_authorized": False,
        "new_m4_authorized": False,
        "pgd_authorized": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    })
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    if protocol.get("schema") != "STAGE_VIII_R0_RELATIVE_TIMING_IDENTIFIABILITY_PROTOCOL_V1" or protocol.get("status") != "FROZEN_BEFORE_R0_EXECUTION":
        raise SystemExit("PROTOCOL_NOT_FROZEN_R0")
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{output}")
    if git("status", "--porcelain"):
        raise SystemExit("WORKTREE_NOT_CLEAN")
    labels, label_bindings = load_labels(protocol)
    source_results: dict[str, Any] = {}
    parent_records: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for name, spec in protocol["score_sources"].items():
        try:
            loaded = load_source(name, spec, labels, protocol)
            result, parents = source_metrics(loaded, protocol)
            source_results[name] = result
            parent_records.extend({"source": name, **parent} for parent in parents)
        except AuditError as exc:
            source_results[name] = {
                "name": name,
                "status": "NOT_IDENTIFIABLE_SCORE_SOURCE",
                "scope": spec["scope"],
                "root": str(Path(spec["root"])),
            }
            errors[name] = str(exc)
    identifiable = [value for value in source_results.values() if value.get("status") == "IDENTIFIABLE_EXACT"]
    passing = [value["name"] for value in identifiable if value.get("promotion_gate", {}).get("pass")]
    relative = [value["name"] for value in identifiable if value.get("relative_signal_reference")]
    if passing:
        decision = "STAGE_VIII_RELATIVE_TIMING_IDENTIFIABILITY_ESTABLISHED"
    elif relative:
        decision = "STAGE_VIII_WEAK_RELATIVE_SIGNAL_ONLY"
    elif not identifiable:
        decision = "STAGE_VIII_R0_DATA_NOT_IDENTIFIABLE"
    else:
        decision = "STAGE_VIII_RELATIVE_TIMING_NOT_IDENTIFIABLE"
    output.mkdir(parents=True)
    (output / "STAGE_VIII_R0_PROTOCOL_V1.json").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
    with (output / "R0_PARENT_METRICS.jsonl").open("w", encoding="utf-8") as handle:
        for row in parent_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema": "STAGE_VIII_R0_RELATIVE_TIMING_IDENTIFIABILITY_V1",
        "status": decision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "label_bindings": label_bindings,
        "score_sources": source_results,
        "source_errors": errors,
        "decision": {
            "passing_sources": passing,
            "relative_signal_sources": relative,
            "r1_authorized": decision == "STAGE_VIII_RELATIVE_TIMING_IDENTIFIABILITY_ESTABLISHED",
            "pass_gate": protocol["promotion_gate"],
        },
        "new_model_training": False,
        "new_m4": False,
        "intervention": False,
        "pgd_rollout": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    write_json(output / "PROVENANCE.json", {
        "schema": "STAGE_VIII_R0_PROVENANCE_V1",
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "source_script": summary["source_script"],
        "source_script_sha256": summary["source_script_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "input_label_bindings": label_bindings,
        "input_score_bindings": {name: {"root": value.get("root"), "predictions": value.get("predictions")} for name, value in source_results.items()},
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    })
    seal(output, summary)
    print(json.dumps({"status": decision, "output_root": str(output), "passing_sources": passing, "relative_signal_sources": relative, "protected_counters": COUNTERS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
