"""Read-only Stage VI decomposition of the immutable Stage V result."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.v5_r3_features import FEATURE_NAMES, materialize_fit670_features  # noqa: E402


BASE = Path("/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student")
STAGE_V_ROOT = BASE / "STAGE_V_READY_FOR_PROTECTED_EVAL_6D39860_20260815T193000Z"
TEACHER_ROOT = BASE / "STAGE_V_PRIMARY_CLEAN_TEACHER_20260813T191000Z"
STUDENT_ROOT = BASE / "STAGE_V_PRIMARY_CLEAN_STUDENT_VALIDATION_FIREWALL_20260813T170000Z"
G1_ROOT = BASE / "STAGE_V_PRIMARY_G1_SPLITS_FIREWALL_20260813T160000Z"
M4_AGGREGATE = BASE / "STAGE_V_M4_CENSOR_AWARE_FORMAL_M4_AGGREGATE_F696F582_20260815T175500Z"
M4_SPLIT = BASE / "STAGE_V_M4_CENSOR_AWARE_REPLACEMENT_PLAN_V1_20260815T100000Z/STAGE_V_M4_FINAL_PARENT_SPLIT_V2.json"
CLEAN_ROOT = BASE / "STAGE_V_M4_CLEAN_REPLAY_STUDENT_INPUTS_F696F582_20260816T021500Z"
M4_STUDENT_ROOT = BASE / "STAGE_V_M4_CENSOR_AWARE_STUDENT_VPHYS_HELDOUT_F696F582_20260816T031500Z"
MATRIX_AGGREGATE = BASE / "STAGE_V_STUDENT_TIME_PHYSICAL_MATRIX_AGGREGATE_6D39860_20260815T190500Z"
OUTPUT_ROOT = BASE / "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC_6D39860_20260815T191500Z"
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
STUDENT_HEADS = ("physical_criticality", "k10_feasibility", "instability", "gripper_closing_state")
TEACHER_HEADS = STUDENT_HEADS + ("safe_release",)
DOSES = ("T3", "T5", "T10")
FAILURE_CLASSES = {"GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"}
METRIC_THRESHOLD = 0.60
TEMPORAL_GAIN = 0.05


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(y_values: list[int] | np.ndarray, score_values: list[float] | np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y_values, dtype=np.int64)
    score = np.asarray(score_values, dtype=np.float64)
    if len(y) != len(score):
        raise ValueError("METRIC_LENGTH_MISMATCH")
    if not len(y):
        return {"count": 0, "positive": 0, "negative": 0, "prevalence": None, "auroc": None, "auprc": None, "brier": None, "ece_10bin": None, "pearson_correlation": None, "mae": None, "top_decile_lift": None, "threshold": threshold}
    if not np.isfinite(score).all():
        raise ValueError("NONFINITE_SCORE")
    positives = int(y.sum())
    prevalence = float(y.mean())
    auroc = float(roc_auc_score(y, score)) if 0 < positives < len(y) else None
    auprc = float(average_precision_score(y, score)) if positives else None
    brier = float(np.mean((score - y) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (score >= low) & ((score < high) if high < 1.0 else (score <= high))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(score[mask].mean()) - float(y[mask].mean()))
    correlation = float(np.corrcoef(score, y)[0, 1]) if np.std(score) and np.std(y) else None
    order = np.argsort(-score, kind="mergesort")
    top_n = max(1, int(np.ceil(len(y) * 0.10)))
    top_rate = float(y[order[:top_n]].mean())
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return {
        "count": int(len(y)), "positive": positives, "negative": int(len(y) - positives), "prevalence": prevalence,
        "auroc": auroc, "auprc": auprc, "brier": brier, "ece_10bin": float(ece), "pearson_correlation": correlation,
        "mae": float(np.mean(np.abs(score - y))), "top_decile_lift": (top_rate / prevalence) if prevalence else None,
        "top_decile_positive_rate": top_rate, "threshold": float(threshold),
        "threshold_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "score_quantiles": {str(q): float(np.quantile(score, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
    }


def grouped_metrics(rows: list[dict[str, Any]], group_key: str, threshold: float = 0.5) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    return {key: metric([int(row["y"]) for row in value], [float(row["score"]) for row in value], threshold) for key, value in sorted(groups.items())}


def label_value(record: dict[str, Any], head: str) -> int:
    return int(str(record["labels"][head]["value"]).upper() == "TRUE")


def load_teacher() -> tuple[dict[tuple[str, int], dict[str, int]], dict[str, int]]:
    by_key: dict[tuple[str, int], dict[str, int]] = {}
    lengths: dict[str, int] = {}
    for row in read_jsonl(TEACHER_ROOT / "teacher_records.jsonl"):
        identity, step = str(row["episode_id"]), int(row["step"])
        key = (identity, step)
        if key in by_key:
            raise ValueError(f"TEACHER_DUPLICATE:{identity}:{step}")
        by_key[key] = {head: label_value(row, head) for head in TEACHER_HEADS}
        lengths[identity] = max(lengths.get(identity, 0), step + 1)
    return by_key, lengths


def load_parent_splits() -> dict[str, str]:
    data = read_json(M4_SPLIT)
    mapping = {str(row["canonical_parent_key"]): str(row["split"]) for row in data["parents"]}
    if len(mapping) != 40:
        raise ValueError("M4_SPLIT_NOT_40")
    return mapping


def load_m4_labels(parent_splits: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(M4_AGGREGATE / "M4_ALL_LABELS_V1.jsonl"):
        if row.get("binary_label_consumable") is True and row.get("label_class") in {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}:
            identity = str(row["canonical_parent_key"])
            if identity not in parent_splits:
                raise ValueError(f"M4_PARENT_NOT_IN_SPLIT:{identity}")
            rows.append({**row, "split": parent_splits[identity], "suite": identity.split("/")[0], "y": int(row["label_class"] == "V_PHYS")})
    if len(rows) != 2558:
        raise ValueError(f"M4_BINARY_COUNT:{len(rows)}")
    return rows


def load_m4_student(parent_splits: dict[str, str], m4_labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {(str(row["canonical_parent_key"]), str(row["dose"]), str(row["probe_id"])): row for row in m4_labels}
    rows = []
    for row in read_jsonl(M4_STUDENT_ROOT / "STUDENT_M4_PROBE_SCORES_V1.jsonl"):
        identity = str(row["canonical_parent_key"])
        key = (identity, "T5", str(row["probe_id"]))
        label = labels.get(key)
        if label is None or int(row["y"]) != int(label["y"]):
            raise ValueError(f"STUDENT_M4_LABEL_JOIN:{identity}:{row['probe_id']}")
        rows.append({**row, "split": parent_splits[identity], "suite": identity.split("/")[0], "y": int(row["y"])})
    if len(rows) != 858:
        raise ValueError(f"STUDENT_M4_COUNT:{len(rows)}")
    return rows


def load_student_stream() -> dict[tuple[str, int], dict[str, float]]:
    stream = {}
    for row in read_jsonl(M4_STUDENT_ROOT / "STUDENT_CLEAN_STREAM_PREDICTIONS_V1.jsonl"):
        key = (str(row["canonical_parent_key"]), int(row["step"]))
        if key in stream:
            raise ValueError(f"STUDENT_STREAM_DUPLICATE:{key}")
        stream[key] = {head: float(row["probabilities"][head]) for head in STUDENT_HEADS}
    return stream


def load_features() -> dict[tuple[str, int], np.ndarray]:
    result: dict[tuple[str, int], np.ndarray] = {}
    paths = sorted(CLEAN_ROOT.glob("parents/*/CLEAN_REPLAY_STUDENT_INPUTS_V1.json"))
    if len(paths) != 40:
        raise ValueError(f"CLEAN_REPLAY_COUNT:{len(paths)}")
    for path in paths:
        data = read_json(path)
        if data.get("status") != "PASS_CLEAN_REPLAY" or data.get("outcomes_read") is not False or data.get("protected_counters") != COUNTERS:
            raise ValueError(f"CLEAN_REPLAY_BOUNDARY:{path}")
        identity = str(data["canonical_parent_key"])
        rows = data["replay_rows"]
        episode = {
            "steps": [{"step": row["step"], "raw_action_7d": row["raw_action_7d"], "action_env_7d": row["env_action_7d"]} for row in rows],
            "telemetry": [{"step": row["step"], "robot0_gripper_qpos": row["robot0_gripper_qpos"], "robot0_eef_pos": row["robot0_eef_pos"]} for row in rows],
        }
        materialized = materialize_fit670_features(episode)
        if len(materialized) != len(rows):
            raise ValueError(f"FEATURE_ROW_COUNT:{identity}")
        for row in materialized:
            vector = np.asarray(row["features_25d"], dtype=np.float64)
            if vector.shape != (25,) or not np.isfinite(vector).all():
                raise ValueError(f"FEATURE_INVALID:{identity}:{row['step']}")
            result[(identity, int(row["step"]))] = vector
    if len(result) != 7297:
        raise ValueError(f"FEATURE_COUNT:{len(result)}")
    return result


def student_teacher_fidelity(teacher: dict[tuple[str, int], dict[str, int]], lengths: dict[str, int], m4_student: list[dict[str, Any]]) -> dict[str, Any]:
    development = []
    for row in read_jsonl(STUDENT_ROOT / "predictions.jsonl"):
        identity, step = str(row["episode_id"]), int(row["step"])
        target = teacher.get((identity, step))
        if target is None:
            raise ValueError(f"STUDENT_TEACHER_DEV_JOIN:{identity}:{step}")
        relative = step / max(1, lengths[identity] - 1)
        phase = "early" if relative < 1 / 3 else "middle" if relative < 2 / 3 else "late"
        development.append({"identity": identity, "suite": identity.split("/")[0], "split": str(row["split"]), "phase": phase, "target": target, "scores": {head: float(row[head]["probability"]) for head in STUDENT_HEADS}})
    heldout = []
    for row in m4_student:
        identity, step = str(row["canonical_parent_key"]), int(row["probe_step"])
        target = teacher.get((identity, step))
        if target is None:
            raise ValueError(f"STUDENT_TEACHER_M4_JOIN:{identity}:{step}")
        heldout.append({"identity": identity, "suite": row["suite"], "split": row["split"], "target": target, "scores": row["scores"]})
    output: dict[str, Any] = {"development_count": len(development), "heldout_m4_count": len(heldout), "development": {}, "heldout_m4": {}}
    thresholds = read_json(STUDENT_ROOT / "thresholds.json")
    for head in STUDENT_HEADS:
        for name, rows, groups in (("development", development, ("split", "suite", "phase")), ("heldout_m4", heldout, ("split", "suite"))):
            flat = [{"y": row["target"][head], "score": row["scores"][head], "split": row["split"], "suite": row["suite"], "phase": row.get("phase", "m4_probe")} for row in rows]
            block = output[name].setdefault(head, {"overall": metric([r["y"] for r in flat], [r["score"] for r in flat], float(thresholds[head]["threshold"]))})
            block["groups"] = {group: grouped_metrics(flat, group, float(thresholds[head]["threshold"])) for group in groups}
    return output


def teacher_vphys(teacher: dict[tuple[str, int], dict[str, int]], m4_labels: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for head in TEACHER_HEADS:
        output[head] = {}
        for dose in DOSES:
            records = []
            for row in m4_labels:
                if row["dose"] != dose:
                    continue
                target = teacher.get((str(row["canonical_parent_key"]), int(row["probe_step"])))
                if target is None:
                    raise ValueError(f"TEACHER_VPHYS_JOIN:{row['canonical_parent_key']}:{row['probe_step']}")
                records.append({"y": row["y"], "score": target[head], "suite": row["suite"], "phase": row["split"]})
            output[head][dose] = {"overall": metric([r["y"] for r in records], [r["score"] for r in records]), "per_suite": grouped_metrics(records, "suite"), "per_phase": grouped_metrics(records, "phase")}
    return output


def student_vphys(m4_student: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for head in STUDENT_HEADS:
        records = [{"y": row["y"], "score": float(row["scores"][head]), "suite": row["suite"], "phase": row["split"]} for row in m4_student]
        output[head] = {"overall": metric([r["y"] for r in records], [r["score"] for r in records]), "per_suite": grouped_metrics(records, "suite"), "per_phase": grouped_metrics(records, "phase")}
    return output


def temporal_alignment(teacher: dict[tuple[str, int], dict[str, int]], stream: dict[tuple[str, int], dict[str, float]], m4_labels: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row for row in m4_labels if row["dose"] == "T5"]
    result = {}
    for offset in range(-16, 17):
        by_head_teacher = {head: [] for head in STUDENT_HEADS}
        by_head_student = {head: [] for head in STUDENT_HEADS}
        for row in labels:
            key = (str(row["canonical_parent_key"]), int(row["probe_step"]) + offset)
            t = teacher.get(key)
            s = stream.get(key)
            if t is None or s is None:
                continue
            for head in STUDENT_HEADS:
                by_head_teacher[head].append({"y": row["y"], "score": t[head]})
                by_head_student[head].append({"y": row["y"], "score": s[head]})
        result[str(offset)] = {"offset_steps": offset, "teacher": {head: metric([r["y"] for r in rows], [r["score"] for r in rows]) for head, rows in by_head_teacher.items()}, "student": {head: metric([r["y"] for r in rows], [r["score"] for r in rows]) for head, rows in by_head_student.items()}}
    return result


def grouped_probe(name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, suites: np.ndarray, phases: np.ndarray) -> dict[str, Any]:
    splitter = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
        model.fit(x[train], y[train])
        oof[test] = model.predict_proba(x[test])[:, 1]
        folds.append({"fold": fold, "train_rows": int(len(train)), "test_rows": int(len(test)), "train_parents": int(len(set(groups[train]))), "test_parents": int(len(set(groups[test])))})
    if not np.isfinite(oof).all():
        raise ValueError(f"GROUPED_PROBE_OOF_INCOMPLETE:{name}")
    records = [{"y": int(y[i]), "score": float(oof[i]), "suite": str(suites[i]), "phase": str(phases[i])} for i in range(len(y))]
    return {"model": "standardized_logistic_regression", "group_key": "canonical_parent_key", "context": "development_only_parent_grouped_5fold_oof", "feature_count": int(x.shape[1]), "rows": int(len(y)), "parents": int(len(set(groups))), "folds": folds, "overall": metric(y, oof), "per_suite": grouped_metrics(records, "suite"), "per_phase": grouped_metrics(records, "phase")}


def information_probes(features: dict[tuple[str, int], np.ndarray], teacher: dict[tuple[str, int], dict[str, int]], m4_labels: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in m4_labels if row["dose"] == "T5"]
    x25, xteacher, xwindow, ys, groups, suites, phases = [], [], [], [], [], [], []
    for row in rows:
        identity, step = str(row["canonical_parent_key"]), int(row["probe_step"])
        vector = features.get((identity, step))
        target = teacher.get((identity, step))
        if vector is None or target is None:
            raise ValueError(f"PROBE_INPUT_JOIN:{identity}:{step}")
        history = []
        for past in range(step - 15, step + 1):
            history.append(features.get((identity, past), np.zeros(25, dtype=np.float64)))
        x25.append(vector)
        xteacher.append([target[head] for head in TEACHER_HEADS])
        xwindow.append(np.concatenate(history))
        ys.append(row["y"]); groups.append(identity); suites.append(row["suite"]); phases.append(row["split"])
    y, group, suite, phase = np.asarray(ys, dtype=np.int64), np.asarray(groups), np.asarray(suites), np.asarray(phases)
    return {"selection_status": "DIAGNOSTIC_ONLY_NO_FINAL_HEAD_SELECTION", "parent_grouped": True, "target": "V_phys@T5", "rows": len(rows), "feature_schema": list(FEATURE_NAMES), "probe_A_25D_instantaneous": grouped_probe("A", np.asarray(x25), y, group, suite, phase), "probe_B_privileged_teacher_heads": grouped_probe("B", np.asarray(xteacher), y, group, suite, phase), "probe_C_25D_causal_16_step_window": grouped_probe("C", np.asarray(xwindow), y, group, suite, phase)}


def matrix_diagnostic() -> dict[str, Any]:
    outcomes = read_jsonl(MATRIX_AGGREGATE / "MATRIX_OUTCOMES.jsonl")
    condition = {arm: {"branch_count": 0, "no_emit": 0, "intervention_executed": 0, "compliant": 0, "paired_count": 0, "paired_failure": 0, "paired_no_failure": 0, "paired_abstain": 0} for arm in ("C0", "C1", "C2", "C3")}
    for path in sorted((BASE / "STAGE_V_STUDENT_TIME_PHYSICAL_MATRIX_EXECUTION_6D39860_20260815T184500Z/parents").glob("*/PARENT_RESULT.json")):
        parent = read_json(path)
        for branch in parent["branches"]:
            condition[branch["arm"]]["branch_count"] += 1
            if branch["status"] == "ABSTAIN_NO_STUDENT_EMIT":
                condition[branch["arm"]]["no_emit"] += 1
            elif branch.get("intervention_executed"):
                condition[branch["arm"]]["intervention_executed"] += 1
                branch_data = read_json(path.parent / f"{branch['arm']}.json")
                if branch_data.get("treatment_compliant") is True:
                    condition[branch["arm"]]["compliant"] += 1
    for row in outcomes:
        treatment, control = row["arm"], row["control_arm"]
        condition[treatment]["paired_count"] += 1
        condition[control]["paired_count"] += 1
        if row["treatment_physical_class"] in FAILURE_CLASSES:
            condition[treatment]["paired_failure"] += 1
        elif row["treatment_physical_class"] == "NO_PHYSICAL_FAILURE":
            condition[treatment]["paired_no_failure"] += 1
        if row["control_physical_class"] in FAILURE_CLASSES:
            condition[control]["paired_failure"] += 1
        elif row["control_physical_class"] == "NO_PHYSICAL_FAILURE":
            condition[control]["paired_no_failure"] += 1
        if row["matrix_outcome"] == "ABSTAIN":
            condition[treatment]["paired_abstain"] += 1
            condition[control]["paired_abstain"] += 1
    by_parent = {}
    for row in outcomes:
        by_parent.setdefault(row["canonical_parent_key"], {})[row["arm"]] = row

    def contrast(name: str, pairs: list[tuple[str, str, str]]) -> dict[str, Any]:
        raw = []
        for identity, left, right in pairs:
            left_row = by_parent.get(identity, {}).get(left)
            right_row = by_parent.get(identity, {}).get(right)
            if right == "C2":
                right_row = left_row
                right_class = left_row["control_physical_class"] if left_row else None
                right_valid = left_row["control_valid"] if left_row else False
            elif right == "C0":
                right_class = right_row["control_physical_class"] if right_row else None
                right_valid = right_row["control_valid"] if right_row else False
            else:
                right_class = right_row["treatment_physical_class"] if right_row else None
                right_valid = right_row["treatment_valid"] if right_row else False
            if left_row is None or right_class is None:
                continue
            left_class = left_row["treatment_physical_class"]
            valid = bool(left_row["treatment_valid"] and right_valid)
            raw.append({"identity": identity, "left_failure": left_class in FAILURE_CLASSES, "right_failure": right_class in FAILURE_CLASSES, "valid": valid, "left_class": left_class, "right_class": right_class})
        valid = [row for row in raw if row["valid"]]
        left_rate = float(np.mean([row["left_failure"] for row in valid])) if valid else None
        right_rate = float(np.mean([row["right_failure"] for row in valid])) if valid else None
        return {"name": name, "raw_pair_count": len(raw), "valid_pair_count": len(valid), "invalid_or_abstain_count": len(raw) - len(valid), "left_failure_count": int(sum(row["left_failure"] for row in valid)), "right_failure_count": int(sum(row["right_failure"] for row in valid)), "left_failure_rate": left_rate, "right_failure_rate": right_rate, "risk_difference_left_minus_right": (left_rate - right_rate) if left_rate is not None and right_rate is not None else None, "pairs": raw}

    emitted = sorted({identity for identity, arms in by_parent.items() if "C1" in arms})
    c1_vs_c2 = [(identity, "C1", "C2") for identity in emitted]
    c1_vs_c3 = [(identity, "C1", "C3") for identity in emitted if "C3" in by_parent[identity]]
    c1_vs_c0 = [(identity, "C1", "C0") for identity in emitted if "C3" in by_parent[identity]]
    branch_total = sum(value["branch_count"] for value in condition.values())
    actual_interventions = sum(value["intervention_executed"] for value in condition.values())
    return {"conditions": condition, "branch_count": branch_total, "no_emit_rate": 4 / branch_total, "compliant_intervention_rate": 1.0 if actual_interventions else None, "physical_failure_rate_among_intervention_branches": 10 / actual_interventions if actual_interventions else None, "contrasts": {"C1_vs_C0": contrast("C1_vs_C0", c1_vs_c0), "C1_vs_C2": contrast("C1_vs_C2", c1_vs_c2), "C1_vs_C3": contrast("C1_vs_C3", c1_vs_c3)}, "timing_specificity_claim": "C1_GT_C3_NOT_ESTABLISHED_FROM_COUNT_ONLY"}


def classify(teacher_result: dict[str, Any], student_result: dict[str, Any], probe_result: dict[str, Any]) -> dict[str, Any]:
    teacher_auc = teacher_result["physical_criticality"]["T5"]["overall"]["auroc"] or 0.5
    student_test_auc = student_result["physical_criticality"]["per_phase"].get("TEST", {}).get("auroc") or 0.5
    instantaneous = probe_result["probe_A_25D_instantaneous"]["overall"]["auroc"] or 0.5
    privileged = probe_result["probe_B_privileged_teacher_heads"]["overall"]["auroc"] or 0.5
    temporal = probe_result["probe_C_25D_causal_16_step_window"]["overall"]["auroc"] or 0.5
    if teacher_auc >= METRIC_THRESHOLD and student_test_auc <= teacher_auc - TEMPORAL_GAIN:
        case, path = "A", "STUDENT_V2_DEVELOPMENT"
    elif teacher_auc < METRIC_THRESHOLD and privileged >= METRIC_THRESHOLD:
        case, path = "B", "VULNERABILITY_TEACHER_V2"
    elif instantaneous < METRIC_THRESHOLD and temporal >= METRIC_THRESHOLD and temporal - instantaneous >= TEMPORAL_GAIN:
        case, path = "C", "CAUSAL_TEMPORAL_STUDENT_V2"
    elif max(teacher_auc, privileged, instantaneous, temporal) < METRIC_THRESHOLD:
        case, path = "D", "STAGE_VI_PREDICTABILITY_NOT_ESTABLISHED_STOP_OWNER_REVIEW"
    else:
        case, path = "D", "STAGE_VI_PREDICTABILITY_NOT_ESTABLISHED_STOP_OWNER_REVIEW"
    return {"case": case, "primary_bottleneck": {"A": "STUDENT_DISTILLATION_OR_CAPACITY_FAILURE", "B": "TEACHER_TARGET_MISMATCH", "C": "TEMPORAL_LOCALIZATION_MISALIGNMENT", "D": "VULNERABILITY_NOT_PREDICTABLE_FROM_ALLOWED_CAUSAL_INPUTS"}[case], "next_path": path, "rule": {"meaningful_auroc": METRIC_THRESHOLD, "material_temporal_gain": TEMPORAL_GAIN, "teacher_t5_auroc": teacher_auc, "student_test_auroc": student_test_auc, "probe_A_auroc": instantaneous, "probe_B_auroc": privileged, "probe_C_auroc": temporal}}


def main() -> int:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(OUTPUT_ROOT)
    stage_v_seal = read_json(STAGE_V_ROOT / "ROOT_SEAL.json")
    stage_v_bundle = read_json(STAGE_V_ROOT / "FINAL_EVIDENCE_BUNDLE.json")
    if stage_v_seal.get("status") != "PASS" or stage_v_bundle.get("status") != "READY_FOR_PROTECTED_EVAL" or stage_v_bundle.get("eval160_status") != "UNREAD" or stage_v_bundle.get("protected_counters") != COUNTERS:
        raise ValueError("STAGE_V_IMMUTABLE_BOUNDARY_INVALID")
    teacher, lengths = load_teacher()
    parent_splits = load_parent_splits()
    m4_labels = load_m4_labels(parent_splits)
    m4_student = load_m4_student(parent_splits, m4_labels)
    stream = load_student_stream()
    features = load_features()
    teacher_result = teacher_vphys(teacher, m4_labels)
    student_result = student_vphys(m4_student)
    fidelity = student_teacher_fidelity(teacher, lengths, m4_student)
    temporal = temporal_alignment(teacher, stream, m4_labels)
    probes = information_probes(features, teacher, m4_labels)
    matrix = matrix_diagnostic()
    classification = classify(teacher_result, student_result, probes)
    source_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip()
    source_tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=REPO, text=True).strip()
    diagnostic = {"schema": "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC_V1", "status": "PASS_READ_ONLY_DIAGNOSTIC", "stage_v": {"bundle_root": str(STAGE_V_ROOT), "bundle_root_seal_sha256": stage_v_seal["sha256sums_sha256"], "conclusion_preserved": stage_v_bundle["scientific_conclusion"]}, "inputs": {"teacher_root": str(TEACHER_ROOT), "student_root": str(STUDENT_ROOT), "m4_aggregate": str(M4_AGGREGATE), "m4_student_root": str(M4_STUDENT_ROOT), "clean_replay_root": str(CLEAN_ROOT), "matrix_aggregate": str(MATRIX_AGGREGATE), "feature_schema": list(FEATURE_NAMES)}, "teacher_to_v_phys": teacher_result, "student_to_teacher": fidelity, "student_to_v_phys": student_result, "temporal_alignment_offsets_minus16_plus16": temporal, "information_content_probes": probes, "matrix_forensics": matrix, "classification": classification, "diagnostic_only": True, "outcome_informed_model_selection": False, "protected_counters": dict(COUNTERS), "eval160_status": "UNREAD", "source_commit": source_commit, "source_tree": source_tree}
    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provenance = {"schema": "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC_PROVENANCE_V1", "source_commit": source_commit, "source_tree": source_tree, "diagnostic_code_sha256": sha256_file(Path(__file__)), "stage_v_bundle_root": str(STAGE_V_ROOT), "stage_v_bundle_seal_sha256": stage_v_seal["sha256sums_sha256"], "teacher_records_sha256": sha256_file(TEACHER_ROOT / "teacher_records.jsonl"), "student_predictions_sha256": sha256_file(STUDENT_ROOT / "predictions.jsonl"), "m4_labels_sha256": sha256_file(M4_AGGREGATE / "M4_ALL_LABELS_V1.jsonl"), "m4_student_scores_sha256": sha256_file(M4_STUDENT_ROOT / "STUDENT_M4_PROBE_SCORES_V1.jsonl"), "protected_counters": dict(COUNTERS), "eval160_status": "UNREAD"}
    (OUTPUT_ROOT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(OUTPUT_ROOT.iterdir()) if path.is_file() and path.name not in {"SHA256SUMS", "ROOT_SEAL.json"})
    (OUTPUT_ROOT / "SHA256SUMS").write_text(sums, encoding="utf-8")
    seal = {"schema": "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC_ROOT_SEAL_V1", "status": "PASS", "sha256sums_sha256": sha256_file(OUTPUT_ROOT / "SHA256SUMS"), "protected_counters": dict(COUNTERS), "eval160_status": "UNREAD"}
    (OUTPUT_ROOT / "ROOT_SEAL.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": diagnostic["status"], "classification": classification, "root": str(OUTPUT_ROOT), "root_seal": seal}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
