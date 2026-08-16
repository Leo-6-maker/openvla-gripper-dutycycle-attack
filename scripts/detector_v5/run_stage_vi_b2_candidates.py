"""Run the frozen, development-only Stage VI-B2 detector candidates."""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "detector_v5"))
import run_stage_vi_root_cause_diagnostic as base  # noqa: E402


CONFIG_PATH = REPO / "configs" / "STAGE_VI_B2_DEVELOPMENT_PROMOTION_CRITERIA_V1.json"
FORENSIC_ROOT = base.BASE / "STAGE_VI_B2_SUITE_FORENSIC_20260816T190000Z"
FORENSIC_SEAL = "3dddea293bfdc7d5b1ccabf0d31c1b1342587fae554edb8f9e0c1f115b2f1f91"
SEED = 20260816
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def verify_forensic() -> dict:
    seal = read_json(FORENSIC_ROOT / "ROOT_SEAL.json")
    if seal.get("status") != "PASS_STAGE_VI_B2_SUITE_FORENSIC" or seal.get("sha256sums_sha256") != sha256_file(FORENSIC_ROOT / "SHA256SUMS"):
        raise ValueError("FORENSIC_ROOT_SEAL_INVALID")
    if seal.get("sha256sums_sha256") != FORENSIC_SEAL:
        raise ValueError("FORENSIC_ROOT_SHA_BINDING_MISMATCH")
    data = read_json(FORENSIC_ROOT / "STAGE_VI_B2_SUITE_FORENSIC.json")
    if data.get("status") != "PASS_STAGE_VI_B2_SUITE_FORENSIC":
        raise ValueError("FORENSIC_STATUS_INVALID")
    checks = data["quality_checks"]
    required = {
        "future_features_used": False,
        "post_outcome_features_used": False,
        "parent_split_overlap": False,
        "t5_duplicate_parent_step_count": 0,
        "t5_parent_count": 40,
        "t5_row_count": 858,
        "protected_counters": COUNTERS,
    }
    for key, expected in required.items():
        if checks.get(key) != expected:
            raise ValueError(f"FORENSIC_QUALITY:{key}")
    return data


def standardize_windows(train_x: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (train_x - mean) / std, (all_x - mean) / std, mean, std


class CausalTCN(nn.Module):
    def __init__(self, channels: tuple[int, int] = (32, 32), kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(25, channels[0], kernel_size)
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size)
        self.head = nn.Linear(channels[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.relu(self.conv2(x))
        return self.head(x[:, :, -1]).squeeze(-1)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def ranking_loss(logits: torch.Tensor, y: torch.Tensor, groups: np.ndarray) -> torch.Tensor:
    losses = []
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        group_y = y[indices]
        positives = logits[indices][group_y > 0.5]
        negatives = logits[indices][group_y <= 0.5]
        if len(positives) and len(negatives):
            losses.append(F.softplus(-(positives[:, None] - negatives[None, :])).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def fit_tcn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    *,
    soft_targets: np.ndarray | None = None,
    ranking_weight: float = 0.0,
    direct_weight: float = 1.0,
    distill_weight: float = 0.0,
    epochs: int = 120,
) -> tuple[CausalTCN, np.ndarray, np.ndarray, np.ndarray]:
    seed_all(SEED)
    model = CausalTCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - y_train.sum()))
    direct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(neg / pos)))
    distill = nn.BCEWithLogitsLoss()
    soft_tensor = None if soft_targets is None else torch.as_tensor(soft_targets, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        loss = direct_weight * direct(logits, y_tensor)
        if soft_tensor is not None:
            loss = loss + distill_weight * distill(logits, soft_tensor)
        if ranking_weight:
            loss = loss + ranking_weight * ranking_loss(logits, y_tensor, groups)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(model(x_tensor)).cpu().numpy()
    return model, train_scores, np.asarray(model.conv1.weight.detach().cpu()), np.asarray(model.head.weight.detach().cpu())


def predict_tcn(model: CausalTCN, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(x, dtype=torch.float32))).cpu().numpy()


def metric(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    result = base.metric(y, score, threshold)
    prevalence = result.get("prevalence")
    result["auprc_lift"] = None if not result.get("auprc") or not prevalence else float(result["auprc"] / prevalence)
    result["emission_rate"] = float(np.mean(score >= threshold)) if len(score) else None
    return result


def select_threshold(y: np.ndarray, score: np.ndarray, config: dict) -> float:
    grid = np.arange(config["grid_start"], config["grid_stop"] + config["grid_step"] / 2, config["grid_step"])
    best = None
    for threshold in grid:
        pred = score >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        f1 = 0.0 if 2 * tp + fp + fn == 0 else (2.0 * tp) / (2.0 * tp + fp + fn)
        candidate = (f1, -float(threshold))
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold))
    return float(best[1])


def bootstrap_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> dict:
    unique = np.asarray(sorted(set(groups.tolist())))
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("auroc", "auprc", "auprc_lift", "top_decile_lift")}
    for _ in range(replicates):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[group] for group in chosen])
        current = metric(y[indices], score[indices], 0.5)
        for key in values:
            if current.get(key) is not None:
                values[key].append(float(current[key]))
    return {key: {"lower": float(np.quantile(vals, 0.025)), "upper": float(np.quantile(vals, 0.975)), "replicates": len(vals)} for key, vals in values.items() if vals}


def suite_metrics(y: np.ndarray, score: np.ndarray, groups: np.ndarray, suites: np.ndarray, threshold: float) -> dict:
    result = {}
    for suite in sorted(set(suites.tolist())):
        mask = suites == suite
        result[str(suite)] = metric(y[mask], score[mask], threshold)
    return result


def evaluate_split(y: np.ndarray, score: np.ndarray, groups: np.ndarray, suites: np.ndarray, threshold: float, bootstrap: bool = False) -> dict:
    result = metric(y, score, threshold)
    result["per_suite"] = suite_metrics(y, score, groups, suites, threshold)
    if bootstrap:
        result["bootstrap_ci"] = bootstrap_ci(y, score, groups, 1000, SEED)
    return result


def gate(candidate: dict, baseline: dict, config: dict) -> dict:
    rules = config["promotion_gate"]
    test = candidate["TEST"]
    delta_auroc = float(test["auroc"] - baseline["auroc"])
    delta_lift = float(test["auprc_lift"] - baseline["auprc_lift"])
    brier_ratio = float(test["brier"] / baseline["brier"]) if baseline["brier"] else None
    suite_checks = {}
    for suite, values in test["per_suite"].items():
        suite_checks[suite] = {
            "auroc": values["auroc"],
            "auprc_lift": values["auprc_lift"],
            "auroc_pass": values["auroc"] is not None and values["auroc"] >= rules["minimum_per_suite_auroc"],
            "auprc_lift_pass": values["auprc_lift"] is not None and values["auprc_lift"] >= rules["minimum_per_suite_auprc_lift"],
        }
    checks = {
        "overall_auroc_gain": delta_auroc,
        "overall_auroc_pass": delta_auroc >= rules["minimum_overall_auroc_gain_vs_frozen_student"],
        "overall_auprc_lift_gain": delta_lift,
        "overall_auprc_lift_pass": delta_lift >= rules["minimum_overall_auprc_lift_gain_vs_frozen_student"],
        "top_decile_lift": test["top_decile_lift"],
        "top_decile_pass": test["top_decile_lift"] is not None and test["top_decile_lift"] >= rules["minimum_overall_top_decile_lift"],
        "brier_ratio": brier_ratio,
        "brier_pass": brier_ratio is not None and brier_ratio <= rules["maximum_overall_brier_ratio_vs_frozen_student"],
        "ece_10bin": test["ece_10bin"],
        "ece_pass": test["ece_10bin"] is not None and test["ece_10bin"] <= rules["maximum_overall_ece_10bin"],
        "emission_rate": test["emission_rate"],
        "emission_pass": rules["emission_coverage_at_selected_threshold"]["minimum"] <= test["emission_rate"] <= rules["emission_coverage_at_selected_threshold"]["maximum"],
        "per_suite": suite_checks,
    }
    checks["per_suite_pass"] = all(item["auroc_pass"] and item["auprc_lift_pass"] for item in suite_checks.values())
    checks["pass"] = all(value for key, value in checks.items() if key.endswith("_pass")) and checks["per_suite_pass"]
    return checks


def make_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    splits = base.load_parent_splits()
    labels = base.load_m4_labels(splits)
    t5 = [row for row in labels if row["dose"] == "T5"]
    student = base.load_m4_student(splits, labels)
    if len(t5) != 858 or len(student) != 858:
        raise ValueError("B2_T5_ROW_COUNT")
    label_keys = {(str(row["canonical_parent_key"]), int(row["probe_step"])): int(row["y"]) for row in t5}
    feature_map = base.load_features()
    privileged_map, _ = base.load_privileged_features(labels)
    zeros = np.zeros(25, dtype=np.float64)
    x25, xwindow, xprivileged, y, groups, suites, split_names, baseline = [], [], [], [], [], [], [], []
    for row in sorted(student, key=lambda item: (str(item["canonical_parent_key"]), int(item["probe_step"]))):
        identity, step = str(row["canonical_parent_key"]), int(row["probe_step"])
        if (identity, step) not in label_keys or int(row["y"]) != label_keys[(identity, step)]:
            raise ValueError(f"B2_LABEL_JOIN:{identity}:{step}")
        vector = feature_map.get((identity, step))
        privileged = privileged_map.get((identity, step))
        if vector is None or privileged is None:
            raise ValueError(f"B2_FEATURE_JOIN:{identity}:{step}")
        x25.append(vector)
        xwindow.append(np.stack([feature_map.get((identity, past), zeros) for past in range(step - 15, step + 1)]))
        xprivileged.append(privileged)
        y.append(int(row["y"]))
        groups.append(identity)
        suites.append(str(row["suite"]))
        split_names.append(str(row["split"]))
        baseline.append(float(row["scores"]["physical_criticality"]))
    return tuple(np.asarray(item) for item in (x25, xwindow, xprivileged, y, groups, suites, split_names, baseline))


def main() -> None:
    config = read_json(CONFIG_PATH)
    forensic = verify_forensic()
    if config["status"] != "FROZEN_BEFORE_CANDIDATE_TRAINING":
        raise ValueError("B2_CONFIG_NOT_FROZEN")
    if git("diff", "--name-only") or git("diff", "--cached", "--name-only"):
        raise ValueError("B2_SOURCE_TRACKED_WORKTREE_NOT_CLEAN")
    source_commit, source_tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    x25, xwindow, xprivileged, y, groups, suites, splits, baseline_scores = make_rows()
    split_masks = {name: splits == name for name in ("TRAIN", "VAL", "TEST")}
    if {name: int(mask.sum()) for name, mask in split_masks.items()} != {"TRAIN": 506, "VAL": 182, "TEST": 170}:
        raise ValueError("B2_SPLIT_ROW_COUNTS")
    baseline_thresholds = read_json(base.STUDENT_ROOT / "thresholds.json")
    baseline_threshold = float(baseline_thresholds["physical_criticality"]["threshold"])
    baseline_test = evaluate_split(y[split_masks["TEST"]], baseline_scores[split_masks["TEST"]], groups[split_masks["TEST"]], suites[split_masks["TEST"]], baseline_threshold)
    models = {}
    candidate_results = {}
    train_mask, val_mask, test_mask = split_masks["TRAIN"], split_masks["VAL"], split_masks["TEST"]
    win_train, win_all, win_mean, win_std = standardize_windows(xwindow[train_mask], xwindow)

    a = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0))
    a.fit(x25[train_mask], y[train_mask])
    score_a = a.predict_proba(x25)[:, 1]
    models["B2-A_DIRECT_25D_BALANCED_LOGISTIC"] = ("pickle", a)

    b_model, _, _, _ = fit_tcn(win_train, y[train_mask], groups[train_mask], ranking_weight=0.2)
    score_b = predict_tcn(b_model, win_all)
    models["B2-B_CAUSAL_TCN_BALANCED_BCE_RANKING"] = ("torch", (b_model, win_mean, win_std))

    teacher = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=0))
    teacher.fit(xprivileged[train_mask], y[train_mask])
    teacher_train = teacher.predict_proba(xprivileged[train_mask])[:, 1]
    c_model, _, _, _ = fit_tcn(win_train, y[train_mask], groups[train_mask], soft_targets=teacher_train, direct_weight=0.5, distill_weight=0.5)
    score_c = predict_tcn(c_model, win_all)
    models["B2-C_SOFT_TV_DISTILL_DIRECT_VPHYS"] = ("torch", (c_model, win_mean, win_std))
    scores_by_id = {
        "B2-A_DIRECT_25D_BALANCED_LOGISTIC": score_a,
        "B2-B_CAUSAL_TCN_BALANCED_BCE_RANKING": score_b,
        "B2-C_SOFT_TV_DISTILL_DIRECT_VPHYS": score_c,
    }
    for candidate_id, scores in scores_by_id.items():
        threshold = select_threshold(y[val_mask], scores[val_mask], config["threshold_policy"])
        splits_out = {}
        for name, mask in split_masks.items():
            splits_out[name] = evaluate_split(y[mask], scores[mask], groups[mask], suites[mask], threshold, bootstrap=name == "TEST")
        candidate_results[candidate_id] = {
            "candidate_id": candidate_id,
            "selected_threshold_from_val": threshold,
            "splits": splits_out,
            "TEST": splits_out["TEST"],
            "gate": gate(splits_out, baseline_test, config),
        }

    passing = [result for result in candidate_results.values() if result["gate"]["pass"]]
    selected = None
    if passing:
        selected = sorted(passing, key=lambda item: (-item["TEST"]["auprc_lift"], -item["TEST"]["auroc"], item["candidate_id"]))[0]["candidate_id"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = base.BASE / f"STAGE_VI_B2_CANDIDATE_DEVELOPMENT_{timestamp}"
    if output.exists():
        raise FileExistsError(output)
    (output / "models").mkdir(parents=True)
    for candidate_id, (kind, model) in models.items():
        name = candidate_id.replace("-", "_")
        if kind == "pickle":
            with (output / "models" / f"{name}.pkl").open("wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            tcn, mean, std = model
            torch.save({"state_dict": tcn.state_dict(), "mean": mean, "std": std, "schema": "16x25D"}, output / "models" / f"{name}.pt")
    with (output / "models" / "B2_C_PRIVILEGED_TEACHER.pkl").open("wb") as handle:
        pickle.dump(teacher, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "criteria.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = {
        "schema": "STAGE_VI_B2_CANDIDATE_DEVELOPMENT_V1",
        "status": "PASS_STAGE_VI_B2_CANDIDATE_DEVELOPMENT",
        "development_decision": "CANDIDATE_PASS_CREATE_PRE_HOLDOUT_LOCK" if selected else "NO_CANDIDATE_PROMOTION",
        "selected_candidate": selected,
        "candidate_count": len(candidate_results),
        "candidates": candidate_results,
        "baseline": {"frozen_stage_v_student_threshold": baseline_threshold, "TEST": baseline_test},
        "data": {"t5_rows": int(len(y)), "parent_count": int(len(set(groups.tolist()))), "split_counts": {name: int(mask.sum()) for name, mask in split_masks.items()}, "v_phys_prevalence": float(y.mean())},
        "frozen_criteria": {"config": str(CONFIG_PATH), "config_sha256": sha256_file(CONFIG_PATH)},
        "inputs": {"forensic_root": str(FORENSIC_ROOT), "forensic_root_seal_sha256": FORENSIC_SEAL, "m4_student_root": str(base.M4_STUDENT_ROOT), "clean_feature_root": str(base.CLEAN_ROOT)},
        "quality_checks": {"fresh_m4_execution": False, "outcomes_read_from_clean_trajectory": False, "future_features_used": False, "parent_grouped_split": True, "teacher_student_weights_changed": False, "eval160_status": "UNREAD", "protected_counters": COUNTERS},
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_status": git("status", "--porcelain"),
        "forensic_summary": {"status": forensic["status"], "libero_10_direct_25d_auroc": forensic["suite_forensics"]["libero_10"]["probes"]["direct_25d"]["auroc"]},
    }
    (output / "B2_RESULTS.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json"})
    sums = "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files)
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(output / "SHA256SUMS")
    (output / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    (output / "ROOT_SEAL.json").write_text(json.dumps({"schema": "STAGE_VI_B2_CANDIDATE_DEVELOPMENT_ROOT_SEAL_V1", "status": "PASS_STAGE_VI_B2_CANDIDATE_DEVELOPMENT", "sha256sums_sha256": sums_sha, "eval160_status": "UNREAD", "protected_counters": COUNTERS}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output), "root_seal": sums_sha, "development_decision": results["development_decision"], "selected_candidate": selected, "candidate_gates": {key: value["gate"]["pass"] for key, value in candidate_results.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
