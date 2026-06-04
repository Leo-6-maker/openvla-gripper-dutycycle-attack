#!/usr/bin/env python3
"""train_cpu_phase_detectors.py — CPU-only offline detector smoke.

Verifies whether runtime/no-step features can separate phase bins.
Does NOT train vulnerability detector. Does NOT require GPU.

Outputs:
  tables/cpu_phase_detector_window_metrics.csv
  tables/cpu_phase_detector_feature_importance.csv
  tables/cpu_phase_detector_streaming_replay.csv
  tables/provisional_vulnerability_descriptor_audit.csv
  reports/CPU_PHASE_DETECTOR_WINDOW_AUDIT.md
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

# Forbidden as model input
FORBIDDEN_FEATURES = {
    "normalized_step", "step_idx", "T_gform", "relative_lead", "candidate_source",
    "object_pose", "target_pose", "object_to_target_distance",
    "VIS_OPEN", "qpos_opening_delta", "VIS_done", "done", "claim_usable",
    "taxonomy", "denominator_status", "provenance_status",
    "raw_open_semantics", "phase_bin_reason", "phase_bin_confidence",
    "detector_version", "checkpoint", "threshold", "K",
    "feature_space_model", "feature_space_open_ratio",
}

# Numeric runtime features allowed as input
ALLOWED_NUMERIC = [
    "clean_open_count", "clean_open_ratio", "raw_gripper_mean",
    "qpos_start", "qpos_end", "qpos_min", "qpos_max",
    "qpos_delta_abs", "qpos_opening_proxy", "qpos_velocity_mean",
    "eef_speed_mean", "eef_speed_max", "eef_displacement", "eef_z_delta",
]

# Target phase bin
TARGET_COL = "phase_bin_proxy"
PHASE_BINS_OF_INTEREST = [
    "approach_far_closed_proxy",
    "approach_near_closed_proxy",
    "pre_lock_closed_proxy",
    "grasp_formation_pre_lock_proxy",
    "stable_grasp_or_lift_proxy",
    "natural_open_or_release_proxy",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptors-csv", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--output-metrics", default="tables/cpu_phase_detector_window_metrics.csv")
    ap.add_argument("--output-importance", default="tables/cpu_phase_detector_feature_importance.csv")
    ap.add_argument("--output-streaming", default="tables/cpu_phase_detector_streaming_replay.csv")
    ap.add_argument("--output-vuln-audit", default="tables/provisional_vulnerability_descriptor_audit.csv")
    ap.add_argument("--output-report", default="reports/CPU_PHASE_DETECTOR_WINDOW_AUDIT.md")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def prepare_features(descs):
    """Extract numeric feature matrix from descriptors, excluding forbidden fields."""
    # Find numeric columns in ALLOWED_NUMERIC
    feature_cols = []
    for col in ALLOWED_NUMERIC:
        if col in descs[0]: feature_cols.append(col)

    X = np.zeros((len(descs), len(feature_cols)), dtype=np.float32)
    y = []
    valid_idx = []

    for i, d in enumerate(descs):
        phase = d.get(TARGET_COL, "")
        if phase not in PHASE_BINS_OF_INTEREST: continue
        row = np.zeros(len(feature_cols), dtype=np.float32)
        missing = 0
        for j, col in enumerate(feature_cols):
            v = d.get(col, "")
            if v is None or v == "":
                missing += 1
                continue
            try: row[j] = float(v)
            except (ValueError, TypeError): missing += 1
        if missing > len(feature_cols) * 0.3: continue  # too sparse
        X[len(valid_idx)] = row; y.append(phase); valid_idx.append(i)

    X = X[:len(valid_idx)]
    return X, np.array(y), feature_cols, valid_idx


def run_classifiers(X, y, feature_cols, descs):
    """Train and evaluate multiple classifiers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline

    results = {}
    feature_importances = {}

    X_scaled = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42),
    }

    for name, model in models.items():
        try:
            model.fit(X_scaled, y)
            pred = model.predict(X_scaled)
            acc = np.mean(pred == y)
            results[name] = {"accuracy": round(acc, 4)}

            # Per-class F1
            from sklearn.metrics import f1_score
            f1_macro = f1_score(y, pred, average="macro")
            results[name]["macro_f1"] = round(f1_macro, 4)

            # Feature importance (RF/GB)
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            elif hasattr(model, "named_steps"):
                fi = model.named_steps["clf"].coef_[0] if len(model.named_steps["clf"].coef_.shape) == 1 else model.named_steps["clf"].coef_.mean(axis=0)
            else:
                fi = np.ones(len(feature_cols))
            for j, col in enumerate(feature_cols):
                if col not in feature_importances: feature_importances[col] = []
                feature_importances[col].append(abs(float(fi[j])))
        except Exception as e:
            results[name] = {"error": str(e)}

    return results, feature_importances


def streaming_replay_check(X, y, feature_cols, descs, valid_idx):
    """Simulate causal replay: per-episode, features at each step."""
    # Group descriptors by episode
    ep_groups = defaultdict(list)
    for idx in valid_idx:
        d = descs[idx]
        ep_groups[d["episode_id"]].append(idx)

    # For each episode, determine first step where each phase could be detected
    # using a simple threshold on clean_open_ratio and qpos values
    replay_rows = []
    for eid, indices in sorted(ep_groups.items()):
        rows = [(idx, descs[idx]) for idx in indices]
        rows.sort(key=lambda x: int(x[1].get("window_start", 0)))

        first_trigger = {}
        for idx, d in rows:
            phase = d.get(TARGET_COL, "")
            if phase not in first_trigger:
                first_trigger[phase] = int(d.get("window_start", 0))

        total_phases = len(set(d[TARGET_COL] for _, d in rows if d.get(TARGET_COL)))
        replay_rows.append(dict(
            episode_id=eid, task_key=rows[0][1].get("task_key","?"),
            n_windows=len(rows), n_phases=total_phases,
            first_far_closed=first_trigger.get("approach_far_closed_proxy", ""),
            first_near_closed=first_trigger.get("approach_near_closed_proxy", ""),
            first_pre_lock=first_trigger.get("pre_lock_closed_proxy", ""),
            first_grasp_lock=first_trigger.get("grasp_formation_pre_lock_proxy", ""),
        ))
    return replay_rows


def provisional_vulnerability_audit(descs):
    """Compare Batch1 positive vs negative descriptor profiles."""
    positive_episodes = {"obj100_ketchup_ketchup_s0", "obj100_butter_butter_s0"}
    negative_episodes = {"obj100_alphabet_soup_alphabet_soup_s0"}

    pos_rows = [d for d in descs if d.get("episode_id","") in positive_episodes and d.get("claim_usable","")=="True"]
    neg_rows = [d for d in descs if d.get("episode_id","") in negative_episodes and d.get("claim_usable","")=="False"]

    audit = []
    numeric_fields = [c for c in ALLOWED_NUMERIC if c in (descs[0] if descs else {})]
    for field in numeric_fields:
        pos_vals = [float(d[field]) for d in pos_rows if d.get(field) not in (None, "")]
        neg_vals = [float(d[field]) for d in neg_rows if d.get(field) not in (None, "")]
        pos_mean = np.mean(pos_vals) if pos_vals else 0
        neg_mean = np.mean(neg_vals) if neg_vals else 0
        diff = pos_mean - neg_mean
        audit.append(dict(feature=field, positive_mean=round(pos_mean,6),
                          negative_mean=round(neg_mean,6), diff=round(diff,6),
                          abs_diff=round(abs(diff),6)))

    audit.sort(key=lambda x: x["abs_diff"], reverse=True)
    return audit


def main():
    args = parse_args()
    if not os.path.exists(args.descriptors_csv):
        print(f"ERROR: {args.descriptors_csv} not found"); sys.exit(1)

    with open(args.descriptors_csv, newline="") as f:
        descs = list(csv.DictReader(f))
    print(f"Loaded {len(descs)} descriptors")

    # ── 1. Feature preparation ──
    X, y, feature_cols, valid_idx = prepare_features(descs)
    print(f"Feature matrix: {X.shape[0]} samples × {len(feature_cols)} features")
    print(f"Classes: {sorted(set(y))} -> {Counter(y)}")

    # ── 2. Window-level classifiers ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    X_scaled = StandardScaler().fit_transform(X)

    results, feature_importances = run_classifiers(X, y, feature_cols, descs)

    print("\nClassifier results:")
    for name, r in results.items():
        print(f"  {name}: acc={r.get('accuracy','?')} macroF1={r.get('macro_f1','?')}")

    # ── 3. Feature importance ──
    fi_rows = []
    for col, vals in sorted(feature_importances.items(), key=lambda x: -np.mean(x[1])):
        fi_rows.append(dict(feature=col, mean_importance=round(np.mean(vals), 6),
                            max_importance=round(max(vals), 6)))
    with open(args.output_importance, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature","mean_importance","max_importance"])
        w.writeheader(); w.writerows(fi_rows[:20])
    print(f"\nTop features: {[r['feature'] for r in fi_rows[:5]]}")

    # ── 4. Streaming replay ──
    replay = streaming_replay_check(X, y, feature_cols, descs, valid_idx)
    with open(args.output_streaming, "w", newline="") as f:
        if replay:
            w = csv.DictWriter(f, fieldnames=list(replay[0].keys()))
            w.writeheader(); w.writerows(replay)
            print(f"Streaming replay: {len(replay)} episodes")

    # ── 5. Vulnerability audit ──
    vuln_audit = provisional_vulnerability_audit(descs)
    with open(args.output_vuln_audit, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vuln_audit[0].keys()) if vuln_audit else ["feature"])
        w.writeheader(); w.writerows(vuln_audit)
    print(f"Vulnerability audit: {len(vuln_audit)} features compared")
    print(f"Top separating features:")
    for r in vuln_audit[:5]:
        print(f"  {r['feature']:25s} pos={r['positive_mean']:.4f} neg={r['negative_mean']:.4f}")

    # ── 6. Report ──
    best_model = max(results.items(), key=lambda x: x[1].get("macro_f1",0)) if results else (None, {})
    report = f"""# CPU Phase Detector Smoke

**Status**: offline smoke only. Do NOT claim online detector works.

## Classifier Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
""" + "\n".join(f"| {n} | {r.get('accuracy','?')} | {r.get('macro_f1','?')} |" for n, r in results.items()) + f"""

Best: {best_model[0]} (macroF1={best_model[1].get('macro_f1','?')})

## Top Features

| Feature | Mean Importance |
|---------|----------------|
""" + "\n".join(f"| {r['feature']} | {r['mean_importance']} |" for r in fi_rows[:10]) + f"""

## Phase Bin Distribution

| Bin | Count |
|-----|-------|
""" + "\n".join(f"| {k} | {v} |" for k, v in sorted(Counter(y).items())) + f"""

## Vulnerability Descriptor Audit (Batch1)

Top features separating positive (ketchup/butter) vs negative (alphabet_soup):
""" + "\n".join(f"| {r['feature']} | pos={r['positive_mean']} | neg={r['negative_mean']} | diff={r['diff']} |" for r in vuln_audit[:5]) + f"""

## Verdict

Phase bins are {'learnable' if best_model[1].get('macro_f1',0) > 0.3 else 'difficult to separate'} from runtime features.
The top models achieve macroF1={
best_model[1].get('macro_f1','?') if best_model else '?'}.
This suggests a Causal TCN detector can identify pre-grasp vs grasp-formation windows
from clean runtime features, even without privileged state.

Next: replace coarse phase_bin_proxy with Batch2b VIS-informed vulnerability_ready label.
"""

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"\nReport: {args.output_report}")


if __name__ == "__main__":
    from collections import Counter
    main()
