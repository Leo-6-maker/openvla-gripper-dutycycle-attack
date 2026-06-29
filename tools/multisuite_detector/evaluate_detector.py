#!/usr/bin/env python3
"""Multi-suite detector evaluation using shared score_fsm_legacy_v1.

Strict checkpoint loading: fails on any state_dict mismatch.
Thresholds bound from checkpoint, CLI override rejected in formal mode.
Input SHAs cross-bound in output for provenance.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, torch
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strict_loader import (
    load_episode_index, load_features, load_teacher_events, VALID_SUITES,
)
from gripper_attack.sc5mlp_v1 import SC5MLPV1, SC5_FEATURES, SC5_PHASES, N_FEATURES, N_PHASES
from score_fsm_legacy_v1 import run_fsm_legacy_v1, model_to_scores


# Dynamically extract expected state keys from reference model
_ref = SC5MLPV1()
SC5MLPV1_STATE_KEYS = set(_ref.state_dict().keys())
del _ref

SHA_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_HEX_RE = re.compile(r"^[0-9a-f]{40}$")
CHECKPOINT_SCHEMA_VERSION = "sc5mlp_v1.0"

REQUIRED_CHECKPOINT_KEYS = {
    "schema_version", "model_type",
    "model_state", "mean", "std", "feature_names", "phase_classes",
    "tau_corridor", "tau_release", "guard", "K",
    "fsm_version",
    "split_mode", "normalization_source", "normalization_sha256",
    "feature_csv_sha256", "label_csv_sha256",
    "episode_index_sha256", "split_file_sha256", "split_definition_sha256",
    "config_sha256",
    "seed", "epoch", "n_train", "n_val", "n_suites",
    "checkpoint_metric", "cohort",
    "repo_commit", "git_dirty",
}

CHECKPOINT_SHA_BINDINGS = [
    ("feature_csv_sha256", "feature_csv"),
    ("label_csv_sha256", "label_csv"),
    ("episode_index_sha256", "episode_index"),
    ("split_file_sha256", "split_file"),
]


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_sha_field(value, field_name):
    if not isinstance(value, str):
        raise ValueError("{} must be string, got {}".format(field_name, type(value).__name__))
    if not SHA_HEX_RE.match(value):
        raise ValueError("{} must be 64-char hex SHA-256: {}".format(field_name, value[:32]))


def _validate_commit_field(value, field_name):
    if not isinstance(value, str):
        raise ValueError("{} must be string".format(field_name))
    if not COMMIT_HEX_RE.match(value):
        raise ValueError("{} must be 40-char hex: {}".format(field_name, value[:20]))


def _validate_finite_float(value, field_name, lo=None, hi=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValueError("{} must be numeric (not bool), got {}".format(field_name, type(value).__name__))
    fv = float(value)
    if not np.isfinite(fv):
        raise ValueError("{} must be finite, got {}".format(field_name, fv))
    if lo is not None and fv < lo:
        raise ValueError("{}={} < {}".format(field_name, fv, lo))
    if hi is not None and fv > hi:
        raise ValueError("{}={} > {}".format(field_name, fv, hi))
    return fv


def _validate_int(value, field_name, lo=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("{} must be integer (not bool), got {}: {}".format(field_name, type(value).__name__, value))
    iv = int(value)
    if float(value) != float(iv):
        raise ValueError("{}={} is not an exact integer".format(field_name, value))
    if lo is not None and iv < lo:
        raise ValueError("{}={} < {}".format(field_name, iv, lo))
    return iv


VALID_CHECKPOINT_METRICS = {"val_loss", "val_suite_macro_event_f1"}


def compute_normalization_sha(mean, std):
    artifact = {
        "feature_names": list(SC5_FEATURES),
        "dtype": "float32",
        "mean": [float(x) for x in np.asarray(mean, dtype=np.float32)],
        "std": [float(x) for x in np.asarray(std, dtype=np.float32)],
    }
    return hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()


def load_checkpoint_strict(path: str, provided_files: dict = None):
    """Load checkpoint with comprehensive strict validation.

    Validates: schema version, all required fields, SHA format+content,
    parameter types+ranges, normalization digest, git_dirty=false,
    fsm_version=legacy_v1, model state exact match.
    SHA cross-verification is unconditional (empty strings rejected).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Schema version and model type
    if ckpt.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("schema_version={} expected={}".format(
            ckpt.get("schema_version"), CHECKPOINT_SCHEMA_VERSION))
    if ckpt.get("model_type") != "SC5MLPV1":
        raise ValueError("model_type must be SC5MLPV1")

    # All required keys present
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(ckpt.keys()))
    if missing:
        raise ValueError("Missing required keys: {}".format(missing))

    # Git must be clean (strict type: only bool False accepted)
    gd = ckpt["git_dirty"]
    if not (isinstance(gd, bool) and gd is False):
        raise ValueError("git_dirty must be bool False, got {}: {}".format(type(gd).__name__, gd))
    _validate_commit_field(ckpt["repo_commit"], "repo_commit")

    # Checkpoint metric allowlist
    cm = ckpt.get("checkpoint_metric")
    if cm not in VALID_CHECKPOINT_METRICS:
        raise ValueError("checkpoint_metric={} not in {}".format(cm, VALID_CHECKPOINT_METRICS))

    # FSM version
    if ckpt.get("fsm_version") != "legacy_v1":
        raise ValueError("fsm_version={} expected=legacy_v1".format(ckpt.get("fsm_version")))

    # String constants
    if ckpt["split_mode"] != "frozen":
        raise ValueError("split_mode={}".format(ckpt["split_mode"]))
    if ckpt["normalization_source"] != "train_only":
        raise ValueError("normalization_source={}".format(ckpt["normalization_source"]))
    if ckpt.get("cohort") not in ("primary_eligible", "safety_abstention", "all"):
        raise ValueError("cohort={}".format(ckpt.get("cohort")))

    # All SHA fields: strict format + unconditional cross-verification
    sha_fields = ["feature_csv_sha256", "label_csv_sha256", "episode_index_sha256",
                  "split_file_sha256", "split_definition_sha256",
                  "config_sha256", "normalization_sha256"]
    for sf in sha_fields:
        _validate_sha_field(ckpt[sf], sf)

    # SHA cross-verification (unconditional — empty strings are already rejected above)
    if provided_files:
        for ckpt_key, file_key in CHECKPOINT_SHA_BINDINGS:
            file_path = provided_files.get(file_key)
            if file_path:
                actual = sha256_file(str(file_path))
                expected = ckpt[ckpt_key]
                if actual != expected:
                    raise ValueError(
                        "SHA MISMATCH: {}: checkpoint={} file={}".format(
                            ckpt_key, expected[:16], actual[:16]))
        # Re-verify split definition SHA
        if "split_file" in provided_files:
            with open(provided_files["split_file"]) as f:
                split_data = json.load(f)
            canonical = json.dumps(split_data, sort_keys=True).encode()
            actual_def = hashlib.sha256(canonical).hexdigest()
            expected_def = ckpt["split_definition_sha256"]
            if actual_def != expected_def:
                raise ValueError(
                    "SPLIT_DEFINITION SHA MISMATCH: checkpoint={} recomputed={}".format(
                        expected_def[:16], actual_def[:16]))

    # Parameter validation with type+ranges
    tau_c = _validate_finite_float(ckpt["tau_corridor"], "tau_corridor", 0.0, 1.0)
    tau_r = _validate_finite_float(ckpt["tau_release"], "tau_release", 0.0, 1.0)
    guard = _validate_int(ckpt["guard"], "guard", 0)
    K = _validate_int(ckpt["K"], "K", 1)
    seed = _validate_int(ckpt["seed"], "seed")
    epoch = _validate_int(ckpt["epoch"], "epoch", 1)
    n_train = _validate_int(ckpt["n_train"], "n_train", 1)
    n_val = _validate_int(ckpt["n_val"], "n_val", 1)
    n_suites = _validate_int(ckpt["n_suites"], "n_suites", 1)

    # Feature/phase consistency
    if list(ckpt["feature_names"]) != list(SC5_FEATURES):
        raise ValueError("feature_names mismatch")
    if list(ckpt["phase_classes"]) != list(SC5_PHASES):
        raise ValueError("phase_classes mismatch")

    # Normalization
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    if mean.shape != (25,) or std.shape != (25,):
        raise ValueError("mean/std shape != (25,)")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("NaN/Inf in mean/std")
    if not np.all(std > 0):
        raise ValueError("Zero/negative std")
    # Recompute and verify normalization SHA
    actual_norm_sha = compute_normalization_sha(mean, std)
    if actual_norm_sha != ckpt["normalization_sha256"]:
        raise ValueError("normalization_sha256 mismatch: checkpoint={} recomputed={}".format(
            ckpt["normalization_sha256"][:16], actual_norm_sha[:16]))

    # Strict model state
    state = ckpt["model_state"]
    expected_keys = SC5MLPV1_STATE_KEYS
    actual_keys = set(state.keys())
    if actual_keys != expected_keys:
        extra = sorted(actual_keys - expected_keys)
        missing_keys = sorted(expected_keys - actual_keys)
        parts = []
        if extra:
            parts.append("extra: {}".format(extra))
        if missing_keys:
            parts.append("missing: {}".format(missing_keys))
        raise ValueError("model_state key mismatch: {}".format("; ".join(parts)))

    ref_state = SC5MLPV1().state_dict()
    for k in expected_keys:
        if state[k].shape != ref_state[k].shape:
            raise ValueError("Shape mismatch {}: ckpt={} ref={}".format(
                k, tuple(state[k].shape), tuple(ref_state[k].shape)))
        if state[k].dtype != ref_state[k].dtype:
            raise ValueError("Dtype mismatch {}: ckpt={} ref={}".format(
                k, state[k].dtype, ref_state[k].dtype))
        if not torch.isfinite(state[k]).all():
            raise ValueError("Non-finite values in tensor: {}".format(k))

    model = SC5MLPV1()
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, mean, std, tau_c, tau_r, guard, K, ckpt


def evaluate_episode(model, features, teacher_info, tau_c, tau_r, guard, K=10):
    feats = torch.from_numpy(features)
    cp, rp, phase_names = model_to_scores(model, feats)
    emitted, emit_step = run_fsm_legacy_v1(cp, rp, phase_names, tau_c, tau_r, guard)

    anchor = teacher_info.get("anchor", -1)
    wstart = teacher_info.get("window_start", anchor)
    wend = teacher_info.get("window_end", anchor + K)
    has_event = teacher_info.get("has_event", anchor >= 0)
    in_window = bool(emitted and anchor >= 0 and wstart <= emit_step <= wend)

    return {
        "emitted": emitted, "emit_step": emit_step, "n_steps": len(features),
        "teacher_anchor": anchor,
        "anchor_error": emit_step - anchor if (emitted and anchor >= 0) else None,
        "absolute_error": abs(emit_step - anchor) if (emitted and anchor >= 0) else None,
        "in_window": in_window,
        "early": bool(emitted and anchor >= 0 and emit_step < wstart),
        "late": bool(emitted and anchor >= 0 and emit_step > wend),
        "k_contained": bool(emitted and anchor >= 0 and wstart <= emit_step <= wstart + K),
        "has_teacher_event": has_event,
        "event_type": teacher_info.get("event_type", "unknown"),
    }


def compute_metrics(results):
    n = max(1, len(results))
    emitted = [r for r in results if r["emitted"]]
    no_emit = [r for r in results if not r["emitted"]]
    has_event = [r for r in results if r.get("has_teacher_event")]
    no_event = [r for r in results if not r.get("has_teacher_event")]

    tp = sum(1 for r in results if r["has_teacher_event"] and r["emitted"] and r["in_window"])
    fp = sum(1 for r in results if r["emitted"] and not r["in_window"])
    fn = sum(1 for r in results if r["has_teacher_event"] and not r["in_window"])
    tn = sum(1 for r in results if not r["has_teacher_event"] and not r["emitted"])
    in_window_emit = [r for r in emitted if r.get("in_window")]
    k_hit = [r for r in emitted if r.get("k_contained")]
    correct_abstention = sum(1 for r in no_emit if not r.get("has_teacher_event"))

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(0.001, prec + rec)
    m = {
        "n_episodes": len(results), "n_emitted": len(emitted),
        "n_no_emission": len(no_emit),
        "n_teacher_positive": len(has_event), "n_teacher_negative": len(no_event),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "emission_rate": len(emitted) / n,
        "event_precision": prec,
        "event_recall": rec,
        "event_f1": f1,
        "k_containment_rate": len(k_hit) / max(1, len(emitted)),
        "false_emits_per_episode": fp / n,
        "correct_abstention_rate": correct_abstention / max(1, len(no_event)) if no_event else 0,
    }
    errors = [r["absolute_error"] for r in emitted if r.get("absolute_error") is not None]
    if errors:
        m["median_absolute_error"] = float(np.median(errors))
        m["mean_absolute_error"] = float(np.mean(errors))
        m["p90_absolute_error"] = float(np.percentile(errors, 90))
    early = sum(1 for r in emitted if r.get("early"))
    late = sum(1 for r in emitted if r.get("late"))
    m["early_rate"] = early / max(1, len(emitted))
    m["late_rate"] = late / max(1, len(emitted))
    return m


def _validate_f1_evidence(ckpt_meta):
    """Strict F1 evidence validation: no defaults, cross-verify aggregates."""
    f1_req = ["val_suite_macro_event_f1", "val_suite_macro_event_f1_details",
               "multi_event_excluded_total", "multi_event_excluded_keys_sha256",
               "selection_false_emits_per_episode", "selection_tie_breaker"]
    missing_f1 = sorted(set(f1_req) - set(ckpt_meta.keys()))
    if missing_f1:
        raise ValueError("F1 checkpoint missing: {}".format(missing_f1))

    _validate_sha_field(ckpt_meta["multi_event_excluded_keys_sha256"], "multi_event_excluded_keys_sha256")
    macro_f1 = _validate_finite_float(ckpt_meta["val_suite_macro_event_f1"], "val_suite_macro_event_f1", 0.0, 1.0)
    excl_total = _validate_int(ckpt_meta["multi_event_excluded_total"], "multi_event_excluded_total", 0)
    false_emits = _validate_finite_float(ckpt_meta["selection_false_emits_per_episode"], "selection_false_emits_per_episode", 0.0)

    f1d = ckpt_meta["val_suite_macro_event_f1_details"]
    if not isinstance(f1d, dict) or len(f1d) == 0:
        raise ValueError("val_suite_macro_event_f1_details is empty or not dict")

    sum_excl = 0
    sum_scored = 0
    sum_fp = 0
    suite_f1s = []
    required_per_suite = {"tp", "fp", "fn", "tn", "n_input_episodes", "n_scored_episodes",
                           "excluded_multi_event", "precision", "recall", "f1"}

    for s, d in sorted(f1d.items()):
        if not isinstance(d, dict):
            raise ValueError("Suite {} details is not dict: {}".format(s, type(d).__name__))
        missing_keys = required_per_suite - set(d.keys())
        if missing_keys:
            raise ValueError("Suite {} missing keys: {}".format(s, sorted(missing_keys)))

        tp = _validate_int(d["tp"], "suite.{}.tp".format(s), 0)
        fp = _validate_int(d["fp"], "suite.{}.fp".format(s), 0)
        fn = _validate_int(d["fn"], "suite.{}.fn".format(s), 0)
        tn = _validate_int(d["tn"], "suite.{}.tn".format(s), 0)
        n_in = _validate_int(d["n_input_episodes"], "suite.{}.n_input".format(s), 0)
        n_sc = _validate_int(d["n_scored_episodes"], "suite.{}.n_scored".format(s), 1)
        n_ex = _validate_int(d["excluded_multi_event"], "suite.{}.excluded".format(s), 0)
        prec = _validate_finite_float(d["precision"], "suite.{}.precision".format(s), 0.0, 1.0)
        rec = _validate_finite_float(d["recall"], "suite.{}.recall".format(s), 0.0, 1.0)
        f1 = _validate_finite_float(d["f1"], "suite.{}.f1".format(s), 0.0, 1.0)

        if n_in != n_sc + n_ex:
            raise ValueError("Suite {}: n_input({}) != n_scored({}) + excluded({})".format(s, n_in, n_sc, n_ex))
        if n_sc == 0:
            raise ValueError("Suite {}: n_scored_episodes=0".format(s))

        # Cross-verify derived metrics
        exp_prec = tp / max(1, tp + fp)
        exp_rec = tp / max(1, tp + fn)
        exp_f1 = 2 * exp_prec * exp_rec / max(0.001, exp_prec + exp_rec)
        if abs(prec - exp_prec) > 0.001 or abs(rec - exp_rec) > 0.001 or abs(f1 - exp_f1) > 0.001:
            raise ValueError("Suite {}: stored metrics (p={:.4f} r={:.4f} f1={:.4f}) "
                             "diverge from TP/FP/FN (p={:.4f} r={:.4f} f1={:.4f})".format(
                                 s, prec, rec, f1, exp_prec, exp_rec, exp_f1))
        if fp + fn != n_sc - tp - tn:
            raise ValueError("Suite {}: fp({})+fn({})+tp({})+tn({}) != n_scored({})".format(s, fp, fn, tp, tn, n_sc))

        sum_excl += n_ex
        sum_scored += n_sc
        sum_fp += fp
        suite_f1s.append(f1)

    if sum_excl != excl_total:
        raise ValueError("excluded_total({}) != sum per-suite({})".format(excl_total, sum_excl))
    exp_macro_f1 = float(np.mean(suite_f1s))
    if abs(macro_f1 - exp_macro_f1) > 0.001:
        raise ValueError("macro F1({:.4f}) != mean per-suite F1({:.4f})".format(macro_f1, exp_macro_f1))
    exp_false = sum_fp / max(1, sum_scored)
    if abs(false_emits - exp_false) > 0.001:
        raise ValueError("false_emits({:.4f}) != sum_fp/scored({:.4f})".format(false_emits, exp_false))


def per_suite_metrics(results, suite_map):
    by_suite = defaultdict(list)
    for r in results:
        by_suite[suite_map.get(r["episode_key"], "unknown")].append(r)
    out = {}
    for s in sorted(by_suite):
        out[s] = compute_metrics(by_suite[s])
        out[s]["n_episodes"] = len(by_suite[s])
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate multi-suite detector")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--episode_index", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--split_key", default="test")
    ap.add_argument("--output", default="-")
    ap.add_argument("--tau_corridor", type=float, default=None,
                    help="Override (must equal checkpoint value in formal mode)")
    ap.add_argument("--tau_release", type=float, default=None)
    ap.add_argument("--guard", type=int, default=None)
    args = ap.parse_args()

    # Strict checkpoint load with SHA cross-verification
    provided = {
        "feature_csv": args.feature_csv,
        "label_csv": args.label_csv,
        "episode_index": args.episode_index,
        "split_file": args.split_file,
    }
    model, mean, std, tau_c, tau_r, guard, K, ckpt_meta = load_checkpoint_strict(args.checkpoint, provided)

    # Conditional F1 evidence: required when checkpoint_metric == val_suite_macro_event_f1
    if ckpt_meta.get("checkpoint_metric") == "val_suite_macro_event_f1":
        _validate_f1_evidence(ckpt_meta)

    # Thresholds: CLI override must match checkpoint, or be unset
    for cli_val, ckpt_val, name in [
        (args.tau_corridor, tau_c, "tau_corridor"),
        (args.tau_release, tau_r, "tau_release"),
        (args.guard, guard, "guard"),
    ]:
        if cli_val is not None and cli_val != ckpt_val:
            sys.exit("CLI {}={} conflicts with checkpoint value={}".format(name, cli_val, ckpt_val))

    # Formal evaluation: must use test split, K from checkpoint
    if args.split_key != "test":
        sys.exit("Formal evaluation requires --split_key test, got: {}".format(args.split_key))
    split_key = "test"

    print("Checkpoint: tau_c={} tau_r={} guard={} K={}".format(tau_c, tau_r, guard, K))
    print("Checkpoint metric: {}".format(ckpt_meta.get("checkpoint_metric", "unknown")))

    print("Loading episode index: {}".format(args.episode_index))
    ep_index = load_episode_index(args.episode_index)
    print("Loading features: {}".format(args.feature_csv))
    features = load_features(args.feature_csv)
    print("Loading teacher events: {}".format(args.label_csv))
    events = load_teacher_events(args.label_csv)
    print("Loading split: {}".format(args.split_file))
    with open(args.split_file) as f:
        split = json.load(f)

    eval_keys = split["splits"].get(split_key, [])
    if not eval_keys:
        sys.exit("Empty test partition in split — formal evaluation requires non-empty test set")
    if split_key not in split["splits"]:
        sys.exit("Missing '{}' partition in split".format(split_key))
    if args.split_key != "test":
        sys.exit("Formal evaluation requires --split_key test, got: {}".format(args.split_key))
    missing_feat = sorted([e for e in eval_keys if e not in features])
    missing_event = sorted([e for e in eval_keys if e not in events])
    errors = []
    if missing_feat:
        errors.append("{} eval episodes MISSING features".format(len(missing_feat)))
    if missing_event:
        errors.append("{} eval episodes MISSING teacher events".format(len(missing_event)))
    if errors:
        for e in errors:
            print("FAIL: {}".format(e))
        sys.exit(1)

    print("Evaluating {} episodes".format(len(eval_keys)))

    # Normalize with checkpoint mean/std
    for ek in list(features.keys()):
        features[ek] = (features[ek] - mean) / np.maximum(std, 1e-8)

    suite_map = {}
    for ek in eval_keys:
        s = ep_index[ek]["suite"]
        if s not in VALID_SUITES:
            sys.exit("Invalid suite: {} in {}".format(s, ek))
        suite_map[ek] = s

    results = []
    for ek in eval_keys:
        teacher_info = events[ek]
        r = evaluate_episode(model, features[ek], teacher_info, tau_c, tau_r, guard, K)
        r["episode_key"] = ek
        results.append(r)

    metrics = compute_metrics(results)
    metrics["split_key"] = split_key
    metrics["checkpoint_sha256"] = sha256_file(args.checkpoint)
    metrics["checkpoint_metric"] = ckpt_meta.get("checkpoint_metric", "unknown")
    metrics["checkpoint_epoch"] = ckpt_meta.get("epoch", -1)
    metrics["tau_corridor"] = tau_c
    metrics["tau_release"] = tau_r
    metrics["guard"] = guard
    metrics["K"] = K
    metrics["input_binding"] = {
        "feature_csv_sha256": sha256_file(args.feature_csv),
        "label_csv_sha256": sha256_file(args.label_csv),
        "episode_index_sha256": sha256_file(args.episode_index),
        "split_file_sha256": sha256_file(args.split_file),
    }
    metrics["per_suite"] = per_suite_metrics(results, suite_map)
    suite_f1s = [m2["event_f1"] for m2 in metrics["per_suite"].values() if m2["n_episodes"] > 0]
    metrics["micro_event_f1"] = metrics.pop("event_f1")
    metrics["suite_macro_event_f1"] = float(np.mean(suite_f1s)) if suite_f1s else 0.0
    if suite_f1s:
        metrics["worst_suite_event_f1"] = float(np.min(suite_f1s))
        metrics["best_suite_event_f1"] = float(np.max(suite_f1s))

    out = json.dumps(metrics, indent=2)
    if args.output == "-":
        print(out)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(out + "\n")

    print("\nEpisodes: {}  TP: {}  FP: {}  FN: {}  TN: {}".format(
        metrics["n_episodes"], metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]))
    print("F1: micro={:.3f} macro={:.3f} worst={:.3f}  Precision: {:.3f}  Recall: {:.3f}".format(
        metrics.get("micro_event_f1", 0.0), metrics.get("suite_macro_event_f1", 0.0),
        metrics.get("worst_suite_event_f1", 0.0),
        metrics["event_precision"], metrics["event_recall"]))
    print("K={}: containment {:.3f}  Abstention: {:.3f}".format(
        K, metrics["k_containment_rate"], metrics["correct_abstention_rate"]))
    for s, m in sorted(metrics.get("per_suite", {}).items()):
        print("  {}: n={} TP={} FP={} FN={} prec={:.3f} rec={:.3f}".format(
            s, m["n_episodes"], m["tp"], m["fp"], m["fn"], m["event_precision"], m["event_recall"]))


if __name__ == "__main__":
    main()
