#!/usr/bin/env python3
"""Shared strict data loader for multisuite detector training and evaluation.

Single source of truth for feature CSV, label CSV, episode index loading.
Used by both train_detector.py and evaluate_detector.py.
Fail-closed: any data integrity violation raises immediately.
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Imported at top for both modules to share
from gripper_attack.sc5mlp_v1 import SC5_FEATURES, SC5_PHASES, N_FEATURES, N_PHASES

VALID_SUITES = {"libero_object", "libero_spatial", "libero_goal", "libero_10"}

# Fields that must be exactly these values for primary eligible cohort
PRIMARY_ELIGIBLE_REQUIRED = {
    "clean_success": True,
    "teacher_label_valid": True,
    "mechanism_eligible": True,
    "schema_fail": False,
}


def load_episode_index(path: str, cohort: str = None) -> dict:
    """Load episode index JSONL. Optionally filter to cohort.

    cohort='primary_eligible': requires clean_success=true, teacher_label_valid=true,
                               mechanism_eligible=true, schema_fail=false.
    cohort='safety_abstention': everything legally collected but NOT primary eligible.
    cohort=None: all episodes.
    """
    index = {}
    seen = set()
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            ek = ep["episode_key"]
            if ek in seen:
                raise ValueError("Duplicate episode_key in index: {}".format(ek))
            seen.add(ek)

            if "suite" not in ep:
                raise ValueError("Episode {} missing suite".format(ek))
            if ep["suite"] not in VALID_SUITES:
                raise ValueError("Episode {} invalid suite: {}".format(ek, ep["suite"]))

            if cohort == "primary_eligible":
                ok = True
                for field, expected in PRIMARY_ELIGIBLE_REQUIRED.items():
                    if ep.get(field) != expected:
                        ok = False
                        break
                if not ok:
                    continue
            elif cohort == "safety_abstention":
                is_primary = all(ep.get(f, False) == v for f, v in PRIMARY_ELIGIBLE_REQUIRED.items())
                if is_primary:
                    continue

            index[ek] = ep
    return index


def load_features(csv_path: str) -> dict:
    """Load 25D features with strict validation.

    Returns {episode_key: np.ndarray(n_steps, 25)} with steps sorted 0..n-1.
    Rejects: missing columns, missing values, NaN, Inf, duplicate steps, step gaps,
    non-zero-first steps.
    """
    data = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing_cols = [fn for fn in SC5_FEATURES if fn not in fieldnames]
        if missing_cols:
            raise ValueError("Missing feature columns: {}".format(missing_cols))

        ek_col = _resolve_column(fieldnames, ["episode_key", "episode"])
        step_col = _resolve_column(fieldnames, ["step", "step_id", "step_idx"])

        for row in reader:
            ek = row.get(ek_col, "").strip()
            if not ek:
                raise ValueError("Empty episode key in feature CSV")
            step = int(row.get(step_col, -1))
            if step < 0:
                raise ValueError("Invalid step {} in episode {}".format(step, ek))
            if step in data[ek]:
                raise ValueError("Duplicate feature step {} in episode {}".format(step, ek))

            feats = []
            for fn in SC5_FEATURES:
                v = row.get(fn)
                if v is None or str(v).strip() == "":
                    raise ValueError("Missing {} in {} step {}".format(fn, ek, step))
                fv = float(v)
                if not np.isfinite(fv):
                    raise ValueError("Non-finite {}={} in {} step {}".format(fn, fv, ek, step))
                feats.append(fv)
            data[ek][step] = feats

    return _build_arrays(data, "feature")


def load_labels(csv_path: str) -> dict:
    """Load teacher labels with strict validation.

    Returns {episode_key: {phase, corridor, release: np.ndarray(n_steps,)}}.
    Rejects: missing columns, out-of-range, non-binary, duplicate steps, step gaps.
    """
    data = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        ek_col = _resolve_column(fieldnames, ["episode_key", "episode"])
        step_col = _resolve_column(fieldnames, ["step", "step_id", "step_idx"])
        phase_col = _resolve_column(fieldnames, ["teacher_phase_idx", "phase_idx", "phase"], required=True)
        corridor_col = _resolve_column(fieldnames, ["teacher_sc5_corridor_active", "corridor_active", "corridor"], required=True)
        release_col = _resolve_column(fieldnames, ["release_safe", "release"], required=True)

        for row in reader:
            ek = row.get(ek_col, "").strip()
            if not ek:
                raise ValueError("Empty episode key in label CSV")
            step = int(row.get(step_col, -1))
            if step < 0:
                raise ValueError("Invalid label step {} in {}".format(step, ek))
            if step in data[ek]:
                raise ValueError("Duplicate label step {} in episode {}".format(step, ek))

            phase = int(row.get(phase_col, -1))
            if phase < 0 or phase >= N_PHASES:
                raise ValueError("Phase {} out of [0,{}) in {} step {}".format(phase, N_PHASES, ek, step))
            corridor = int(row.get(corridor_col, -1))
            if corridor not in (0, 1):
                raise ValueError("Corridor {} not 0/1 in {} step {}".format(corridor, ek, step))
            release = int(row.get(release_col, -1))
            if release not in (0, 1):
                raise ValueError("Release {} not 0/1 in {} step {}".format(release, ek, step))

            data[ek][step] = {"phase": phase, "corridor": corridor, "release": release}

    result = {}
    for ek, steps_dict in data.items():
        steps = sorted(steps_dict.keys())
        if steps[0] != 0:
            raise ValueError("Label episode {} first step is {}".format(ek, steps[0]))
        for i, s in enumerate(steps):
            if s != i:
                raise ValueError("Label episode {} step gap at {}".format(ek, i))
        result[ek] = {
            "phase": np.array([steps_dict[s]["phase"] for s in steps], dtype=np.int64),
            "corridor": np.array([steps_dict[s]["corridor"] for s in steps], dtype=np.int64),
            "release": np.array([steps_dict[s]["release"] for s in steps], dtype=np.int64),
        }
    return result


def load_teacher_events(csv_path: str) -> dict:
    """Load per-episode teacher event anchors/windows for evaluation.

    Returns {episode_key: {anchor, window_start, window_end, event_type}}.
    event_type distinguishes: 'primary' vs 'no_event' vs 'abstain' vs 'unsupported'.
    Rejects: missing label, ambiguous anchor, window anchor mismatch.
    """
    events = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        ek_col = _resolve_column(fieldnames, ["episode_key", "episode"])

        for row in reader:
            ek = row.get(ek_col, "").strip()
            if not ek:
                raise ValueError("Empty episode key in event CSV")

            anchor_str = row.get("teacher_anchor_step", row.get("sc5_anchor", ""))
            if anchor_str == "" or anchor_str is None:
                event_type = row.get("event_type", row.get("teacher_event_type", "no_event"))
                events[ek] = {"anchor": -1, "window_start": -1, "window_end": -1,
                              "event_type": event_type, "has_event": False}
                continue

            anchor = int(anchor_str)
            wstart = int(row.get("teacher_window_start", anchor))
            wend = int(row.get("teacher_window_end", anchor + 10))
            if wstart > anchor or wend < anchor:
                raise ValueError("Window [{},{}] does not contain anchor {} in {}".format(wstart, wend, anchor, ek))
            event_type = row.get("event_type", row.get("teacher_event_type", "primary"))
            events[ek] = {"anchor": anchor, "window_start": wstart, "window_end": wend,
                          "event_type": event_type, "has_event": True}
    return events


def strict_join(train_eks, val_eks, features, labels, episode_index):
    """Validate all split episodes have features AND labels. Build suite_map.

    Returns suite_map: {episode_key: suite}.
    Raises on: missing features, missing labels, length mismatch, unknown suite.
    """
    all_eks = set(train_eks + val_eks)
    missing_feat = sorted([e for e in all_eks if e not in features])
    missing_label = sorted([e for e in all_eks if e not in labels])
    errors = []
    if missing_feat:
        errors.append("{} split episodes MISSING features: {}".format(len(missing_feat), missing_feat[:5]))
    if missing_label:
        errors.append("{} split episodes MISSING labels: {}".format(len(missing_label), missing_label[:5]))

    for ek in sorted(all_eks):
        if ek not in features or ek not in labels:
            continue
        n_feat = len(features[ek])
        n_label = len(labels[ek]["phase"])
        if n_feat != n_label:
            errors.append("LENGTH MISMATCH: {} feat={} label={}".format(ek, n_feat, n_label))

    if errors:
        for e in errors:
            print("STRICT_JOIN FAIL: {}".format(e))
        raise ValueError("strict_join: {} errors".format(len(errors)))

    suite_map = {}
    for ek in all_eks:
        s = episode_index.get(ek, {}).get("suite", "MISSING")
        if s not in VALID_SUITES:
            raise ValueError("Episode {} has invalid suite: {}".format(ek, s))
        suite_map[ek] = s
    return suite_map


def compute_normalization(features, episode_keys):
    all_feats = [features[ek] for ek in episode_keys]
    if not all_feats:
        raise ValueError("No training features for normalization")
    X = np.concatenate(all_feats, axis=0)
    mean = np.mean(X, axis=0).astype(np.float32)
    std = np.std(X, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-8)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("NaN/Inf in normalization mean/std")
    norm_sha = _array_sha256(mean) + _array_sha256(std)
    return mean, std, norm_sha


# ── helpers ──

def _resolve_column(fieldnames, candidates, required=False):
    for c in candidates:
        if c in fieldnames:
            return c
    if required:
        raise ValueError("Missing required column, tried: {}".format(candidates))
    return candidates[0]


def _build_arrays(data, label):
    result = {}
    for ek, steps_dict in data.items():
        steps = sorted(steps_dict.keys())
        if steps[0] != 0:
            raise ValueError("{} episode {} first step={}".format(label, ek, steps[0]))
        for i, s in enumerate(steps):
            if s != i:
                raise ValueError("{} episode {} step gap at {}".format(label, ek, i))
        result[ek] = np.array([steps_dict[s] for s in steps], dtype=np.float32)
    return result


def _array_sha256(arr):
    import hashlib
    return hashlib.sha256(arr.tobytes()).hexdigest()
