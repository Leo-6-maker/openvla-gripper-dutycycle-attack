"""D8 Physical Dataset: load Teacher records, G=3 consolidate, build weights.

Returns per-step tensors: features, labels, masks, weights.
Features are minimal for smoke — canonical 25D integration happens in full CV.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from d8_event_consolidator import (
    consolidate_physical_events,
    build_physical_event_weights,
)

# Simple feature set for smoke: 8 features from Teacher record fields
# Full 25D features to be integrated for D8-2 formal CV
SMOKE_FEATURE_NAMES = [
    "step_norm",          # step / max_steps
    "candidate_close",    # binary
    "is_libero_10",       # suite one-hot
    "is_libero_goal",
    "is_libero_object",
    "is_libero_spatial",
    "task_id_norm",       # task_id / 10
    "state_id_norm",      # state_id / 50
]
N_FEATURES = len(SMOKE_FEATURE_NAMES)

ARTICULATED_TASKS = {"libero_goal/task_00", "libero_goal/task_07"}
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
G = 3


class D8PhysicalDataset(torch.utils.data.Dataset):
    """Dataset returning (features, label, mask, weight) per step.

    Only steps with mask=True (known TRUE/FALSE) are included in __getitem__.
    UNKNOWN/GEOM_NA/RIGHT_CENSORED steps are excluded from training.
    """

    def __init__(
        self,
        teacher_root: Path,
        sidecar: dict,
        ep_labels: dict,
        identity_list: List[str],
        device: str = "cpu",
    ):
        self.features: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.weights: List[torch.Tensor] = []

        for eid in sorted(identity_list):
            labels = ep_labels.get(eid, {})
            relations = sidecar.get(eid, {})
            if not labels:
                continue

            task_key = "/".join(eid.split("/")[:2])
            if task_key in ARTICULATED_TASKS:
                continue

            result = consolidate_physical_events(eid, labels, relations=relations, G=G)
            event_groups = result.get("event_groups", [])
            if not event_groups:
                continue

            max_step = max(labels.keys())
            n = max_step + 1
            labs = np.zeros(n, dtype=np.float32)
            masks = np.zeros(n, dtype=bool)
            rc_arr = np.zeros(n, dtype=bool)
            geom_arr = np.zeros(n, dtype=bool)

            suite_idx = SUITES.index(eid.split("/")[0]) if eid.split("/")[0] in SUITES else 0
            task_id = int(eid.split("/")[1].replace("task_", "")) if "/" in eid else 0
            state_id = int(eid.split("/")[2].replace("state_", "")) if "/" in eid else 0

            for s, lab in labels.items():
                v = lab.get("value", "UNKNOWN")
                m = lab.get("mask", False) and lab.get("valid_mask", False)
                if v == "TRUE":
                    labs[s] = 1.0
                elif v == "FALSE":
                    labs[s] = 0.0
                else:
                    labs[s] = -1.0
                masks[s] = m
                rc_arr[s] = bool(lab.get("right_censored", False))
                geom_arr[s] = lab.get("reason") == "GEOMETRY_NOT_APPLICABLE"

            weights = build_physical_event_weights(
                labs, masks, result, right_censored=rc_arr, geom_na=geom_arr,
            )

            # Build simple features
            feat = np.zeros((n, N_FEATURES), dtype=np.float32)
            for s in range(n):
                lab = labels.get(s, {})
                feat[s, 0] = s / max(1, n - 1)  # step_norm
                feat[s, 1] = float(lab.get("candidate_close", False)) if isinstance(lab, dict) else 0.0
                feat[s, 2] = 1.0 if suite_idx == 0 else 0.0
                feat[s, 3] = 1.0 if suite_idx == 1 else 0.0
                feat[s, 4] = 1.0 if suite_idx == 2 else 0.0
                feat[s, 5] = 1.0 if suite_idx == 3 else 0.0
                feat[s, 6] = task_id / 10.0
                feat[s, 7] = state_id / 50.0

            # Only include steps with effective mask
            effective_mask = masks & (~rc_arr) & (~geom_arr)
            for s in range(n):
                if effective_mask[s]:
                    self.features.append(torch.from_numpy(feat[s]).float().to(device))
                    self.labels.append(torch.tensor(labs[s], dtype=torch.float32, device=device))
                    self.weights.append(torch.tensor(weights[s], dtype=torch.float32, device=device))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


def compute_normalization(dataset: D8PhysicalDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute feature mean/std from training data only."""
    all_feat = torch.stack([f for f, _, _ in dataset], dim=0)
    mean = all_feat.mean(dim=0)
    std = all_feat.std(dim=0)
    std[std < 1e-8] = 1.0
    return mean, std
