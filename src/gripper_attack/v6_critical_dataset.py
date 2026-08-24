"""V2 Gripper-Critical Dataset Adapter.

Loads: 25D causal history + policy_9d + gripper_9d + Teacher K10 labels.
Generates: K10_startability target + secure_grasp target + manipulation target.

Excludes episodes with parser contract contradictions:
  - K10 feasible but phase labels never known
  - K10 all zero with no phase labels
"""
from __future__ import annotations

import json, os, hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


K10_WINDOW = 10
FEATURE_25D_DIM = 25
POLICY_9D_DIM = 9
GRIPPER_9D_DIM = 9


def sha256_file(path: str) -> str:
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            d.update(chunk)
    return d.hexdigest()


class CriticalEpisode:
    """One episode with all features and labels."""

    def __init__(self, eid: str, split: str,
                 features_25d: np.ndarray,     # [T, 25]
                 policy_9d: np.ndarray,         # [T, 9]
                 gripper_9d: np.ndarray,        # [T, 9]
                 k10_startable: np.ndarray,     # [T] bool
                 k10_known: np.ndarray,         # [T] bool
                 grasp_label: np.ndarray,       # [T] bool
                 grasp_known: np.ndarray,       # [T] bool
                 manipulation_label: np.ndarray, # [T] bool
                 manipulation_known: np.ndarray, # [T] bool
                 candidate_close: np.ndarray,    # [T] bool
                 has_opportunity: bool,
                 absence_reason: str,
                 T: int):
        self.eid = eid
        self.split = split
        self.features_25d = features_25d
        self.policy_9d = policy_9d
        self.gripper_9d = gripper_9d
        self.k10_startable = k10_startable
        self.k10_known = k10_known
        self.grasp_label = grasp_label
        self.grasp_known = grasp_known
        self.manipulation_label = manipulation_label
        self.manipulation_known = manipulation_known
        self.candidate_close = candidate_close
        self.has_opportunity = has_opportunity
        self.absence_reason = absence_reason
        self.T = T

    @property
    def valid_window(self) -> int:
        """Last step index where K10 window is valid."""
        return max(0, self.T - K10_WINDOW)

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert to tensors for model input."""
        return {
            'x_25d': torch.tensor(self.features_25d, dtype=torch.float32),
            'x_policy': torch.tensor(self.policy_9d, dtype=torch.float32),
            'x_gripper': torch.tensor(self.gripper_9d, dtype=torch.float32),
            'k10_startable': torch.tensor(self.k10_startable, dtype=torch.float32).unsqueeze(-1),
            'k10_known': torch.tensor(self.k10_known, dtype=torch.bool).unsqueeze(-1),
            'grasp_label': torch.tensor(self.grasp_label, dtype=torch.float32).unsqueeze(-1),
            'grasp_known': torch.tensor(self.grasp_known, dtype=torch.bool).unsqueeze(-1),
            'manipulation_label': torch.tensor(self.manipulation_label, dtype=torch.float32).unsqueeze(-1),
            'manipulation_known': torch.tensor(self.manipulation_known, dtype=torch.bool).unsqueeze(-1),
            'candidate_close': torch.tensor(self.candidate_close, dtype=torch.bool).unsqueeze(-1),
            'T': self.T,
        }


def load_v2_episodes(
    feature_root: str,
    teacher_root: str,
    manifest: dict,
    exclude_parser_contradictions: bool = True,
) -> List[CriticalEpisode]:
    """Load episodes for V2 training.

    Args:
        feature_root: path to clean/ directory with step_records.jsonl
        teacher_root: path to factorized teacher labels
        manifest: split manifest with identity lists
        exclude_parser_contradictions: exclude episodes where K10 labels contradict phase labels

    Returns:
        List of CriticalEpisode objects
    """
    episodes = []
    excluded = []

    for split_key in sorted(manifest.get('splits', {})):
        identities = manifest['splits'][split_key].get('identities',
                     manifest['splits'][split_key].get('heldout_l3',
                     manifest['splits'][split_key].get('calibrator_fit',
                     manifest['splits'][split_key].get('policy_selection', []))))

        for eid in sorted(identities):
            parts = eid.split('/')
            feat_path = os.path.join(feature_root, parts[0], parts[1], parts[2], 'step_records.jsonl')
            teach_path = os.path.join(teacher_root, parts[0], parts[1], parts[2],
                                      'factorized_teacher_v1.jsonl')
            if not os.path.isfile(feat_path) or not os.path.isfile(teach_path):
                continue

            recs = [json.loads(l) for l in open(feat_path).read().splitlines() if l.strip()]
            tr = [json.loads(l) for l in open(teach_path).read().splitlines() if l.strip()]
            tr.sort(key=lambda r: r['step'])
            T = len(recs)
            max_t = min(T, T - K10_WINDOW + 1)

            # Extract features
            f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
            p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(POLICY_9D_DIM)) for r in recs],
                           dtype=np.float32)
            g9d = np.array([[r.get('clean_close_probability_mass', 0),
                             r.get('clean_open_probability_mass', 0),
                             r.get('clean_top1_is_close', 0),
                             r.get('clean_top1_is_open', 0),
                             r.get('clean_top1_probability', 0),
                             r.get('clean_best_close_rank_normalized', 0),
                             r.get('clean_best_open_rank_normalized', 0),
                             r.get('clean_action_token_entropy_normalized', 0),
                             r.get('clean_open_minus_close_log_mass', 0)] for r in recs],
                           dtype=np.float32)

            # Extract labels
            k10_startable = np.zeros(T, dtype=bool)
            k10_known = np.zeros(T, dtype=bool)
            grasp_label = np.zeros(T, dtype=bool)
            grasp_known = np.zeros(T, dtype=bool)
            manip_label = np.zeros(T, dtype=bool)
            manip_known = np.zeros(T, dtype=bool)
            cc = np.zeros(T, dtype=bool)

            for t in range(T):
                tr_t = tr[min(t, len(tr)-1)]
                if t < max_t:
                    k10_startable[t] = tr_t.get('strict_k10_feasible', False)
                    k10_known[t] = tr_t.get('strict_k10_known_mask', False)
                grasp_label[t] = tr_t.get('grasp_established', False)
                grasp_known[t] = tr_t.get('grasp_established_known_mask', False)
                manip_label[t] = tr_t.get('manipulation_active', False)
                manip_known[t] = tr_t.get('manipulation_active_known_mask', False)
                cc[t] = tr_t.get('candidate_close', False)

            has_opp = bool(k10_startable[:max_t].any() and k10_known[:max_t].any())

            # Parser contradiction check
            n_k10_pos = int(k10_startable[:max_t].sum())
            n_grasp_known = int(grasp_known[:max_t].sum())
            n_manip_known = int(manip_known[:max_t].sum())
            n_k10_known = int(k10_known[:max_t].sum())
            n_grasp_pos = int((grasp_label[:max_t] & grasp_known[:max_t]).sum())
            n_manip_pos = int((manip_label[:max_t] & manip_known[:max_t]).sum())

            # Classify parser issues
            is_parser_issue = False
            parser_reason = ''
            if n_k10_pos > 0 and n_grasp_known == 0 and n_manip_known == 0:
                is_parser_issue = True
                parser_reason = 'K10_FEASIBLE_BUT_PHASE_NEVER_KNOWN'
            elif n_k10_known > 0 and n_k10_pos == 0 and n_grasp_known == 0:
                is_parser_issue = True
                parser_reason = 'K10_ALL_ZERO_NO_PHASE_LABELS'

            if exclude_parser_contradictions and is_parser_issue:
                excluded.append((eid, parser_reason))
                continue

            # Absence reason
            absence_reason = 'OPPORTUNITY_PRESENT'
            if not has_opp:
                if n_k10_known == 0:
                    absence_reason = 'F1_TASK_STRUCTURAL_ZERO'
                elif n_grasp_pos == 0 and n_grasp_known > 0:
                    absence_reason = 'F4_NO_STABLE_GRASP'
                elif n_manip_pos == 0 and n_manip_known > 0:
                    absence_reason = 'F3_NO_MANIPULATION'
                elif n_k10_pos == 0:
                    absence_reason = 'F6_PARSER_DECODER_ZERO'
                else:
                    absence_reason = 'OPPORTUNITY_PRESENT'  # has K10 after all

            ep = CriticalEpisode(
                eid=eid, split=split_key,
                features_25d=f25d, policy_9d=p9d, gripper_9d=g9d,
                k10_startable=k10_startable, k10_known=k10_known,
                grasp_label=grasp_label, grasp_known=grasp_known,
                manipulation_label=manip_label, manipulation_known=manip_known,
                candidate_close=cc,
                has_opportunity=has_opp, absence_reason=absence_reason,
                T=T,
            )
            episodes.append(ep)

    if excluded:
        print(f'Excluded {len(excluded)} parser-contradiction episodes:')
        for eid, reason in excluded[:5]:
            print(f'  {eid}: {reason}')
        if len(excluded) > 5:
            print(f'  ... and {len(excluded)-5} more')

    return episodes


class CriticalEpisodeDataset(Dataset):
    """PyTorch Dataset for V2 training. Returns per-episode tensors."""

    def __init__(self, episodes: List[CriticalEpisode]):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx].to_tensors()

    @property
    def opportunity_rate(self) -> float:
        return sum(1 for ep in self.episodes if ep.has_opportunity) / max(len(self.episodes), 1)

    @property
    def absence_summary(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for ep in self.episodes:
            counts[ep.absence_reason] += 1
        return dict(counts)


def collate_v2_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad episodes to same length for batch processing."""
    max_T = max(item['T'] for item in batch)
    B = len(batch)

    keys_2d = ['x_25d', 'x_policy', 'x_gripper',
               'k10_startable', 'k10_known', 'grasp_label', 'grasp_known',
               'manipulation_label', 'manipulation_known', 'candidate_close']
    dims = {
        'x_25d': FEATURE_25D_DIM, 'x_policy': POLICY_9D_DIM, 'x_gripper': GRIPPER_9D_DIM,
        'k10_startable': 1, 'k10_known': 1, 'grasp_label': 1, 'grasp_known': 1,
        'manipulation_label': 1, 'manipulation_known': 1, 'candidate_close': 1,
    }

    batch_out = {'episode_lengths': torch.tensor([item['T'] for item in batch], dtype=torch.long)}

    for key in keys_2d:
        d = dims[key]
        padded = torch.zeros(B, max_T, d)
        for b, item in enumerate(batch):
            t = item[key]
            T_b = item['T']
            padded[b, :T_b, :] = t[:T_b]
        batch_out[key] = padded

    return batch_out
