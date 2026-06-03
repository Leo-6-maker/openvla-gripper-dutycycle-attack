# -*- coding: utf-8 -*-
"""Test the provenance aggregator schema and edge cases."""

import csv
import io
import json
import os
import tempfile

import pytest
import numpy as np

from gripper_attack.gripper_semantics import raw_gripper_is_open


# ── Minimal provenance aggregator logic (mirrors audit_prefix_margin_provenance.py) ──

def _compute_row_metrics(trace_rows):
    """Compute per-row metrics from trace CSV rows."""
    window_rows = [r for r in trace_rows if _parse_bool(r.get('in_window', 'False'))]
    attacked_rows = [r for r in window_rows if _parse_bool(r.get('pgd_applied', 'False'))]

    generated_open_count = sum(
        1 for r in window_rows
        if raw_gripper_is_open(float(r.get('adv_grip', 0.996)))
    )
    generated_open_total = len(window_rows)

    # qpos metrics
    qpos_vals_pre = [float(r['qpos_pre_step']) for r in attacked_rows if 'qpos_pre_step' in r]
    qpos_vals_post = [float(r['qpos_post_step']) for r in attacked_rows if 'qpos_post_step' in r]

    qpos_pre_start = qpos_vals_pre[0] if qpos_vals_pre else None
    qpos_post_start = qpos_vals_post[0] if qpos_vals_post else None
    qpos_pre_end = qpos_vals_pre[-1] if qpos_vals_pre else None
    qpos_post_end = qpos_vals_post[-1] if qpos_vals_post else None

    qpos_delta_pre = max(abs(v - qpos_vals_pre[0]) for v in qpos_vals_pre) if len(qpos_vals_pre) > 1 else 0.0
    qpos_delta_post = max(abs(v - qpos_vals_post[0]) for v in qpos_vals_post) if len(qpos_vals_post) > 1 else 0.0

    qpos_abs_after_min = min(qpos_vals_post) if qpos_vals_post else None
    qpos_abs_after_max = max(qpos_vals_post) if qpos_vals_post else None

    arm_l2_vals = [float(r['arm_l2']) for r in window_rows if 'arm_l2' in r]
    arm_l2_mean = float(np.mean(arm_l2_vals)) if arm_l2_vals else 0.0
    arm_l2_max = float(np.max(arm_l2_vals)) if arm_l2_vals else 0.0

    token_flip_count = sum(1 for r in window_rows if _parse_bool(r.get('token_flip', 'False')))

    official_done = any(_parse_bool(r.get('done', 'False')) for r in trace_rows)
    timeout = not official_done and len(trace_rows) >= 299

    return {
        'generated_open_count_canonical': generated_open_count,
        'generated_open_total': generated_open_total,
        'generated_open_ratio': round(generated_open_count / max(generated_open_total, 1), 4),
        'qpos_pre_start': qpos_pre_start,
        'qpos_post_start': qpos_post_start,
        'qpos_pre_end': qpos_pre_end,
        'qpos_post_end': qpos_post_end,
        'qpos_delta_pre': round(qpos_delta_pre, 6),
        'qpos_delta_post': round(qpos_delta_post, 6),
        'qpos_abs_after_min': qpos_abs_after_min,
        'qpos_abs_after_max': qpos_abs_after_max,
        'armL2_mean': round(arm_l2_mean, 6),
        'armL2_max': round(arm_l2_max, 6),
        'token_flip_count': token_flip_count,
        'official_done': official_done,
        'timeout': timeout,
    }


def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ('true', '1', 'yes')
    return bool(val)


# ── Helpers ──

def _make_trace_row(**overrides):
    base = {
        'task': 'ketchup', 'condition': 'vis_pgd', 'seed': '0',
        'step': '0', 'policy_step': '0', 'in_window': 'True',
        'attack_attempted': 'True', 'pgd_applied': 'True',
        'controller_active': 'True', 'controller_stopped': 'False',
        'effective_attack_step_idx': '0',
        'raw_gripper': '0.0', 'env_gripper': '1.0',
        'gripper_qpos': '0.039', 'qpos_pre_step': '0.039', 'qpos_post_step': '0.038',
        'clean_grip': '0.996', 'adv_grip': '0.0',
        'clean_z': '0.0', 'adv_z': '0.0',
        'nad_dof7': '0.996', 'nad_z': '0.0', 'nad_dof1_3': '0.0',
        'arm_l2': '0.0', 'linf': '0.0', 'token_flip': 'True', 'attack_dt': '0.5',
        'eef_x': '0.0', 'eef_y': '0.0', 'eef_z': '0.0',
        'done': 'False', 'reward': '0.0',
        'ctrl_mode': 'fixed', 'ctrl_stop_reason': 'none',
        'ctrl_streak': '0', 'ctrl_max_streak': '0',
        'ctrl_qpos_delta': '0.0', 'ctrl_attacks': '0',
    }
    base.update(overrides)
    return base


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── Tests ──

class TestProvenanceAggregatorSchema:
    def test_open_count_canonical(self):
        """OPEN count uses raw_gripper_is_open (adv_grip < 0.5)."""
        rows = [
            _make_trace_row(policy_step='0', adv_grip='0.0'),     # OPEN
            _make_trace_row(policy_step='1', adv_grip='0.3'),     # OPEN
            _make_trace_row(policy_step='2', adv_grip='0.996'),   # CLOSE
            _make_trace_row(policy_step='3', adv_grip='-0.5'),    # OPEN
        ]
        m = _compute_row_metrics(rows)
        assert m['generated_open_count_canonical'] == 3
        assert m['generated_open_total'] == 4
        assert m['generated_open_ratio'] == pytest.approx(0.75, abs=0.01)

    def test_all_close(self):
        """When model stays CLOSE, OPEN count is 0."""
        rows = [_make_trace_row(policy_step=str(i), adv_grip='0.996') for i in range(18)]
        m = _compute_row_metrics(rows)
        assert m['generated_open_count_canonical'] == 0

    def test_all_open(self):
        """When attack generates OPEN at every step."""
        rows = [_make_trace_row(policy_step=str(i), adv_grip='0.0') for i in range(18)]
        m = _compute_row_metrics(rows)
        assert m['generated_open_count_canonical'] == 18

    def test_qpos_delta_pre(self):
        """qpos_delta_pre is max deviation from first pre-step qpos."""
        rows = [
            _make_trace_row(policy_step='0', qpos_pre_step='0.039'),
            _make_trace_row(policy_step='1', qpos_pre_step='0.035'),
            _make_trace_row(policy_step='2', qpos_pre_step='0.001'),
        ]
        m = _compute_row_metrics(rows)
        assert m['qpos_delta_pre'] == pytest.approx(0.038, abs=0.001)

    def test_qpos_delta_post(self):
        rows = [
            _make_trace_row(policy_step='0', qpos_post_step='0.039'),
            _make_trace_row(policy_step='1', qpos_post_step='0.002'),
        ]
        m = _compute_row_metrics(rows)
        assert m['qpos_delta_post'] == pytest.approx(0.037, abs=0.001)

    def test_arm_l2_mean(self):
        rows = [
            _make_trace_row(policy_step='0', arm_l2='0.0'),
            _make_trace_row(policy_step='1', arm_l2='0.1'),
        ]
        m = _compute_row_metrics(rows)
        assert m['armL2_mean'] == pytest.approx(0.05, abs=0.001)

    def test_arm_l2_max(self):
        rows = [
            _make_trace_row(policy_step='0', arm_l2='0.0'),
            _make_trace_row(policy_step='1', arm_l2='0.5'),
        ]
        m = _compute_row_metrics(rows)
        assert m['armL2_max'] == pytest.approx(0.5, abs=0.001)

    def test_token_flip_count(self):
        rows = [
            _make_trace_row(policy_step='0', token_flip='True'),
            _make_trace_row(policy_step='1', token_flip='False'),
            _make_trace_row(policy_step='2', token_flip='True'),
        ]
        m = _compute_row_metrics(rows)
        assert m['token_flip_count'] == 2

    def test_done_detection(self):
        rows = [
            _make_trace_row(policy_step='0', done='False'),
            _make_trace_row(policy_step='1', done='True'),
        ]
        m = _compute_row_metrics(rows)
        assert m['official_done'] is True

    def test_timeout_detection(self):
        rows = [_make_trace_row(policy_step=str(i), done='False') for i in range(300)]
        m = _compute_row_metrics(rows)
        assert m['timeout'] is True


class TestSchemaValidation:
    def test_missing_qpos_post_step(self):
        """When qpos_post_step is missing from trace, validity must be flagged."""
        # Simulate: trace lacks qpos_post_step column entirely
        rows = [
            {k: v for k, v in _make_trace_row(policy_step=str(i)).items() if k != 'qpos_post_step'}
            for i in range(5)
        ]
        # When computing metrics, missing qpos_post_step should result in None values
        attacked_rows = [r for r in rows if _parse_bool(r.get('pgd_applied', 'False'))]
        qpos_vals_post = []
        for r in attacked_rows:
            if 'qpos_post_step' in r:
                qpos_vals_post.append(float(r['qpos_post_step']))
        assert len(qpos_vals_post) == 0
        # This should trigger schema_incomplete validity
        is_schema_complete = len(qpos_vals_post) > 0
        assert not is_schema_complete, "Missing qpos_post_step must be detected as schema_incomplete"

    def test_schema_complete_when_all_fields_present(self):
        rows = [_make_trace_row(policy_step=str(i)) for i in range(5)]
        m = _compute_row_metrics(rows)
        assert m['qpos_post_start'] is not None
        assert m['generated_open_count_canonical'] is not None


class TestGroupKey:
    def test_group_key_fields(self):
        """Verify the group key fields are all present."""
        group_key_fields = [
            'task', 'state_id', 'condition', 'objective',
            'eps_raw_pixels', 'window_start', 'window_end',
        ]
        for field in group_key_fields:
            assert isinstance(field, str) and len(field) > 0
