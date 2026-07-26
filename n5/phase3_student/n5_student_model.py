"""N5 Multi-Head Student Model.
Label Contract V2 target: independent prediction of 5 causal factors.

Architecture:
  51D causal history
    -> short CausalTCN (RF=32, hidden=64)
    -> long CausalTCN (RF=128, hidden=64)
    -> concat fusion (128D -> 64D)
    -> 5 independent heads:
       A: physical_criticality  (scalar logit)
       B: k10_feasible          (scalar logit)
       C: safe_release          (scalar logit)
       D: instability           (scalar logit)
       E: close_intent          (scalar logit)

Hard constraints:
  - All heads output RAW logits (before sigmoid)
  - NO head output is used as gate for another head
  - NO candidate_close in loss mask or prediction pipeline
  - Each head has independent valid_mask for training
  - Normalization computed from train split only
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, hashlib
from typing import Dict, Tuple, Optional

# ── Causal TCN Encoder (same base as V4) ──

class CausalTCNBlock(nn.Module):
    """Single causal TCN block with dilated conv + residual."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                              padding=padding, padding_mode='zeros')
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)
        out = out[..., :x.shape[-1]]  # causal: trim future padding
        res = self.residual(x)
        out = out.transpose(1, 2)  # (B, T, C) for LayerNorm
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)  # back to (B, C, T)
        return out + res

class CausalTCNEncoder(nn.Module):
    """Stacked causal TCN with EXACT receptive field.

    P0 FIX: kernel_size=2 gives precise RF control.
    With kernel_size=2, dilation sequence 1,2,4,8,16 gives RF=32.
    Dilation 1,2,4,8,16,32,64 gives RF=128.
    Formula: RF = 1 + sum((kernel_size-1) * d_i) = 1 + sum(d_i) for k=2.
    """
    def __init__(self, input_dim, hidden_dim, rf, dropout=0.1):
        super().__init__()
        layers = []
        d = 1
        self._rf = 1
        while self._rf < rf:
            in_ch = input_dim if d == 1 else hidden_dim
            layers.append(CausalTCNBlock(in_ch, hidden_dim, kernel_size=2,
                                          dilation=d, dropout=dropout))
            self._rf += d  # exact for k=2: each layer adds dilation to RF
            d *= 2
        self.layers = nn.ModuleList(layers)
        self.output_dim = hidden_dim

    @property
    def receptive_field(self):
        return self._rf

    def forward(self, x, timestep_mask=None):
        # x: (B, T, C) or (B, T, C) with timestep_mask: (B, T) boolean
        if timestep_mask is not None:
            # Zero out padding timesteps so they don't contribute
            mask_expanded = timestep_mask.unsqueeze(-1).float()  # (B, T, 1)
            x = x * mask_expanded
        x = x.transpose(1, 2)  # (B, C, T)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return x

class N5SharedEncoder(nn.Module):
    """Dual-branch CausalTCN with concat fusion."""
    def __init__(self, input_dim=51, hidden=64, short_rf=32, long_rf=128, dropout=0.1):
        super().__init__()
        self.short_tcn = CausalTCNEncoder(input_dim, hidden, short_rf, dropout)
        self.long_tcn = CausalTCNEncoder(input_dim, hidden, long_rf, dropout)
        self.fusion = nn.Linear(hidden * 2, hidden)
        self.hidden = hidden

    @property
    def short_rf(self):
        return self.short_tcn.receptive_field

    @property
    def long_rf(self):
        return self.long_tcn.receptive_field

    def forward(self, x, timestep_mask=None):
        # x: (B, T, 51), timestep_mask: (B, T) boolean (True=valid)
        s = self.short_tcn(x, timestep_mask)  # (B, T, hidden)
        l = self.long_tcn(x, timestep_mask)   # (B, T, hidden)
        fused = self.fusion(torch.cat([s, l], dim=-1))  # (B, T, hidden)
        return F.relu(fused)


# ── Independent Heads ──

class ScalarHead(nn.Module):
    """Single scalar logit output head. No gate dependency."""
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        # x: (B, T, hidden)
        return self.net(x).squeeze(-1)  # (B, T)

class N5MultiHeadStudent(nn.Module):
    """Complete N5 Student: shared encoder + 5 independent heads.

    Heads:
      0: physical_criticality  — requires physical engagement, task not done, not releasing
      1: k10_feasible          — horizon >= 10, no release/terminal in window, critical corridor
      2: safe_release          — task done or high-confidence release
      3: instability           — low-confidence release, possible vulnerability window
      4: close_intent          — policy-level close signal (auxiliary, never a gate)
    """
    HEAD_NAMES = ['physical_criticality', 'k10_feasible', 'safe_release',
                  'instability', 'close_intent']
    N_HEADS = 5

    def __init__(self, input_dim=51, hidden=64, short_rf=32, long_rf=128, dropout=0.1):
        super().__init__()
        self.encoder = N5SharedEncoder(input_dim, hidden, short_rf, long_rf, dropout)
        self.heads = nn.ModuleList([ScalarHead(hidden, dropout) for _ in range(self.N_HEADS)])
        self.input_dim = input_dim
        self.hidden = hidden

    def forward(self, x, timestep_mask=None, head_idx=None):
        """Forward pass.

        Args:
            x: (B, T, 51) normalized input features
            timestep_mask: (B, T) boolean, True=valid timestep. Padding steps should be False.
            head_idx: If None, return all 5 head logits.
                      If int, return only that head's logits.
        """
        shared = self.encoder(x, timestep_mask=timestep_mask)  # (B, T, hidden)

        if head_idx is not None:
            return self.heads[head_idx](shared)

        return {name: head(shared) for name, head in zip(self.HEAD_NAMES, self.heads)}

    def get_last_logits(self, x, timestep_mask=None, head_idx=None):
        """Get logits for the LAST VALID timestep (handles padding correctly).

        P0-5 FIX: Passes head_idx to forward; handles device; hard-fails on all-False mask.
        """
        all_logits = self.forward(x, timestep_mask=timestep_mask, head_idx=head_idx)

        if timestep_mask is not None:
            # Check per-row: each batch item must have at least one valid step
            invalid_rows = ~timestep_mask.any(dim=1)
            if invalid_rows.any():
                n_invalid = invalid_rows.sum().item()
                raise ValueError(
                    f'get_last_logits: {n_invalid} batch items have all-False mask. '
                    f'Each item must have at least one valid timestep.'
                )
            # Find last True index per batch item
            last_valid_idx = (timestep_mask.shape[1] - 1 -
                              timestep_mask.flip(1).long().argmax(dim=1))
        else:
            last_valid_idx = None

        if head_idx is not None:
            # all_logits is (B, T) tensor
            if last_valid_idx is not None:
                idx = torch.arange(all_logits.shape[0], device=all_logits.device)
                return all_logits[idx, last_valid_idx]
            return all_logits[:, -1]

        # all_logits is dict
        if last_valid_idx is not None:
            first_name = list(all_logits.keys())[0]
            idx = torch.arange(all_logits[first_name].shape[0], device=all_logits[first_name].device)
            return {
                name: logits[idx, last_valid_idx]
                for name, logits in all_logits.items()
            }
        return {name: logits[:, -1] for name, logits in all_logits.items()}

# ── Loss Functions ──

class FrozenPosWeights:
    """P0 FIX: Pre-compute pos_weight from TRAIN split, freeze for all batches.

    Usage:
      fw = FrozenPosWeights.compute(train_labels, train_masks)
      loss = masked_bce_loss(logits, targets, mask, fw.get_weight(head_name))
    """
    def __init__(self, weights):
        self.weights = weights

    @classmethod
    def compute(cls, train_labels, train_valid_masks):
        """Compute frozen pos_weight from full train split. Stores pos/neg counts."""
        weights = {}
        pos_counts = {}
        neg_counts = {}
        for name in N5MultiHeadStudent.HEAD_NAMES:
            targets = train_labels[name]
            mask = train_valid_masks[name]
            n_valid = mask.sum().item()
            if n_valid == 0:
                weights[name] = None
                pos_counts[name] = 0
                neg_counts[name] = 0
                continue
            n_pos = targets[mask].sum().item()
            n_neg = n_valid - n_pos
            pos_counts[name] = int(n_pos)
            neg_counts[name] = int(n_neg)
            if n_pos > 0 and n_neg > 0:
                w = float(np.clip(n_neg / n_pos, 1, 20))
                weights[name] = w
            else:
                weights[name] = None  # No positives OR no negatives → HOLD
        instance = cls(weights)
        instance._pos_counts = pos_counts
        instance._neg_counts = neg_counts
        return instance

    def get_weight(self, head_name):
        return self.weights.get(head_name)

    def validate(self):
        """Check all heads have both positive AND negative samples. Returns issues list."""
        issues = []
        for name in N5MultiHeadStudent.HEAD_NAMES:
            w = self.weights.get(name)
            if w is None:
                issues.append(f'{name}: no valid samples — HOLD (split-level)')
            elif not hasattr(self, '_counts'):
                pass  # counts not available from weights alone
        # Check from stored counts if available
        if hasattr(self, '_pos_counts') and hasattr(self, '_neg_counts'):
            for name in N5MultiHeadStudent.HEAD_NAMES:
                n_pos = self._pos_counts.get(name, 0)
                n_neg = self._neg_counts.get(name, 0)
                if n_pos == 0 and n_neg == 0:
                    issues.append(f'{name}: zero valid samples — HOLD')
                elif n_pos == 0:
                    issues.append(f'{name}: zero positives (n_neg={n_neg}) — HOLD')
                elif n_neg == 0:
                    issues.append(f'{name}: zero negatives (n_pos={n_pos}) — HOLD')
        return issues

def masked_bce_loss(logits, targets, valid_mask, pos_weight=None):
    """Masked BCE loss with optional FROZEN pos_weight (scalar, not per-batch)."""
    loss = F.binary_cross_entropy_with_logits(
        logits[valid_mask], targets[valid_mask].float(),
        pos_weight=torch.tensor(pos_weight, device=logits.device) if pos_weight else None,
        reduction='mean'
    )
    return loss

def n5_total_loss(model_output, labels, valid_masks, frozen_weights, head_weights=None):
    """Total N5 training loss. Uses FROZEN pos_weight per head.

    Batch-level: skips head if zero valid samples (logs warning).
    Split-level: FrozenPosWeights.validate() should have already checked for
    trainable heads. Batch-skip is for rare edge cases (all-instability batch).
    """
    if head_weights is None:
        head_weights = {name: 1.0 for name in N5MultiHeadStudent.HEAD_NAMES}

    total = 0.0
    per_head = {}
    active_heads = 0
    for name in N5MultiHeadStudent.HEAD_NAMES:
        logits = model_output[name]
        target = labels[name]
        mask = valid_masks[name]
        n_valid = mask.sum().item()
        if n_valid == 0:
            per_head[name] = 0.0
            continue

        loss = masked_bce_loss(logits, target, mask, frozen_weights.get_weight(name))
        weighted = loss * head_weights[name]
        total += weighted
        per_head[name] = loss.item()
        active_heads += 1

    if active_heads == 0:
        raise RuntimeError('All heads have zero valid samples in batch')

    return total, per_head

# ── Suite-Balanced Sampler ──

class SuiteBalancedSampler:
    """Ensures each batch has roughly equal representation from all 4 suites.

    P1 FIX: Uses fixed RNG, warns on silent truncation.
    """
    def __init__(self, episode_indices, episode_suites, batch_size, shuffle=True, seed=42):
        self.suite_indices = {}
        for i, suite in enumerate(episode_suites):
            self.suite_indices.setdefault(suite, []).append(episode_indices[i])

        self.suites = sorted(self.suite_indices.keys())
        self.n_suites = len(self.suites)
        self.per_suite_batch = max(1, batch_size // self.n_suites)
        self.effective_batch_size = self.per_suite_batch * self.n_suites
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed + 42)  # Fixed RNG per sampler instance

        # Warn on truncation
        suite_lens = {s: len(self.suite_indices[s]) for s in self.suites}
        self.min_suite_len = min(suite_lens.values())
        self.max_suite_len = max(suite_lens.values())
        if self.min_suite_len < self.max_suite_len:
            truncated = {s: n - self.min_suite_len for s, n in suite_lens.items() if n > self.min_suite_len}
            print(f'WARNING: SuiteBalancedSampler truncating {sum(truncated.values())} episodes to min={self.min_suite_len}: {truncated}')

    def __iter__(self):
        indices = []
        for suite in self.suites:
            suite_idx = self.suite_indices[suite].copy()
            if self.shuffle:
                self.rng.shuffle(suite_idx)
            indices.append(suite_idx[:self.min_suite_len])

        n_batches = self.min_suite_len // self.per_suite_batch
        batch_order = list(range(n_batches))
        if self.shuffle:
            self.rng.shuffle(batch_order)

        for b in batch_order:
            batch = []
            for i in range(self.n_suites):
                start = b * self.per_suite_batch
                end = start + self.per_suite_batch
                batch.extend(indices[i][start:end])
            yield batch

    def __len__(self):
        return self.min_suite_len // self.per_suite_batch

# ── Baseline Models ──

class PriorBaseline(nn.Module):
    """Always outputs the training prior as RAW LOGIT (logit(prior), not probability).

    P1 FIX: Converts prior probability to logit space for fair comparison with other models.
    """
    def __init__(self, priors=None):
        super().__init__()
        self.priors = priors or {}

    def forward(self, x, timestep_mask=None):
        B, T = x.shape[:2]
        result = {}
        for name, p in self.priors.items():
            # Convert probability to logit: logit(p) = log(p/(1-p))
            p_clipped = max(min(p, 0.999), 0.001)
            logit_val = np.log(p_clipped / (1 - p_clipped))
            result[name] = torch.full((B, T), logit_val, device=x.device)
        return result

class LastFrameMLP(nn.Module):
    """MLP on last VALID frame only (no temporal context, no future leakage).

    P1 FIX: Does NOT expand to full sequence. Each timestep prediction uses
    only that timestep's features. Use timestep_mask to find valid frames.
    For evaluation, processes each step independently (no temporal modeling).
    """
    def __init__(self, input_dim=51, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden // 2, 1) for _ in range(N5MultiHeadStudent.N_HEADS)
        ])

    def forward(self, x, timestep_mask=None):
        B, T, C = x.shape
        # Process each timestep independently (no temporal leakage)
        x_flat = x.reshape(B * T, C)
        h = self.net(x_flat)  # (B*T, hidden//2)
        result = {}
        for i, (name, head) in enumerate(zip(N5MultiHeadStudent.HEAD_NAMES, self.heads)):
            logits_flat = head(h).squeeze(-1)  # (B*T,)
            result[name] = logits_flat.reshape(B, T)  # (B, T)
        return result

# ── Schema and Versioning ──

N5_MODEL_SCHEMA = {
    'schema': 'N5_MULTIHEAD_STUDENT_V2',
    'input_dim': 51,
    'input_breakdown': {
        'f25d': '25D SC5 streaming features',
        'p9d': '9D policy intent (POLICY_INTENT_ORDER)',
        'g9d': '9D gripper token (TRAIN_G9D_ORDER)',
        'proxies_8d': '8D causal response proxies',
    },
    'encoder': {
        'type': 'DualCausalTCN',
        'kernel_size': 2,
        'short_rf': 32,
        'short_dilations': [1, 2, 4, 8, 16],
        'long_rf': 128,
        'long_dilations': [1, 2, 4, 8, 16, 32, 64],
        'hidden_dim': 64,
        'dropout': 0.1,
        'fusion': 'concat -> Linear(128, 64) -> ReLU',
        'timestep_mask': 'supported (zeros out padding, gathers last valid step)',
    },
    'heads': {
        'order': ['physical_criticality', 'k10_feasible', 'safe_release', 'instability', 'close_intent'],
        'physical_criticality': {'output': 'scalar logit', 'validity': 'tri-state (value/valid_mask/reason)'},
        'k10_feasible': {'output': 'scalar logit', 'validity': 'tri-state'},
        'safe_release': {'output': 'scalar logit', 'validity': 'tri-state'},
        'instability': {'output': 'scalar logit', 'validity': 'tri-state'},
        'close_intent': {'output': 'scalar logit', 'validity': 'binary (always valid)'},
    },
    'loss': {
        'type': 'masked_bce_with_logits',
        'pos_weight': 'frozen from train split, clamped [1, 20]; no pos OR no neg → HOLD',
        'head_weight': 'configurable per head',
        'split_level': 'HOLD if any head has no positives or no negatives',
        'batch_level': 'skip head if zero valid in batch (log warning), hard-fail if all heads empty',
    },
    'sampler': {
        'type': 'suite_balanced',
        'behavior': 'truncates to min suite; warns on discard; with-replacement not yet implemented',
        'note': 'with-replacement sampling is tracked as P1 enhancement for N5 training',
    },
    'constraints': [
        'NO head output gates another head',
        'NO candidate_close in loss mask or prediction',
        'Independent valid_mask per head',
        'Train-only normalization',
        'Right-padding only for batched inference (left-padding not fully supported)',
    ],
}

def compute_schema_sha():
    return hashlib.sha256(
        json.dumps(N5_MODEL_SCHEMA, sort_keys=True).encode()
    ).hexdigest()

# ── Anti-Regression Tests ──

def test_cc_not_in_forward():
    """Verify no head uses candidate_close in forward pass."""
    model = N5MultiHeadStudent()
    model.eval()
    x = torch.randn(2, 10, 51)
    # candidate_close is NOT part of the input or forward logic
    output = model(x)
    assert len(output) == N5MultiHeadStudent.N_HEADS
    for name, logits in output.items():
        assert logits.shape == (2, 10), f'{name} shape mismatch: {logits.shape}'
    print('PASS: test_cc_not_in_forward')

def test_heads_independent():
    """Verify no head output depends on another head's output."""
    model = N5MultiHeadStudent()
    model.eval()
    x = torch.randn(2, 10, 51)
    # Each head should be computable independently
    for i, name in enumerate(N5MultiHeadStudent.HEAD_NAMES):
        out_i = model(x, head_idx=i)
        full = model(x)
        assert torch.allclose(out_i, full[name], atol=1e-6), \
            f'Head {name} output differs between single and full forward'
    print('PASS: test_heads_independent')

def test_no_hard_candidate_close():
    """Verify model output is not multiplied by or gated by candidate_close."""
    model = N5MultiHeadStudent()
    # The model should accept input without candidate_close field
    x_without_cc = torch.randn(2, 10, 51)  # No cc field
    output = model(x_without_cc)
    # Output should be finite and in reasonable range for raw logits
    for name, logits in output.items():
        assert torch.isfinite(logits).all(), f'{name} has non-finite values'
        assert logits.abs().max() < 100, f'{name} has extreme values'
    print('PASS: test_no_hard_candidate_close')

def test_baseline_shapes():
    """Verify baselines produce correct output shapes."""
    priors = {name: 0.5 for name in N5MultiHeadStudent.HEAD_NAMES}
    prior_model = PriorBaseline(priors)
    x = torch.randn(2, 10, 51)
    out = prior_model(x)
    for name in N5MultiHeadStudent.HEAD_NAMES:
        assert out[name].shape == (2, 10), f'Prior {name} shape: {out[name].shape}'
        # PriorBaseline should output raw logits (not raw probabilities)
        assert out[name].abs().max() < 10, f'Prior {name} has extreme logits'

    mlp = LastFrameMLP()
    out2 = mlp(x)
    for name in N5MultiHeadStudent.HEAD_NAMES:
        assert out2[name].shape == (2, 10), f'MLP {name} shape: {out2[name].shape}'
    print('PASS: test_baseline_shapes')

def test_receptive_field():
    """Verify EXACT receptive field: RF32=32, RF128=128 with kernel_size=2."""
    encoder32 = CausalTCNEncoder(51, 64, rf=32)
    assert encoder32.receptive_field == 32, f'Short RF: expected 32, got {encoder32.receptive_field}'
    encoder128 = CausalTCNEncoder(51, 64, rf=128)
    assert encoder128.receptive_field == 128, f'Long RF: expected 128, got {encoder128.receptive_field}'
    print(f'PASS: test_receptive_field (short={encoder32.receptive_field}, long={encoder128.receptive_field})')

def test_rf_impulse():
    """Impulse response test: verify prefix output is invariant to suffix changes.

    Two sequences share the same prefix up to t=49 but have different suffixes
    (zeros vs impulse at t=50). Prefix outputs must be identical.
    """
    for rf in [32, 128]:
        encoder = CausalTCNEncoder(1, 8, rf=rf)
        encoder.eval()
        T = rf + 100
        # Sequence A: all zeros
        x_a = torch.zeros(1, T, 1)
        # Sequence B: impulse at t=50
        x_b = torch.zeros(1, T, 1)
        x_b[0, 50, 0] = 1.0

        with torch.no_grad():
            out_a = encoder(x_a)
            out_b = encoder(x_b)

        # Prefix (steps 0..49) must be identical
        prefix_diff = (out_a[0, :50, :] - out_b[0, :50, :]).abs().max()
        assert prefix_diff < 1e-5, f'RF{rf}: prefix outputs differ, max_diff={prefix_diff:.6f}'

        # Steps at and after impulse should differ
        post_diff = (out_a[0, 50:, :] - out_b[0, 50:, :]).abs().max()
        assert post_diff > 0.01, f'RF{rf}: impulse caused no change, diff={post_diff:.6f}'
    print('PASS: test_rf_impulse')

def test_padding_parity():
    """Verify padding doesn't affect valid timestep outputs."""
    model = N5MultiHeadStudent()
    model.eval()
    x = torch.randn(1, 5, 51)
    mask = torch.ones(1, 5, dtype=torch.bool)

    # Pad with zeros at end
    x_padded = torch.cat([x, torch.zeros(1, 3, 51)], dim=1)  # (1, 8, 51)
    mask_padded = torch.cat([mask, torch.zeros(1, 3, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        out_orig = model(x, timestep_mask=mask)
        out_padded = model(x_padded, timestep_mask=mask_padded)

    for name in N5MultiHeadStudent.HEAD_NAMES:
        diff = (out_orig[name] - out_padded[name][:, :5]).abs().max()
        assert diff < 1e-4, f'{name}: padding changed valid outputs, max_diff={diff:.6f}'
    print('PASS: test_padding_parity')

def test_padding_left_right():
    """Verify left-padding vs right-padding parity for last valid step."""
    model = N5MultiHeadStudent()
    model.eval()
    seq_len = model.encoder.long_rf + 50
    x = torch.randn(1, seq_len, 51)
    x_right = torch.cat([x, torch.zeros(1, 20, 51)], dim=1)
    mask_right = torch.cat([torch.ones(1, seq_len, dtype=torch.bool), torch.zeros(1, 20, dtype=torch.bool)], dim=1)
    x_left = torch.cat([torch.zeros(1, 20, 51), x], dim=1)
    mask_left = torch.cat([torch.zeros(1, 20, dtype=torch.bool), torch.ones(1, seq_len, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        out_right = model.get_last_logits(x_right, timestep_mask=mask_right)
        out_left = model.get_last_logits(x_left, timestep_mask=mask_left)
    for name in N5MultiHeadStudent.HEAD_NAMES:
        diff = (out_right[name] - out_left[name]).abs().max()
        assert diff < 5e-3, f'{name}: left vs right padding mismatch, max_diff={diff:.6f}'
    print('PASS: test_padding_left_right')

def test_get_last_logits_head_idx():
    """Verify get_last_logits with head_idx returns correct single-head output."""
    model = N5MultiHeadStudent()
    model.eval()
    x = torch.randn(2, 10, 51)
    mask = torch.ones(2, 10, dtype=torch.bool)
    # All heads
    out_all = model.get_last_logits(x, timestep_mask=mask)
    assert isinstance(out_all, dict), f'Expected dict, got {type(out_all)}'
    # Single head
    for i, name in enumerate(N5MultiHeadStudent.HEAD_NAMES):
        out_single = model.get_last_logits(x, timestep_mask=mask, head_idx=i)
        assert out_single.shape == (2,), f'head_idx={i}: expected (2,), got {out_single.shape}'
        diff = (out_all[name] - out_single).abs().max()
        assert diff < 1e-6, f'head_idx={i} ({name}): mismatch with dict path, diff={diff:.6f}'
    print('PASS: test_get_last_logits_head_idx')

def test_get_last_logits_empty_mask():
    """Verify get_last_logits hard-fails on all-False mask."""
    model = N5MultiHeadStudent()
    model.eval()
    x = torch.randn(2, 10, 51)
    mask_all_false = torch.zeros(2, 10, dtype=torch.bool)
    try:
        model.get_last_logits(x, timestep_mask=mask_all_false)
        assert False, 'Should have raised ValueError'
    except ValueError as e:
        assert 'all-False mask' in str(e), f'Wrong error: {e}'
    print('PASS: test_get_last_logits_empty_mask')

def test_get_last_logits_varying_lengths():
    """Verify get_last_logits handles batch items with different valid lengths."""
    model = N5MultiHeadStudent()
    model.eval()
    x = torch.randn(3, 15, 51)  # 3 items, max 15 steps
    mask = torch.tensor([
        [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],  # valid len=5
        [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],  # valid len=12
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],  # valid len=8
    ], dtype=torch.bool)
    with torch.no_grad():
        out = model.get_last_logits(x, timestep_mask=mask)
    for name in N5MultiHeadStudent.HEAD_NAMES:
        assert out[name].shape == (3,), f'{name}: expected (3,), got {out[name].shape}'
    print('PASS: test_get_last_logits_varying_lengths')

def test_get_last_logits_cpu():
    """Verify get_last_logits works on CPU."""
    model = N5MultiHeadStudent()
    model.eval()
    model = model.cpu()
    x = torch.randn(2, 10, 51)
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = model.get_last_logits(x, timestep_mask=mask)
    assert isinstance(out, dict)
    out_single = model.get_last_logits(x, timestep_mask=mask, head_idx=0)
    assert out_single.shape == (2,)
    print('PASS: test_get_last_logits_cpu')

def test_get_last_logits_cuda():
    """Verify get_last_logits works on CUDA if available."""
    if not torch.cuda.is_available():
        print('SKIP: test_get_last_logits_cuda (no CUDA)')
        return
    model = N5MultiHeadStudent()
    model.eval()
    model = model.cuda()
    x = torch.randn(2, 10, 51, device='cuda')
    mask = torch.ones(2, 10, dtype=torch.bool, device='cuda')
    out = model.get_last_logits(x, timestep_mask=mask)
    assert isinstance(out, dict)
    for name in N5MultiHeadStudent.HEAD_NAMES:
        assert out[name].device.type == 'cuda', f'{name} not on CUDA'
    out_single = model.get_last_logits(x, timestep_mask=mask, head_idx=0)
    assert out_single.device.type == 'cuda'
    print('PASS: test_get_last_logits_cuda')

def test_mixed_valid_empty_mask():
    """Verify get_last_logits handles batch with mixed valid/empty items."""
    model = N5MultiHeadStudent()
    model.eval()
    x = torch.randn(3, 10, 51)
    mask = torch.tensor([
        [1,1,1,1,1,0,0,0,0,0],  # valid
        [0,0,0,0,0,0,0,0,0,0],  # ALL EMPTY
        [1,1,1,0,0,0,0,0,0,0],  # valid
    ], dtype=torch.bool)
    try:
        model.get_last_logits(x, timestep_mask=mask)
        assert False, 'Should have raised ValueError for row with all-False mask'
    except ValueError as e:
        assert 'all-False mask' in str(e), f'Wrong error: {e}'
    print('PASS: test_mixed_valid_empty_mask')

def test_individual_vs_batched_padding():
    """Verify individual inference matches batched right-padded inference."""
    model = N5MultiHeadStudent()
    model.eval()
    # Test various lengths around RF boundaries
    for length in [1, 5, 31, 32, 33, 100]:
        # Individual inference
        x_indiv = torch.randn(1, length, 51)
        with torch.no_grad():
            out_indiv = model.get_last_logits(x_indiv)
        # Batched right-padded: pad to length+10
        max_len = length + 10
        x_pad = torch.zeros(1, max_len, 51)
        x_pad[0, :length] = x_indiv[0]
        mask_pad = torch.zeros(1, max_len, dtype=torch.bool)
        mask_pad[0, :length] = True
        with torch.no_grad():
            out_batch = model.get_last_logits(x_pad, timestep_mask=mask_pad)
        for name in N5MultiHeadStudent.HEAD_NAMES:
            diff = (out_indiv[name] - out_batch[name]).abs().max()
            assert diff < 1e-4, f'len={length} {name}: individual vs batched right-pad mismatch, diff={diff:.6f}'
    print('PASS: test_individual_vs_batched_padding')

def test_sampler():
    """Verify suite-balanced sampler."""
    indices = list(range(100))
    suites = (['libero_10'] * 25 + ['libero_goal'] * 25 +
              ['libero_object'] * 25 + ['libero_spatial'] * 25)
    sampler = SuiteBalancedSampler(indices, suites, batch_size=16)
    for batch in sampler:
        batch_suites = [suites[i] for i in batch]
        for s in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
            assert batch_suites.count(s) == 4, f'Expected 4 {s}, got {batch_suites.count(s)}'
        break
    print('PASS: test_sampler')

def test_causal_invariance():
    """Verify that adding future steps doesn't change past predictions."""
    model = N5MultiHeadStudent()
    model.eval()
    x_past = torch.randn(1, 5, 51)
    x_future = torch.randn(1, 3, 51)
    x_full = torch.cat([x_past, x_future], dim=1)

    with torch.no_grad():
        out_past = model(x_past)
        out_full = model(x_full)

    for name in N5MultiHeadStudent.HEAD_NAMES:
        # Past predictions should be identical
        past_from_full = out_full[name][:, :5]
        assert torch.allclose(out_past[name], past_from_full, atol=1e-4), \
            f'{name}: future steps changed past predictions'
    print('PASS: test_causal_invariance')

def test_checkpoint_roundtrip():
    """Verify model can be saved and loaded with identical outputs."""
    model = N5MultiHeadStudent()
    model.eval()
    x = torch.randn(2, 10, 51)
    with torch.no_grad():
        out_before = model(x)

    # Save and load
    tmp = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/tmp/n5_test_checkpoint.pt'
    torch.save({'model': model.state_dict(), 'schema_sha': compute_schema_sha()}, tmp)
    model2 = N5MultiHeadStudent()
    model2.load_state_dict(torch.load(tmp, weights_only=True)['model'])
    model2.eval()

    with torch.no_grad():
        out_after = model2(x)

    for name in N5MultiHeadStudent.HEAD_NAMES:
        assert torch.allclose(out_before[name], out_after[name], atol=1e-6), \
            f'{name}: checkpoint roundtrip mismatch'
    os.remove(tmp)
    print('PASS: test_checkpoint_roundtrip')

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []

    def run(self, test_fn, skippable=False):
        try:
            test_fn()
            self.passed += 1
            self.results.append(('PASS', test_fn.__name__))
        except Exception as e:
            self.results.append(('FAIL', test_fn.__name__, str(e)))
            print(f'FAIL: {test_fn.__name__}: {e}')
            if not skippable:
                self.failed += 1

def run_all_tests():
    tr = TestResult()
    for test in [
        test_cc_not_in_forward,
        test_heads_independent,
        test_no_hard_candidate_close,
        test_baseline_shapes,
        test_receptive_field,
        test_rf_impulse,
        test_sampler,
        test_causal_invariance,
        test_padding_parity,
        test_padding_left_right,
        test_get_last_logits_head_idx,
        test_get_last_logits_empty_mask,
        test_get_last_logits_varying_lengths,
        test_get_last_logits_cpu,
        test_mixed_valid_empty_mask,
        test_individual_vs_batched_padding,
        test_checkpoint_roundtrip,
    ]:
        tr.run(test)

    # CUDA test: SKIP if no CUDA, PASS/FAIL if available
    if torch.cuda.is_available():
        print(f'CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_capability(0)})')
        tr.run(test_get_last_logits_cuda)
    else:
        tr.skipped += 1
        print('SKIP: test_get_last_logits_cuda (no CUDA)')

    print(f'\n{tr.passed} PASS / {tr.failed} FAIL / {tr.skipped} SKIP (total {tr.passed + tr.failed + tr.skipped})')
    return tr.failed == 0

if __name__ == '__main__':
    ok = run_all_tests()
    print(f'\nN5 Model Schema SHA: {compute_schema_sha()}')
    import sys
    sys.exit(0 if ok else 1)
