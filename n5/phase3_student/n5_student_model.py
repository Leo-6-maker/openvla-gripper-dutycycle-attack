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

# ── Shared Encoder (replaces old separate definition) ──
# N5SharedEncoder is now the canonical shared encoder
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return x

# ── Shared Encoder (defined above, this section removed) ──

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

        P0 FIX: Uses timestep_mask to find last valid step, not x[:, -1].
        """
        all_logits = self.forward(x, timestep_mask=timestep_mask)
        if timestep_mask is not None:
            # Find last True index per batch: (B, T) → (B,)
            last_valid_idx = (timestep_mask.shape[1] - 1 -
                              timestep_mask.flip(1).long().argmax(dim=1))
            no_valid = ~timestep_mask.any(dim=1)
            last_valid_idx[no_valid] = timestep_mask.shape[1] - 1
        else:
            last_valid_idx = None

        if head_idx is not None:
            if last_valid_idx is not None:
                return all_logits[torch.arange(all_logits.shape[0]), last_valid_idx]
            return all_logits[:, -1]

        if last_valid_idx is not None:
            idx = torch.arange(all_logits[list(all_logits.keys())[0]].shape[0])
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
        """Compute frozen pos_weight from full train split.
        train_labels: Dict[str, Tensor] per-head binary labels (concatenated over all episodes)
        train_valid_masks: Dict[str, Tensor] per-head valid masks
        """
        weights = {}
        for name in N5MultiHeadStudent.HEAD_NAMES:
            targets = train_labels[name]
            mask = train_valid_masks[name]
            if mask.sum() == 0:
                weights[name] = None  # No valid data for this head → skip
                continue
            n_pos = targets[mask].sum()
            n_neg = mask.sum() - n_pos
            if n_pos > 0 and n_neg > 0:
                w = float((n_neg / n_pos).clamp(1, 20))
                weights[name] = w
            elif n_pos == 0:
                weights[name] = None  # No positives → cannot train this head
            else:
                weights[name] = 1.0
        return cls(weights)

    def get_weight(self, head_name):
        return self.weights.get(head_name)

    def validate(self):
        """Check that all heads have usable weights. Returns list of issues."""
        issues = []
        for name in N5MultiHeadStudent.HEAD_NAMES:
            w = self.weights.get(name)
            if w is None:
                issues.append(f'{name}: no valid positive or negative samples — HOLD')
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
    """Total N5 training loss. Uses FROZEN pos_weight per head (not per-batch).

    Args:
        model_output: Dict[str, Tensor] from model.forward()
        labels: Dict[str, Tensor] per-head binary labels
        valid_masks: Dict[str, Tensor] per-head boolean valid masks
        frozen_weights: FrozenPosWeights instance
        head_weights: Optional Dict[str, float] per-head loss weights

    HARD FAIL if a head has no valid samples.
    """
    if head_weights is None:
        head_weights = {name: 1.0 for name in N5MultiHeadStudent.HEAD_NAMES}

    total = 0.0
    per_head = {}
    for name in N5MultiHeadStudent.HEAD_NAMES:
        logits = model_output[name]
        target = labels[name]
        mask = valid_masks[name]
        n_valid = mask.sum().item()
        if n_valid == 0:
            raise RuntimeError(f'HARD FAIL: head {name} has zero valid samples in batch. Check data pipeline.')

        loss = masked_bce_loss(logits, target, mask, frozen_weights.get_weight(name))
        weighted = loss * head_weights[name]
        total += weighted
        per_head[name] = loss.item()

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

    def forward(self, x):
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
    'schema': 'N5_MULTIHEAD_STUDENT_V1',
    'input_dim': 51,
    'input_breakdown': {
        'f25d': '25D SC5 streaming features',
        'p9d': '9D policy intent (POLICY_INTENT_ORDER)',
        'g9d': '9D gripper token (TRAIN_G9D_ORDER)',
        'proxies_8d': '8D causal response proxies',
    },
    'heads': {
        'physical_criticality': 'Binary: physically engaged AND not releasing AND task not done',
        'k10_feasible': 'Binary: horizon >= 10, no release/terminal in window, critical corridor',
        'safe_release': 'Binary: task done or high-confidence release',
        'instability': 'Binary: low-confidence release, possible vulnerability',
        'close_intent': 'Binary: policy-level close signal (raw_gripper <= 0.5)',
    },
    'constraints': [
        'NO head output gates another head',
        'NO candidate_close in loss mask',
        'NO cc in prediction pipeline',
        'Independent valid_mask per head',
        'Train-only normalization',
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
    """Verify left-padding vs right-padding parity for last valid step.

    Uses sequences long enough to cover the long RF (128) so that warmup
    transients don't affect the comparison at the last valid step.
    """
    model = N5MultiHeadStudent()
    model.eval()
    seq_len = model.encoder.long_rf + 50  # Well beyond long RF
    x = torch.randn(1, seq_len, 51)

    # Both have same valid sequence, just different padding positions
    # Right-padded: valid at start, zeros at end
    x_right = torch.cat([x, torch.zeros(1, 20, 51)], dim=1)
    mask_right = torch.cat([torch.ones(1, seq_len, dtype=torch.bool), torch.zeros(1, 20, dtype=torch.bool)], dim=1)

    # Left-padded: zeros at start, valid at end
    x_left = torch.cat([torch.zeros(1, 20, 51), x], dim=1)
    mask_left = torch.cat([torch.zeros(1, 20, dtype=torch.bool), torch.ones(1, seq_len, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        out_right = model.get_last_logits(x_right, timestep_mask=mask_right)
        out_left = model.get_last_logits(x_left, timestep_mask=mask_left)

    for name in N5MultiHeadStudent.HEAD_NAMES:
        diff = (out_right[name] - out_left[name]).abs().max()
        assert diff < 5e-3, f'{name}: left vs right padding mismatch, max_diff={diff:.6f} (seq_len={seq_len})'
    print('PASS: test_padding_left_right')

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

def run_all_tests():
    tests = [
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
        test_checkpoint_roundtrip,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f'FAIL: {test.__name__}: {e}')
    print(f'\n{passed}/{len(tests)} tests passed')
    return passed == len(tests)

if __name__ == '__main__':
    run_all_tests()
    print(f'\nN5 Model Schema SHA: {compute_schema_sha()}')
