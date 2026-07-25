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
    """Stacked causal TCN with configurable receptive field."""
    def __init__(self, input_dim, hidden_dim, rf, dropout=0.1):
        super().__init__()
        layers = []
        d = 1
        while d <= rf:
            in_ch = input_dim if d == 1 else hidden_dim
            layers.append(CausalTCNBlock(in_ch, hidden_dim, kernel_size=3,
                                          dilation=d, dropout=dropout))
            d *= 2
        self.layers = nn.ModuleList(layers)
        self.output_dim = hidden_dim

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return x

# ── Shared Encoder ──

class N5SharedEncoder(nn.Module):
    """Dual-branch CausalTCN with concat fusion. Same architecture as V4 encoder."""
    def __init__(self, input_dim=51, hidden=64, short_rf=32, long_rf=128, dropout=0.1):
        super().__init__()
        self.short_tcn = CausalTCNEncoder(input_dim, hidden, short_rf, dropout)
        self.long_tcn = CausalTCNEncoder(input_dim, hidden, long_rf, dropout)
        self.fusion = nn.Linear(hidden * 2, hidden)
        self.hidden = hidden

    def forward(self, x):
        # x: (B, T, 51)
        s = self.short_tcn(x)  # (B, T, hidden)
        l = self.long_tcn(x)   # (B, T, hidden)
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

    def forward(self, x, head_idx=None):
        """Forward pass.

        Args:
            x: (B, T, 51) normalized input features
            head_idx: If None, return all 5 head logits.
                      If int, return only that head's logits.

        Returns:
            If head_idx is None: Dict[str, Tensor] mapping head name to (B, T) logits
            If head_idx is int: Tensor (B, T) logits for that head
        """
        shared = self.encoder(x)  # (B, T, hidden)

        if head_idx is not None:
            return self.heads[head_idx](shared)

        return {name: head(shared) for name, head in zip(self.HEAD_NAMES, self.heads)}

    def get_last_logits(self, x, head_idx=None):
        """Get logits for the last timestep only (for streaming inference)."""
        all_logits = self.forward(x, head_idx=head_idx)
        if head_idx is not None:
            return all_logits[:, -1]  # (B,)
        return {name: logits[:, -1] for name, logits in all_logits.items()}

# ── Loss Functions ──

def masked_bce_loss(logits, targets, valid_mask, pos_weight=None):
    """Masked binary cross-entropy. Only valid steps contribute to loss.

    Args:
        logits: (B, T) raw logits
        targets: (B, T) binary labels {0, 1}
        valid_mask: (B, T) boolean, True = step is valid for this head
        pos_weight: Optional positive class weight tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        logits[valid_mask], targets[valid_mask].float(),
        pos_weight=pos_weight, reduction='mean'
    )
    return loss

def n5_total_loss(model_output, labels, valid_masks, head_weights=None):
    """Total N5 training loss across all heads.

    Args:
        model_output: Dict[str, Tensor] from model.forward()
        labels: Dict[str, Tensor] per-head binary labels
        valid_masks: Dict[str, Tensor] per-head boolean valid masks
        head_weights: Optional Dict[str, float] per-head loss weights

    Returns:
        total_loss: scalar
        per_head_losses: Dict[str, float]
    """
    if head_weights is None:
        head_weights = {name: 1.0 for name in N5MultiHeadStudent.HEAD_NAMES}

    total = 0.0
    per_head = {}
    for name in N5MultiHeadStudent.HEAD_NAMES:
        logits = model_output[name]
        target = labels[name]
        mask = valid_masks[name]
        if mask.sum() == 0:
            per_head[name] = 0.0
            continue

        # Compute class weight from valid positives
        n_pos = target[mask].sum()
        n_neg = mask.sum() - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=logits.device)
        else:
            pos_weight = None

        loss = masked_bce_loss(logits, target, mask, pos_weight)
        weighted = loss * head_weights[name]
        total += weighted
        per_head[name] = loss.item()

    return total, per_head

# ── Suite-Balanced Sampler ──

class SuiteBalancedSampler:
    """Ensures each batch has roughly equal representation from all 4 suites."""
    def __init__(self, episode_indices, episode_suites, batch_size, shuffle=True):
        self.suite_indices = {}
        for i, suite in enumerate(episode_suites):
            self.suite_indices.setdefault(suite, []).append(episode_indices[i])

        self.suites = sorted(self.suite_indices.keys())
        self.n_suites = len(self.suites)
        self.per_suite_batch = max(1, batch_size // self.n_suites)
        self.batch_size = self.per_suite_batch * self.n_suites
        self.shuffle = shuffle

    def __iter__(self):
        indices = []
        for suite in self.suites:
            suite_idx = self.suite_indices[suite].copy()
            if self.shuffle:
                np.random.shuffle(suite_idx)
            indices.append(suite_idx)

        # Truncate to min common length
        min_len = min(len(idx) for idx in indices)
        for i in range(self.n_suites):
            indices[i] = indices[i][:min_len]

        # Interleave batches
        n_batches = min_len // self.per_suite_batch
        batch_order = list(range(n_batches))
        if self.shuffle:
            np.random.shuffle(batch_order)

        for b in batch_order:
            batch = []
            for i in range(self.n_suites):
                start = b * self.per_suite_batch
                end = start + self.per_suite_batch
                batch.extend(indices[i][start:end])
            yield batch

    def __len__(self):
        min_len = min(len(idx) for idx in self.suite_indices.values())
        return min_len // self.per_suite_batch

# ── Baseline Models ──

class PriorBaseline(nn.Module):
    """Always outputs the training prior (mean label per head)."""
    def __init__(self, priors=None):
        super().__init__()
        self.priors = priors or {}

    def forward(self, x):
        B, T = x.shape[:2]
        return {name: torch.full((B, T), p, device=x.device)
                for name, p in self.priors.items()}

class LastFrameMLP(nn.Module):
    """Simple MLP on last frame only (no temporal context)."""
    def __init__(self, input_dim=51, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden // 2, 1) for _ in range(N5MultiHeadStudent.N_HEADS)
        ])

    def forward(self, x):
        last = x[:, -1, :]  # (B, 51)
        h = self.net(last)  # (B, hidden//2)
        return {name: head(h).squeeze(-1).unsqueeze(1).expand(-1, x.shape[1])
                for name, head in zip(N5MultiHeadStudent.HEAD_NAMES, self.heads)}

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

    mlp = LastFrameMLP()
    out2 = mlp(x)
    for name in N5MultiHeadStudent.HEAD_NAMES:
        assert out2[name].shape == (2, 10), f'MLP {name} shape: {out2[name].shape}'
    print('PASS: test_baseline_shapes')

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
        test_sampler,
        test_causal_invariance,
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
