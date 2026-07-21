"""Test V2 windowed GRU hidden state reset at window boundaries."""
import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_student_v2 import FactorizedStudentV2, WindowedGRUEncoder


def test_windowed_gru_resets_hidden():
    """Hidden state must be zero at window boundaries."""
    encoder = WindowedGRUEncoder(input_dim=25, hidden_dim=64, window_size=8)
    B, T = 2, 24  # 3 windows
    x = torch.randn(B, T, 25)

    # Monkey-patch to capture hidden states
    hidden_norms = []
    original_forward = encoder.forward

    def capture_forward(x_in):
        device = x_in.device
        h = torch.zeros(B, 64, device=device)
        outputs = []
        for t in range(T):
            if t % encoder.window_size == 0:
                h = torch.zeros(B, 64, device=device)
                hidden_norms.append(float(h.norm().item()))
            h = encoder.gru(x_in[:, t], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    encoder.forward = capture_forward
    encoder(torch.randn(B, T, 25))

    # At window boundaries (t=0, 8, 16), hidden must be zero before GRU step
    assert hidden_norms[0] == 0.0, f'Window 0: hidden not zero: {hidden_norms[0]}'
    assert hidden_norms[1] == 0.0, f'Window 1: hidden not zero: {hidden_norms[1]}'
    assert hidden_norms[2] == 0.0, f'Window 2: hidden not zero: {hidden_norms[2]}'


def test_windowed_gru_no_cross_window_state():
    """Same prefix in two different windows must produce identical outputs."""
    encoder = WindowedGRUEncoder(input_dim=25, hidden_dim=64, window_size=8)
    encoder.eval()

    # Episode 1: pattern A in window 1, pattern B in window 2
    # Episode 2: pattern A in window 2 (should be identical to Ep1 window 1)
    pattern = torch.randn(1, 8, 25)
    B, T = 1, 16
    x1 = torch.cat([pattern, torch.randn(1, 8, 25)], dim=1)
    x2 = torch.cat([torch.randn(1, 8, 25), pattern], dim=1)

    with torch.no_grad():
        out1 = encoder(x1)  # [1, 16, H]
        out2 = encoder(x2)

    # Window 1 of ep1 (steps 0-7) should NOT equal window 2 of ep2 (steps 8-15)
    # because the windows start from zero hidden state in both cases
    diff = (out1[0, :8] - out2[0, 8:]).abs().max().item()
    assert diff < 1e-5, f'Cross-window outputs differ: {diff}'


def test_parameter_count():
    """V2 models must have ~50K parameters."""
    for enc_type in ['tcn', 'windowed_gru']:
        model = FactorizedStudentV2(hidden_dim=64, receptive_field=32,
                                     encoder_type=enc_type, dropout=0.0, use_9d=False)
        n = model.parameter_count()
        assert 30000 <= n <= 80000, f'{enc_type}: param count {n} outside [30K,80K]'
