"""V2.2 Student: strict-K10-now + within-H + onset-band auxiliary heads.

43D V2-B CausalTCN backbone, no RGB, no V1 phase FSM.
"""
import torch, torch.nn as nn
from typing import Dict, Optional, List

class CausalTCNEncoder(nn.Module):
    def __init__(self, input_dim=43, hidden_dim=64, receptive_field=32, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        layers = []; current_dim = input_dim; dilation = 1
        while dilation < receptive_field:
            layers.append(nn.Conv1d(current_dim, hidden_dim, 2, dilation=dilation, padding=0))
            layers.append(nn.ReLU()); layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim; dilation *= 2
            if dilation * 2 > receptive_field * 2: break
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x):
        x_t = x.transpose(1,2); B,D,T = x_t.shape
        total_pad = sum(m.dilation[0]*(m.kernel_size[0]-1) for m in self.conv_stack if isinstance(m,nn.Conv1d))
        if total_pad > 0: x_t = nn.functional.pad(x_t, (total_pad, 0))
        out = self.conv_stack(x_t)
        if out.shape[2] > T: out = out[:,:,-T:]
        return out.transpose(1,2)

class LocalizationStudentV22(nn.Module):
    """V2.2: 43D → CausalTCN → shared hidden → {k10_now, within_H, onset_band}"""
    def __init__(self, hidden_dim=64, receptive_field=32, dropout=0.1,
                 use_within_h=True, use_onset_band=True, within_h=5):
        super().__init__()
        self.encoder = CausalTCNEncoder(43, hidden_dim, receptive_field, dropout)
        self.hidden_dim = hidden_dim; self.within_h = within_h
        self.use_within_h = use_within_h; self.use_onset_band = use_onset_band

        self.head_now = nn.Linear(hidden_dim, 1)
        if use_within_h: self.head_within = nn.Linear(hidden_dim, 1)
        if use_onset_band: self.head_onset = nn.Linear(hidden_dim, 1)

    def forward(self, x_43d):
        h = self.encoder(x_43d)
        out = {'k10_now': self.head_now(h)}
        if self.use_within_h: out['within_H'] = self.head_within(h)
        if self.use_onset_band: out['onset_band'] = self.head_onset(h)
        return out
