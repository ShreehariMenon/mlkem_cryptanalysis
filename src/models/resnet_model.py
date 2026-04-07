"""
resnet_model.py
---------------
1D Residual Network for ML-KEM ciphertext statistical analysis.

Architecture
------------
Input  : (batch, 1, ct_bits)  — bit-sliced ciphertext as 1D signal
Output : (batch, 2)           — binary class logits

Design choices
--------------
- 1D convolutions treat the bit-stream as a signal, allowing the network
  to learn local structural patterns (e.g. NTT coefficient boundaries)
  rather than global correlations only, as an MLP would.
- Batch normalisation after every conv layer stabilises training on
  binary (0/1) inputs which have very low variance per channel.
- Max-pooling between ResBlocks reduces sequence length progressively,
  making the deeper layers attend to coarser structure.
- The head uses Dropout(0.3) to prevent overfitting on the label
  distribution (MSB is 50/50 by construction).
- Parameter count is logged at init so you can verify model size
  before committing to a long training run.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ResBlock1D(nn.Module):
    """
    One residual block: two Conv1D layers with BN and ReLU,
    plus a skip connection.

    If in_channels ≠ out_channels, a 1×1 conv projects the residual.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_block(x) + self.skip(x))


class CryptoResNet1D(nn.Module):
    """
    Full 1D ResNet for ML-KEM ciphertext classification.

    Parameters
    ----------
    input_len   : number of bits in the ciphertext (6144 / 8704 / 12544)
    n_classes   : 2 for binary classification
    base_ch     : base channel width (default 32, increase for more capacity)
    """

    def __init__(
        self,
        input_len: int,
        n_classes: int = 2,
        base_ch:   int = 32,
    ):
        super().__init__()
        self.input_len = input_len

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )

        # ── Residual stages with progressive downsampling ─────────────────────
        # Each stage: ResBlock → MaxPool(4) → halves channel count grows
        self.stage1 = nn.Sequential(ResBlock1D(base_ch,    base_ch),    nn.MaxPool1d(4))
        self.stage2 = nn.Sequential(ResBlock1D(base_ch,    base_ch*2),  nn.MaxPool1d(4))
        self.stage3 = nn.Sequential(ResBlock1D(base_ch*2,  base_ch*2),  nn.MaxPool1d(4))
        self.stage4 = nn.Sequential(ResBlock1D(base_ch*2,  base_ch*4),  nn.MaxPool1d(4))

        # ── Head ──────────────────────────────────────────────────────────────
        # Compute flattened size dynamically (avoids hardcoding for each variant)
        with torch.no_grad():
            dummy  = torch.zeros(1, 1, input_len)
            dummy  = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(dummy)))))
            flat   = dummy.flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # global average pooling → (batch, ch, 1)
            nn.Flatten(),
            nn.Linear(base_ch * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

        self._log_params()

    def _log_params(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  CryptoResNet1D | input={self.input_len} bits | "
              f"params={n:,} ({n/1e6:.2f}M)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, ct_bits)  →  unsqueeze channel dim  →  (batch, 1, ct_bits)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_model(input_len: int, device: str = "cpu") -> CryptoResNet1D:
    model = CryptoResNet1D(input_len=input_len).to(device)
    return model
