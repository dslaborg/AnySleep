"""
Early fusion AnySleep variant (AnySleepEF).

This module implements an attention-based sleep staging architecture where
channel attention is applied BEFORE the U-Net encoder. The attention mechanism
learns to weight and combine multiple input channels into a fixed number of
"virtual channels" before any feature extraction occurs.

Architecture Overview:
    Input (batch, time, channels)
        ↓
    ChannelAttention → Virtual channels (batch, n_virtual, time)
        ↓
    U-Net Encoder (12 blocks)
        ↓
    Connector
        ↓
    U-Net Decoder (11 blocks)
        ↓
    SegmentClassifier → Sleep stage predictions

Key Difference from AnySleep (base anysleep.py):
    - AnySleep applies attention on skip connections (mid-fusion)
    - AnySleepEF applies attention on raw input (early fusion)
    - Early fusion reduces computational cost by processing fewer channels
"""

from logging import getLogger

import torch
from torch import nn

from base.config import Config
from base.model.base_model import BaseModel, _weights_init
from base.model.usleep_base_components import (
    SegmentClassifier,
    USleepEncoderBlock,
    USleepDecoderBlock,
)
from base.model.utilities import calc_filters

logger = getLogger(__name__)


class AnySleepEF(BaseModel):
    """
    Early fusion attention U-Sleep model (AnySleepEF).

    This architecture applies channel attention at the input level, combining
    multiple input channels into virtual channels BEFORE the U-Net encoder.
    This is in contrast to AnySleep which applies attention on skip connections.

    The model first uses a learned attention mechanism to weight and combine
    input channels, then processes the resulting virtual channels through a
    standard U-Net architecture for sleep stage classification.

    Args:
        path (str, optional): Path to pre-trained weights. Loaded after initialization.
        seed (int, optional): Random seed for weight initialization reproducibility.
        depth (int): Number of encoder/decoder blocks. Default: 12.
        complexity (float): Base filter multiplier. Default: 1.2923.
        scale (float): Filter scaling factor between blocks. Default: 1.4142.
        num_virt_channels (int): Number of virtual channels to produce from
            attention. Each virtual channel learns a different weighting of
            input channels. Default: 1.
        att_filters (int): Number of filters in attention encoder CNN. Default: 32.
        att_hidden_size (int): Hidden layer size in attention MLPs. Default: 32.
        sleep_stage_frequency (int): Sleep stages per 30s epoch. Default: 1.
        **kwargs: Additional arguments (ignored, for config compatibility).

    Example:
        >>> model = AnySleepEF(
        ...     depth=12,
        ...     num_virt_channels=2,
        ...     att_filters=32
        ... )
        >>> # Input: (batch=4, time=17280, channels=6)
        >>> x = torch.randn(4, 17280, 6)
        >>> output = model(x)  # (4, 4, 5) - 4 epochs, 5 sleep stages
    """

    identifier = "anysleep_early_fusion"

    def __init__(
        self,
        path=None,
        seed: int = None,
        depth: int = 12,
        complexity: float = 1.2923,
        scale: float = 1.4142,
        num_virt_channels: int = 1,
        att_filters: int = 32,
        att_hidden_size: int = 32,
        sleep_stage_frequency: int = 1,  # measured as sleep stages per epoch
        **kwargs,
    ):
        super(AnySleepEF, self).__init__(seed)

        cs_enc, cs_dec = calc_filters(num_virt_channels, depth, complexity, scale)

        self.channel_attention = ChannelAttention(
            num_virt_channels, att_filters, att_hidden_size
        )

        self.encoders = nn.ModuleList(
            [
                USleepEncoderBlock(cs_enc[i - 1], cs_enc[i])
                for i in range(1, len(cs_enc))
            ]
        )
        self.connector = nn.Sequential(
            nn.Conv1d(cs_enc[-1], cs_dec[0], 9, padding="same"),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(cs_dec[0]),
        )
        self.decoders = nn.ModuleList(
            [
                USleepDecoderBlock(cs_dec[i - 1], cs_dec[i])
                for i in range(1, len(cs_dec))
            ]
        )

        self.segment_classifier = SegmentClassifier(sleep_stage_frequency, cs_dec[-1])

        self.apply(_weights_init)

        # always load model at the end of the initialization
        self.load(path)

    def forward(self, x):
        """
        Forward pass with early channel attention fusion.

        The input channels are first combined via learned attention weights,
        then processed through the U-Net encoder-decoder architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, channels).
                Time dimension should be samples_per_epoch * num_epochs.

        Returns:
            torch.Tensor: Sleep stage logits of shape (batch, epochs, 5).
                5 classes: Wake, N1, N2, N3, REM.
        """
        # x shape: (batch, time, channels)

        _cfg = Config.get()
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        # x shape: (batch, channels, time)

        # attention
        x = self.channel_attention(x)
        # x shape: (batch, virt_channels, time)

        # encoder
        x_res = []
        for i in range(len(self.encoders)):
            x_r, x = self.encoders[i](x)
            x_res.append(x_r)

        x = self.connector(x)

        # decoder
        for i in range(len(self.decoders)):
            x = self.decoders[i](x, x_res.pop())

        x = self.segment_classifier(x)

        return x.transpose(1, 2)


class ChannelAttention(nn.Module):
    """
    Early fusion channel attention module for raw signals.

    This module learns to combine multiple input channels into a fixed number
    of "virtual channels" by computing attention weights from the raw signals.
    Each virtual channel learns a different soft weighting of the input channels.

    Architecture:
        1. Encode each channel independently with a small CNN
        2. Global average pool to get per-channel features
        3. For each virtual channel, an MLP produces attention scores
        4. Softmax normalizes scores across input channels
        5. Weighted sum produces the virtual channel output

    Args:
        n_virtual_channels (int): Number of virtual channels to produce.
            Each has its own MLP to learn different channel weightings.
        encoder_filters (int): Number of filters in the per-channel CNN encoder.
        mlp_hidden_size (int): Hidden layer size in the attention MLPs.

    Note:
        The CNN kernel sizes depend on the sampling rate from Config.
        With 128 Hz: kernel_size=64, stride=32 (covers 0.5s with 0.25s stride).
    """

    def __init__(self, n_virtual_channels, encoder_filters, mlp_hidden_size):
        super(ChannelAttention, self).__init__()

        _cfg = Config.get()
        sampling_rate = _cfg.data.sampling_rate

        self.encoder = nn.Sequential(
            nn.Conv1d(1, encoder_filters, sampling_rate // 2, sampling_rate // 4),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(encoder_filters),
            nn.Conv1d(encoder_filters, encoder_filters, 9),
        )

        self.mlps = nn.ModuleList()
        for channel in range(n_virtual_channels):
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(encoder_filters, mlp_hidden_size),
                    nn.ELU(),
                    nn.BatchNorm1d(mlp_hidden_size),
                    nn.Linear(mlp_hidden_size, 1),
                )
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Compute attention-weighted virtual channels from raw input.

        Args:
            x (torch.Tensor): Input signals of shape (batch, channels, time).

        Returns:
            torch.Tensor: Virtual channels of shape (batch, n_virtual_channels, time).
                Each virtual channel is a learned weighted sum of input channels.
        """
        # x shape (batch, channels, time)
        batch = x.shape[0]

        x_att = x.reshape(-1, 1, x.shape[-1])
        # x_att shape (batch * channels, 1, time)

        x_att = self.encoder(x_att)
        # x_att shape (batch * channels, filters, time)

        x_att = x_att.mean(dim=-1)
        # x_att shape (batch * channels, filters)

        results = []
        for mlp in self.mlps:
            alpha = mlp(x_att).reshape(batch, -1)
            # alpha shape (batch, channels)
            alpha = self.softmax(alpha).unsqueeze(-1)
            # alpha shape (batch, channels, 1)

            x_out = x * alpha
            x_out = x_out.sum(1)
            # x_out shape (batch, time)

            results.append(x_out.unsqueeze(1))

        return torch.cat(results, dim=1)
