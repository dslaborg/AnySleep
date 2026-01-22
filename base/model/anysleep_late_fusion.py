"""
Late fusion AnySleep variant.

This module implements an attention-based sleep staging architecture where
channel attention is applied AFTER the U-Net decoder. Each input channel is
processed independently through the full U-Net, then attention combines the
decoded feature representations.

Architecture Overview:
    Input (batch, time, channels)
        ↓
    Reshape to (batch * channels, 1, time)
        ↓
    U-Net Encoder (12 blocks) - processes each channel independently
        ↓
    Connector
        ↓
    U-Net Decoder (11 blocks) - processes each channel independently
        ↓
    Reshape to (batch, channels, features, time)
        ↓
    ChannelAttention → Virtual channels (batch, n_virtual * features, time)
        ↓
    SegmentClassifier → Sleep stage predictions

Key Difference from other variants:
    - AnySleep (base anysleep.py): attention on skip connections (mid-fusion)
    - AnySleepEF (early_fusion.py): attention on raw input (early fusion)
    - AnySleepLF (this file): attention on decoded features (late fusion)
"""

from logging import getLogger

import torch
from torch import nn

from base.config import Config
from base.model.base_model import BaseModel, _weights_init
from base.model.usleep_base_components import (
    SegmentClassifier,
    USleepDecoderBlock,
    USleepEncoderBlock,
)
from base.model.utilities import calc_filters

logger = getLogger(__name__)


class AnySleepLF(BaseModel):
    """
    Late fusion attention U-Sleep model (AnySleepLF).

    This architecture processes each input channel independently through the
    full U-Net encoder-decoder, then applies channel attention on the decoded
    feature maps. This allows the attention mechanism to operate on learned
    features rather than raw signals.

    Each channel goes through its own forward pass of the U-Net (with shared
    weights), producing per-channel feature maps. The attention module then
    learns to weight and combine these feature maps into virtual channels.

    Trade-offs:
        - More expressive: attention sees learned features, not just raw signals
        - More expensive: full U-Net computation for each input channel
        - Shared U-Net weights: each channel uses the same feature extractor

    Args:
        path (str, optional): Path to pre-trained weights. Loaded after initialization.
        seed (int, optional): Random seed for weight initialization reproducibility.
        depth (int): Number of encoder/decoder blocks. Default: 12.
        complexity (float): Base filter multiplier. Default: 1.2923.
        scale (float): Filter scaling factor between blocks. Default: 1.4142.
        att_hidden_size (int): Hidden layer size in attention MLPs. Default: 32.
        att_filters (int): Number of filters in attention encoder CNN. Default: 32.
        sleep_stage_frequency (int): Sleep stages per 30s epoch. Default: 1.
        **kwargs: Additional arguments (ignored, for config compatibility).

    Example:
        >>> model = AnySleepLF(depth=12, att_filters=32)
        >>> # Input: (batch=4, time=17280, channels=6)
        >>> x = torch.randn(4, 17280, 6)
        >>> output = model(x)  # (4, 4, 5) - 4 epochs, 5 sleep stages
    """

    identifier = "anysleep_late_fusion"

    def __init__(
        self,
        path=None,
        seed: int = None,
        depth: int = 12,
        complexity: float = 1.2923,
        scale: float = 1.4142,
        att_hidden_size: int = 32,
        att_filters: int = 32,
        sleep_stage_frequency: int = 1,  # measured as sleep stages per epoch
        **kwargs,
    ):
        super(AnySleepLF, self).__init__(seed)

        cs_enc, cs_dec = calc_filters(1, depth, complexity, scale)

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

        self.channel_attention = ChannelAttention(
            1, cs_dec[-1], att_filters, att_hidden_size
        )

        self.segment_classifier = SegmentClassifier(sleep_stage_frequency, cs_dec[-1])

        self.apply(_weights_init)

        # always load model at the end of the initialization
        self.load(path)

    def forward(self, x):
        """
        Forward pass with late channel attention fusion.

        Each input channel is processed independently through the U-Net,
        then attention combines the decoded feature maps.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, channels).
                Time dimension should be samples_per_epoch * num_epochs.

        Returns:
            torch.Tensor: Sleep stage logits of shape (batch, epochs, 5).
                5 classes: Wake, N1, N2, N3, REM.
        """
        # x shape: (batch, time, channels)
        batch, _, channels = x.shape

        _cfg = Config.get()
        x = x.transpose(1, 2).reshape(batch * channels, 1, -1)
        # x shape: (batch * channels, n_filters, time)

        # encoder
        x_res = []
        for i in range(len(self.encoders)):
            x_r, x = self.encoders[i](x)
            # x_r/x shape: (batch * channels, n_filters, time)
            x_res.append(x_r)

        x = self.connector(x)
        # x shape: (batch * channels, n_filters, time)

        # decoder
        for i in range(len(self.decoders)):
            x = self.decoders[i](x, x_res.pop())
            # x shape: (batch * channels, n_filters, time)

        # attention
        # x shape: (batch * channels, n_filters, time)
        x = x.reshape(batch, channels, *x.shape[1:])
        # x shape: (batch, channels, n_filters, time)

        x = self.channel_attention(x).reshape(batch, -1, x.shape[-1])
        # x shape: (batch, n_virt_channels * n_filters, time)

        x = self.segment_classifier(x)

        return x.transpose(1, 2)


class ChannelAttention(nn.Module):
    """
    Late fusion channel attention module for decoded features.

    This module learns to combine multiple channel feature maps (after U-Net
    decoding) into virtual channels. Unlike the early fusion variant, this
    operates on high-level learned features rather than raw signals.

    Architecture:
        1. Encode each channel's feature map with a CNN
        2. Global average pool to get per-channel feature vectors
        3. For each virtual channel, an MLP produces attention scores
        4. Softmax normalizes scores across input channels
        5. Weighted sum of feature maps produces virtual channel features

    Args:
        n_virtual_channels (int): Number of virtual channels to produce.
            Each has its own MLP to learn different channel weightings.
        in_filters (int): Number of input feature channels from the decoder.
        encoder_filters (int): Number of filters in the attention CNN encoder.
        mlp_hidden_size (int): Hidden layer size in the attention MLPs.

    Note:
        The key difference from early fusion ChannelAttention is that this
        operates on multi-channel feature maps (in_filters) rather than
        single-channel raw signals.
    """

    def __init__(
        self, n_virtual_channels, in_filters, encoder_filters, mlp_hidden_size
    ):
        super(ChannelAttention, self).__init__()

        _cfg = Config.get()
        sampling_rate = _cfg.data.sampling_rate

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_filters, encoder_filters, sampling_rate // 2, sampling_rate // 4
            ),
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
        Compute attention-weighted virtual channels from decoded features.

        Args:
            x (torch.Tensor): Decoded feature maps of shape
                (batch, channels, filters, time).

        Returns:
            torch.Tensor: Virtual channel features of shape
                (batch, n_virtual_channels, filters, time).
                Each virtual channel is a learned weighted sum of input channels.
        """
        # x shape (batch, channels, filters, time)
        batch, channels, _, _ = x.shape

        x_att = x.reshape(batch * channels, x.shape[-2], x.shape[-1])
        # x_att shape (batch * channels, filters, time)

        x_att = self.encoder(x_att)
        # x_att shape (batch * channels, filters, time)

        x_att = x_att.mean(dim=-1)
        # x_att shape (batch * channels, filters)

        results = []
        for mlp in self.mlps:
            alpha = mlp(x_att).reshape(batch, -1)
            # alpha shape (batch, channels)
            alpha = self.softmax(alpha).unsqueeze(-1).unsqueeze(-1)
            # alpha shape (batch, channels, 1, 1)

            x_out = x * alpha
            x_out = x_out.sum(1)
            # x_out shape (batch, filters, time)

            results.append(x_out.unsqueeze(1))

        return torch.cat(results, dim=1)
