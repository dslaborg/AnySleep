"""
AnySleep: Attention-based U-Sleep for variable channel sleep staging.

This module implements AnySleep, an extension of U-Sleep that uses attention
mechanisms to combine information from multiple input channels. Unlike the
standard U-Sleep which uses exactly 2 channels, AnySleep can process any
number of EEG and EOG channels and learn to weight them appropriately.

Key innovation:
- Processes each channel independently through the encoder
- Uses learned attention weights to combine channel features at skip connections
- Allows the model to adaptively focus on the most informative channels
"""

import os.path
from logging import getLogger

import numpy as np
import torch
from torch import nn

from base.model.base_model import BaseModel, _weights_init
from base.model.usleep_base_components import (
    USleepDecoderBlock,
    USleepEncoderBlock,
    SegmentClassifier,
)
from base.model.utilities import calc_filters

logger = getLogger(__name__)


class AnySleep(BaseModel):
    """
    Attention-based U-Sleep for multi-channel sleep staging.

    AnySleep extends U-Sleep by processing each input channel independently
    and using learned attention weights to combine their features. This allows
    the model to handle variable numbers of channels and learn which channels
    are most informative.

    Architecture:
    - Encoder: Shared across channels, processes each channel independently
    - Skip Connections: Attention-weighted combination of channel features
    - Decoder: Standard U-Sleep decoder with attention-weighted skip connections
    - Classifier: Standard segment classifier

    Args:
        path (str, optional): Path to pretrained weights.
        seed (int, optional): Random seed for reproducibility.
        depth (int): Number of encoder/decoder blocks. Default: 12.
        complexity (float): Filter count multiplier. Default: 1.2923.
        scale (float): Filter growth rate between layers. Default: 1.4142.
        hidden_size (int): Hidden layer size in attention MLP. Default: 40.
        sleep_stage_frequency (int): Predictions per epoch. Default: 1.
        save_att_weights (bool): If True, save attention weights for analysis.
        **kwargs: Additional arguments (ignored).

    Input shape:
        (batch_size, time_samples, num_channels)
        where num_channels can vary between batches

    Output shape:
        (batch_size, num_epochs, 5)
    """

    identifier = "anysleep"

    def __init__(
        self,
        path=None,
        seed: int = None,
        depth: int = 12,
        complexity: float = 1.2923,
        scale: float = 1.4142,
        hidden_size: int = 40,
        sleep_stage_frequency: int = 1,
        save_att_weights=False,
        **kwargs,
    ):
        super(AnySleep, self).__init__(seed)

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

        self.segment_classifier = SegmentClassifier(sleep_stage_frequency, cs_dec[-1])

        self.skip_connections = nn.ModuleList(
            [
                SkipConnectionBlock(cs_enc[i], hidden_size, save_att_weights, i)
                for i in range(1, len(cs_enc))
            ]
            + [
                SkipConnectionBlock(
                    cs_dec[0], hidden_size, save_att_weights, len(cs_enc)
                )
            ]
        )

        self.apply(_weights_init)

        # always load model at the end of the initialization
        self.load(path)

    def forward(self, x):
        """
        Forward pass through AnySleep.

        Process each channel independently through the encoder, apply attention
        at skip connections to combine channel information, then decode.

        Args:
            x (torch.Tensor): Input of shape (batch, time, channels).

        Returns:
            torch.Tensor: Predictions of shape (batch, epochs, 5).
        """
        # x shape: (batch, time, channels)
        batch, _, channels = x.shape

        x = x.transpose(1, 2).reshape(batch * channels, 1, -1)
        # x shape: (batch * channels, n_filters, time)

        # encoder: process each channel independently
        x_res = []
        for i in range(len(self.encoders)):
            x_r, x = self.encoders[i](x)
            # x_r/x shape: (batch * channels, n_filters, time)
            x_r = x_r.reshape(batch, channels, *x_r.shape[1:])
            # x_r shape: (batch, channels, n_filters, time)
            x_res.append(self.skip_connections[i](x_r))
            # x_res shape: (batch, n_filters, time) - attention-weighted

        # attention for connector
        x = self.connector(x)
        x = x.reshape(batch, channels, *x.shape[1:])
        # x shape: (batch, channels, n_filters, time)

        x = self.skip_connections[-1](x)
        # x shape: (batch, n_filters, time)

        # decoder: standard U-Sleep decoding with attention-weighted skips
        for i in range(len(self.decoders)):
            x = self.decoders[i](x, x_res.pop())

        x = self.segment_classifier(x)

        return x.transpose(1, 2)


class SkipConnectionBlock(nn.Module):
    """
    Attention mechanism for combining multi-channel features.

    This block learns attention weights for each input channel and produces
    a weighted combination of features. The attention weights sum to 1
    across channels (via softmax) and indicate channel importance.

    Architecture:
    1. Global average pooling over time
    2. MLP: Linear → BatchNorm → ReLU → Linear
    3. Softmax over channels
    4. Weighted sum of channel features

    Args:
        n_filters (int): Number of feature channels.
        hidden_size (int): Hidden layer size in the attention MLP.
        save_att_weights (bool): If True, save attention weights to file.
        attention_layer_idx (int): Layer index for saving (used in filename).

    Example:
        >>> block = SkipConnectionBlock(n_filters=64, hidden_size=40)
        >>> x = torch.randn(8, 4, 64, 1000)  # batch, channels, filters, time
        >>> y = block(x)  # Shape: (8, 64, 1000)
    """

    def __init__(
        self,
        n_filters: int,
        hidden_size: int,
        save_att_weights=False,
        attention_layer_idx=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(n_filters, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.softmax = nn.Softmax(dim=1)
        self.save_att_weights = save_att_weights
        self.attention_layer_idx = attention_layer_idx

    def forward(self, x: torch.Tensor):
        """
        Compute attention-weighted combination of channel features.

        Args:
            x (torch.Tensor): Input of shape (batch, channels, filters, time).

        Returns:
            torch.Tensor: Weighted output of shape (batch, filters, time).
        """
        # x shape (batch, channels, filters, time)

        # Global average pooling over time
        x_att = x.mean(dim=-1).reshape(-1, x.shape[2])
        # x_att shape (batch * channels, filters)

        # Compute attention weights
        alpha = self.mlp(x_att).reshape(x.shape[0], -1)
        # alpha shape (batch, channels)
        alpha = self.softmax(alpha).unsqueeze(-1).unsqueeze(-1)
        # alpha shape (batch, channels, 1, 1)

        # Optionally save attention weights for analysis
        if self.save_att_weights:
            save_file_name = f"attention_weights.npz"
            save_key = f"layer_{self.attention_layer_idx}"
            if not os.path.exists(save_file_name):
                np.savez(save_file_name)
            old_content = dict(np.load(save_file_name))
            if save_key not in old_content:
                old_content[save_key] = np.zeros((0, 32))
            filled_alpha = np.zeros((alpha.shape[0], 32)) - 1
            filled_alpha[:, : alpha.shape[1]] = (
                alpha.detach().cpu().numpy().squeeze(-1).squeeze(-1)
            )
            old_content[save_key] = np.append(
                old_content[save_key], filled_alpha, axis=0
            )
            np.savez(save_file_name, **old_content)

        # Apply attention weights and sum across channels
        x_out = x * alpha
        x_out = x_out.sum(1)
        # x_out shape (batch, filters, time)

        return x_out
