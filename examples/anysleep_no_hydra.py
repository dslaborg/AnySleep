"""
Standalone AnySleep model implementation without Hydra dependencies.

This file provides a self-contained implementation of the AnySleep
model that can be used independently of the Hydra configuration system. It is
intended for:
- Quick experimentation without setting up the full training pipeline
- Integration into external projects
- Understanding the model architecture in a single file

The model accepts variable numbers of input channels and uses learned attention
on skip connections to combine channel information.

See Also:
    - base/model/anysleep.py: Full implementation with Hydra config support
    - predict_edf_file_logits_plain.py: Example usage script
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def calc_filters(
    num_channels: int, depth: int, complexity: float, scale: float
) -> (np.ndarray[int], np.ndarray[int]):
    filter_sizes = np.zeros(depth + 2, dtype=int)
    filter_sizes[:2] = [num_channels, 5]
    for i in range(2, depth + 2):
        filter_sizes[i] = int(filter_sizes[i - 1] * scale)

    for i in range(1, len(filter_sizes)):
        filter_sizes[i] = int(filter_sizes[i] * complexity)

    return filter_sizes[:-1], filter_sizes[:0:-1]


class AnySleep(nn.Module):
    """
    Standalone AnySleep model for variable-channel sleep staging.

    This is a self-contained implementation without Hydra dependencies.
    The model uses attention on skip connections to handle any number of
    input channels.

    Args:
        path (str, optional): Path to pre-trained weights (.pth file).
            If provided, weights are loaded after initialization.
        depth (int): Number of encoder/decoder blocks. Default: 12.
        complexity (float): Base filter multiplier. Default: 1.2923.
        scale (float): Filter scaling factor between blocks. Default: 1.4142.
        hidden_size (int): Hidden layer size for attention MLPs. Default: 40.
        sleep_stage_frequency (int): Predictions per 30s epoch. Default: 1.
            Set higher for finer temporal resolution (e.g., 128 for per-sample).

    Note:
        The model expects input sampled at 128 Hz with 30-second epochs
        (3840 samples per epoch).
    """

    def __init__(
        self,
        path=None,
        depth: int = 12,
        complexity: float = 1.2923,
        scale: float = 1.4142,
        hidden_size: int = 40,
        sleep_stage_frequency: int = 1,  # measured as sleep stages per epoch
    ):
        super(AnySleep, self).__init__()

        sampling_rate = 128
        epoch_length = sampling_rate * 30

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

        pool_size = int(epoch_length / sleep_stage_frequency)
        self.segment_classifier = nn.Sequential(
            nn.Conv1d(cs_dec[-1], cs_dec[-1], 1, padding="same"),
            nn.Tanh(),
            nn.AvgPool1d(pool_size, stride=pool_size),
            nn.Conv1d(cs_dec[-1], 5, 1, padding="same"),
            nn.ELU(inplace=True),
            nn.Conv1d(5, 5, 1, padding="same"),
        )

        self.skip_connections = nn.ModuleList(
            [SkipConnectionBlock(cs_enc[i], hidden_size) for i in range(1, len(cs_enc))]
            + [SkipConnectionBlock(cs_dec[0], hidden_size)]
        )

        # always load model at the end of the initialization
        self.load(path)

    def forward(self, x):
        """
        Forward pass through the AnySleep model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, channels).
                Time should be divisible by (3840 / sleep_stage_frequency).

        Returns:
            torch.Tensor: Logits of shape (batch, epochs, 5).
                5 classes: Wake=0, N1=1, N2=2, N3=3, REM=4.
        """
        # x shape: (batch, time, channels)
        batch, _, channels = x.shape

        x = x.transpose(1, 2).reshape(batch * channels, 1, -1)
        # x shape: (batch * channels, n_filters, time)

        # encoder
        x_res = []
        for i in range(len(self.encoders)):
            x_r, x = self.encoders[i](x)
            # x_r/x shape: (batch * channels, n_filters, time)
            x_r = x_r.reshape(batch, channels, *x_r.shape[1:])
            # x_r shape: (batch, channels, n_filters, time)
            x_res.append(self.skip_connections[i](x_r))
            # x_res shape: (batch, n_filters, time)

        # attention for connector
        x = self.connector(x)
        x = x.reshape(batch, channels, *x.shape[1:])
        # x shape: (batch, channels, n_filters, time)

        x = self.skip_connections[-1](x)
        # x shape: (batch, n_filters, time)

        # decoder
        for i in range(len(self.decoders)):
            x = self.decoders[i](x, x_res.pop())

        x = self.segment_classifier(x)

        return x.transpose(1, 2)

    def load(self, path):
        if path is None:
            return

        model_state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(model_state["state_dict"])


class USleepEncoderBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, kernel_size=9):
        super(USleepEncoderBlock, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding="same"),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(n_filters_out),
        )
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x_res = self.enc(x)
        if x_res.shape[-1] % 2 != 0:
            x_res = F.pad(x_res, (1, 0))
        x = self.pool(x_res)

        return x_res, x


class USleepDecoderBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, kernel_size=9):
        super(USleepDecoderBlock, self).__init__()
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # kernel size see https://github.com/perslev/U-Time/blob/0217665224eda37467c40610879c751b2fe36970/utime/models/usleep.py#L266
            nn.Conv1d(n_filters_in, n_filters_out, 2, padding="same"),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(n_filters_out),
        )

        self.dec2 = nn.Sequential(
            nn.Conv1d(n_filters_out * 2, n_filters_out, kernel_size, padding="same"),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(n_filters_out),
        )

    def forward(self, x, x_res):
        x = self.dec1(x)
        x = self.crop(x, x_res)  # crop
        x = torch.cat((x_res, x), 1)
        x = self.dec2(x)
        return x

    def crop(self, x, x_res):
        diff = max(0, x.shape[-1] - x_res.shape[-1])
        start = diff // 2 + diff % 2
        return x[:, :, start : start + x_res.shape[-1]]


class SkipConnectionBlock(nn.Module):
    def __init__(
        self,
        n_filters: int,
        hidden_size: int,
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

    def forward(self, x: torch.Tensor):
        # x shape (batch, channels, filters, time)

        x_att = x.mean(dim=-1).reshape(-1, x.shape[2])
        # x_att shape (batch * channels, filters)

        alpha = self.mlp(x_att).reshape(x.shape[0], -1)
        # alpha shape (batch, channels)
        alpha = self.softmax(alpha).unsqueeze(-1).unsqueeze(-1)
        # alpha shape (batch, channels, 1, 1)

        x_out = x * alpha
        x_out = x_out.sum(1)
        # x_out shape (batch, filters, time)

        return x_out
