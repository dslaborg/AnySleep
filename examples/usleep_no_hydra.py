"""
U-Sleep neural network architecture for sleep stage classification.

U-Sleep is a fully convolutional neural network based on the U-Net architecture,
adapted for 1D time-series data. It processes multi-channel EEG/EOG signals and
outputs sleep stage predictions for each 30-second epoch.

Architecture:
    Input → Encoder → Connector → Decoder → Segment Classifier → Output

The encoder progressively downsamples while increasing feature channels.
Skip connections pass information from encoder to decoder levels.
The decoder upsamples back to the original temporal resolution.
The segment classifier pools to epoch resolution and predicts sleep stages.

Reference:
    Perslev, M., et al. (2021). U-Sleep: resilient high-frequency sleep staging.
    npj Digital Medicine, 4(1), 72.
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


class USleep(nn.Module):
    """
    U-Sleep architecture for sleep stage classification.

    A U-Net style encoder-decoder network for processing polysomnography
    signals. Processes 2-channel input (EEG + EOG) and outputs per-epoch
    sleep stage predictions.

    The network consists of:
    - Encoder: Stack of conv-ELU-BN-pool blocks, increasing filters
    - Connector: Bottleneck convolution
    - Decoder: Stack of upsample-conv-concat blocks with skip connections
    - Classifier: Pools to epoch resolution and predicts 5 classes

    Args:
        path (str, optional): Path to pretrained weights. None for random init.
        seed (int, optional): Random seed for reproducibility.
        depth (int): Number of encoder/decoder blocks. Default: 12.
        complexity (float): Filter count multiplier. Default: 1.2923.
        scale (float): Filter growth rate between layers. Default: 1.4142 (sqrt(2)).
        sleep_stage_frequency (int): Predictions per epoch. Default: 1.
        **kwargs: Additional arguments (ignored, for config compatibility).

    Input shape:
        (batch_size, time_samples, 2)
        where time_samples = num_epochs * sampling_rate * epoch_duration
    """

    def __init__(
        self,
        path=None,
        depth: int = 12,
        complexity: float = 1.2923,
        scale: float = 1.4142,
        sleep_stage_frequency: int = 1,
        **kwargs,
    ):
        super(USleep, self).__init__()

        num_channels = 2

        cs_enc, cs_dec = calc_filters(num_channels, depth, complexity, scale)

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

        # always load model at the end of the initialization
        self.load(path)

    def forward(self, x):
        """
        Forward pass through the U-Sleep network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, channels).

        Returns:
            torch.Tensor: Predictions of shape (batch, epochs, 5).
        """
        # x shape: (batch, time, channels)

        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        # x shape: (batch, channels, time)

        # encoder: progressively downsample, save residuals for skip connections
        x_res = []
        for i in range(len(self.encoders)):
            x_r, x = self.encoders[i](x)
            x_res.append(x_r)

        x = self.connector(x)

        # decoder: upsample and concatenate with skip connections
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
        x = self.crop(x, x_res)
        x = torch.cat((x_res, x), 1)
        x = self.dec2(x)
        return x

    def crop(self, x, x_res):
        diff = max(0, x.shape[-1] - x_res.shape[-1])
        start = diff // 2 + diff % 2
        return x[:, :, start : start + x_res.shape[-1]]


class SegmentClassifier(nn.Module):
    """
    Classification head that pools to epoch resolution and predicts sleep stages.

    Performs:
    1. Conv1d → Tanh (feature transformation)
    2. AvgPool to pool from sample to epoch resolution
    3. Conv1d → ELU → Conv1d (classification layers)

    The pooling size is calculated as:
        pool_size = sampling_rate * epoch_duration / sleep_stage_frequency

    For 128 Hz, 30s epochs, and 1 prediction per epoch: pool_size = 3840

    Args:
        sleep_stage_frequency (int): Number of predictions per epoch.
        n_filters (int): Number of input feature channels.
    """

    def __init__(self, sleep_stage_frequency, n_filters):
        super(SegmentClassifier, self).__init__()

        sampling_rate = 128
        epoch_length = sampling_rate * 30

        pool_size = int(epoch_length / sleep_stage_frequency)
        self.segment_classifier = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, 1, padding="same"),
            nn.Tanh(),
            nn.AvgPool1d(pool_size, stride=pool_size),
            nn.Conv1d(n_filters, 5, 1, padding="same"),
            nn.ELU(inplace=True),
            nn.Conv1d(5, 5, 1, padding="same"),
            # nn.Softmax(dim=1) # Not needed as CrossEntropyLoss already applies softmax
        )

    def forward(self, x):
        """
        Forward pass through the segment classifier.

        Args:
            x (torch.Tensor): Input of shape (batch, filters, time_samples).

        Returns:
            torch.Tensor: Logits of shape (batch, 5, num_epochs).
        """
        return self.segment_classifier(x)
