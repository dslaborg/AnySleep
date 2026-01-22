import torch
import torch.nn.functional as F
from torch import nn

from base.config import Config


class USleepEncoderBlock(nn.Module):
    """
    Encoder block that extracts features while downsampling.

    Each block performs:
    1. Conv1d → ELU → BatchNorm (feature extraction)
    2. MaxPool with stride 2 (downsampling)

    Returns both the pre-pooling features (for skip connections) and
    the downsampled output (for the next encoder level).

    Args:
        n_filters_in (int): Number of input channels.
        n_filters_out (int): Number of output channels.
        kernel_size (int): Convolution kernel size. Default: 9.
    """

    def __init__(self, n_filters_in, n_filters_out, kernel_size=9):
        super(USleepEncoderBlock, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding="same"),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(n_filters_out),
        )
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        """
        Forward pass through the encoder block.

        Args:
            x (torch.Tensor): Input of shape (batch, channels, time).

        Returns:
            tuple: (x_res, x_downsampled)
                - x_res: Pre-pooling features for skip connection
                - x_downsampled: Pooled output for next encoder level
        """
        x_res = self.enc(x)
        if x_res.shape[-1] % 2 != 0:
            x_res = F.pad(x_res, (1, 0))
        x = self.pool(x_res)

        return x_res, x


class USleepDecoderBlock(nn.Module):
    """
    Decoder block that upsamples and combines with skip connections.

    Each block performs:
    1. Upsample by factor 2 → Conv1d → ELU → BatchNorm
    2. Crop to match skip connection size
    3. Concatenate with skip connection
    4. Conv1d → ELU → BatchNorm (feature fusion)

    Args:
        n_filters_in (int): Number of input channels.
        n_filters_out (int): Number of output channels.
        kernel_size (int): Convolution kernel size. Default: 9.
    """

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
        """
        Forward pass with skip connection.

        Args:
            x (torch.Tensor): Input from previous decoder/connector.
            x_res (torch.Tensor): Skip connection from encoder.

        Returns:
            torch.Tensor: Upsampled and fused features.
        """
        x = self.dec1(x)
        x = self.crop(x, x_res)
        x = torch.cat((x_res, x), 1)
        x = self.dec2(x)
        return x

    def crop(self, x, x_res):
        """Center-crop x to match x_res dimensions."""
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

        _cfg = Config.get()
        sampling_rate = _cfg.data.sampling_rate
        epoch_length = sampling_rate * _cfg.data.epoch_duration

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
