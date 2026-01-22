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

from logging import getLogger

from torch import nn

from base.config import Config
from base.model.base_model import BaseModel, _weights_init
from base.model.usleep_base_components import (
    USleepDecoderBlock,
    USleepEncoderBlock,
    SegmentClassifier,
)
from base.model.utilities import calc_filters

logger = getLogger(__name__)


class USleep(BaseModel):
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

    Output shape:
        (batch_size, num_epochs, 5)
        where 5 is the number of sleep stages

    Example:
        >>> model = USleep(seed=42, depth=12)
        >>> x = torch.randn(8, 35 * 30 * 128, 2)  # 35 epochs, 128 Hz
        >>> y = model(x)  # Shape: (8, 35, 5)
    """

    identifier = "usleep"

    def __init__(
        self,
        path=None,
        seed: int = None,
        depth: int = 12,
        complexity: float = 1.2923,
        scale: float = 1.4142,
        sleep_stage_frequency: int = 1,
        **kwargs,
    ):
        super(USleep, self).__init__(seed)

        _cfg = Config.get()
        num_channels = _cfg.data.channel_num

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

        self.apply(_weights_init)

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

        _cfg = Config.get()
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
