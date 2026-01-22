import numpy as np


def calc_filters(
    num_channels: int, depth: int, complexity: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate encoder and decoder filter sizes for U-Sleep architecture.

    Computes filter counts that grow geometrically through the encoder
    and shrink symmetrically through the decoder.

    Based on the original U-Sleep implementation:F
    https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py

    Args:
        num_channels (int): Number of input channels (e.g., 2 for EEG+EOG).
        depth (int): Number of encoder/decoder blocks.
        complexity (float): Multiplier for all filter counts.
        scale (float): Growth factor between consecutive layers.

    Returns:
        tuple: (encoder_filters, decoder_filters)
            - encoder_filters: Array of filter counts for encoder layers
            - decoder_filters: Array of filter counts for decoder layers (reversed)

    Example:
        >>> enc, dec = calc_filters(num_channels=2, depth=12, complexity=1.29, scale=1.41)
        >>> print(enc[:4])  # First 4 encoder filter sizes
        [  2   6   9  13]
        >>> print(dec[:4])  # First 4 decoder filter sizes (reversed from encoder)
        [381 270 191 135]
    """
    filter_sizes = np.zeros(depth + 2, dtype=int)
    filter_sizes[:2] = [num_channels, 5]
    for i in range(2, depth + 2):
        filter_sizes[i] = int(filter_sizes[i - 1] * scale)

    for i in range(1, len(filter_sizes)):
        filter_sizes[i] = int(filter_sizes[i] * complexity)

    return filter_sizes[:-1], filter_sizes[:0:-1]
