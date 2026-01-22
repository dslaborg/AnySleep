import threading
from logging import getLogger

import numpy as np

from base.config import Config

logger = getLogger(__name__)


class DelayedExecutor:
    """
    Executes a function after a specified delay, canceling pending calls.

    This class is useful for batching frequent operations like file writes.
    When called multiple times within the delay window, only the last call
    is executed. This prevents excessive I/O during training while ensuring
    results are eventually written.

    Attributes:
        delay (float): Time in seconds to wait before executing.
        timer (threading.Timer): The current timer object, if any.

    Example:
        >>> executor = DelayedExecutor(delay=1.0)
        >>> def save_results():
        ...     print("Saving...")
        >>> # These calls within 1 second will be batched
        >>> executor.execute(save_results)  # Canceled
        >>> executor.execute(save_results)  # Canceled
        >>> executor.execute(save_results)  # This one runs after 1 second
    """

    def __init__(self, delay=3):
        """
        Initialize the DelayedExecutor.

        Args:
            delay (float): Time in seconds to wait before executing the callable.
                Defaults to 3 seconds.
        """
        self.delay = delay
        self.timer = None

    def execute(self, func, *args, **kwargs):
        """
        Schedule a function to be executed after the delay.

        If called again before the delay has passed, the previous timer is
        canceled and a new one starts. This effectively batches multiple
        calls into a single execution.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        if self.timer is not None and self.timer.is_alive():
            self.timer.cancel()

        # Create a new timer
        self.timer = threading.Timer(self.delay, func, args=args, kwargs=kwargs)
        self.timer.start()


def calc_mode(data, axis=0):
    """
    Compute the mode (most frequent value) along an axis with random tie-breaking.

    This function is used for majority voting across channel predictions.
    When multiple values have the same count (a tie), one is chosen randomly.
    This random tie-breaking ensures unbiased consensus in case of disagreement.

    Args:
        data (np.ndarray): Input array of integer predictions.
            Values should be in range [0, n_stages-1] where n_stages is
            configured in cfg.data.stages.
        axis (int): Axis along which to compute the mode. Defaults to 0.

    Returns:
        np.ndarray: Array of mode values with the specified axis removed.

    Example:
        >>> # Three channels predicting sleep stages for 4 epochs
        >>> predictions = np.array([
        ...     [0, 2, 2, 3],  # Channel 1
        ...     [0, 2, 1, 3],  # Channel 2
        ...     [1, 2, 2, 3],  # Channel 3
        ... ])
        >>> consensus = calc_mode(predictions, axis=0)
        >>> # Result: [0, 2, 2, 3] - majority vote per epoch
    """
    _cfg = Config.get()
    n_stages = len(_cfg.data.stages)
    bins = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=n_stages), axis=axis, arr=data
    )
    # resolve ties randomly by adding a small random number to the counts
    rand_mat = np.random.random_sample(bins.shape)
    bins = bins + rand_mat
    return np.argmax(bins, axis=axis)


def choice(listable, n, random_state=None, replace=True):
    """
    Randomly select n items from a list-like object.

    This is a convenience wrapper around numpy's random choice that works
    with any list-like object (not just arrays) and supports reproducibility
    via a random state.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    idx = random_state.choice(len(listable), n, replace=replace)
    return [listable[i] for i in idx]
