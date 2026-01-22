import json
import os
from os.path import *

import numpy as np
from hydra.utils import to_absolute_path

from base.config import Config
from base.results.json_utils import NumpyEncoder, remove_empty_values
from base.utils import DelayedExecutor


class BaseResultTracker:
    """
    Abstract base class for tracking and persisting evaluation metrics.

    Result trackers accumulate metrics during training/evaluation and
    periodically write them to JSON files. They use delayed writing to
    batch frequent updates and reduce I/O overhead.

    The metrics are stored in a hierarchical dictionary structure that
    can be accessed using "/" separated paths (e.g., "train/loss").

    Args:
        filename (str): Name of the JSON file for storing results.

    Attributes:
        results (dict): Hierarchical dictionary of tracked metrics.
        file (str): Full path to the results file.
        stages (dict): Sleep stage configuration from config.
        write_scheduler (DelayedExecutor): Manages delayed file writes.

    Example:
        >>> tracker = MyResultTracker("results.json")
        >>> tracker.add_to_results("train/loss", 0.5)
        >>> tracker.add_to_results("train/loss", 0.4)
        >>> tracker.schedule_write()  # Will write after delay
    """

    def __init__(self, filename: str):
        _cfg = Config.get()

        if _cfg.general.results_dir is None:
            results_dir = os.getcwd()
        else:
            results_dir = to_absolute_path(_cfg.general.results_dir)
            if not exists(results_dir):
                os.makedirs(results_dir)

        self.results = {}
        self.file = join(results_dir, filename)
        self.stages = _cfg.data.stages

        self.write_scheduler = DelayedExecutor(delay=1)

    def track_result(
        self,
        log_name,
        loss: float,
        predictions: np.ndarray,
        labels: np.ndarray,
        s_ids: np.ndarray,
    ):
        """
        Track metrics from an evaluation run.

        Args:
            log_name (str): Name for logging (e.g., 'train', 'val/test').
            loss (float): Loss value from this evaluation.
            predictions (np.ndarray): Model predictions.
            labels (np.ndarray): Ground truth labels.
            s_ids (np.ndarray): Sample identifiers [dataset, subject, channels].

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def schedule_write(self):
        """Schedule a delayed write of results to file."""
        self.write_scheduler.execute(self.write)

    def write(self):
        """Write results to JSON file immediately."""
        with open(self.file, "w") as f:
            f.write(
                json.dumps(
                    remove_empty_values(self.results),
                    indent=None,
                    sort_keys=True,
                    cls=NumpyEncoder,
                )
                + "\n"
            )

    def add_to_results(self, metric_name, metric_value):
        """
        Add a metric value to the results dictionary.

        Creates nested dictionaries as needed for hierarchical paths.
        Values are appended to lists to track history over epochs.

        Args:
            metric_name (str): "/" separated path (e.g., "train/loss").
            metric_value: Value to append to the metric's list.
        """
        metric_key_parts = metric_name.split("/")
        results_pointer = self.results
        for i in range(len(metric_key_parts) - 1):
            if metric_key_parts[i] not in results_pointer:
                results_pointer[metric_key_parts[i]] = {}
            results_pointer = results_pointer[metric_key_parts[i]]

        if metric_key_parts[-1] not in results_pointer:
            results_pointer[metric_key_parts[-1]] = []
        results_pointer[metric_key_parts[-1]].append(metric_value)

    def read_from_results(self, metric_name):
        """
        Read a metric value from the results dictionary.

        Args:
            metric_name (str): "/" separated path (e.g., "train/loss").

        Returns:
            Value at the path, or None if not found.
        """
        metric_key_parts = metric_name.split("/")
        results_pointer = self.results
        for i in range(len(metric_key_parts)):
            if metric_key_parts[i] not in results_pointer:
                return None
            results_pointer = results_pointer[metric_key_parts[i]]

        return results_pointer

    def get_latest_es_score(self):
        """
        Get the latest score for early stopping.

        Returns:
            float: Score to use for early stopping decisions.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
