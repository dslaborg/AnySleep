from logging import getLogger

import numpy as np

from base.results.sleep_stage_tracker import SleepStageResultTracker

logger = getLogger(__name__)


class AnySleepNChannelsSSResultTracker(SleepStageResultTracker):
    """
    Result tracker that handles duplicate recording IDs in AnySleep evaluation.

    When evaluating AnySleep models with different channel configurations,
    the same recording may appear multiple times (once per channel subset).
    This tracker prefixes each occurrence with a unique index to prevent
    metrics from being incorrectly merged.

    For example, if subject "S001" is evaluated with 3 different channel
    combinations, the recording IDs become: "0_S001", "1_S001", "2_S001".

    Args:
        filename (str): Output JSON filename.
        track_datasplit (bool): Track datasplit-level metrics.
        track_datasets (bool): Track per-dataset metrics.
        track_channels (bool): Track per-channel metrics.
        track_recordings (bool): Track per-recording metrics.
        do_majority_voting (bool): Enable majority voting metrics.
    """

    def __init__(
        self,
        filename,
        track_datasplit,
        track_datasets,
        track_channels,
        track_recordings,
        do_majority_voting,
    ):
        super().__init__(
            filename,
            track_datasplit,
            track_datasets,
            track_channels,
            track_recordings,
            do_majority_voting,
        )

    def track_result(
        self,
        log_name,
        loss,
        predictions: np.ndarray,
        labels: np.ndarray,
        s_ids: np.ndarray,
    ):
        """
        Track metrics with unique prefixes for duplicate recordings.

        Detects when the same recording ID appears multiple times (indicating
        different channel evaluations) and prefixes each occurrence with a
        running index before delegating to the parent tracker.

        Args:
            log_name (str): Name for logging.
            loss (float): Loss value.
            predictions (np.ndarray): Model predictions.
            labels (np.ndarray): Ground truth labels.
            s_ids (np.ndarray): Sample identifiers [dataset, subject, channels...].
        """
        # add running index to recording ids to handle duplicates
        # first find switches between recordings
        rec_switches = np.where(s_ids[:-1, 1] != s_ids[1:, 1])[0] + 1
        rec_switches = np.insert(rec_switches, 0, 0)
        rec_switches = np.insert(rec_switches, len(rec_switches), len(s_ids))
        # then add running index to each recording
        rec_prefixed = []
        for i in range(len(rec_switches) - 1):
            rec_prefixed.extend(
                [
                    f"{i}_{rid}"
                    for rid in s_ids[rec_switches[i] : rec_switches[i + 1], 1]
                ]
            )
        s_ids[:, 1] = rec_prefixed

        super().track_result(log_name, loss, predictions, labels, s_ids)
