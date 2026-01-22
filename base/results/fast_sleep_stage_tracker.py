from logging import getLogger

import numpy as np

from base.results.sleep_stage_tracker import (
    add_f1_to_results,
    SleepStageResultTracker,
)

logger = getLogger(__name__)


class FastSleepStageResultTracker(SleepStageResultTracker):
    """
    Lightweight result tracker for training metrics.

    A simplified version of SleepStageResultTracker that only computes
    datasplit-level concatenation metrics. This reduces computational
    overhead during training when detailed per-dataset metrics are not needed.

    Disabled features:
    - Per-dataset metrics
    - Per-recording metrics
    - Per-channel metrics
    - Majority voting

    Args:
        filename (str): Output JSON filename.
        **kwargs: Ignored (for compatibility).
    """

    def __init__(self, filename, **kwargs):
        super().__init__(filename, True, False, False, False, False)

    def track_result(
        self,
        log_name,
        loss,
        predictions: np.ndarray,
        labels: np.ndarray,
        s_ids: np.ndarray,
    ):
        """
        Only computes overall datasplit metrics by concatenating all predictions.
        """
        self.add_to_results("loss", loss)

        tps, fps, fns = self._get_supports(labels, predictions)
        res = self._calc_metrics_all_stages(tps, fps, fns)
        results = {}
        add_f1_to_results(results, "datasplit/concat", res)

        for res_k, res_v in results.items():
            self.add_to_results(res_k, res_v)

        self.schedule_write()

        logger.info(f"{log_name}: {self.create_log_msg()}")
