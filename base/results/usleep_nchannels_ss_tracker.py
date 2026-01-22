from concurrent.futures.thread import ThreadPoolExecutor
from logging import getLogger

import numpy as np

from base.results.sleep_stage_tracker import SleepStageResultTracker, add_f1_to_results
from base.utils import calc_mode, choice

logger = getLogger(__name__)


class USleepNChannelsSSResultTracker(SleepStageResultTracker):
    """
    Tracker for analyzing majority voting performance across channel counts.

    This tracker evaluates how model performance changes when using different
    numbers of EEG channels for majority voting. For each specified channel
    count, it randomly samples that many channels (with a fixed EOG) and
    computes the majority-voted prediction.

    Useful for answering questions like:
    - How does accuracy improve from 1 to N EEG channels?
    - What's the minimum number of channels needed to match full-channel performance?
    - Is there diminishing returns beyond a certain channel count?

    Args:
        filename (str): Output JSON filename.
        n_channels_list (list[int]): List of channel counts to evaluate
            (e.g., [1, 2, 4, 8] to test 1, 2, 4, and 8 EEG channels).
        n_samples (int): Number of subject samples for Monte Carlo estimation.
            Subjects are sampled with replacement.
        track_recordings (bool): Whether to track per-recording metrics.
        seed (int, optional): Random seed for reproducibility.

    Example:
        >>> tracker = USleepNChannelsSSResultTracker(
        ...     "channel_analysis.json",
        ...     n_channels_list=[1, 2, 4, 8, 16],
        ...     n_samples=1000,
        ...     track_recordings=False,
        ...     seed=42
        ... )

    Note:
        - Subjects with fewer EEG channels than requested are skipped
        - One EOG channel is randomly selected and kept fixed
        - EEG channels are randomly sampled without replacement
    """

    def __init__(
        self, filename, n_channels_list, n_samples, track_recordings=False, seed=None
    ):
        super().__init__(filename, True, False, False, track_recordings, True)
        self.n_channels_list = n_channels_list
        self.n_samples = n_samples
        self.track_recordings = track_recordings
        self.random_state = np.random.RandomState(seed)

    def track_result(
        self,
        log_name,
        loss,
        predictions: np.ndarray,
        labels: np.ndarray,
        s_ids: np.ndarray,
    ):
        """
        Track metrics for each channel count configuration.

        For each channel count in n_channels_list, samples random channel
        subsets and computes majority-voted metrics.

        Args:
            log_name (str): Name for logging.
            loss (float): Loss value (unused, kept for API compatibility).
            predictions (np.ndarray): Model predictions per channel combination.
            labels (np.ndarray): Ground truth labels.
            s_ids (np.ndarray): Sample identifiers [dataset, subject, eeg, eog].
        """
        n_stages = len(self.stages)
        datasplit_sup_majority = {
            n: np.zeros((n_stages, 3)) for n in self.n_channels_list
        }  # tp, fp, fn
        channel_combs = np.array(["#".join(ch_comb) for ch_comb in s_ids[:, 2:]])

        # Create a lock for thread-safe operations
        futures = []
        results = {}

        # sample dataset, subject pairs
        ds_subj_pairs = []
        for dataset_id in np.unique(s_ids[:, 0]):
            for recording_id in np.unique(s_ids[s_ids[:, 0] == dataset_id, 1]):
                ds_subj_pairs.append((dataset_id, recording_id))
        ds_subj_pairs = choice(
            ds_subj_pairs, self.n_samples, random_state=self.random_state, replace=True
        )

        # Process each recording in a separate thread
        with ThreadPoolExecutor() as executor:
            for dataset_id, recording_id in ds_subj_pairs:
                future = executor.submit(
                    self._process_dataset_rec,
                    dataset_id,
                    recording_id,
                    s_ids,
                    predictions,
                    labels,
                    channel_combs,
                    n_stages,
                )
                futures.append(future)

            # Collect results
            for i, future in enumerate(futures):
                recording_sup, recording_id = future.result()
                for n_channels in self.n_channels_list:
                    datasplit_sup_majority[n_channels] += recording_sup[
                        f"majority_{n_channels}"
                    ]
                    if self.track_recordings:
                        add_f1_to_results(
                            results,
                            f"recordings/majority_{n_channels}/{i}_{recording_id}",
                            self._calc_metrics_all_stages(
                                *recording_sup[f"majority_{n_channels}"].T
                            ),
                        )

        for n_channels in self.n_channels_list:
            add_f1_to_results(
                results,
                f"datasplit/majority_{n_channels}",
                self._calc_metrics_all_stages(*datasplit_sup_majority[n_channels].T),
            )

        for res_k, res_v in results.items():
            self.add_to_results(res_k, res_v)

        self.schedule_write()

        logger.info(f"{log_name}: {self.create_log_msg()}")

    def create_log_msg(self):
        """Generate log message showing macro F1 for each channel count."""
        log_msg = ""
        for n_channels in self.n_channels_list:
            dsp_f1s = self.read_from_results(f"datasplit/majority_{n_channels}")
            dsp_mf1 = np.mean([v["f1"][-1] for v in dsp_f1s.values()])
            log_msg += (
                f"Datasplit voted over {n_channels} Channels MF1: {dsp_mf1:.3f}\n"
            )
        return log_msg

    def get_latest_es_score(self):
        """Not implemented - this tracker is for analysis, not early stopping."""
        raise NotImplementedError

    def _process_dataset_rec(
        self,
        dataset_id,
        recording_id,
        s_ids,
        predictions,
        labels,
        channel_combs,
        n_stages,
    ):
        """
        Process a single recording for all channel count configurations.

        For each channel count, randomly samples EEG channels (with a fixed
        random EOG) and computes majority-voted metrics.

        Args:
            dataset_id: Dataset identifier.
            recording_id: Recording/subject identifier.
            s_ids: Full sample identifiers array.
            predictions: All predictions array.
            labels: All labels array.
            channel_combs: Pre-computed channel combination strings.
            n_stages: Number of sleep stages.

        Returns:
            tuple: (recording_sup dict, recording_id)
                recording_sup maps "majority_{n}" to (n_stages, 3) support array
        """
        recording_sup = {
            f"majority_{n_channels}": np.zeros((n_stages, 3))
            for n_channels in self.n_channels_list
        }
        recording_mask = (s_ids[:, 0] == dataset_id) & (s_ids[:, 1] == recording_id)

        channels, n_epochs = np.unique(
            channel_combs[recording_mask], return_counts=True
        )
        per_channel_predictions = np.empty((len(channels), max(n_epochs)), dtype=int)
        per_channel_labels = np.empty((per_channel_predictions.shape[1]), dtype=int)
        eeg_channels = set([c.split("#")[0] for c in channels])
        eog_channels = set([c.split("#")[1] for c in channels])

        for i, channel_id in enumerate(channels):
            channel_mask = recording_mask & (channel_combs == channel_id)
            per_channel_predictions[i, : np.sum(channel_mask)] = predictions[
                channel_mask
            ]
            per_channel_labels = labels[channel_mask]

        for n_channels in self.n_channels_list:
            if n_channels > len(eeg_channels):
                # skipping subjects with less than n_channels eeg channels should result in no tps, fps, fns
                # which means that this subject has no impact on the datasplit scores
                logger.warning(
                    f"skipping subject {dataset_id} {recording_id} with less than {n_channels} channels,"
                    f" available channels: {eeg_channels}, {eog_channels}"
                )
                continue

            # Sample one EOG channel and keep it fixed
            sampled_eog = self.random_state.choice(list(eog_channels), 1)
            eog_mask = [c.split("#")[1] == sampled_eog[0] for c in channels]
            per_channel_predictions_w_eog = per_channel_predictions[eog_mask, :]

            # Sample n_channels EEG channels without replacement
            pcp_n_channels = per_channel_predictions_w_eog[
                self.random_state.choice(sum(eog_mask), n_channels, replace=False), :
            ]
            majority_voted = calc_mode(pcp_n_channels, axis=0)
            tps, fps, fns = self._get_supports(per_channel_labels, majority_voted)
            recording_sup[f"majority_{n_channels}"] = np.array([tps, fps, fns]).T

        return recording_sup, recording_id
