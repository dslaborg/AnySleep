from concurrent.futures.thread import ThreadPoolExecutor
from logging import getLogger

import numpy as np
from sklearn.metrics import confusion_matrix

from base.results.base_result_tracker import BaseResultTracker
from base.utils import calc_mode

logger = getLogger(__name__)


class SleepStageResultTracker(BaseResultTracker):
    """
    Comprehensive metrics tracker for sleep stage classification.

    Tracks precision, recall, and F1 score at multiple granularity levels:
    - Datasplit: Overall performance on train/val/test
    - Dataset: Per-dataset performance (e.g., MASS, Sleep-EDF)
    - Recording: Per-subject performance
    - Channel: Per-channel combination performance

    Supports two aggregation strategies:
    - Concatenation: Pool all predictions and compute metrics
    - Majority voting: Consensus across channels per epoch

    Args:
        filename (str): Output JSON filename.
        track_datasplit (bool): Track datasplit-level metrics.
        track_datasets (bool): Track per-dataset metrics.
        track_channels (bool): Track per-channel metrics.
        track_recordings (bool): Track per-recording metrics.
        do_majority_voting (bool): Enable majority voting metrics.
        log_channel_names (bool): Track channel names of evaluated recordings and save them to file.
    """
    def __init__(
        self,
        filename,
        track_datasplit,
        track_datasets,
        track_channels,
        track_recordings,
        do_majority_voting,
        log_channel_names=False,
    ):
        super().__init__(filename)
        self.track_datasplit = track_datasplit
        self.track_datasets = track_datasets
        self.track_channels = track_channels
        self.track_recordings = track_recordings
        self.do_majority_voting = do_majority_voting
        self.log_channel_names = log_channel_names

    def track_result(
        self,
        log_name,
        loss,
        predictions: np.ndarray,
        labels: np.ndarray,
        s_ids: np.ndarray,
    ):
        self.add_to_results("loss", loss)

        n_stages = len(self.stages)
        datasplit_sup_concat = np.zeros((n_stages, 3))  # tp, fp, fn
        datasplit_sup_majority = np.zeros((n_stages, 3))
        channel_combs = np.array(["_".join(ch_comb) for ch_comb in s_ids[:, 2:]])

        if self.log_channel_names:
            log_filename = "channel_names.npy"
            s_ids_wo_duplicates = np.unique(s_ids, axis=0)
            channel_combs_wo_duplicates = np.array(
                ["_".join(ch_comb) for ch_comb in s_ids_wo_duplicates[:, 2:]]
            )
            np.save(log_filename, channel_combs_wo_duplicates)

        # Create a lock for thread-safe operations
        futures = []
        results = {}

        # Process each dataset in a separate thread
        with ThreadPoolExecutor() as executor:
            for dataset_id in np.unique(s_ids[:, 0]):
                future = executor.submit(
                    self._process_dataset,
                    dataset_id,
                    s_ids,
                    predictions,
                    labels,
                    channel_combs,
                    n_stages,
                )
                futures.append(future)

            # Collect results
            for future in futures:
                dataset_results, dataset_sup = future.result()
                results.update(dataset_results)
                datasplit_sup_concat += dataset_sup["concat"]
                if self.do_majority_voting:
                    datasplit_sup_majority += dataset_sup["majority"]

        if self.track_datasplit:
            add_f1_to_results(
                results,
                "datasplit/concat",
                self._calc_metrics_all_stages(*datasplit_sup_concat.T),
            )
            if self.do_majority_voting:
                add_f1_to_results(
                    results,
                    "datasplit/majority",
                    self._calc_metrics_all_stages(*datasplit_sup_majority.T),
                )

        for res_k, res_v in results.items():
            self.add_to_results(res_k, res_v)

        self.schedule_write()

        logger.info(f"{log_name}: {self.create_log_msg()}")

    def create_log_msg(self):
        log_msg = ""
        grp_key = "majority" if self.do_majority_voting else "concat"
        if self.track_datasplit:
            dsp_f1s = self.read_from_results("datasplit/" + grp_key)
            dsp_mf1 = np.mean([v["f1"][-1] for v in dsp_f1s.values()])
            log_msg += f"Datasplit MF1: {dsp_mf1:.3f}\n"

        if self.track_datasets:
            ds_f1s = self.read_from_results("datasets")
            for k, v_ds in ds_f1s.items():
                ds_mf1 = np.mean([v["f1"][-1] for v in v_ds[grp_key].values()])
                log_msg += f"Dataset {k} MF1: {ds_mf1:.3f}\n"
        return log_msg

    def get_latest_es_score(self):
        if (
            (self.track_datasplit and self.read_from_results("datasplit") is None)
            or (self.track_datasets and self.read_from_results("datasets") is None)
            or (self.track_recordings and self.read_from_results("recordings") is None)
        ):
            return 0
        key = "majority" if self.do_majority_voting else "concat"
        if self.track_datasplit:
            # return latest macro f1 of datasplit majority
            dsp_f1s = self.read_from_results("datasplit/" + key)
            return np.mean([v["f1"][-1] for v in dsp_f1s.values()])
        elif self.track_datasets:
            # return mean of latest macro f1s of all datasets
            return np.mean(
                [
                    v_ss["f1"][-1]
                    for v_ds in self.read_from_results("datasets").values()
                    for v_ss in v_ds[key].values()
                ]
            )
        elif self.track_recordings:
            # return mean of latest macro f1s of all recordings
            return np.mean(
                [
                    v_ss["f1"][-1]
                    for v_ds in self.read_from_results("recordings").values()
                    for v_rec in v_ds.values()
                    for v_ss in v_rec[key].values()
                ]
            )
        else:
            raise ValueError("Early Stopping Result Tracker not tracking anything")

    def _process_dataset(
        self,
        dataset_id,
        s_ids,
        predictions,
        labels,
        channel_combs,
        n_stages,
    ):
        results = {}
        dataset_mask = s_ids[:, 0] == dataset_id
        channels = np.unique(channel_combs[dataset_mask])
        dataset_sup = {
            "concat": np.zeros((n_stages, 3)),
        }
        if self.track_channels:
            dataset_sup.update({c: np.zeros((n_stages, 3)) for c in channels})
        if self.do_majority_voting:
            dataset_sup["majority"] = np.zeros((n_stages, 3))

        recordings = np.unique(s_ids[dataset_mask, 1])
        for recording_id in recordings:
            recording_mask = dataset_mask & (s_ids[:, 1] == recording_id)

            channels, n_epochs = np.unique(
                channel_combs[recording_mask], return_counts=True
            )
            per_channel_predictions = np.empty(
                (len(channels), max(n_epochs)), dtype=int
            )
            per_channel_labels = np.empty((per_channel_predictions.shape[1]), dtype=int)

            for i, channel_id in enumerate(channels):
                channel_mask = recording_mask & (channel_combs == channel_id)
                per_channel_predictions[i, : np.sum(channel_mask)] = predictions[
                    channel_mask
                ]
                per_channel_labels = labels[channel_mask]

                tps, fps, fns = self._get_supports(
                    per_channel_labels, predictions[channel_mask]
                )
                dataset_sup["concat"] += np.array([tps, fps, fns]).T
                if self.track_channels:
                    dataset_sup[channel_id] += np.array([tps, fps, fns]).T
                if self.track_recordings:
                    add_f1_to_results(
                        results,
                        f"recordings/{dataset_id}/{recording_id}/{channel_id}",
                        self._calc_metrics_all_stages(tps, fps, fns),
                    )

            if self.do_majority_voting:
                majority_voted = calc_mode(per_channel_predictions, axis=0)
                tps, fps, fns = self._get_supports(per_channel_labels, majority_voted)
                dataset_sup["majority"] += np.array([tps, fps, fns]).T
                if self.track_recordings:
                    add_f1_to_results(
                        results,
                        f"recordings/{dataset_id}/{recording_id}/majority",
                        self._calc_metrics_all_stages(tps, fps, fns),
                    )

        if self.track_datasets:
            for k in dataset_sup.keys():
                add_f1_to_results(
                    results,
                    f"datasets/{dataset_id}/{k}",
                    self._calc_metrics_all_stages(*dataset_sup[k].T),
                )

        return results, dataset_sup

    def _get_supports(self, y_true: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_true, y_pred, labels=list(self.stages.keys()))
        tps = cm.diagonal()
        fps = cm.sum(axis=0) - tps
        fns = cm.sum(axis=1) - tps
        return tps, fps, fns

    def _calc_metrics_all_stages(self, tps, fps, fns) -> dict:
        result = {}
        sleep_stages = list(self.stages.keys())

        for i, stage in enumerate(sleep_stages):
            precision, recall, f1 = calc_metrics(tps[i], fps[i], fns[i])
            result[self.stages[stage]] = {"p": precision, "r": recall, "f1": f1}

        return result


def calc_metrics(tp, fp, fn):
    """
    Calculate precision, recall, and F1 score from confusion matrix counts.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        tuple: (precision, recall, f1) scores.

    Note:
        If a class has no true or predicted instances (tp + fp + fn == 0),
        F1 is set to 1.0 (perfect score for correctly predicting absence).
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    # if a class does not exist and was not predicted, f1 is 1
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1
    return precision, recall, f1


def add_f1_to_results(result_dict, base_key, metrics):
    """
    Add per-stage metrics to a results dictionary.

    Args:
        result_dict (dict): Dictionary to update.
        base_key (str): Base path for metrics (e.g., "datasplit/concat").
        metrics (dict): Stage name to {"p", "r", "f1"} mapping.
    """
    for stage, stage_vals in metrics.items():
        result_dict[f"{base_key}/{stage}/p"] = stage_vals["p"]
        result_dict[f"{base_key}/{stage}/r"] = stage_vals["r"]
        result_dict[f"{base_key}/{stage}/f1"] = stage_vals["f1"]
