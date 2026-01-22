"""
Training dataset for AnySleep model with variable channel counts.

This module extends the U-Sleep sampling strategy for attention-based models
that can handle variable numbers of input channels. Unlike USleep which always
uses exactly 2 channels (1 EEG + 1 EOG), AnySleep can process any number of
channels and learn attention weights to combine them.

Key difference from UsleepTrainDataset:
- Batches are constructed internally (not by DataLoader) to ensure all samples
  in a batch have the same number of channels
- Channel count varies between batches but is fixed within a batch
- All available EEG and EOG channels can be used simultaneously

Reference:
    Perslev, M., et al. (2021). U-Sleep: resilient high-frequency sleep staging.
    npj Digital Medicine, 4(1), 72.
"""

from logging import getLogger

import numpy as np
import torch
from omegaconf import OmegaConf

from base.config import Config
from base.data.base_dataset import BaseDataset

logger = getLogger(__name__)


class AnySleepTrainDataset(BaseDataset):
    """
    Training dataset for attention-based models with variable channel counts.

    This dataset extends the U-Sleep sampling strategy for models that use
    attention mechanisms across multiple channels. Each batch has a fixed
    number of channels (to allow batching), but different batches can have
    different channel counts.

    The channel count per batch is sampled to favor having more channels,
    encouraging the model to learn robust attention weights.

    Each sample consists of:
    - Window of multi-channel signal data (variable channels)
    - Corresponding sleep stage labels
    - Identifier string with all channel names

    Attributes:
        window_size (int): Number of 30-second epochs per sample.
        batch_size (int): Samples per batch (handled internally).
        max_channels (int): Maximum channels available across all datasets.
        num_of_samples (int): Total samples per training epoch.

    Note:
        The __len__ returns num_batches, not num_samples, since batching
        is handled internally by __getitem__.
    """

    def __init__(
        self, h5_path: str, window_size: int, num_of_samples: int, batch_size: int
    ) -> None:
        """
        On creation the h5_file is read into the archive property.
        Additionally, the names of all top level dataset are made available
        in the dataset_names property.
        The full number of subjects in the file is calculated and saved in the property num_subjects.
        We need to create the batches inside the dataset instead of the dataloader because each batch needs a fixed number of channels.


        Args:
            - h5_path (string): Path to the h5 file
            - window_size : size of window
            for Train Datasampling is applied for the other values no sampling is applied
            - num_of_samples : number of samples to generate
            - batch_size : batch size
        """
        super().__init__(h5_path, "train")

        # Load config
        _cfg = Config.get()

        # General Parameters
        self.window_size = window_size
        self.batch_size = batch_size

        # extract the channels available for all subjects
        self.max_channels = self._get_max_channels()

        # make sure that the number of samples is divisible by the batch size
        # the current dataloader doesn't allow for incomplete batches
        self.num_of_samples = int(
            np.ceil(num_of_samples / self.batch_size) * self.batch_size
        )

    def __len__(self):
        # the number of samples has to be divided by the batch size because we batch manually
        return int(self.num_of_samples // self.batch_size)

    def _get_max_channels(self) -> int:
        """
        Returns the maximum number of channels available for all subjects in all datasets.
        """
        max_channels = 0
        for dataset_name in self.get_dataset_names():
            eeg_channels: np.ndarray = self.archive[dataset_name].attrs["eeg_channels"]
            eog_channels: np.ndarray = self.archive[dataset_name].attrs["eog_channels"]
            n_channels = len(eeg_channels) + len(eog_channels)
            max_channels = max(max_channels, n_channels)
        return max_channels

    def calc_channels_per_batch(self):
        arranged = np.arange(1, self.max_channels + 1)
        p = np.array([1 / np.sum(n / arranged) for n in arranged])
        p[-1] = 1 - np.sum(p[:-1])
        return self.random_state.choice(arranged, p=p)

    def __getitem__(self, _):
        """Generates a sample for the mode training.
        The datasampling is implemented as described by Perslev et al. ( 2021 ).
        For details see the docstring of the subfunctions.
        """

        # Load config
        _cfg = Config.get()

        c_batch = self.calc_channels_per_batch()

        data = torch.empty(
            (
                self.batch_size,
                _cfg.data.sampling_rate * _cfg.data.epoch_duration * self.window_size,
                c_batch,
            ),
            dtype=torch.float32,
        )
        labels = torch.empty((self.batch_size, self.window_size))
        identifiers = []
        for i in range(self.batch_size):
            data[i], labels[i], idx = self.sample_item(c_batch)
            identifiers.append(idx)
        return data, labels, identifiers

    def sample_item(self, c_batch: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        # Load config
        _cfg = Config.get()
        sample_rate = _cfg.data.sampling_rate
        epoch_duration = _cfg.data.epoch_duration
        possible_stages = len(_cfg.data.stages)
        selected_phase = str(self.random_state.randint(possible_stages))

        tries, max_tries = 0, 100
        while tries < max_tries:
            # Generate sample
            dataset_name: str = self.sample_dataset()
            subject_name = self.choice(list(self.data_filter[dataset_name]))

            try:
                start_time, end_time, phase_labels = self.sample_segment(
                    dataset_name,
                    subject_name,
                    sample_rate,
                    epoch_duration,
                    selected_phase,
                )
            except KeyError:
                # sleep stage not found in this subject
                tries += 1
                continue

            # channel sampling
            eeg, eog = self.get_channels(dataset_name, subject_name)
            channels = eeg + eog
            with_replacement = len(channels) < c_batch
            sampled_channels = self.random_state.choice(
                channels, c_batch, replace=with_replacement
            )

            # Extract data
            data = np.empty((end_time - start_time, c_batch), dtype=np.float32)
            for i, channel_name in enumerate(sampled_channels):
                data[:, i] = self.archive[dataset_name][subject_name]["PSG"][
                    channel_name
                ][start_time:end_time]

            # Build output
            identifier = f"{dataset_name}#{subject_name}#{'&'.join(sampled_channels)}"  # #{start_time}#{end_time}

            return torch.from_numpy(data), torch.from_numpy(phase_labels), identifier
        raise RuntimeError(f"could not sample a segment for class {selected_phase}")

    def sample_dataset(self, alpha: float = 0.5) -> str:
        """
        Based on dataset sampling as described by Perslev et. al. (2021)

        This function randomly selects one of the available training datasets based on a probability defined by a combination of two factors:
        1. Discrete uniform sampling (all datasets are sampled with equal probability)
        2. Sampling according to the size of the dataset (larger datasets are sampled more often)

        The probability of selecting a dataset, D, is given by p(D) = alpha*p1(D) + (1 -alpha)*p2(D), where:
        - p1(D) is the probability under discrete uniform sampling (i.e., 1/N, where N is the number of datasets)
        - p2(D) is the probability of sampling a dataset according to its size (i.e., the number of PSG records in the dataset)
        - alpha is a parameter to weigh p1(D) and p2(D) (set to 0.5 by default to equally weigh p1(D) and p2(D))

        This sampling policy ensures that all datasets (and thus clinical cohorts) are considered equally important in training,
        independent of their size, while individual PSG records are equally important,
        independent of the size of the dataset from which they originated.

        Parameters:
        alpha (float): The weight for the two probabilities p1(D) and p2(D). Default is 0.5.

        Returns:
        chosen_element (str): The name of the chosen dataset.
        """
        dataset_names = list(self.data_filter.keys())
        n_datasets = len(self.data_filter)
        p1 = np.full(n_datasets, 1 / n_datasets)
        p2 = np.array(
            [
                (len(self.data_filter[dataset]) / self.num_subjects)
                for dataset in dataset_names
            ]
        )

        p = alpha * p1 + (1 - alpha) * p2

        chosen_element = self.random_state.choice(dataset_names, p=p)

        return chosen_element

    def choice(self, l: list):
        return l[self.random_state.randint(0, len(l))]

    def sample_segment(
        self,
        dataset_name: str,
        subject_name: str,
        sample_rate,
        epoch_duration,
        selected_phase: str,
    ):
        """
        Based on segment sampling as described by Perslev et. al. (2021)

        This method samples a segment of length T from the chosen EEG-EOG channel combination from PSG record SD.

        The temporal placement of the segment is selected by following these steps:
        1. Uniformly sample a class from the label set {W, N1, N2, N3, REM}.
        2. Select a random sleep period of 30 s that the human annotator scored to be of the sampled class.
        3. Randomly position a sleep segment of length T so that it containst he choosen sleep period.

        This scheme ensures that even very rare sleep stages are visited. However, this approach does not fully balance
        the batches, as the T−1 remaining segments of the input window are still subject to class imbalance, and some
        PSG records might not display a given minority class at all.

        Parameters:
        dataset_name (str): Name of the dataset.
        subject_name (str): Name of the subject.
        T (int): Size of the shift window in the hypnogram (default is 35, corresponding to 17.5 min).

        Returns:
        phase_index (int): The index of the phase within the window.
        """
        # Get the indices of all 30s segments / epochs that were labeled as the choosen sleep phase from the hypnogram
        phase_indices = self.archive[dataset_name][subject_name]["class_to_index"][
            selected_phase
        ]

        if len(phase_indices) == 0:
            raise KeyError("No phase indices found for selected stage.")

        # Sample a random epoch of the chosen sleep phase
        phase_idx = self.choice(phase_indices)

        # Calculate the start and end indices in the hypnogram
        len_h = self.archive[dataset_name][subject_name]["hypnogram"].shape[0]

        start_idx, end_idx = self._sample_start_end(len_h, int(phase_idx))

        # Get the label vector
        labels = self.archive[dataset_name][subject_name]["hypnogram"][
            start_idx:end_idx
        ]

        # Get the start- and end-time for the EEG / EOG data based on the sampling_rate and epoch_duration
        start_time = start_idx * epoch_duration * sample_rate
        end_time = end_idx * epoch_duration * sample_rate

        # Return the sampling results
        return start_time, end_time, labels

    def _sample_start_end(self, len_h: int, phase_idx: int):
        """Window sampling as described by Perslev et. al.

        Based on the given max. epoch num (len_h), the phase to be sampled (phase_idx)
        and the window length (T) a start and an end index for a randomly positioned
        window in the hypnogram are calculated.
        The function ensures that the window is always exactly of length T.

        """
        min_left_idx = max(0, int(phase_idx) - self.window_size + 1)
        max_left_idx = min(len_h - self.window_size, phase_idx)

        start_idx = self.random_state.randint(min_left_idx, max_left_idx + 1)
        return start_idx, start_idx + self.window_size
