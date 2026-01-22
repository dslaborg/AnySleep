"""
Reference:
    Perslev, M., et al. (2021). U-Sleep: resilient high-frequency sleep staging.
    npj Digital Medicine, 4(1), 72.
"""

from logging import getLogger

import numpy as np
import torch

from base.config import Config
from base.data.base_dataset import BaseDataset

logger = getLogger(__name__)


class UsleepTrainDataset(BaseDataset):
    """
    Training dataset with balanced sleep stage sampling for USleep.

    This dataset implements the sampling strategy from Perslev et al. (2021)
    to address class imbalance in sleep staging. Rather than iterating through
    recordings sequentially, it samples windows on-the-fly ensuring all sleep
    stages are represented equally.

    Each sample consists of:
    - A window of EEG+EOG signal data (2 channels)
    - Corresponding sleep stage labels
    - An identifier string for tracking

    The dataset samples:
    1. A dataset (weighted by size and uniform probability)
    2. A random subject from that dataset
    3. A random sleep stage
    4. A window containing at least one epoch of that stage
    5. One random EEG and one random EOG channel

    Attributes:
        window_size (int): Number of 30-second epochs per sample.
        num_of_samples (int): Total samples to generate per training epoch.
    """

    def __init__(self, h5_path: str, window_size: int, num_of_samples: int) -> None:
        """
        On creation the h5_file is read into the archive property.
        Additionally, the names of all top level dataset are made available
        in the dataset_names property.
        The full number of subjects in the file is calculated and saved in the property num_subjects.


        Args:
            - h5_path (string): Path to the h5 file
            - window_size : size of window
            for Train Datasampling is applied for the other values no sampling is applied
            - num_of_samples : number of samples to generate
        """
        super().__init__(h5_path, "train")

        # Load config
        _cfg = Config.get()

        # General Parameters
        self.window_size = window_size
        self.num_of_samples = num_of_samples

    def __getitem__(self, _):
        """Generates a sample for the mode training.
        The datasampling is implemented as described by Perslev et al. ( 2021 ).
        For details see the docstring of the subfunctions.
        """

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
            eeg_name = self.choice(eeg)
            eog_name = self.choice(eog)

            # Extract data
            eeg = self.archive[dataset_name][subject_name]["PSG"][eeg_name][
                start_time:end_time
            ]
            eog = self.archive[dataset_name][subject_name]["PSG"][eog_name][
                start_time:end_time
            ]

            # Build output
            identifier = f"{dataset_name}#{subject_name}#{eeg_name}#{eog_name}"  # #{start_time}#{end_time}

            x = np.column_stack([eeg, eog])
            return torch.from_numpy(x), torch.from_numpy(phase_labels), identifier
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
