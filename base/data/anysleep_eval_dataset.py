"""
Evaluation dataset for AnySleep model with flexible channel selection.

This module provides evaluation datasets for attention-based models that can
handle variable numbers of input channels. It supports various evaluation
scenarios:
- All available channels
- Specific channel sets
- Limited number of EEG/EOG channels
- Random subsets for robustness testing
"""

import itertools
from logging import getLogger

import numpy as np
import torch

from base.config import Config
from base.data.base_dataset import BaseDataset
from base.utils import choice

logger = getLogger(__name__)


class AnySleepEvalDataset(BaseDataset):
    """
    Evaluation dataset for attention-based models with flexible channel selection.

    This dataset supports various channel configurations for comprehensive
    evaluation of attention-based sleep staging models:

    1. All channels: Use every available EEG and EOG channel
    2. Specific channels: Use a predefined set of channels
    3. Limited channels: Use n EEG + m EOG channels, iterating all combinations
    4. Random subset: Sample a limited number of subjects with fixed channels

    Each sample contains a complete recording with the selected channels,
    allowing evaluation of the model's attention mechanism across different
    channel configurations.

    Attributes:
        n_eeg_channels (int): Number of EEG channels to use (-1 for all).
        n_eog_channels (int): Number of EOG channels to use (-1 for all).
        channels (list): Specific channels to use (overrides n_* settings).
        identifiers (list): List of sample metadata dictionaries.
        num_of_epochs (int): Total epochs across all samples.

    Example:
        >>> # Evaluate with all channels
        >>> dataset = AnySleepEvalDataset(h5_path, "val", n_eeg_channels=-1, n_eog_channels=-1)
        >>> # Evaluate with 2 EEG + 1 EOG
        >>> dataset = AnySleepEvalDataset(h5_path, "val", n_eeg_channels=2, n_eog_channels=1)
    """

    def __init__(
        self,
        h5_path: str,
        datasplit: str,
        n_eeg_channels: int = -1,
        n_eog_channels: int = -1,
        channels: list[str] = None,
        datasets_to_load=None,
        limit_num_samples_to: int = None,
    ) -> None:
        """
        On creation the h5_file is read into the archive property.
        Additionally, the names of all top level dataset are made available
        in the dataset_names property.
        The full number of subjects in the file is calculated and saved in the property num_subjects.


        Args:
            - h5_path (string): Path to the h5 file
            - datasplit (str) : Values can be test, val
            for Train Datasampling is applied for the other values no sampling is applied
            - n_eeg_channels (int): Number of EEG channels to use
            - n_eog_channels (int): Number of EOG channels to use
            - channels: List of channels to use, if None, n_eeg_channels and n_eog_channels are used
            - datasets_to_load: List of datasets to load. If None, all datasets are loaded
            - limit_num_samples_to: Limit the number of samples to load to specified number
        """
        assert datasplit in ["test", "val"], f"invalid mode {datasplit}"
        super().__init__(h5_path, datasplit)

        # Load config
        _cfg = Config.get()

        if n_eeg_channels * n_eog_channels < 0:
            raise ValueError(
                "All of n_eeg_channels and n_eog_channels must either be -1 or >=0."
            )
        if (
            n_eeg_channels < 0
            and n_eog_channels < 0
            and limit_num_samples_to is not None
        ):
            raise ValueError(
                "When limiting the number of samples, both n_eeg_channels and n_eog_channels must be set to a value >=0."
            )
        self.n_eeg_channels = n_eeg_channels
        self.n_eog_channels = n_eog_channels
        self.channels = channels
        self.datasets_to_load = datasets_to_load
        self.limit_num_samples_to = limit_num_samples_to

        self.identifiers, self.num_of_epochs = (
            self.calculate_total_samples_records_only()
        )
        self.num_of_samples = len(self.identifiers)
        print(f"Validating on {self.num_of_samples} samples")

    def calculate_total_samples_records_only(self) -> tuple[list[dict], int]:
        """Creates an index to all possible samples in the dataset. A sample consists of a full recording of a subject.

        Note that the data is not randomised and during unlimited validation all available data will be used
        in always the same order.

        """

        indices = []

        ds_subj_list = []
        # Iterate through all datasets
        for dataset_name, subjects in self.data_filter.items():
            if (
                self.datasets_to_load is not None
                and dataset_name not in self.datasets_to_load
            ):
                continue
            # Iterate through all subjects
            for subject_name in subjects:
                ds_subj_list.append((dataset_name, subject_name))

        if self.limit_num_samples_to is not None:
            ds_subj_list = choice(
                ds_subj_list, self.limit_num_samples_to, random_state=self.random_state
            )

        for dataset_name, subject_name in ds_subj_list:
            # Get all possible channels for the current dataset
            eeg_channels, eog_channels = self.get_channels(dataset_name, subject_name)
            channels = eeg_channels + eog_channels

            if self.limit_num_samples_to is not None:
                if (
                    len(eeg_channels) < self.n_eeg_channels
                    or len(eog_channels) < self.n_eog_channels
                ):
                    logger.warning(
                        f"Dataset {dataset_name} for subject {subject_name} has too few channels: {eeg_channels}, {eog_channels}"
                    )
                sampled_eeg_ch = self.random_state.choice(
                    eeg_channels,
                    size=self.n_eeg_channels,
                    replace=len(eeg_channels) < self.n_eeg_channels,
                )
                sampled_eog_ch = self.random_state.choice(
                    eog_channels,
                    size=self.n_eog_channels,
                    replace=len(eog_channels) < self.n_eog_channels,
                )
                channel_set = set(sampled_eeg_ch).union(sampled_eog_ch)
                if len(channel_set) > 0:
                    indices.append(
                        {
                            "Dataset": dataset_name,
                            "Subject": subject_name,
                            "Channels": channel_set,
                        }
                    )
            elif self.channels is not None:
                if any(ch not in channels for ch in self.channels):
                    logger.warning(
                        f"Channels {self.channels} not available in dataset {dataset_name} "
                        f"for subject {subject_name}, skipping"
                    )
                    continue
                indices.append(
                    {
                        "Dataset": dataset_name,
                        "Subject": subject_name,
                        "Channels": self.channels,
                    }
                )
            elif self.n_eeg_channels == -1 and self.n_eog_channels == -1:
                self.random_state.shuffle(channels)
                indices.append(
                    {
                        "Dataset": dataset_name,
                        "Subject": subject_name,
                        "Channels": channels,
                    }
                )
            else:
                n_eeg = min(self.n_eeg_channels, len(eeg_channels))
                n_eog = min(self.n_eog_channels, len(eog_channels))
                sampled_channel_combs = []
                # iterate over all channel combinations with n_eeg eeg channels and n_eog eog channels

                for eeg_comb in itertools.combinations(eeg_channels, n_eeg):
                    for eog_comb in itertools.combinations(eog_channels, n_eog):
                        comb_channels = set(eeg_comb).union(eog_comb)
                        # we assume that this dataloader is only used for attention models that are invariant to
                        # channel order
                        if comb_channels in sampled_channel_combs:
                            continue
                        sampled_channel_combs.append(comb_channels)
                        indices.append(
                            {
                                "Dataset": dataset_name,
                                "Subject": subject_name,
                                "Channels": comb_channels,
                            }
                        )

        num_of_epochs = sum(
            [
                self.archive[idx["Dataset"]][idx["Subject"]]["hypnogram"].len()
                for idx in indices
            ]
        )

        return indices, num_of_epochs

    def __getitem__(self, index):
        """
        Returns the datawindow([eeg,eog], label_vector, identifier)
        at the index for the modes validation and testing.

        Assumes that identifiers is a dictionary like
        1 : {'Dataset': 'abc',
            'Subject': '900001',
            'Channels': ['F3-M2', 'E1-M2']},
        """
        # Generates a logging message to see the progress of the validation
        step = int(self.num_of_samples * 0.1)
        if step != 0 and (index + 1) % step == 0:
            percentage = (index + 1) * 100 // (len(self))
            logger.info(
                f"[Validation] Processed samples:  {percentage}% of {self.num_of_samples} samples."
            )

        # Load config
        _cfg = Config.get()

        # Load item
        id = self.identifiers[index]

        # Extract data
        dataset_name = id["Dataset"]
        subject_name = id["Subject"]
        channels = id["Channels"]

        # Get labels
        sleep_stage_labels = torch.from_numpy(
            self.archive[dataset_name][subject_name]["hypnogram"][:]
        )

        # Get channel data
        channel_data = [
            self.archive[dataset_name][subject_name]["PSG"][s][:] for s in channels
        ]
        x = np.column_stack(channel_data)

        # Build output
        identifier = f"{dataset_name}#{subject_name}#{'&'.join(channels)}"
        return torch.from_numpy(x), sleep_stage_labels, identifier
