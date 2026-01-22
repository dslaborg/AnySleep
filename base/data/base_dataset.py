"""
Base dataset class for loading polysomnography (PSG) data from HDF5 files.

This module defines the abstract base class for all dataset implementations in
the AnySleep framework. It handles HDF5 file access, data filtering by
train/validation/test splits, and provides common utilities for channel selection.

The HDF5 file structure expected is:
    File
    ├── Dataset_Name/
    │   ├── Subject_Name/
    │   │   ├── PSG/
    │   │   │   ├── EEG_Channel (e.g., 'F3-M2')
    │   │   │   └── EOG_Channel (e.g., 'E1-M2')
    │   │   ├── hypnogram      # Sleep stage labels per 30s epoch
    │   │   └── class_to_index/  # Indices grouped by sleep stage
    │   └── ...
    └── ...
"""

from functools import lru_cache
from logging import getLogger

import h5py
import hydra
import numpy as np
from torch.utils.data import Dataset

from base.config import Config

logger = getLogger(__name__)


class BaseDataset(Dataset):
    """Torch Dataset
    Reads in Datasets from a h5 file and iterates over EEG / EOG segments

    Expects the following file format:

    File
    |--- Dataset_Name_1
    |      |--- Subject_Name_1
    |      |      |--- PSG
    |      |      |     |--- Channel_1
    |      |      |     |--- Channel_2
    |      |      |     |--- ...
    |      |      |
    |      |      |--- hypnogram
    |      |      |
    |      |      |--- (class_to_index)
    |      |            |--- 0
    |      |            |--- 1
    |      |            |---...
    |      |--- ...
    |
    |--- Dataset_Name_2
    |      |---
    |---...
    """

    def __init__(self, h5_path: str, datasplit: str) -> None:
        """
        On creation the h5_file is read into the archive property.
        Additionally, the names of all top level dataset are made available
        in the dataset_names property.
        The full number of subjects in the file is calculated and saved in the property num_subjects.


        Args:
            - h5_path (string): Path to the h5 file
            - datasplit (str) : Values can be test, val, train
                for Train Datasampling is applied for the other values no sampling is applied
        """

        # Load config
        _cfg = Config.get()

        # Output
        logger.info(f"Instantiated dataset for split '{datasplit}'")

        # Data loading
        self._h5_path = hydra.utils.to_absolute_path(h5_path)
        self._archive = h5py.File(self._h5_path, "r")

        if datasplit == "test":
            self.data_filter = convert_to_string(_cfg.datasets.test)
        elif datasplit == "val":
            self.data_filter = convert_to_string(_cfg.datasets.valid)
        elif datasplit == "train":
            self.data_filter = convert_to_string(_cfg.datasets.train)
        else:
            raise ValueError(f"unknown mode {datasplit}")
        logger.info(f"Using the following datasets:\n {self.get_dataset_names()}'")

        #  Check if all configured sets and subjects are actually available in the loaded data
        for dataset in self.data_filter.keys():
            if dataset not in self.archive:
                raise ValueError(
                    f"Dataset '{dataset}' is not available. The following dataset are available: "
                    f"\n{list(self.archive.keys())} "
                )

            for subject in self.data_filter[dataset]:
                if subject not in self.archive[dataset].keys():
                    raise ValueError(
                        f"Subject '{subject}' was not found in dataset '{dataset}'. "
                        f"The following subjects are available in {dataset}: "
                        f"\n {list(self.archive[dataset].keys())}"
                    )

        # General Parameters
        self.num_of_samples = 0
        self.num_subjects = np.sum([len(v) for v in self.data_filter.values()])

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, index):
        raise NotImplementedError

    @lru_cache(maxsize=12800)
    def get_channels(
        self, dataset_name: str, subject_name: str
    ) -> tuple[list[str], list[str]] | tuple[list[str], list[str], list[str]]:
        """
        Returns all channels, split by EEG and EOG, for a given subject in a given dataset.
        """
        eeg_channels: np.ndarray = self.archive[dataset_name].attrs["eeg_channels"]
        eog_channels: np.ndarray = self.archive[dataset_name].attrs["eog_channels"]

        subj_channels = set(self.archive[dataset_name][subject_name]["PSG"].keys())

        available_eeg_channels = list(
            filter(lambda channel: channel in subj_channels, eeg_channels)
        )
        available_eog_channels = list(
            filter(lambda channel: channel in subj_channels, eog_channels)
        )

        return available_eeg_channels, available_eog_channels

    @property
    def archive(self) -> h5py.File:
        """Full h5 file"""
        if self._archive is None:  # lazy loading here!
            self._archive = h5py.File(self._h5_path, "r")
        return self._archive

    def get_dataset_names(self):
        """List of the names of all contained datasetss"""
        return list(self.data_filter.keys())

    @property
    def random_state(self):
        """Random state for the dataset, is set in CustomDataloader worker_init_fn (if num_workers > 0)"""
        if not hasattr(self, "_random_state") or self._random_state is None:
            self._random_state = np.random.RandomState()
        return self._random_state

    def __str__(self):
        """Returns the file structure of the dataset"""
        # datasets
        text = ""
        for dataset_name in self.get_dataset_names():
            eegs = self.archive[dataset_name].attrs["eeg_channels"]
            eogs = self.archive[dataset_name].attrs["eog_channels"]
            subjects = list(self.data_filter[dataset_name])

            text += f"-> Dataset : {dataset_name} \n"

            text += f"  -> EEG-channels: {eegs}\n"
            text += f"  -> EOG-channels: {eogs}\n"

            text += f"  -> Subj.       : {subjects}\n"

        return text


def convert_to_string(set_dict):
    """
    Convert configuration dict keys and values to strings.

    Hydra configs may contain integer keys/values that need to be converted
    to strings for HDF5 group access.

    Args:
        set_dict (dict): Dictionary from config with datasets and subject lists.

    Returns:
        dict: Same structure with all keys and values as strings.
    """
    string_dict = {}
    for dataset in set_dict.keys():
        string_dict[str(dataset)] = []
        for subject in set_dict[str(dataset)]:
            string_dict[str(dataset)].append(str(subject))

    return string_dict
