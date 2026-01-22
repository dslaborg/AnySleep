from logging import getLogger

import numpy as np
import torch

from base.config import Config
from base.data.base_dataset import BaseDataset

logger = getLogger(__name__)


class UsleepEvalDataset(BaseDataset):
    """
    Evaluation dataset returning complete recordings for USleep.

    Unlike the training dataset which samples windows, this dataset returns
    full recordings for comprehensive evaluation. Each sample contains an
    entire night's data for one subject with one channel combination.

    Features:
    - Iterates through all configured subjects deterministically
    - Can evaluate all EEG/EOG channel combinations or random single pair
    - Returns complete recordings with all epochs
    - Supports filtering to specific datasets

    Each sample consists of:
    - Complete EEG+EOG signal data (2 channels)
    - All sleep stage labels for the recording
    - Identifier string (dataset#subject#eeg#eog)

    Attributes:
        all_channels (bool): If True, iterate all channel combinations.
        eeg_eog_only (bool): If True, only use valid EEG+EOG pairs.
        identifiers (dict): Mapping from index to sample metadata.
        num_of_epochs (int): Total epochs across all samples.
    """

    def __init__(
        self,
        h5_path: str,
        datasplit: str,
        all_channels: bool = True,
        eeg_eog_only=True,
        datasets_to_load=None,
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
            - eeg_eog_only: If True, only valid EEG and EOG channel combinations are used (EEG first, EOG second).
                            If False, all channels are used.
            - all_channels: If True the recordings are sampled multiple times, once for each possible channel combination
            - datasets_to_load: List of datasets to load. If None, all datasets are loaded
        """
        assert datasplit in ["test", "val"], f"invalid mode {datasplit}"
        super().__init__(h5_path, datasplit)

        # Load config
        _cfg = Config.get()

        self.eeg_eog_only = eeg_eog_only
        self.all_channels = all_channels
        self.datasets_to_load = datasets_to_load

        self.identifiers, self.num_of_epochs = (
            self.calculate_total_samples_records_only(
                single_random=not self.all_channels
            )
        )
        self.num_of_samples = len(self.identifiers)
        print(f"Validating on {self.num_of_samples} samples")

    def calculate_total_samples_records_only(
        self, single_random: bool = False
    ) -> tuple[dict[int, dict], int]:
        """Creates an index to all possible samples in the dataset. A sample consists of a full recording of a subject.

        Note that the data is not randomised and during unlimited validation all available data will be used
        in always the same order.
        """

        total_samples = 0
        num_of_epochs = 0
        indices = {}

        # Iterate through all datasets
        for dataset_name, subjects in self.data_filter.items():
            if (
                self.datasets_to_load is not None
                and dataset_name not in self.datasets_to_load
            ):
                continue

            # Iterate through all subjects
            for subject_name in subjects:

                # Get all possible channels for the current dataset
                eeg_channels, eog_channels = self.get_channels(
                    dataset_name, subject_name
                )

                if single_random:
                    eeg_name = eeg_channels[
                        self.random_state.randint(0, len(eeg_channels))
                    ]
                    eog_name = eog_channels[
                        self.random_state.randint(0, len(eog_channels))
                    ]
                    indices[total_samples] = {
                        "Dataset": dataset_name,
                        "Subject": subject_name,
                        "C1": eeg_name,
                        "C2": eog_name,
                    }
                    total_samples += 1
                    num_of_epochs += len(
                        self.archive[dataset_name][subject_name]["hypnogram"]
                    )
                else:
                    if self.eeg_eog_only:
                        c1_list = eeg_channels
                        c2_list = eog_channels
                    else:
                        c1_list = eeg_channels + eog_channels
                        c2_list = eeg_channels + eog_channels

                    # Iterate through all channel combinations
                    for c1 in c1_list:
                        for c2 in c2_list:
                            indices[total_samples] = {
                                "Dataset": dataset_name,
                                "Subject": subject_name,
                                "C1": c1,
                                "C2": c2,
                            }
                            total_samples += 1
                            num_of_epochs += self.archive[dataset_name][subject_name][
                                "hypnogram"
                            ].len()

        return indices, num_of_epochs

    def __getitem__(self, index):
        """
        Returns the datawindow([eeg,eog],label_vector, identifier)
        at the index for the modes validation and testing.

        Assumes that identifiers is a dictionary like
        1 : {'Dataset': 'abc',
            'Subject': '900001',
            'C1': 'F3-M2',
            'C2': 'E1-M2'},
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
        c1_name, c2_name = id["C1"], id["C2"]

        # Get labels
        sleep_stage_labels = torch.from_numpy(
            self.archive[dataset_name][subject_name]["hypnogram"][:]
        )

        # Get channel data
        eeg = self.archive[dataset_name][subject_name]["PSG"][c1_name][:]
        eog = self.archive[dataset_name][subject_name]["PSG"][c2_name][:]

        # Build output
        identifier = f"{dataset_name}#{subject_name}#{c1_name}#{c2_name}"

        x = np.column_stack([eeg, eog])
        return torch.from_numpy(x), sleep_stage_labels, identifier
