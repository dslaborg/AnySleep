"""
Generate per-subject sleep stage predictions with majority voting.

This script runs inference on a dataset and saves predictions for each subject,
applying majority voting across channel combinations when multiple channels are
evaluated for the same subject.

Usage: see run.md in config folder

The script:
1. Loads the trained model specified in the config
2. Runs inference on the dataset specified in high_freq_predict.dataloader
3. For each subject, collects predictions from all channel combinations
4. Applies majority voting to get consensus predictions per epoch
5. Saves predictions and labels to compressed numpy files

Args (Hydra overrides):
    -cn: Config name (must have high_freq_predict.dataloader configured)

Output files:
    - predictions.npz: Majority-voted predictions per subject (keys: dataset_subject)
    - labels.npz: Ground truth labels per subject (keys: dataset_subject)
"""

import os
import shutil
import sys
from logging import getLogger
from os.path import realpath, dirname, join

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, realpath(join(dirname(__file__), "..")))

from base.config import Config
from base.utils import calc_mode

logger = getLogger(__name__)


def initialize(cfg):
    Config.initialize(cfg)


@hydra.main(config_path="../config", version_base="1.2")
def main(cfg: DictConfig):
    initialize(cfg)
    hydra_cfg = HydraConfig.get()
    logger.info(f"overrides:\n{OmegaConf.to_yaml(hydra_cfg.overrides)}")

    if not OmegaConf.is_missing(hydra_cfg.job, "num") and hydra_cfg.job.num == 0:
        shutil.copytree(
            join(os.getcwd(), str(hydra_cfg.output_subdir)),
            join(os.getcwd(), "..", str(hydra_cfg.output_subdir)),
            ignore=shutil.ignore_patterns("overrides.yaml"),
        )

    model = instantiate(cfg.model)
    model.eval()
    model.to(cfg.general.device)

    dataloader_cfg = cfg.high_freq_predict.dataloader
    dataloader = instantiate(dataloader_cfg)

    predictions_to_save = {}
    labels_to_save = {}

    with torch.no_grad():
        for loaded in dataloader:  # iterate over recordings
            data, labels, name = loaded
            predict = model(data.to(cfg.general.device))
            np_predict = predict.detach().cpu().numpy()
            np_predict = np.argmax(np_predict.reshape(-1, np_predict.shape[-1]), axis=1)

            dataset_name = name[0].split("#")[0]
            subject_name = name[0].split("#")[1]

            if f"{dataset_name}_{subject_name}" not in predictions_to_save:
                predictions_to_save[f"{dataset_name}_{subject_name}"] = []
            predictions_to_save[f"{dataset_name}_{subject_name}"].append(np_predict)

            labels_to_save[f"{dataset_name}_{subject_name}"] = labels.numpy().flatten()

    model.cpu()
    # majority vote predictions for each subject
    for ds_subj_key, subject_data in predictions_to_save.items():
        if len(subject_data) == 1:
            majority_vote = subject_data[0]
        else:
            majority_vote = calc_mode(subject_data, axis=0)
        predictions_to_save[ds_subj_key] = majority_vote

    np.savez_compressed("predictions.npz", **predictions_to_save)
    np.savez_compressed("labels.npz", **labels_to_save)


if __name__ == "__main__":
    main()
