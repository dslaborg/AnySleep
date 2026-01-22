"""
Generate per-recording confusion matrices for detailed error analysis. Used to create Figure 1a in our manuscript.

This script runs inference on a dataset and computes a confusion matrix for
each recording and channel combination, enabling fine-grained analysis of
model performance across different subjects and channels.

Usage: see run.md in config folder

The script:
1. Loads the trained model specified in the config
2. Runs inference on the dataset specified in predict_cm.dataloader
3. For each recording, computes a 5x5 confusion matrix (rows=true, cols=pred)
4. Saves all confusion matrices to a compressed numpy file

Args (Hydra overrides):
    -cn: Config name (must have predict_cm.dataloader configured)

Output files:
    - pred_cms.npz: Confusion matrices per recording (keys: dataset#subject#channels)
                    Shape per key: (5, 5) - confusion matrix for 5 sleep stages
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
from sklearn.metrics import confusion_matrix

sys.path.insert(0, realpath(join(dirname(__file__), "..")))

from base.config import Config

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

    dataloader_cfg = cfg.predict_cm.dataloader
    dataloader = instantiate(dataloader_cfg)

    cms_to_save = {}

    with torch.no_grad():
        for loaded in dataloader:  # iterate over recordings
            data, labels, name = loaded
            predict = model(data.to(cfg.general.device))
            np_predict = predict.detach().cpu().numpy()
            np_predict = np.argmax(np_predict.reshape(-1, np_predict.shape[-1]), axis=1)

            cms_to_save[name[0]] = confusion_matrix(
                labels.flatten(), np_predict, labels=range(5)
            )

    model.cpu()

    np.savez_compressed("pred_cms.npz", **cms_to_save)


if __name__ == "__main__":
    main()
