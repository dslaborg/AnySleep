"""
Train a sleep stage classification model.

This script trains a neural network model (USleep, AnySleep, etc.) for sleep
stage classification using the configuration specified via Hydra.

Usage: see run.md in config folder

The script:
1. Loads the model, dataset, and trainer configuration from the specified config
2. Trains the model with early stopping based on validation macro F1 score
3. Saves the best model checkpoint to ./models/
4. Logs training progress and saves results to ./logs/<config_name>/<timestamp>/

Args (Hydra overrides):
    -cn: Config name (e.g., exp001/exp001a, exp002/exp002a)
    Any config parameter can be overridden with key=value syntax

Returns:
    float: Best macro F1 score achieved (used for hyperparameter optimization)

Output files:
    - logs/<config>/<timestamp>/results.json: Training and validation metrics
    - logs/<config>/<timestamp>/*.log: Training logs
    - models/<config>-<identifier>-<timestamp>-final.pth: Model checkpoint
"""

import os
import shutil
import sys
from logging import getLogger
from os.path import realpath, dirname, join

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, realpath(join(dirname(__file__), "..")))

from base.config import Config

logger = getLogger(__name__)


def initialize(cfg):
    Config.initialize(cfg)


@hydra.main(config_path="../config", version_base="1.2")
def main(cfg: DictConfig):
    initialize(cfg)
    if cfg.name is not None and cfg.name != "":
        logger.info("Running training tagged with '%s'", cfg.name)

    hydra_cfg = HydraConfig.get()
    logger.info(f"overrides:\n{OmegaConf.to_yaml(hydra_cfg.overrides)}")

    if not OmegaConf.is_missing(hydra_cfg.job, "num") and hydra_cfg.job.num == 0:
        shutil.copytree(
            join(os.getcwd(), hydra_cfg.output_subdir),
            join(os.getcwd(), "..", hydra_cfg.output_subdir),
            ignore=shutil.ignore_patterns("overrides.yaml"),
        )

    trainer = instantiate(cfg.training.trainer)
    best_macro_f1_score = trainer.train()

    # use macro f1 score as optimization criterion for nevergrad sweeps
    # cast to float, because OmegaConf doesn't like numpy datatypes
    return float(best_macro_f1_score)


if __name__ == "__main__":
    main()
