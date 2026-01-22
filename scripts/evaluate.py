"""
Evaluate a trained sleep stage classification model.

This script loads a pre-trained model and evaluates it on validation or test
datasets, computing per-class and aggregate metrics.

Usage: see run.md in config folder

The script:
1. Loads the trained model specified in the config (model.path)
2. Runs evaluation on all configured evaluators (validation, test, etc.)
3. Computes metrics: per-class precision, recall, F1; macro F1; confusion matrices
4. Saves results to JSON files in the log directory

Args (Hydra overrides):
    -cn: Config name (e.g., exp001/exp001a)
    +training.trainer.evaluators.test: Add test evaluator (not included by default)
    Any config parameter can be overridden with key=value syntax

Output files:
    - logs/<config>/<timestamp>/results_<evaluator>.json: Evaluation metrics
    - logs/<config>/<timestamp>/*.log: Evaluation logs
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
    # ensure that general.async_results is False, otherwise the evaluator will fail
    cfg.general.async_results = False
    Config.initialize(cfg)
    return cfg


@hydra.main(config_path="../config", version_base="1.2")
def main(cfg: DictConfig):
    cfg = initialize(cfg)
    hydra_cfg = HydraConfig.get()
    logger.info(f"overrides:\n{OmegaConf.to_yaml(hydra_cfg.overrides)}")

    if not OmegaConf.is_missing(hydra_cfg.job, "num") and hydra_cfg.job.num == 0:
        shutil.copytree(
            join(os.getcwd(), hydra_cfg.output_subdir),
            join(os.getcwd(), "..", hydra_cfg.output_subdir),
            ignore=shutil.ignore_patterns("overrides.yaml"),
        )

    model = instantiate(cfg.model)
    model.eval()
    model.to(cfg.general.device)
    for eval_name, evaluator in cfg.training.trainer.evaluators.items():
        logger.info(f'Evaluator "{eval_name}"')
        evaluator_inst = instantiate(evaluator)
        evaluator_inst.evaluate(model)


if __name__ == "__main__":
    main()
