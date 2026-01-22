import time
from logging import getLogger

import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn

from base.config import Config
from base.evaluation.base_evaluator import BaseEvaluator
from base.model.base_model import BaseModel
from base.results.base_result_tracker import BaseResultTracker

logger = getLogger(__name__)

# Value that marks artifacts in labels (excluded from loss computation)
ARTEFACT: int = 9


class ClasTrainer:
    """
    Training orchestrator for sleep stage classification.

    Manages the training loop including:
    - Model training with cross-entropy loss
    - Validation after each epoch using configured evaluators
    - Early stopping based on macro F1 score
    - Model checkpointing (best model and periodic saves)
    - Learning rate scheduling
    - Gradient clipping

    Args:
        epochs (int): Maximum number of training epochs.
        model (BaseModel): The neural network model to train.
        dataloader (DataLoader): Training data loader.
        log_interval (int): Logging frequency as percentage of epoch.
        clip_gradients (float | bool): Gradient clipping threshold, or False.
        lr (float): Learning rate.
        train_result_tracker (BaseResultTracker): Tracks training metrics.
        evaluators (dict): Dictionary of evaluators by name.
        early_stopping_epochs (int): Stop if no improvement for this many epochs.
        save_every_epoch (bool | int): Save checkpoint every N epochs, or False.
        seed (int, optional): Random seed for reproducibility.

    Attributes:
        optimizer: PyTorch optimizer (instantiated from config).
        lr_scheduler: Learning rate scheduler (instantiated from config).
    """

    def __init__(
        self,
        epochs,
        model,
        dataloader,
        log_interval,
        clip_gradients,
        lr,
        train_result_tracker: BaseResultTracker,
        evaluators: dict[str, BaseEvaluator],
        early_stopping_epochs,
        save_every_epoch: bool | int = False,
        seed: int = None,
    ):
        self.epochs = epochs
        self.model: BaseModel = model
        self.dataloader = dataloader
        self.log_interval = log_interval
        self.clip_gradients = clip_gradients
        self.lr = lr
        self.train_result_tracker: BaseResultTracker = train_result_tracker
        self.evaluators = evaluators
        self.early_stopping_epochs = early_stopping_epochs
        self.save_every_epoch = save_every_epoch

        _cfg = Config.get()

        self.model.to(_cfg.general.device)
        self.optimizer = instantiate(
            _cfg.training.optimizer,
            _convert_="all",
            params=self.model.get_parameters(self.lr),
        )
        self.lr_scheduler = instantiate(
            _cfg.training.lr_scheduler, optimizer=self.optimizer
        )

        # frequency of logs (every LOG_INTERVAL% of data in dataloader)
        self.log_fr = max(int(self.log_interval / 100.0 * len(self.dataloader)), 1)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def train(self):
        """
        Run the full training loop.

        Trains the model for up to `epochs` epochs, evaluating after each.
        Uses early stopping to terminate training if validation performance
        doesn't improve for `early_stopping_epochs` consecutive epochs.

        The best model (by validation macro F1) is saved automatically.

        Returns:
            float: Best macro F1 score achieved on validation set.
        """
        # save best results for early stopping and snapshot creation of best model
        best_epoch = 0
        best_macro_f1_score = 0

        logger.info(f"metrics before training (epoch 0):")
        for eval_name, evaluator in self.evaluators.items():
            evaluator.evaluate(self.model)

        for epoch in range(1, self.epochs + 1):
            start = time.time()  # measure time each epoch takes

            self.train_epoch(epoch)

            self.lr_scheduler.step()
            for eval_name, evaluator in self.evaluators.items():
                evaluator.evaluate(self.model)

            end = time.time()
            logger.info(
                f"[epoch {epoch:3d}] execution time: {end - start:.2f}s\tmetrics:"
            )

            f1_score_es = (
                self.evaluators["early_stopping"]
                .result_tracker["sleep_stages"]
                .get_latest_es_score()
            )
            if f1_score_es > best_macro_f1_score:
                best_macro_f1_score = f1_score_es
                best_epoch = epoch
                self.model.save()

            # save_every_epoch denotes either a number of epochs or True to save every epoch
            if self.save_every_epoch and epoch % int(self.save_every_epoch) == 0:
                self.model.save(epoch)

            # early stopping, stop training if the f1 score on the early stopping data has not increased
            # over the last x epochs
            if epoch - best_epoch >= self.early_stopping_epochs:
                break

        logger.info("finished training")
        logger.info(
            f"best model on epoch: {best_epoch} \tf1-score: {best_macro_f1_score:.4f}"
        )

        return best_macro_f1_score

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Iterates through all training batches, computing loss and updating
        model weights. Artifacts (label=9) are excluded from loss computation.

        Args:
            epoch (int): Current epoch number (for logging).
        """
        _cfg = Config.get()

        self.model.train()

        num_of_epochs = (
            self.dataloader.dataset.num_of_samples * self.dataloader.dataset.window_size
        )
        # save predicted and actual labels for results
        predictions: torch.Tensor = torch.empty(
            num_of_epochs, dtype=torch.int8, device=_cfg.general.device
        )
        actual_labels: torch.Tensor = torch.empty(
            num_of_epochs, dtype=torch.int8, device=_cfg.general.device
        )
        s_ids = []
        loss: torch.Tensor = torch.zeros(1)

        current = 0
        for i, data in enumerate(self.dataloader, 0):
            # get the inputs; data is a list of [inputs, labels, s_ids]
            features, labels, s_id = data
            features = features.to(_cfg.general.device)

            labels = labels.long().flatten().to(_cfg.general.device)

            # skip batches with only one sample, because otherwise the BN layers do not work
            if features.shape[0] == 1:
                continue

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(features)
            outputs = outputs.reshape(-1, outputs.size(-1))

            if not torch.all(labels == ARTEFACT):

                criterion = nn.CrossEntropyLoss()
                loss = criterion(
                    outputs[labels != ARTEFACT], labels[labels != ARTEFACT]
                )  # ignore artefacts (artefact)
                loss.backward()
                if self.clip_gradients:
                    # clip the gradients to avoid exploding gradients
                    if isinstance(self.clip_gradients, float):
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_gradients, "inf"
                        )
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 0.1, "inf")
                self.optimizer.step()

            # determine predicted labels and save them together with actual labels
            _, predicted_labels_i = torch.max(outputs, dim=1)

            predictions[current : current + len(predicted_labels_i)] = (
                predicted_labels_i.detach()
            )
            actual_labels[current : current + len(predicted_labels_i)] = labels
            s_ids.extend(
                np.repeat(
                    [s.split("#") for s in s_id],
                    self.dataloader.dataset.window_size,
                    axis=0,
                ).tolist()
            )
            current += len(predicted_labels_i)

            # log various information every log_fr minibatches
            if i % self.log_fr == self.log_fr - 1:
                logger.info(
                    f"train epoch: {epoch} [{i * len(features)}/{len(self.dataloader.dataset)} "
                    f"({100. * i / len(self.dataloader):.0f}%)], "
                    f'lr: {[f"{lr:.2e}" for lr in self.lr_scheduler.get_last_lr()]}, '
                    f"loss: {loss.item():.6f}"
                )

        artefacts_mask = actual_labels != ARTEFACT
        actual_labels = actual_labels[artefacts_mask].cpu()
        predictions = predictions[artefacts_mask].cpu()
        s_ids = np.array(s_ids)[artefacts_mask.cpu()]

        self.train_result_tracker.track_result(
            "train",
            loss.item(),
            predictions.numpy(),
            actual_labels.numpy(),
            s_ids,
        )
