from logging import getLogger

import numpy as np
import torch
from torch import nn

from base.config import Config
from base.evaluation.base_evaluator import BaseEvaluator
from base.results.base_result_tracker import BaseResultTracker

logger = getLogger(__name__)

# Value that marks artifacts in labels (excluded from evaluation)
ARTEFACT: int = 9


class ClasEvaluator(BaseEvaluator):
    """
    Evaluator for sleep stage classification models.

    Runs inference on the configured dataset and computes:
    - Cross-entropy loss (excluding artifact labels)
    - Per-epoch sleep stage predictions

    Results are passed to configured result trackers for metric computation
    and logging.

    Args:
        name (str): Evaluator name (e.g., 'validation', 'test').
        dataloader (DataLoader): Evaluation dataset loader.
        result_tracker (dict): Tracker name to BaseResultTracker mapping.

    Note:
        Artifacts (label=9) are excluded from both loss computation and
        metric calculation.
    """

    def __init__(self, name, dataloader, result_tracker: dict[str, BaseResultTracker]):
        super(ClasEvaluator, self).__init__(name, dataloader, result_tracker)

    def evaluate(self, model):
        """
        Evaluate the model on the configured dataset.

        Runs inference with gradients disabled, computes loss and predictions,
        filters artifacts, and passes results to all configured trackers.

        Args:
            model (nn.Module): The model to evaluate. Will be set to eval mode.
        """
        _cfg = Config.get()

        model.eval()

        num_of_epochs = self.dataloader.dataset.num_of_epochs
        predictions = torch.empty(num_of_epochs, dtype=torch.int8)
        actual_labels = torch.empty(num_of_epochs, dtype=torch.int8)
        s_ids = []

        loss: torch.Tensor = torch.zeros(1).to(_cfg.general.device)

        current = 0
        with torch.no_grad():
            for idx, data in enumerate(self.dataloader):
                # Data loading and transformation
                features, labels, name = data

                features = features.to(_cfg.general.device)
                labels = labels.flatten().to(_cfg.general.device)

                # Transformation of predicted data
                outputs = model(features)
                outputs = outputs.reshape(-1, outputs.size(-1))

                # Loss calculation
                if not torch.all(labels == ARTEFACT):
                    criterion = nn.CrossEntropyLoss()
                    loss += criterion(
                        outputs[labels != ARTEFACT], labels[labels != ARTEFACT]
                    )

                # Prediction transformation
                pred = torch.softmax(outputs, dim=1).argmax(dim=1).cpu()

                predictions[current : current + len(pred)] = pred
                actual_labels[current : current + len(labels)] = labels.cpu()
                s_ids.extend(
                    np.repeat([name[0].split("#")], len(pred), axis=0).tolist()
                )
                current += len(labels)
        predictions = predictions.numpy()
        actual_labels = actual_labels.numpy()

        # Filter and save
        artefacts_mask = actual_labels != ARTEFACT
        actual_labels = actual_labels[artefacts_mask]
        predictions = predictions[artefacts_mask]
        s_ids = np.array(s_ids)[artefacts_mask]

        # Updating result tracker with loss and metrics
        for t_name, tracker in self.result_tracker.items():
            tracker.track_result(
                f"{self.name}/{t_name}",
                loss.item() / len(self.dataloader),
                predictions,
                actual_labels,
                s_ids,
            )
