from base.results.base_result_tracker import BaseResultTracker


class BaseEvaluator:
    """
    Abstract base class for model evaluators.

    Evaluators run inference on a dataset and track metrics using
    result trackers. Concrete implementations should override the
    evaluate() method.

    Args:
        name (str): Name of this evaluator (e.g., 'validation', 'test').
        dataloader (DataLoader): Data loader for evaluation data.
        result_tracker (dict): Dictionary mapping tracker names to
            BaseResultTracker instances.

    Attributes:
        name (str): Evaluator name, used in logging and result keys.
        dataloader (DataLoader): Evaluation data source.
        result_tracker (dict): Metric trackers.
    """

    def __init__(self, name, dataloader, result_tracker: dict[str, BaseResultTracker]):
        self.name = name
        self.dataloader = dataloader
        self.result_tracker = result_tracker

    def evaluate(self, model):
        """
        Evaluate the model on this evaluator's dataset.

        Args:
            model (nn.Module): The model to evaluate.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
