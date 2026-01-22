from logging import getLogger
from os.path import join

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch import nn

from base.config import Config

logger = getLogger(__name__)


def _weights_init(m):
    """
    Initialize network weights using He (Kaiming) initialization.

    He initialization is designed for networks with ReLU activations and helps
    prevent vanishing/exploding gradients in deep networks.

    Handles the following layer types:
    - nn.Linear: Kaiming normal, bias = 0
    - nn.Conv1d: Kaiming normal
    - nn.BatchNorm1d: weight = 1, bias = 0
    - nn.LSTM / nn.GRU: Kaiming normal for weights, bias = 0

    Args:
        m (nn.Module): The module to initialize.

    Reference:
        He, K., et al. (2015). Delving Deep into Rectifiers: Surpassing
        Human-Level Performance on ImageNet Classification.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for layer_p in m._all_weights:
            for p in layer_p:
                if "weight" in p:
                    nn.init.kaiming_normal_(
                        m.__getattr__(p), mode="fan_out", nonlinearity="relu"
                    )
                if "bias" in p and m.__getattr__(p) is not None:
                    m.__getattr__(p).data.fill_(0)


class BaseModel(nn.Module):
    """
    Abstract base class for all neural network models.

    This class provides common functionality for model saving, loading,
    and initialization. All model architectures should inherit from this class.

    Class Attributes:
        identifier (str): Unique identifier for the model type, used in
            checkpoint filenames. Subclasses should override this.

    Attributes:
        All standard PyTorch nn.Module attributes.

    Example:
        >>> class MyModel(BaseModel):
        ...     identifier = "my_model"
        ...     def __init__(self, seed=None):
        ...         super().__init__(seed)
        ...         self.layers = nn.Sequential(...)
        ...     def forward(self, x):
        ...         return self.layers(x)
    """

    identifier = "base_model"

    def __init__(self, seed: int):
        """
        Initialize the base model with optional random seed.

        Args:
            seed (int, optional): Random seed for reproducibility. If provided,
                sets both PyTorch and NumPy random seeds.
        """
        super(BaseModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def save(self, epoch=None):
        """
        Save model weights to a checkpoint file.

        The filename includes the config name, model identifier, timestamp,
        and optionally the epoch number.

        Args:
            epoch (int, optional): Epoch number to include in filename.
                If None, saves as 'final' checkpoint.
        """
        model_state = {"state_dict": self.state_dict()}
        snapshot_file = self.create_save_file_name(epoch)

        torch.save(model_state, snapshot_file)
        logger.info(f"snapshot saved to {snapshot_file}")

    def create_save_file_name(self, epoch=None):
        """
        Generate a unique filename for saving model checkpoints.

        Format: {config_name}[-e{epoch}][-m{job_num}]-{identifier}-{timestamp}-final.pth

        Args:
            epoch (int, optional): Epoch number to include in filename.

        Returns:
            str: Full path to the checkpoint file.
        """
        _hydra_cfg = HydraConfig.get()
        _cfg = Config.get()

        snapshot_dir = to_absolute_path(_cfg.general.snapshot_dir)
        config_name = _hydra_cfg.job.config_name.split("/")[-1]
        timestamp = _hydra_cfg.run.dir.split("/")[-1]

        save_file_name = f"{config_name}"
        if epoch is not None:
            save_file_name += f"-e{epoch}"
        if not OmegaConf.is_missing(_hydra_cfg.job, "num"):
            save_file_name += f"-m{_hydra_cfg.job.num}"
        save_file_name += f"-{self.identifier}-{timestamp}-final.pth"

        return join(snapshot_dir, save_file_name)

    def load(self, path):
        """
        Load model weights from a checkpoint file.

        Args:
            path (str, optional): Relative path to checkpoint within snapshot_dir.
                If None, skips loading (useful for training from scratch).
        """
        if path is None:
            return

        _cfg = Config.get()
        snapshot_dir = to_absolute_path(_cfg.general.snapshot_dir)
        model_state = torch.load(
            join(snapshot_dir, path), map_location=torch.device(_cfg.general.device)
        )
        self.load_state_dict(model_state["state_dict"])

    def get_parameters(self, lr):
        """
        Get parameter groups for the optimizer.

        This method can be overridden in subclasses to implement different
        learning rates for different parts of the model.

        Args:
            lr (float): Learning rate for all parameters.

        Returns:
            list: List of parameter group dictionaries for the optimizer.
        """
        parameters = [{"params": self.parameters(), "lr": lr}]
        return parameters
