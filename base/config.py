"""
Configuration singleton for global access to Hydra configuration.

This module provides a singleton pattern implementation for managing the application's
configuration throughout the codebase. The configuration is typically initialized
once at the start of training/evaluation scripts with a Hydra configuration object.

Example:
    >>> from base.config import Config
    >>> # In your main script, after Hydra initializes:
    >>> Config.initialize(hydra_cfg)
    >>> # Anywhere else in the codebase:
    >>> cfg = Config.get()
    >>> sampling_rate = cfg.data.sampling_rate
"""


class Config:
    """
    Singleton class for managing application-wide configuration.

    This class ensures that there is only one configuration instance throughout
    the application lifetime. It is designed to work with Hydra's configuration
    system but can hold any configuration object.

    The singleton pattern prevents multiple configuration instances from existing,
    which could lead to inconsistent behavior across different parts of the code.

    Attributes:
        __instance: Private class variable holding the singleton instance.

    Raises:
        Exception: If attempting to instantiate directly (use initialize() instead).
        ValueError: If get() is called before initialize().
    """

    __instance = None

    def __init__(self):
        """Private constructor - do not instantiate directly."""
        raise Exception("This class is a singleton! Use Config.initialize() instead.")

    @staticmethod
    def get():
        """
        Retrieve the singleton configuration instance.

        Returns:
            The configuration object that was set via initialize().

        Raises:
            ValueError: If the configuration has not been initialized yet.

        Example:
            >>> cfg = Config.get()
            >>> print(cfg.data.sampling_rate)
            128
        """
        if Config.__instance is None:
            raise ValueError('Config was not initialized. Call Config.initialize() first.')
        return Config.__instance

    @staticmethod
    def initialize(config):
        """
        Initialize the singleton configuration instance.

        This method should be called once at application startup, typically
        in the main training or evaluation script after Hydra loads the config.

        Args:
            config: The configuration object to store (typically a Hydra DictConfig).

        Example:
            >>> import hydra
            >>> @hydra.main(config_path="config", config_name="base_config")
            ... def main(cfg):
            ...     Config.initialize(cfg)
            ...     # Rest of your code can now use Config.get()
        """
        Config.__instance = config
