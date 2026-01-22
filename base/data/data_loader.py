import numpy as np
import torch
from torch.utils.data import DataLoader


def random_worker_init_fn(worker_id):
    """
    Initialize random state for each DataLoader worker process.

    This function is called when a worker process starts. It sets up the
    dataset's random state based on the worker's seed to ensure reproducible
    but different random sampling across workers.

    Args:
        worker_id (int): The worker process ID (0 to num_workers-1).
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = worker_info.seed
    worker_seed = worker_seed if worker_seed is None else worker_seed % (2**32 - 1)
    dataset = worker_info.dataset
    dataset._random_state = np.random.RandomState(worker_seed)


class CustomDataloader(DataLoader):
    """
    PyTorch DataLoader with reproducible random sampling.

    This class extends the standard DataLoader to ensure that copies of the dataloader in
    different workers have their own random seed. Otherwise, each worker would produce the
    same random datapoints, effectively reducing the dataset size.

    The seed controls:
    - The PyTorch generator for batch shuffling
    - Each worker's numpy random state for data sampling

    Args:
        seed (int, optional): Random seed for reproducibility. If None, a random
            seed is generated.
        *args: Positional arguments passed to DataLoader.
        **kwargs: Keyword arguments passed to DataLoader. Must include 'dataset'.
    """

    def __init__(self, seed, *args, **kwargs):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        if kwargs.get("num_workers", 0) == 0:
            # set random state here, since no workers are spawned and random_worker_init_fn is not called
            kwargs["dataset"]._random_state = np.random.RandomState(seed)

        super().__init__(
            *args,
            **kwargs,
            worker_init_fn=random_worker_init_fn,
            generator=torch.Generator().manual_seed(seed)
        )
