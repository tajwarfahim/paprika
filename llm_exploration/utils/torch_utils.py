import numpy as np
import torch
import random


def set_seed_everywhere(
    seed: int,
) -> None:
    """
    Given a seed, sets it to be the seed for numpy, torch and random.

    Input:
        seed (int):
            The selected seed

    Output:
        None
    """
    assert isinstance(seed, int) and seed >= 0

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
