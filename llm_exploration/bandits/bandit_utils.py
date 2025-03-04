import numpy as np
from typing import List

from llm_exploration.bandits.bandit import Bandit
from llm_exploration.utils.stats_utils import (
    get_cumulative_sum,
)


def get_empirical_regret(
    bandit_env: Bandit,
) -> List[float]:
    """
    Returns regrets per time step, for a single iteration of
    the bandit problem.

    Definition of regret used:
        Regret(i) = regret at time step i
                  = average reward from the best arm
                    - actual average reward until timestep i
                  = mean of the best arm's Bernoulli distribution
                    - actual average reward until timestep i

    Input:
        bandit_env (Bandit):
            Instantiation of the bandit environment
            Must have "probability_list" and "all_rewards" attributes

    Output:
        regrets (List[float]):
            A list where regrets[i] is the regret at time step i,
            calculated using the above formulat
    """
    best_arm_average_reward = np.max(bandit_env.get_probability_list())
    all_rewards = bandit_env.get_rewards()

    upper_bound = [best_arm_average_reward * i for i in range(1, len(all_rewards) + 1)]

    cum_sum_reward = get_cumulative_sum(arr=all_rewards)
    regrets = [upper_bound[i] - cum_sum_reward[i] for i in range(len(all_rewards))]

    return regrets
