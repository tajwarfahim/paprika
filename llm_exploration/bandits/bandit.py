import numpy as np
from typing import List, Optional, Union


class Bandit:
    """
    Barebone implementation of the basic bandit class,
    upon which other variations of bandit will be built.

    Example Usage:
        arms_list = ["blue", "red", "green"]
        probability_list = [0.7, 0.3, 0.3]
        T = 10

        bandit_env = Bandit(
            arms_list=arms_list,
            probability_list=probability_list,
        )

        # run an example algorithm here
        _ = run_ucb(
            bandit_env=bandit_env,
            T=T,
        )
    """

    def __init__(
        self,
        arms_list: List[str],
        probability_list: List[float],
    ):
        """
        Input:
            arms_list (List[str]):
                List of arms in text, should be name of colors
                e.g., ["blue", "red", "green"]

            probability_list (List[float]):
                List of probability for the bernoulli distribution
                probability_list[i] -> the mean of the bernoulli distribution for arm_list[i]
                Should have same length as arms_list
        """
        self.arms_list = arms_list
        self.probability_list = probability_list
        self.history = []
        self.values = [0 for i in range(len(self.arms_list))]
        self.counts = [0 for i in range(len(self.arms_list))]
        self.all_rewards = []
        self.default_not_valid_action = -100
        self.not_valid_action_string = "Given action is not valid"

    def reset(
        self,
        arms_list: Optional[List[str]] = None,
        probability_list: Optional[List[float]] = None,
    ) -> None:
        """
        Resets the bandit environment with a different list of arms, probability and T.
        If any of the arguments is None, we do not reset it.

        Input:
            arms_list (List[str]):
                List of arms in text, should be name of colors
                e.g., ["blue", "red", "green"]
                Defaults to None

            probability_list (List[float]):
                List of probability for the bernoulli distribution
                probability_list[i] -> the mean of the bernoulli distribution for arm_list[i]
                Should have same length as arms_list
                Defaults to None

        Output:
            None
        """
        if arms_list is not None:
            self.arms_list = arms_list

        if probability_list is not None:
            self.probability_list = probability_list

        self.history = []
        self.values = [0 for i in range(len(self.arms_list))]
        self.counts = [0 for i in range(len(self.arms_list))]
        self.all_rewards = []

    def step(self, arm: int) -> Union[float, int]:
        """
        Takes one step in the environment, and returns the corresponding reward.
        Takes an integer index to the arms_list, and not a text version of the arm.
        See step_text_action() function for the other version.

        Input:
            arm (int):
                Index of the arms_list, i.e., the arm picked

        Output:
            reward (float or int):
                The reward obtained from this step, typically sampled from the
                independent bernoulli distribution associated with the chosen arm.

                This is typically either 0 or 1,
                but we leave the option of it being other real numbers as well.
        """
        if arm != self.default_not_valid_action:
            self.history.append(self.arms_list[arm])

            prob = self.probability_list[arm]
            if prob == 0:
                return 0
            elif prob == 1:
                return 1.0

            random_num = np.random.uniform(low=0.0, high=1.0, size=None)
            reward = 1.0 if random_num < prob else 0.0

            self.values[arm] += reward
            self.counts[arm] += 1

        else:
            self.history.append(self.not_valid_action_string)
            reward = 0.0

        self.all_rewards.append(reward)
        return reward

    def get_rewards(self) -> List[float]:
        """
        Returns a list of rewards per timestep.

        Input:
            None

        Output:
            self.all_rewards
        """
        return self.all_rewards

    def get_probability_list(self) -> List[float]:
        """
        Returns a list, probability_list, where
            probability_list[i] = mean of the Bernoulli distribution for self.arms_list[i]

        Input:
            None

        Output:
            self.probability_list
        """
        return self.probability_list
