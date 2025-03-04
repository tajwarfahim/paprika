import math
import numpy as np
from typing import List, Union, Optional
from abc import ABC, abstractmethod

from llm_exploration.bandits.bandit import Bandit


class BanditAlgorithm(ABC):
    """
    Class Template for running bandit algorithms such as
    UCB, Thompson sampling or epsilon-greedy.
    """

    def __init__(
        self,
        num_arms: int,
    ):
        """
        Instantiates an object of this class,
        used to run a particular algorithm.

        Input:
            num_arms (int):
                number of arms in the particular bandit problem.

                Note: the reset() method can be used to reset this.

        """
        self.num_arms = num_arms
        self.counts = [0 for _ in range(self.num_arms)]
        self.values = [0 for _ in range(self.num_arms)]

    def reset(
        self,
        num_arms: Optional[int] = None,
    ) -> None:
        """
        Reset the algorithms internal representation/states.

        Input:
            num_arms (int):
                number of arms in the particular bandit problem.
                Default: None, in which case it is not changed
        """
        if num_arms is not None:
            self.num_arms = num_arms

        self.counts = [0 for _ in range(self.num_arms)]
        self.values = [0 for _ in range(self.num_arms)]

    @abstractmethod
    def select_arm(self) -> int:
        """
        Template method, used to choose a particular arm for this timestep
        """
        raise NotImplementedError

    def run_algorithm(
        self,
        bandit_env: Bandit,
        T: int,
    ) -> List[Union[float, int]]:
        """
        Runs the UCB algorithm on a given bandit environment for
        horizon T.

        Input:
            bandit_env (Bandit):
                Environment to run UCB on

            T (int):
                Number of time steps to run UCB on

        Output:
            rewards (List of float or int):
                rewards[i] is the reward obtained at time step i.
        """
        rewards = []

        for _ in range(T):
            selected_arm = self.select_arm()
            reward = bandit_env.step(arm=selected_arm)

            self._update_params(
                selected_arm=selected_arm,
                reward=reward,
            )

            rewards.append(reward)

        return rewards

    @abstractmethod
    def _update_params(
        self,
        selected_arm: int,
        reward: Union[int, float],
    ) -> None:
        """
        Updates the internal parameters used by the algorithm.
        This is algorithm specific.

        Input:
            selected_arm (int):
                the index of arms_list, that was chosen in this time step

            reward (int or float):
                reward obtained in this time step
        """
        raise NotImplementedError


class UCB(BanditAlgorithm):
    def __init__(
        self,
        num_arms: int,
        alpha: float = 0.1,
    ):
        """
        Instantiates an object to run UCB algorithm.

        Input:
            num_arms (int):
                number of arms in the particular bandit problem.

                Note: the reset() method can be used to reset this.

            alpha (float):
                The parameter that controls exploration/exploitation tradeoff
                alpha is used for the exploration bonus:

                effective average reward for arms_list[i]
                    = average reward for arms_list[i] + alpha * sqrt{2 * log(t) / N_i}

                where
                    t = number of time steps elapsed
                    N_i = within t timesteps, how many times arm_list[i] was chosen

                higher alpha ---> more exploration
                lower alpha ---> more exploitation

                Note: the reset() method can be used to reset this.
        """
        super().__init__(num_arms=num_arms)
        self.alpha = alpha

    def reset(
        self,
        num_arms: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """
        Reset the algorithms internal representation/states.

        Input:
            num_arms (int):
                number of arms in the particular bandit problem.
                Default: None, in which case it is not changed

            alpha (float):
                The parameter that controls exploration/exploitation tradeoff
                Default: None, in which case it is not changed
        """
        super().reset(num_arms=num_arms)
        if alpha is not None:
            self.alpha = alpha

    def select_arm(self) -> int:
        for arm in range(len(self.counts)):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for _ in range(len(self.counts))]
        total_counts = sum(self.counts)

        for arm in range(len(self.counts)):
            bonus = self.alpha * math.sqrt(
                (2 * math.log(total_counts)) / float(self.counts[arm])
            )
            ucb_values[arm] = (self.values[arm] / float(self.counts[arm])) + bonus

        return np.argmax(ucb_values)

    def _update_params(
        self,
        selected_arm: int,
        reward: Union[int, float],
    ) -> None:
        self.counts[selected_arm] += 1
        self.values[selected_arm] += reward


class EpsilonGreedy(BanditAlgorithm):
    def __init__(
        self,
        num_arms: int,
        epsilon: float,
    ):
        """
        Instantiates an object to run epsilon-greedy algorithm.

        Input:
            num_arms (int):
                number of arms in the particular bandit problem.

                Note: the reset() method can be used to reset this.

            epsilon (float):
                The parameter that controls exploration/exploitation tradeoff

                With probability epsilon, the algorithm chooses a random arm
                with probability 1 - epsilon, the algorithm chooses the best arm so far.

                higher epsilon ---> more exploration
                lower epsilon ---> more exploitation

                Note: the reset() method can be used to reset this.
        """
        super().__init__(num_arms=num_arms)
        self.epsilon = epsilon

    def reset(
        self,
        num_arms: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """
        Reset the algorithms internal representation/states.

        Input:
            num_arms (int):
                number of arms in the particular bandit problem.
                Default: None, in which case it is not changed

            epsilon (float):
                The parameter that controls exploration/exploitation tradeoff
                Default: None, in which case it is not changed
        """
        super().reset(num_arms=num_arms)
        if epsilon is not None:
            self.epsilon = epsilon

    def select_arm(self) -> int:
        random_num = np.random.uniform(low=0.0, high=1.0, size=None)
        if random_num > 1.0 - self.epsilon:
            return np.argmax(self.values)
        else:
            return np.random.choice(a=len(self.counts))

    def _update_params(
        self,
        selected_arm: int,
        reward: Union[int, float],
    ) -> None:
        self.counts[selected_arm] += 1
        self.values[selected_arm] += reward


class ThompsonSampling(BanditAlgorithm):
    def __init__(
        self,
        num_arms: int,
    ):
        """
        Instantiates an object of this class,
        used to run a particular algorithm.

        Input:
            num_arms (int):
                number of arms in the particular bandit problem.

                Note: the reset() method can be used to reset this.

        """
        super().__init__(num_arms=num_arms)
        self.alphas = [1.0 for _ in range(self.num_arms)]
        self.betas = [1.0 for _ in range(self.num_arms)]

    def reset(
        self,
        num_arms: Optional[int] = None,
    ) -> None:
        """
        Reset the algorithms internal representation/states.

        Input:
            num_arms (int):
                number of arms in the particular bandit problem.
                Default: None, in which case it is not changed
        """
        super().__init__(num_arms=num_arms)
        self.alphas = [1.0 for _ in range(self.num_arms)]
        self.betas = [1.0 for _ in range(self.num_arms)]

    def select_arm(self) -> int:
        thetas = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(self.num_arms)]

        return np.argmax(thetas)

    def _update_params(
        self,
        selected_arm: int,
        reward: Union[int, float],
    ) -> None:
        self.alphas[selected_arm] += reward
        self.betas[selected_arm] += 1.0 - reward
