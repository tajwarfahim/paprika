from typing import (
    List,
    Dict,
    Optional,
    Any,
)
import time
import numpy as np

from llm_exploration.inference.inference_engine import LLMInferenceEngine


class BanditInferenceEngine(LLMInferenceEngine):
    """
    LLM inference engine to run the LLM bandit game.
    """

    difficulty_to_p_parameter_map = {
        "easy": {
            "low": 0.25,
            "high": 0.75,
        },
        "hard": {
            "low": 0.4,
            "high": 0.6,
        },
    }

    def __init__(
        self,
        difficulty: str,
        randomize_arm_probabilities: bool,
    ):
        """
        Inference engine, that calculates the arm rewards
        using hidden Bernoulli parameters, and lets us simulate the
        bandit game played by LLMs described in this paper:

        https://arxiv.org/pdf/2403.15371

        Input:
            Difficulty (str):
                Difficulty of the setup, options are "easy" and "hard"

                In "easy", one arm has Bernoulli parameter 0.75,
                the other arms have Bernoulli parameter 0.25

                In "hard", one arm has Bernoulli parameter 0.6
                the other arms have Bernoulli parameter 0.4

            randomize_arm_probabilities (bool):
                Whether to select the arm rewards uniformly at random,
                or from fixed values
        """
        self.reset()
        self.difficulty = difficulty
        self.randomize_arm_probabilities = randomize_arm_probabilities

        if self.difficulty not in ["easy", "hard"]:
            raise ValueError(f"Given difficulty {difficulty} not supported.")

    def reset(self) -> None:
        self.steps = 0
        self.rewards = []
        self.arm_names = None
        self.probabilities = None
        super().reset()

    def get_rewards(self) -> Any:
        return {
            "rewards_per_timestep": self.rewards,
            "probability": self.probabilities.tolist(),
        }

    def set_probability_list(self) -> None:
        """
        Sets a list with per arm Bernoulli parameter list
        as the class's internal attribute

        self.probabilities[i] = probability of sampling reward 1
                                (as opposed to reward 0)
                                for arms[i]

        Input:
            None

        Output:
            None
        """
        if self.randomize_arm_probabilities:
            self.probabilities = np.random.uniform(
                low=0.0,
                high=1.0,
                size=len(self.arm_names),
            )

        else:
            random_index = np.random.randint(low=0.0, high=len(self.arm_names))

            low_parameter = self.difficulty_to_p_parameter_map[self.difficulty]["low"]
            high_parameter = self.difficulty_to_p_parameter_map[self.difficulty]["high"]

            self.probabilities = [low_parameter for _ in range(len(self.arm_names))]
            self.probabilities[random_index] = high_parameter

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        time.sleep(0.1)

        if self.steps == 0:
            assert conv[0]["role"] == "system"
            self.arm_names = conv[0]["content"].lower().split(",")
            self.arm_names = [
                self.arm_names[index].strip() for index in range(len(self.arm_names))
            ]
            self.set_probability_list()

        action = conv[-1]["content"].lower().strip()
        self.steps += 1

        try:
            action_index = self.arm_names.index(action)
            random_number = np.random.uniform(
                low=0.0,
                high=1.0,
                size=None,
            )

            reward = 0
            if random_number <= self.probabilities[action_index]:
                reward = 1
            self.rewards.append(reward)

        except:
            return "You have made an invalid choice, please choose again!"

        return f"You have received reward {reward}\n"

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        feedbacks = []

        for index in range(len(convs)):
            feedback = self.generate(
                conv=convs[index],
                max_n_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
            )
            feedbacks.append(feedback)

        return feedbacks
