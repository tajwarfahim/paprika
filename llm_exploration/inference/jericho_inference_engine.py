"""
This file implements the inference engine
that is used to let LLMs play games in the Jericho environments.

Link: https://github.com/microsoft/jericho
Paper link: https://arxiv.org/abs/1909.05398

Jericho contains multiple Interactive Fiction (IF) games.

NOTE: This file is incomplete and does not implement the game correctly.
Should not be used without fixing, including implementing a mechanism
to check if the game has finished etc.
"""

from typing import (
    List,
    Dict,
    Optional,
)
import jericho
import os

from llm_exploration.constants import JERICHO_GAMES_DIR
from llm_exploration.inference.inference_engine import LLMInferenceEngine


GAME_ENV_PATHS = {
    "detective": os.path.join(JERICHO_GAMES_DIR, "detective.z5"),
}


class JerichoInferenceEngine(LLMInferenceEngine):
    """
    Inference engine to run the Jericho interactive fiction (IF) games

    Link: https://github.com/microsoft/jericho
    Paper link: https://arxiv.org/abs/1909.05398

    NOTE: This file is incomplete and does not implement the game correctly.
    Should not be used without fixing, including implementing a mechanism
    to check if the game has finished etc.
    """

    def __init__(
        self,
        env_name: str,
    ):
        """
        Initializes the inference engine for Jericho interactive fiction games
        We took inspiration from the ArCHer codebase to design this part of
        our codebase. Specifically, the following file:
        https://github.com/YifeiZhou02/ArCHer/blob/master/archer/environment/adventure_env.py

        Input:
            env_name (str):
                Name of the Jericho environment to load
        """
        if env_name not in GAME_ENV_PATHS:
            raise ValueError(f"Given env_name {env_name} is not supported yet.")

        self.env_load_path = GAME_ENV_PATHS[env_name]
        self.env = jericho.FrotzEnv(self.env_load_path)
        self.steps = 0
        self.done = False
        self.reward = None

        ###
        # NOTE: remove this after fixing the env
        ###
        raise ValueError(f"The Jericho inference engine is not properly set up yet!")

    def get_surroundings(self) -> str:
        """
        Returns the surroundings in the Jericho game

        Input:
            None

        Output:
            obs (str):
                A description of the surroundings in the game,
                in string format
        """
        bkp = self.env.get_state()
        obs, _, _, _ = self.env.step("look")
        self.env.set_state(bkp)

        return obs

    def get_inventory(self) -> str:
        """
        Returns the player inventory in the Jericho game

        Input:
            None

        Output:
            obs (str):
                A description of the player inventory in the game,
                in string format
        """
        bkp = self.env.get_state()
        obs, _, _, _ = self.env.step("inventory")
        self.env.set_state(bkp)

        return obs

    def get_game_observation(self, obs: str) -> str:
        """
        Returns the observation from the game environment,
        that will then be passed to the LLM agent that is playing the game.

        Input:
            obs (str):
                Observation string from the game

        Output:
            game_obs (str):
                Complete observation string, that is to be used by the
                player playing the game
        """
        surrounding = self.get_surroundings()
        inventory = self.get_inventory()
        available_actions_list = self.env.get_valid_actions()
        available_actions = "\nYou have the following valid actions: "
        available_actions += ", ".join(available_actions_list)

        return surrounding + inventory + obs + available_actions

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        if self.steps == 0:
            self.steps += 1
            return self.get_game_observation(obs="")

        assert conv[-1]["role"] == "user"
        action = conv[-1]["content"].split("\n")[0]

        self.steps += 1
        obs, reward, done, _ = self.env.step(action)
        self.done = done
        self.reward = reward

        return self.get_game_observation(obs=obs)

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        game_observations = []

        for index in range(len(convs)):
            game_observation = self.generate(
                conv=convs[index],
                max_n_tokens=max_n_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
            )
            game_observations.append(game_observation)

        return game_observations

    def reset(self) -> None:
        self.env.reset()
        super().reset()
