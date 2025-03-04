from typing import Any, Dict, Optional, Callable, List
import os
import re

from llm_exploration.constants import GAME_CONFIFS_DIR
from llm_exploration.utils.data_utils import read_json
from llm_exploration.game.game_environment_template import GameEnvironmentTemplate


ENVIRONMENT_NAME_TO_CONFIG_PATH = {
    "minesweeper": "minesweeper.json",
    "twenty_questions_cot": "twenty_questions_cot.json",
    "twenty_questions": "twenty_questions.json",
    "guess_my_city": "guess_my_city.json",
    "wordle": "wordle.json",
    "customer_service": "customer_service.json",
    "murder_mystery": "murder_mystery.json",
    "cellular_automata": "cellular_automata.json",
    "jericho": "jericho.json",
    "bandit": "bandit.json",
    "bandit_best_arm_identification_fixed_sampling_budget": (
        "bandit_best_arm_identification_fixed_sampling_budget.json"
    ),
    "mastermind": "mastermind.json",
    "wordle_modified": "wordle_modified.json",
    "battleship": "battleship.json",
}

DEFAULT_MAX_TURNS = 20


class GameEnvironment(GameEnvironmentTemplate):
    """
    Class for Game Environments.

    Each game will be played between an environment (usually an LLM),
    and an agent/player (usually also an LLM).

    However, the instructions for the environment and agent will differ
    based on which game we are playing, and this class will handle that.
    """

    def __init__(self, env_name: str):
        """
        Initializes a game environment for 20 questions.

        Input:
            env_name (str):
                The name of the environment
        """
        game_config_path = self.get_game_config_path(env_name=env_name)

        self.env_name = env_name
        self.game_config: Dict[str, Any] = read_json(game_config_path)
        self.agent_first_message: str = self.game_config["agent"].strip()
        self.env_first_message: str = self.game_config["env"].strip()
        self.judge_prompt_agent: Optional[str] = (
            self.game_config["judge_prompt_agent"].strip()
            if isinstance(self.game_config["judge_prompt_agent"], str)
            else None
        )
        self.judge_prompt_env: Optional[str] = (
            self.game_config["judge_prompt_env"].strip()
            if isinstance(self.game_config["judge_prompt_env"], str)
            else None
        )

    def get_game_config_path(self, env_name: str) -> str:
        """
        Returns the path to the twenty questions config file

        Input:
            env_name (str):
                The name of the environment

        output:
            path_to_config (str):
                Path to the config, which is always calculated with respect to
                the GAME_CONFIFS_DIR constant
        """
        path_to_game_config = ENVIRONMENT_NAME_TO_CONFIG_PATH.get(env_name)
        if path_to_game_config is None:
            raise ValueError(f"Given environment {env_name} is not supported yet.")

        return os.path.join(GAME_CONFIFS_DIR, path_to_game_config)

    def get_game_max_turns(self) -> int:
        """
        Games will typically have a threshold on the number of interactions between
        the agent and the environment for a single gameplay,
        for example, it is 20 for twenty questions, 6 for Wordle.

        This function returns it.

        Input:
            None

        Output:
            max_turns (int):
                The maximum number of turns in a single game play, as specified
                by the game config
        """
        return self.game_config.get("max_turns", DEFAULT_MAX_TURNS)

    def get_game_scenarios(self, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        A single game can have multiple scenarios.
        For example, for the twenty questions games, there is a list of questions/topics
        that the game can be played about.

        This function will retrive this list and return it.

        Input:
            config (Dict[str, Any]):
                The config for retrieving the data. Usually would contain
                "data_type" (e.g., train, eval),
                and "data_subtype" (e.g., "easy", "medium", "hard")

        Output:
            game_scenarios (List[Dict[str, str]]):
                A list of game scenarios. It looks like below:
                [
                    {
                        "agent": <Scenario for the agent>,
                        "env": <Scenario for the env>,
                    }, # First scenario
                    {
                        "agent": <Scenario for the agent>,
                        "env": <Scenario for the env>,
                    }, # Second scenario
                    ...
                ]

        """
        data_type = config["data_type"]
        data_subtype = config.get("data_subtype")

        game_scenarios = self.game_config[data_type]
        if data_subtype is not None:
            game_scenarios = game_scenarios[data_subtype]

        assert isinstance(game_scenarios, list)
        for i in range(len(game_scenarios)):
            assert isinstance(game_scenarios[i], dict)
            assert isinstance(game_scenarios[i].get("agent"), str)
            assert isinstance(game_scenarios[i].get("env"), str)

        return game_scenarios

    def get_env_message(self, config: Dict[str, Any]) -> str:
        """
        Returns the first message that environment LLM should receive
        at the start of the game.

        Input:
            config (Dict[str, Any]):
                config from the game environment.
                Must have "env_input" as a key

        Output:
            env_first_message (str):
                First message for the environment LLM
        """
        return self.env_first_message.format(
            env=config["env_input"],
            agent=config["agent_input"],
        ).strip()

    def get_env_optional_message(self, config: Dict[str, Any]) -> Optional[str]:
        """
        The suffix we wat to add
        to the agent action, before passing it to the game environment
        (eg., "Answer only in Yes/No") to prevent environment hacking.

        Input:
            None

        Output:
            env_optional_message (str):
                Optional message to be passed to the environment at every turn
        """
        env_optional_message: str = self.game_config["env_optional_message"]
        return env_optional_message.format(
            env=config["env_input"],
            agent=config["agent_input"],
        )

    def get_agent_message(self, config: Dict[str, Any]) -> str:
        """
        Returns the first message that agent LLM should receive
        at the start of the game.

        Input:
            config (Dict[str, Any]):
                config from the game environment.
                Must have "agent_input" as a key

        Output:
            agent_first_message (str):
                First message for the agent LLM
        """
        return self.agent_first_message.format(
            agent=config["agent_input"],
            env=config["env_input"],
        ).strip()

    def get_environment_default_response(self) -> Optional[str]:
        """
        Default response (eg., sorry I cannot answer questions that are not yes/no...)
        for special cases such as the agent action being invalid

        Input:
            None

        Output:
            default_response (str):
                Default response for this environment
        """
        return self.game_config["environment_default_response"].strip()

    def get_enviroment_response_extractor(self) -> Optional[Callable[[str], str]]:
        """
        For 20 questions, we need to extract the response from the
        LLM generated environment response.

        This is done to prevent environment hacking/force some constraints on
        the environment response, such as to make it constrained to
        "Yes", "No" or "Goal reached"

        Input:
            None

        Output:
            response_extractor (Callable[[str], str]):
                A function, that takes environment_response (str)
                and returns the extracted response
        """
        if self.env_name == "twenty_questions":

            def extract_env_info_twenty_questions(response: str) -> str:
                """
                Function that extracts the info from env response

                Input:
                    response (str):
                        Given response

                Output:
                    extracted_response (str):
                        The extracted output from the response,
                        restricted to 'Yes', 'No' and 'Goal reached'
                """
                response = response.lower()
                if response == "yes" or response == "yes.":
                    return "Yes"
                elif response == "no" or response == "no.":
                    return "No"
                elif response == "goal reached" or response == "goal reached.":
                    return "Goal reached"
                else:
                    raise ValueError(f"Given response {response} not valid.")

            return lambda env_response: extract_env_info_twenty_questions(env_response)

        else:
            return None

    def get_judge_prompt_agent(self, config: Dict[str, Any]) -> str:
        """
        The system prompt for the LLM judge, in case we want to verify the game conversation
        at the end, by a separate judge, in order to filter away
        false positives, i.e., trajectories where the agent has not reached
        the goal, but the environment erroneously says so.

        Input:
            config (Dict[str, Any]):
                config for the particular game scenario, used to generate
                the judge prompt

                Needs to contain "env_input" as a key, containing the topic for the
                particular 20 questions game instance

        Output:
            judge_prompt (str):
                System prompt for LLM judge
        """
        if self.judge_prompt_agent is not None:
            return self.judge_prompt_agent.format(
                env=config["env_input"],
                agent=config["agent_input"],
            )

        else:
            return None

    def get_judge_prompt_env(self, config: Dict[str, Any]) -> str:
        """
        The system prompt for the LLM judge, in case we want to verify if the
        environment has responded following the correct way

        Input:
            config (Dict[str, Any]):
                config for the particular game scenario, used to generate
                the judge prompt

                Needs to contain "env_input" as a key, containing the topic for the
                particular 20 questions game instance

        Output:
            judge_prompt (str):
                System prompt for LLM judge
        """
        if self.judge_prompt_env is not None:
            return self.judge_prompt_env.format(
                env=config["env_input"],
                agent=config["agent_input"],
            )

        else:
            return None

    def get_verifier_input_generator(
        self,
    ) -> Optional[Callable]:
        """
        For 20 questions, we want to verify the game conversation
        at the end, by a separate judge, in order to filter away
        false positives, i.e., trajectories where the agent has not reached
        the goal, but the environment erroneously says so.

        This returns a function, that would produce the output that would
        then be passed to the LLM-based judge (e.g., GPT-4).

        Input:
            None

        Output:
            verifier_input_generator (Callable):
                A function that generates the input for the LLM-judge
        """
        if self.env_name in ["wordle", "wordle_modified"]:

            def get_input_for_verifier_wordle(config: Dict[str, Any]) -> str:
                """
                Function to generate input for LLM-judge

                Input:
                    config (Dict[str, Any]):
                        config for generating the input
                        Must have "conversation" as key, that maps to a trajectory of
                        the following format:

                        [
                            {
                                "role": "system",
                                "content": <system_prompt>,
                            },
                            {
                                "role": "user",
                                "content": <user_prompt>,
                            },
                            {
                                "role": "assistant",
                                "content": <model_response>,
                            },
                            ....
                        ]

                Output:
                    judge_prompt (str):
                        User prompt for the judge, containing
                        part of the conversation that will be judged.
                """
                return config["conversation"][-2]["content"].strip().lower()

            return lambda config: get_input_for_verifier_wordle(config=config)

        elif self.env_name == "minesweeper":

            def get_input_for_verifier_minesweeper(config: Dict[str, Any]) -> str:
                """
                Function to generate input for LLM-judge

                Input:
                    config (Dict[str, Any]):
                        config for generating the input
                        Must have "conversation" as key, that maps to a trajectory of
                        the following format:

                        [
                            {
                                "role": "system",
                                "content": <system_prompt>,
                            },
                            {
                                "role": "user",
                                "content": <user_prompt>,
                            },
                            {
                                "role": "assistant",
                                "content": <model_response>,
                            },
                            ....
                        ]

                Output:
                    judge_prompt (str):
                        User prompt for the judge, containing
                        part of the conversation that will be judged.
                """
                return config["conversation"][-1]["content"].strip()

            return lambda config: get_input_for_verifier_minesweeper(config=config)

        else:

            def get_input_for_verifier(config: Dict[str, Any]) -> str:
                """
                Function to generate input for LLM-judge

                Input:
                    config (Dict[str, Any]):
                        config for generating the input
                        Must have "conversation" as key, that maps to a trajectory of
                        the following format:

                        [
                            {
                                "role": "system",
                                "content": <system_prompt>,
                            },
                            {
                                "role": "user",
                                "content": <user_prompt>,
                            },
                            {
                                "role": "assistant",
                                "content": <model_response>,
                            },
                            ....
                        ]

                Output:
                    judge_prompt (str):
                        User prompt for the judge, containing
                        part of the conversation that will be judged.
                """
                judge_prompt = "The conversation begins here: \n\n"
                trajectory = config["conversation"]

                num_turns_to_judge = min(2, len(trajectory))
                for i in range(len(trajectory) - num_turns_to_judge, len(trajectory)):
                    role = trajectory[i]["role"]
                    message = trajectory[i]["content"]

                    if role == "assistant":
                        judge_prompt += "Agent: "
                        judge_prompt += message
                        judge_prompt += "\n\n"

                    else:
                        # We ignore the system prompt
                        continue

                judge_prompt += " (End of Agent Turn)\n\n"
                judge_prompt += self.game_config["judge_prompt_suffix"].format(
                    env=config["env_game_scenario"],
                    agent=config["agent_game_scenario"],
                )

                return judge_prompt

            return lambda config: get_input_for_verifier(config=config)

    def get_agent_optional_message(self, config: Dict[str, Any]) -> Optional[str]:
        """
        The suffix we want to add
        to the environment response, before passing it to the agent
        (eg., "Think step-by-step, answer in this format").

        Input:
            None

        Output:
            agent_optional_message (str):
                Optional message to be passed to the agent every turn
        """
        agent_optional_message: Optional[str] = self.game_config.get("agent_optional_message")

        if agent_optional_message is None:
            return None

        else:
            return agent_optional_message.format(
                agent=config["agent_input"],
            )

    def get_agent_response_extractor(self) -> Optional[Callable[[str], str]]:
        """
        For 20 questions with chain-of-thought, we want to
        separate the chain-of-thought, from the actual question the agent
        is asking, so that we can only send the question to environment
        and not the chain-of-thought, so as to confuse the environment LLM
        less.

        This returns a function, that would extract the actual question
        from the agent response

        Input:
            None

        Output:
            response_extractor (Callable[[str], str]):
                A function, that takes agent_response (str)
                and returns the extracted response
        """
        if self.env_name in [
            "wordle",
            "wordle_modified",
            "jericho",
            "bandit",
            "bandit_best_arm_identification_fixed_sampling_budget",
            "mastermind",
        ]:

            def extract_agent_info(response: str) -> str:
                """
                Assumes responses are in this shape:
                <Chain-of-thought> <Answer> answer tokens </Answer>

                We ignore the chain-of-thought part, and only return
                what appears between <Answer> and </Answer>

                Input:
                    response (str):
                        The given agent response, expected in a certain
                        format

                Output:
                    extracted_response (str):
                        The answer extracted from the agent response
                """
                match = re.search(r"<Answer>(.*?)</Answer>", response, re.DOTALL)
                if match:
                    return match.group(1).strip().lower()
                else:
                    raise ValueError(f"Given response {response} is invalid")

            return lambda response: extract_agent_info(response=response)

        elif self.env_name == "twenty_questions_cot":

            def extract_agent_info(response: str) -> str:
                """
                Assumes responses are in this shape:
                <Chain-of-thought> <answer> answer tokens </answer>

                We ignore the chain-of-thought part, and only return
                what appears between <answer> and </answer>

                Input:
                    response (str):
                        The given agent response, expected in a certain
                        format

                Output:
                    extracted_response (str):
                        The answer extracted from the agent response
                """
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if match:
                    return match.group(1).strip().lower()
                else:
                    raise ValueError(f"Given response {response} is invalid")

            return lambda response: extract_agent_info(response=response)

        elif self.env_name == "minesweeper":

            def extract_agent_info_minesweeper(response: str) -> str:
                """
                Assumes responses are in this shape:
                <Chain-of-thought> <Answer> answer tokens </Answer>

                We ignore the chain-of-thought part, and only return
                what appears between <Answer> and </Answer>

                Input:
                    response (str):
                        The given agent response, expected in a certain
                        format

                Output:
                    extracted_response (str):
                        The answer extracted from the agent response
                """
                match = re.search(r"<Answer>(.*?)</Answer>", response, re.DOTALL)
                if match:
                    extracted_response = match.group(1).strip().lower()
                    pattern = r"^reveal\s+(\d+)\s+(\d+)$"
                    match = re.search(pattern, extracted_response)
                    if match:
                        return extracted_response

                    else:
                        raise ValueError(f"Given response {response} is invalid")

                else:
                    raise ValueError(f"Given response {response} is invalid")

            return lambda response: extract_agent_info_minesweeper(response=response)

        elif self.env_name == "battleship":

            def extract_agent_info_battleship(response: str) -> str:
                """
                Assumes responses are in this shape:
                <Chain-of-thought> <Answer> Cell to reveal </Answer>

                We ignore the chain-of-thought part, and only return
                what appears between <Answer> and </Answer>

                Input:
                    response (str):
                        The given agent response, expected in a certain
                        format

                Output:
                    extracted_response (str):
                        The answer extracted from the agent response
                """
                match = re.search(r"<Answer>(.*?)</Answer>", response, re.DOTALL)
                if match:
                    extracted_response = match.group(1).strip()
                    pattern = r"^([A-Z])(\d{1,2})$"
                    match = re.match(pattern, extracted_response)

                    if match:
                        return extracted_response

                    else:
                        raise ValueError(f"Given response {response} is invalid")

                else:
                    raise ValueError(f"Given response {response} is invalid")

            return lambda response: extract_agent_info_battleship(response=response)

        elif self.env_name == "cellular_automata":

            def extract_agent_info_cellular_automata(response: str) -> str:
                """
                Assumes responses are in this shape:
                <Chain-of-thought> <Answer> CA Rule</Answer>

                We ignore the chain-of-thought part, and only return
                what appears between <Answer> and </Answer>

                Input:
                    response (str):
                        The given agent response, expected in a certain
                        format

                Output:
                    extracted_response (str):
                        The answer extracted from the agent response
                """
                match = re.search(r"<Answer>(.*?)</Answer>", response, re.DOTALL)
                if match:
                    text = match.group(1).strip().lower()
                    neighborhoods = [
                        "111",
                        "110",
                        "101",
                        "100",
                        "011",
                        "010",
                        "001",
                        "000",
                    ]

                    for neighborhood in neighborhoods:
                        assert neighborhood in text
                        text_part = text.split(f"{neighborhood}:")[-1]

                        assert "</rule>" in text_part
                        next_cell_state = text_part.split("</rule>")[0].strip()

                        assert len(next_cell_state) == 1

                    return text
                else:
                    raise ValueError(
                        f"Given response {response} is invalid for Cellular Automata."
                    )

            return lambda response: extract_agent_info_cellular_automata(response=response)

        else:
            return None
