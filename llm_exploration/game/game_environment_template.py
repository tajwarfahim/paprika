from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable


class GameEnvironmentTemplate(ABC):
    """
    Template Class for Game Environments.

    Defines the methods that each inherited class should implement,
    plus any common method that all subclasses should be able to use.

    Each game will be played between an environment (usually an LLM),
    and an agent/player (usually also an LLM).

    However, the instructions for the environment and agent will differ
    based on which game we are playing, and this class will handle that.
    """

    @abstractmethod
    def get_env_message(self, config: Dict[str, Any]) -> str:
        """
        Given a config, it generates the message for this particular game,
        that the environment module in the game shall receive.

        Input:
            config (Dict[str, Any]):
                config file for the environment message

        Output:
            env_message (str):
                The first message that the environment will receive
                before starting to play the game.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def get_agent_message(self, config: Dict[str, Any]) -> str:
        """
        Given a config, it generates the message for this particular game,
        that the agent module in the game shall receive.

        Input:
            config (Dict[str, Any]):
                config file for the agent message

        Output:
            agent_message (str):
                The first message that the agent will receive
                before starting to play the game.
        """
        raise NotImplementedError

    def get_environment_default_response(self) -> Optional[str]:
        """
        Some environments may have a default response
        (eg., sorry I cannot answer questions that are not yes/no...)
        for special cases such as the agent action being invalid.

        Input:
            None

        Output:
            default_response (str):
                Default response for this environment
                Default: None, individual game environments should implement this
                    if they want to provide one
        """
        return None

    def get_enviroment_response_extractor(self) -> Optional[Callable[[str], str]]:
        """
        For a particular game, we often need to extract the response from the
        LLM generated environment response.

        This is often done to prevent environment hacking/force some constraints on
        the environment response, such as to make it constrained to "Yes" or "No"

        This function returns a function that will extract the environment answer
        from the entire environment response

        Input:
            None

        Output:
            response_extractor (Callable[[str], str]):
                A function, that takes environment_response (str)
                and returns the extracted response

                Default: None, individual game environments should implement this
                if they want to provide one
        """
        return None

    def get_env_optional_message(self, config: Dict[str, Any]) -> Optional[str]:
        """
        In some environments, we might want to add a suffix
        to the agent action, before passing it to the game environment
        (eg., "Answer only in Yes/No") to prevent environment hacking.

        Input:
            None

        Output:
            env_optional_message (str):
                Optional message to be passed to the environment at every turn
                Default: None, individual game environments should implement this
                if they want to provide one
        """
        return None

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
        raise NotImplementedError

    def get_verifier_input_generator(
        self,
    ) -> Optional[Callable]:
        """
        For some environments, we want to verify the game conversation
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

                Default: None, individual game environments should implement this
                if they want to provide one
        """
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

        Output:
            judge_prompt (str):
                System prompt for the LLM judge

                Default: None, individual game environments should implement this
                if they want to provide one
        """
        return None

    def get_judge_prompt_env(self, config: Dict[str, Any]) -> Optional[str]:
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
        return None

    def get_agent_optional_message(self, config: Dict[str, Any]) -> Optional[str]:
        """
        In some environments, we might want to add a suffix
        to the environment response, before passing it to the agent
        (eg., "Think step-by-step, answer in this format").

        Input:
            None

        Output:
            agent_optional_message (str):
                Optional message to be passed to the agent every turn
                Default: None, individual game environments should implement this
                if they want to provide one
        """
        return None

    def get_agent_response_extractor(self) -> Optional[Callable[[str], str]]:
        """
        For a particular game, we often need to extract the response from the
        LLM generated agent action.

        This is often done to separate out the actual agent action from the chain
        of thought (COT)

        This function returns a function that will extract the agent action
        from the agent response.

        Input:
            None

        Output:
            response_extractor (Callable[[str], str]):
                A function, that takes agent_response (str)
                and returns the extracted response

                Default: None, individual game environments should implement this
                if they want to provide one
        """
        return None
