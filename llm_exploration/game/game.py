from fastchat.model import get_conversation_template
from typing import (
    Dict,
    Any,
    Optional,
    Callable,
    Tuple,
    List,
)
import numpy as np

from llm_exploration.inference import LLMInferenceEngine


class GameSimulator:
    """
    Class to simulate one itertation of a game,
    between an agent and an env. Optionally takes
    a judge to check if the game is successfully finished or not.

    Example Usage:
        simulator = GameSimulator(agent=agent, env=env)
        record = simulator.run_one_iteration(...)
    """

    def __init__(
        self,
        agent: LLMInferenceEngine,
        env: LLMInferenceEngine,
        judge: Optional[LLMInferenceEngine] = None,
    ):
        """
        Initializes a game simulator object.

        Input:
            agent (LLMInferenceEngine):
                The agent role-player in this particular game,
                typically a module that hosts a LLM, and allows
                generate() functionality

            env (LLMInferenceEngine):
                The environment role-player in this particular game
                typically a module that hosts a LLM, and allows
                generate() functionality

            judge (LLMInferenceEngine):
                The LLM-based judge, that is used if we want to filter
                away false positives, i.e, trajectories where the environment
                erroneously decides that the agent has reached the goal,
                but it has not.
        """
        self.agent = agent
        self.env = env
        self.judge = judge

    def reset(self) -> None:
        """
        Resets the agent, environment and judge inference engines
        to the default state.

        Input:
            None

        Output:
            None
        """
        self.agent.reset()
        self.env.reset()
        self.judge.reset()

    def soft_reset(self) -> None:
        """
        Resets the agent, environment and judge inference engines
        to the default state.

        NOTE: for soft reset, as opposed to reset, we only reset
        some of the game states to the default, and keep others the
        same.

        Input:
            None

        Output:
            None
        """
        self.agent.soft_reset()
        self.env.soft_reset()
        self.judge.soft_reset()

    def run_judge_verification(
        self,
        judge_prompt: Optional[str],
        verifier_input_generator: Optional[Callable],
        agent_messages: List[Dict[str, Any]],
        env_game_scenario: str,
        agent_game_scenario: str,
        judge_temperature: float,
        judge_top_p: float,
        judge_min_p: Optional[float],
        judge_max_n_tokens: int,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Given the last response/generation from the agent LLM,
        runs one iteration of the judge verification to check
        if the agent has reached goal or not.

        Input:
            judge_prompt (str):
                The system prompt for the judge, typically a powerful
                API based LLM like gpt-4o-mini

                If None, we do not run judge verification

            verifier_input_generator (Callable):
                Generates the conversation in the correct format (specific to
                each game environment) that will then be used to judge the
                game trajectory (to remove false positives.)

            agent_messages (List[Dict[str, Any]]):
                The game conversation from the agent's perspective.
                Typically has the following format.
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

            env_game_scenario (str):
                The game scenario that the environment receives.
                For example, for 20 questions, the env will receive the topic
                the agent is trying to guess, eg., "Albert Einstein"

            agent_game_scenario (str):
                The game scenario that the agent receives.
                For example, for 20 questions, the agent will receive the topic
                it needs to guess, eg., "Person" or "Location"

            judge_temperature (float):
                The temperature parameter for LLM generation, for the Judge.

            judge_top_p (float):
                top_p for sampling from the Judge LLM.

            judge_min_p (float):
                min_p for sampling from the Judge LLM

            judge_max_n_tokens (int):
                Maximum number of tokens to generate from the Judge

        Output:
            (judge_label, judge_messages) tuple, where:

            judge_label (str):
                Whether or not the judge thinks the agent has successfully
                reached the goal or not

            judge_messages (Dict[str, Any]):
                The conversation/interaction history with the judge,
                has the same format as agent_messages above
        """
        judge_conv = get_conversation_template("gpt-4")
        if judge_prompt is None:
            return True, judge_conv.to_openai_api_messages()

        assert verifier_input_generator is not None
        assert self.judge is not None

        # Prepare the judge
        judge_conv.set_system_message(judge_prompt)

        judge_input = verifier_input_generator(
            config={
                "conversation": agent_messages,
                "agent_game_scenario": agent_game_scenario,
                "env_game_scenario": env_game_scenario,
            },
        )
        judge_conv.append_message(role="user", message=judge_input)

        judge_response = self.judge.generate(
            conv=judge_conv.to_openai_api_messages(),
            max_n_tokens=judge_max_n_tokens,
            temperature=judge_temperature,
            top_p=judge_top_p,
            min_p=judge_min_p,
        ).strip()

        judge_conv.append_message(role="assistant", message=judge_response)
        judge_label = judge_response == "<VALID>"

        return judge_label, judge_conv.to_openai_api_messages()

    def check_if_agent_has_reached_goal(
        self,
        env_message: str,
        judge_prompt: Optional[str],
        verifier_input_generator: Optional[Callable],
        agent_messages: List[Dict[str, Any]],
        env_game_scenario: str,
        agent_game_scenario: str,
        judge_temperature: float,
        judge_top_p: float,
        judge_min_p: Optional[float],
        judge_max_n_tokens: int,
    ) -> bool:
        """
        Checks if the agent has reached the goal in a particular
        game scenario.

        Input:
            env_message (str):
                Output/generation from the environment LLM

            judge_prompt (str):
                The system prompt for the judge, typically a powerful
                API based LLM like gpt-4o-mini

                If None, we do not run judge verification

            verifier_input_generator (Callable):
                Generates the conversation in the correct format (specific to
                each game environment) that will then be used to judge the
                game trajectory (to remove false positives.)

            agent_messages (List[Dict[str, Any]]):
                The game conversation from the agent's perspective.
                Typically has the following format.
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

            env_game_scenario (str):
                The game scenario that the environment receives.
                For example, for 20 questions, the env will receive the topic
                the agent is trying to guess, eg., "Albert Einstein"

            agent_game_scenario (str):
                The game scenario that the agent receives.
                For example, for 20 questions, the agent will receive the topic
                it needs to guess, eg., "Person" or "Location"

            judge_temperature (float):
                The temperature parameter for LLM generation, for the Judge.

            judge_top_p (float):
                top_p for sampling from the Judge LLM.

            judge_min_p (float):
                min_p for sampling from the Judge LLM

            judge_max_n_tokens (int):
                Maximum number of tokens to generate from the Judge

        Output:
            goal_reached (bool):
                Whether the Agent LLM has reached the goal or not,
                as judged by the environment LLM.
        """
        if judge_prompt is None:
            return "goal reached" in env_message.lower()

        # In this guess the agent has lost the game
        # But we return goal_reached = True,
        # and handle success/failure through the judge
        elif "<End> Game over </End>" in env_message:
            return True

        else:
            env_label = "goal reached" in env_message.lower()
            verification_label, _ = self.run_judge_verification(
                judge_prompt=judge_prompt,
                verifier_input_generator=verifier_input_generator,
                agent_messages=agent_messages,
                env_game_scenario=env_game_scenario,
                agent_game_scenario=agent_game_scenario,
                judge_temperature=judge_temperature,
                judge_top_p=judge_top_p,
                judge_min_p=judge_min_p,
                judge_max_n_tokens=judge_max_n_tokens,
            )
            return env_label or verification_label

    def run_one_iteration(
        self,
        env_game_scenario: str,
        agent_game_scenario: str,
        agent_first_message: str,
        env_first_message: str,
        max_turns: int,
        agent_temperature: float,
        agent_top_p: float,
        agent_min_p: Optional[float],
        agent_max_n_tokens: int,
        env_temperature: float,
        env_top_p: float,
        env_min_p: Optional[float],
        env_max_n_tokens: int,
        judge_temperature: float,
        judge_top_p: float,
        judge_min_p: Optional[float],
        judge_max_n_tokens: int,
        terminate_at_first_agent_failure: bool,
        env_optional_message: Optional[str],
        agent_optional_message: Optional[str],
        agent_response_extractor: Optional[Callable[[str], str]],
        env_response_extractor: Optional[Callable[[str], str]],
        env_default_response: Optional[str],
        num_max_env_response_generations: int,
        num_max_agent_response_generations: int,
        judge_prompt_agent: Optional[str],
        judge_prompt_env: Optional[str],
        verifier_input_generator: Optional[Callable],
        agent_model_supports_system_message: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns the conversation from one simulation run
        of the game.

        Input:
            env_game_scenario (str):
                The game scenario that the environment receives.
                For example, for 20 questions, the env will receive the topic
                the agent is trying to guess, eg., "Albert Einstein"

            agent_game_scenario (str):
                The game scenario that the agent receives.
                For example, for 20 questions, the agent will receive the topic
                it needs to guess, eg., "Person" or "Location"

            agent_first_message (str):
                The first message that the agent LLM receives at the
                start of the game.
                This is typically provided by the game environment, and varies based
                on the game.

            env_first_message (str):
                The first message that the environment LLM receives at the
                start of the game.
                This is typically provided by the game environment, and varies based
                on the game.

            max_turns (int):
                Upper bound on the turn of conversations that can happen in this game.

                NOTE: One turn consists of one generation from the agent, and one
                generation from environment.

            agent_temperature (float):
                The temperature parameter for LLM generation, for the Agent.
                Higher temperature means more variability in the generations,
                whereas lower temperature means more deterministic answers.

                See the following link for documentation:
                https://platform.openai.com/docs/api-reference/introduction

            agent_top_p (float):
                top_p for sampling from the Agent LLM.
                An alternative to sampling with temperature, called nucleus sampling,
                where the model considers the results of the tokens
                with top_p probability mass.
                So 0.1 means only the tokens comprising the top
                10% probability mass are considered.

                See the following link for documentation:
                https://platform.openai.com/docs/api-reference/introduction

            agent_min_p (float):
                min_p for sampling from the Agent LLM.
                Alternative way of sampling/decoding from the LLM
                Please see this paper for more details: https://arxiv.org/abs/2407.01082
                Values must be between 0 and 1, with typically 0.01-0.2 being preferred.

                NOTE: Typically useful when one wants to sample at high temperatures,
                e.g., temperature > 1

                NOTE: Also currenlty only supported in huggingface, and not in OpenAI

            agent_max_n_tokens (int):
                Maximum number of tokens to generate from the agent

            env_temperature (float):
                The temperature parameter for LLM generation, for the Environment.

            env_top_p (float):
                top_p for sampling from the environment LLM.

            env_min_p (float):
                min_p for sampling from the environment LLM

            env_max_n_tokens (int):
                Maximum number of tokens to generate from the environment

            judge_temperature (float):
                The temperature parameter for LLM generation, for the Judge.

            judge_top_p (float):
                top_p for sampling from the Judge LLM.

            judge_min_p (float):
                min_p for sampling from the Judge LLM

            judge_max_n_tokens (int):
                Maximum number of tokens to generate from the Judge

            terminate_at_first_agent_failure (bool):
                Whether to end the game, when the agent/player breaks the rule of the game
                (e.g., asking a question in 20 questions that cant be answered with yes/no)
                for the first time.

            env_optional_message (str):
                Optional message, that we add to the end of agent action,
                before passing it back to the environment.
                This helps filtering out actions that try to hack
                the environment, also for a long context problem,
                it helps with the environment judgement.

            agent_optional_message (str):
                Optional message that we add to the end of the environment
                response, before passing it back to the agent.

            agent_response_extractor (Callable[[str], str]):
                For a particular game, we often need to extract the response from the
                LLM generated agent action.

                This is often done to separate the action from the chain-of-thought

            env_response_extractor (Callable[[str], str]):
                For a particular game, we often need to extract the response from the
                LLM generated environment response.

                This is often done to prevent environment hacking/force some constraints on
                the environment response, such as to make it constrained to "Yes" or "No"

            env_default_response (str):
                In case the env_response_extractor cannot successfully extract answers
                from the environment response, i.e., it does not follow the constraints
                we desire, we replace the response with a default one.

                This is done to prevent environment hacking

            num_max_env_response_generations (int):
                Sometimes there is variability in the environment response, and it
                does not follow the format we want it to follow.
                In that case, we want to generate >= 1 number of times to see if the
                environment can potentially generate valid responses

            num_max_agent_response_generations (int):
                Sometimes there is variability in the agent response, and it
                does not follow the format we want it to follow.
                In that case, we want to generate >= 1 number of times to see if the
                agent can potentially generate valid responses

            judge_prompt_agent (str):
                System prompt for the LLM judge, for verifying the agent's response

            judge_prompt_env (str):
                System prompt for the LLM judge, for verifying the environment's response

            verifier_input_generator (Callable):
                Generates the conversation in the correct format (specific to
                each game environment) that will then be used to judge the
                game trajectory (to remove false positives.)

            agent_model_supports_system_message (bool):
                Some LLMs like gemma-2-9b-it does not support a system message. If this flag
                is set to false, then we do not add it during model generation.

        Output:
            record (Dict[str, Any]):
                A dictionary containing the record of this game simulation.
                It looks like the following:
                {
                    "agent_game_scenario": agent_game_scenario,
                    "env_game_scenario": env_game_scenario,
                    "goal_reached": agent_has_reached_goal,
                    "judge_label": judge_label,
                    "num_turns": turn,
                    "max_turns": max_turns,
                    "env_first_message": env_first_message,
                    "conversation": agent_conversation,
                    "env_conversation": env_conversation,
                    "judge_conversation": judge_conversation,
                    "rewards": rewards,
                }

                where each entry in the dictionary is described below:

                agent_game_scenario (str):
                    The game scenario information that the agent receives

                env_game_scenario (str):
                    The game scenario information that the environment receives

                agent_has_reached_goal (bool):
                    Whether or not the agent has successfully solved the task,
                    as judged by the environment (typically another LLM)

                judge_label (bool):
                    Whether or not the trajectory is valid (i.e., either a failure,
                    OR is a success according to the environment AND a judge)

                    judge_label = True
                        Either the trajectory is a failure
                        Or it is a success, and the extra verification by LLM judge
                        also agrees

                    judge_label = False
                        Otherwise

                    NOTE: if no judge is provided, then judge_label is always True

                turn (int):
                    How many turns the game lasted, or for games with fixed turns,
                    but environment-supplied rewards,
                    the negative of total reward accumulated in the game

                max_turns (int):
                    Upper bound on number of turns the game was allowed to run

                env_first_message (str):
                    The first message that the environment receives
                    at the beginning of the game.

                conversation (Dict[str, Any]):
                    Typically has the following format.
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
                    Here the first user prompt comes from the game design,
                    subsequent user prompts come from the environment LLM,
                    and assistant prompts come from the agent.

                    NOTE: It is the conversation seen from agent's perspective.

                env_conversation (Dict[str, Any]):
                    Has the same format as conversation above.

                    NOTE: this is the conversation seen from the environment's perspective.
                    I.e., input for environment is output from the agent, and vice versa

                judge_conversation (Dict[str, Any]):
                    The conversation with the LLM-judge.

                rewards (Any):
                    Either None, or some reward statistics
        """
        # Prepare the agent
        agent_conv = get_conversation_template("gpt-4")
        agent_conv.append_message(role="user", message=agent_first_message)

        # Prepare the environment
        env_conv = get_conversation_template("gpt-4")
        env_conv.set_system_message(env_first_message)

        # Prepare the judge messages
        judge_messages = None

        # run the game iterations
        turn = 0
        agent_has_reached_goal = False
        judge_label = True

        env_generation_config = {
            "max_n_tokens": env_max_n_tokens,
            "top_p": env_top_p,
            "min_p": env_min_p,
            "temperature": env_temperature,
        }

        agent_generation_config = {
            "max_n_tokens": agent_max_n_tokens,
            "top_p": agent_top_p,
            "min_p": agent_min_p,
            "temperature": agent_temperature,
        }

        self.soft_reset()

        while turn < max_turns and not agent_has_reached_goal:
            agent_response_dict = self.generate_response_from_llm_helper(
                llm_inference_engine=self.agent,
                conv=(
                    agent_conv.to_openai_api_messages()
                    if agent_model_supports_system_message
                    else agent_conv.to_openai_api_messages()[1:]
                ),
                generation_config=agent_generation_config,
                max_attempts=num_max_agent_response_generations,
                response_extractor=agent_response_extractor,
            )

            # Agent gave no valid response
            if not agent_response_dict["got_valid_llm_generation"]:
                return None

            agent_action = agent_response_dict["response"]
            extracted_agent_response = agent_response_dict["extracted_response"]

            agent_conv.append_message(role="assistant", message=agent_action)

            # In case there is an optional message we want to pass to the environment
            if env_optional_message is not None:
                extracted_agent_response = (
                    extracted_agent_response + "\n" + env_optional_message
                )

            env_conv.append_message(role="user", message=extracted_agent_response)

            env_response_dict = self.generate_response_from_llm_helper(
                llm_inference_engine=self.env,
                conv=env_conv.to_openai_api_messages(),
                generation_config=env_generation_config,
                max_attempts=num_max_env_response_generations,
                response_extractor=env_response_extractor,
            )

            #  Strict filtering to ensure no environment hacking
            if not env_response_dict["got_valid_llm_generation"]:
                env_response = env_response_dict["response"]
                assert env_default_response is not None
                extracted_env_response = env_default_response

            else:
                env_response = env_response_dict["response"]
                extracted_env_response = env_response_dict["extracted_response"]

            # In case strict filtering is harder to perform, we do it
            # via running an LLM judge
            if judge_prompt_env is not None:
                judge_label_env, _ = self.run_judge_verification(
                    judge_prompt=judge_prompt_env,
                    verifier_input_generator=verifier_input_generator,
                    agent_messages=env_conv.to_openai_api_messages(),
                    agent_game_scenario=agent_game_scenario,
                    env_game_scenario=env_game_scenario,
                    judge_temperature=judge_temperature,
                    judge_top_p=judge_top_p,
                    judge_min_p=judge_min_p,
                    judge_max_n_tokens=judge_max_n_tokens,
                )

                if not judge_label_env:
                    env_response = env_response_dict["response"]
                    assert env_default_response is not None
                    extracted_env_response = env_default_response

            env_conv.append_message(role="assistant", message=env_response)

            if agent_optional_message is not None:
                extracted_env_response = extracted_env_response + "\n" + agent_optional_message

            agent_conv.append_message(role="user", message=extracted_env_response)

            turn += 1

            # Check if we have reached goal
            agent_has_reached_goal = self.check_if_agent_has_reached_goal(
                env_message=extracted_env_response,
                judge_prompt=judge_prompt_agent,
                verifier_input_generator=verifier_input_generator,
                agent_messages=agent_conv.to_openai_api_messages(),
                agent_game_scenario=agent_game_scenario,
                env_game_scenario=env_game_scenario,
                judge_temperature=judge_temperature,
                judge_top_p=judge_top_p,
                judge_min_p=judge_min_p,
                judge_max_n_tokens=judge_max_n_tokens,
            )

            # Extra verification if the environment thinks the game is solved
            if agent_has_reached_goal and judge_prompt_agent is not None:
                judge_label, judge_messages = self.run_judge_verification(
                    judge_prompt=judge_prompt_agent,
                    verifier_input_generator=verifier_input_generator,
                    agent_messages=agent_conv.to_openai_api_messages(),
                    agent_game_scenario=agent_game_scenario,
                    env_game_scenario=env_game_scenario,
                    judge_temperature=judge_temperature,
                    judge_top_p=judge_top_p,
                    judge_min_p=judge_min_p,
                    judge_max_n_tokens=judge_max_n_tokens,
                )

                # Even if it is not actually solved, we terminate the game here,
                # Since the agent things it is solved,
                # but change the label goal_reached -> not goal reached
                break

            elif (
                not env_response_dict["got_valid_llm_generation"]
                and terminate_at_first_agent_failure
            ):
                break

        # NOTE: we use number of turns (lower the better) to choose
        # preferred trajectories. For environment that has fixed turns but
        # environment specified reward, we should use the negative reward
        # to do this
        rewards = self.env.get_rewards()
        if rewards is None:
            num_turns = turn
        else:
            num_turns = float(-np.sum(rewards["rewards_per_timestep"]))

        return {
            "agent_game_scenario": agent_game_scenario,
            "env_game_scenario": env_game_scenario,
            "goal_reached": agent_has_reached_goal,
            "judge_label": judge_label,
            "num_turns": num_turns,
            "max_turns": max_turns,
            "env_first_message": env_first_message,
            "conversation": agent_conv.to_openai_api_messages(),
            "env_conversation": env_conv.to_openai_api_messages(),
            "judge_conversation": judge_messages,
            "rewards": rewards,
        }

    def generate_response_from_llm_helper(
        self,
        llm_inference_engine: LLMInferenceEngine,
        conv: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
        max_attempts: int,
        response_extractor: Optional[Callable[[str], str]],
    ) -> Dict[str, Any]:
        """
        Helper function, to unify generations from the agent and the environment.

        NOTE: In some cases, we want to ensure that the agent or the environment
        responses follow a certain pattern strictly
        (e.g., for 20 questions, all responses are yes/no).
        We use the response extractor (optional) to ensure this.
        We would normally try max_attempts number of times to try to get a valid
        response, otherwise revert to default behavior

        (for agent, if there is no valid response, we terminate the sequence.
        for env, if there is no valid response, we replace it with some environment
        or game specific default string)

        Input:
            llm_inference_engine (LLMInferenceEngine):
                An API that lets us query and get responses from an LLM

            conv (List[Dict[str, Any]]):
                multi-turn conversation for which we will generate the next
                turn of conversation

            generation_config (Dict[str, Any]):
                Configuration/hyper-parameters for generating a response from the LLM
                Usual structure:
                    {
                        "temperature": <temperature> (float),
                        "top_p": <top_p> (float),
                        "min_p": <min_p> (float),
                        "max_n_tokens": <max_n_tokens> (int),
                    }

            max_attempts (int):
                The response from the LLM can often not follow the format we want
                This parameter determines the maximum number of times we try
                to generate a valid response before reverting to default behavior

            response_extractor (Callable[[str], str]):
                Used to extract the part of the LLM generation that we will use
                in the game.
                Used to ensure that agent/env responses follow strict formats
                when we need them to

        Output:
            response_dict (Dict[str, Any]):
                A dictionary of outputs containing LLM generations. Has the following
                structure:

                {
                    "got_valid_llm_generation": got_valid_llm_response (bool),
                    "response": llm_generation (str),
                    "extracted_response": extracted_response (str),
                }

                Each element is described below:

                got_valid_llm_response (bool):
                    Whether the LLM response is valid or not

                llm_generation (str):
                    Raw response from the LLM

                extracted_response (str):
                    The response extracted from the raw LLM generation
        """
        generation_config["conv"] = conv
        if response_extractor is not None:
            got_valid_llm_response = False
            llm_generation = ""

            for _ in range(max_attempts):
                try:
                    llm_generation = llm_inference_engine.generate(**generation_config).strip()
                    extracted_response = response_extractor(llm_generation)
                    got_valid_llm_response = True
                    break

                except:
                    print("Did not get valid response from the agent, trying again.")

            return {
                "got_valid_llm_generation": got_valid_llm_response,
                "response": llm_generation,
                "extracted_response": (
                    None if not got_valid_llm_response else extracted_response
                ),
            }

        else:
            llm_generation = llm_inference_engine.generate(**generation_config).strip()
            return {
                "got_valid_llm_generation": True,
                "response": llm_generation,
                "extracted_response": llm_generation,
            }
