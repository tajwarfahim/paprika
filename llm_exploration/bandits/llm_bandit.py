import math
from typing import List, Optional, Union, Dict

from llm_exploration.bandits.bandit import Bandit


class LLMBandit(Bandit):
    """
    LLM Bandit Environment
    This environment is based on the following paper:
    Can large language models explore in-context? (https://arxiv.org/abs/2403.15371)

    This class provides an API for generating text trajectories based on
    some contextual bandit algorithm running underneath, such as UCB
    """

    allowable_modes = ["original", "exploratory"]
    allowable_reward_history_types = ["original", "summary", "ucb"]

    def __init__(
        self,
        arms_list: List[str],
        probability_list: List[float],
        T: int,
        mode: str = "original",
        reward_history_type: str = "original",
        alpha: float = 0.1,
        include_system_prompt_in_conversation: bool = False,
        include_special_tokens: bool = False,
        include_text_explanation: bool = False,
        name: str = "",
    ):
        """
        Generates an instance of this class.

        Input:
            arms_list (List[str]):
                List of arms in text, should be name of colors
                e.g., ["blue", "red", "green"]

            probability_list (List[float]):
                List of probability for the bernoulli distribution
                probability_list[i] -> the mean of the bernoulli distribution for arms_list[i]
                Should have same length as arms_list

            T (int):
                The horizon of the algorithm.
                Typically this is not provided to bandit environments,
                but we provide this to the LLM following the original paper.

            mode (str):
                mode of the LLM prompts.
                Has to be one of two types:
                1. "original", which uses the default set of prompts.
                2. "exploratory", which asks the LLM explicitly to be exploratory
                    in the first few turns.

                Default: "original"

            reward_history_type (str):
                Type of reward history presented to the LLM at the start of every turn.
                Has to be one of three types:
                1. "original", which just tells the LLM the reward from the last round.
                2. "summary", which summarizes the average reward + number of times each
                    arm has been tried out.
                3. "ucb", which provides the average reward per arm + the UCB exploration
                    bonus, which provides a bonus reward for arms that have been tried out
                    very often. The goal is to see provided this information, will an LLM
                    solve this problem.

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

            include_system_prompt_in_conversation (bool):
                Whether to include system prompt at the start
                of the conversation or not.
                If True, self.generate_history() returns a conversation like:
                    [
                        {"role": "system", "content": <system_prompt>},
                        {"role": "user", "content": <user_prompt>},
                        {"role": "assistant", "content": <assistant_prompt>},
                        ...
                    ]
                If False, then the system prompt (bandit problem description)
                is added to the first user prompt, and the conversation looks like:
                    [
                        {"role": "user", "content": <user_prompt>},
                        {"role": "assistant", "content": <assistant_prompt>},
                        ...
                    ]

            include_special_tokens (bool):
                If True, then the arm COLOR is wrapped by special tokens, i.e.,
                <Answer>COLOR</Answer>

                Otherwise, there is no special tokens.

            include_text_explanation (bool):
                Whether or not to ask the language model to generate text explanation
                for the arm it chooses.

            name (str):
                Name of the bandit environment
                Used for identification purposes only

                Default: "" (empty string)
        """
        super().__init__(arms_list=arms_list, probability_list=probability_list)
        self.T = T

        self.all_arms = ", ".join(arms_list)
        self.env_name = name
        self.text_actions = []

        assert mode in self.allowable_modes
        self.mode = mode

        assert reward_history_type in self.allowable_reward_history_types
        self.reward_history_type = reward_history_type

        self.alpha = alpha
        self.include_system_prompt_in_conversation = include_system_prompt_in_conversation
        self.include_special_tokens = include_special_tokens
        self.include_text_explanation = include_text_explanation

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt, which is slightly modified from the original paper:
        Can large language models explore in-context? (https://arxiv.org/abs/2403.15371)

        Input:
            None

        Output:
            system_prompt (str)
        """
        if self.include_special_tokens:
            """
            Ask the LLM to wrap arm as <Answer>arm</Asnwer>
            """
            system_prompt = (
                f"You are in a room with {len(self.arms_list)} buttons labeled {self.all_arms}. "
                f"Each button is associated with a Bernoulli distribution with a fixed but unknown mean; "
                f"the means for the buttons could be different. "
                f"For each button, when you press it, you will get a reward that is "
                f"sampled from the button's associated distribution. "
                f"You have {self.T} time steps and, on each time step, you can choose any button and receive the reward. "
                f"Your goal is to maximize the total reward over the {self.T} time steps. "
                f"At each time step, I will tell you the reward from the previous round, "
                f"Then you must make the next choice, which must be exactly one of {self.all_arms}. "
                f"You must provide your final answer as <Answer>COLOR</Answer> where "
                f"COLOR is one of {self.all_arms}."
            )

        elif self.mode == "original":
            system_prompt = (
                f"You are in a room with {len(self.arms_list)} buttons labeled {self.all_arms}. "
                f"Each button is associated with a Bernoulli distribution with a fixed but unknown mean; "
                f"the means for the buttons could be different. "
                f"For each button, when you press it, you will get a reward that is "
                f"sampled from the button's associated distribution. "
                f"You have {self.T} time steps and, on each time step, you can choose any button and receive the reward. "
                f"Your goal is to maximize the total reward over the {self.T} time steps. "
                f"At each time step, I will tell you the reward from the previous round, "
                f"Then you must make the next choice, which must be exactly one of {self.all_arms}. "
                f"You must provide your final answer immediately as COLOR where "
                f"COLOR is one of {self.all_arms}, "
                f"with no text explanation."
            )

        else:
            """
            This is similar to the "original" case, except we prompt the model to be exploratory.
            Following are the lines we add to the system prompt, in addition to the original one:

                The ideal strategy is to try each arm a few number of times before deciding the optimal arm.
                i.e., you need to explore each action and not decide on the optimal choice based on
                few/no samples for each arm.
                Ideal algorithms for contextual bandit problems are UCB or Thompson sampling
                so you should try to act like an UCB algorithm.
            """

            system_prompt = (
                f"You are in a room with {len(self.arms_list)} buttons labeled {self.all_arms}. "
                f"Each button is associated with a Bernoulli distribution with a fixed but unknown mean; "
                f"the means for the buttons could be different. "
                f"For each button, when you press it, you will get a reward that is "
                f"sampled from the button's associated distribution. "
                f"You have {self.T} time steps and, on each time step, you can choose any button and receive the reward. "
                f"Your goal is to maximize the total reward over the {self.T} time steps. "
                f"At each time step, I will tell you the reward from the previous round, "
                f"Then you must make the next choice, which must be exactly one of {self.all_arms}. "
                f"The ideal strategy is to try each arm a few number of times before deciding the optimal arm. "
                f"i.e., you need to explore each action and not decide on the optimal choice based on "
                f"few/no samples for each arm. "
                f"Ideal algorithms for contextual bandit problems are UCB or Thompson sampling "
                f"so you should try to act like an UCB algorithm. "
                f"You must provide your final answer immediately as COLOR where COLOR is one of {self.all_arms}, "
                f"with no text explanation."
            )

        return system_prompt

    def reset(
        self,
        arms_list: Optional[List[str]] = None,
        probability_list: Optional[List[float]] = None,
        T: Optional[int] = None,
        mode: Optional[str] = None,
        reward_history_type: Optional[str] = None,
        alpha: Optional[float] = None,
        include_system_prompt_in_conversation: Optional[bool] = None,
        include_special_tokens: Optional[bool] = None,
        include_text_explanation: Optional[bool] = None,
    ) -> None:
        """
        Resets the bandit environment with a different list of arms, probability and T.
        If any of the arguments is None, we do not reset it.

        Input:
            arms_list (List[str]):
                List of arms in text, should be name of colors
                e.g., ["blue", "red", "green"]
                Defaults to None, in which case we do not reset

            probability_list (List[float]):
                List of probability for the bernoulli distribution
                probability_list[i] -> the mean of the bernoulli distribution for arms_list[i]
                Should have same length as arms_list
                Defaults to None, in which case we do not reset

            T (int):
                The horizon of the algorithm.
                Typically this is not provided to bandit environments,
                but we provide this to the LLM following the original paper.
                Defaults to None, in which case we do not reset

            mode (str):
                mode of the LLM prompts.
                Has to be one of two types:
                1. "original", which uses the default set of prompts.
                2. "exploratory", which asks the LLM explicitly to be exploratory in
                the first few turns.
                Defaults to None, in which case we do not reset

            reward_history_type (str):
                Type of reward history presented to the LLM at the start of every turn.
                Has to be one of three types:
                1. "original", which just tells the LLM the reward from the last round.
                2. "summary", which summarizes the average reward + number of times each
                    arm has been tried out.
                3. "ucb", which provides the average reward per arm + the UCB exploration
                    bonus, which provides a bonus reward for arms that have been tried out
                    very often. The goal is to see provided this information, will an LLM
                    solve this problem.

                Defaults to None, in which case we do not reset

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
                Defaults to None, in which case we do not reset

            include_system_prompt_in_conversation (bool):
                Whether to include system prompt at the start
                of the conversation or not.
                If True, self.generate_history() returns a conversation like:
                    [
                        {"role": "system", "content": <system_prompt>},
                        {"role": "user", "content": <user_prompt>},
                        {"role": "assistant", "content": <assistant_prompt>},
                        ...
                    ]
                If False, then the system prompt (bandit problem description)
                is added to the first user prompt, and the conversation looks like:
                    [
                        {"role": "user", "content": <user_prompt>},
                        {"role": "assistant", "content": <assistant_prompt>},
                        ...
                    ]

                Defaults to None, in which case we do not reset

            include_special_tokens (bool):
                If True, then the arm COLOR is wrapped by special tokens, i.e.,
                <Answer>COLOR</Answer>

                Otherwise, there is no special tokens.

                Defaults to None, in which case we do not reset

            include_text_explanation (bool):
                Whether or not to ask the language model to generate text explanation
                for the arm it chooses.

        Output:
            None
        """
        super().reset(arms_list, probability_list)

        if T is not None:
            self.T = T

        if arms_list is not None:
            self.all_arms = ", ".join(arms_list)

        if mode is not None:
            assert mode in self.allowable_modes
            self.mode = mode

        if reward_history_type is not None:
            assert reward_history_type in self.allowable_reward_history_types
            self.reward_history_type = reward_history_type

        if alpha is not None:
            self.alpha = alpha

        if include_system_prompt_in_conversation is not None:
            self.include_system_prompt_in_conversation = include_system_prompt_in_conversation

        if include_special_tokens is not None:
            self.include_special_tokens = include_special_tokens

        if include_text_explanation is not None:
            self.include_text_explanation = include_text_explanation

        self.text_actions = []

    def get_user_prompt(
        self,
        time_step: int,
    ) -> str:
        """
        Returns the user prompt.

        Input:
            time_step (int):
                which time step it is to generate the user prompt for.

        Output:
            user_prompt (str):
                Typically the input to the language model.
        """
        if time_step == 0:
            reward_statement = "This is the first timestep. "
            if not self.include_system_prompt_in_conversation:
                reward_statement = self.get_system_prompt() + reward_statement
        else:
            len(self.all_rewards) >= time_step

            if self.reward_history_type == "original":
                reward_statement = f"You got reward {self.all_rewards[time_step - 1]}. "

            else:
                reward_statement = f"Here is your interaction history so far: \n"

                for arm in range(len(self.arms_list)):
                    average_reward = self.get_average_reward(
                        arm=arm,
                    )

                    if self.reward_history_type == "summary":
                        arm_statement = (
                            f"You have tried the {self.arms_list[arm]} button "
                            f"{self.counts[arm]} times, and got average reward "
                            f"{average_reward} \n"
                        )
                    else:
                        arm_statement = (
                            f"Average reward for {self.arms_list[arm]} "
                            f"is {average_reward} \n"
                        )

                    reward_statement += arm_statement

        arm_string = "COLOR" if not self.include_special_tokens else "<Answer>COLOR</Answer>"

        if self.include_text_explanation:
            user_prompt = (
                f"Which button will you choose next? "
                f"YOU MUST provide your final answer as {arm_string}"
            )

        else:
            user_prompt = (
                f"Which button will you choose next? "
                f"YOU MUST provide your final answer as "
                f"{arm_string} where COLOR is one of {self.all_arms}, "
                f"with no text explanation."
            )

        return reward_statement + user_prompt

    def extract_arm_from_text_action(
        self,
        text_action: str,
    ) -> int:
        """
        Extracts the arm (integer, index of self.arms_list)
        from a text action, generated by an LLM.

        Input:
            text_action (str):
                Action in text/string format.
                It looks like: <Answer>blue</Answer>

        Output:
            arm (int):
                Integer arm index
        """
        for index in range(len(self.arms_list)):
            if self.arms_list[index].lower() in text_action:
                return index

        return self.default_not_valid_action

    def step_text_action(
        self,
        text_action: str,
    ) -> Union[float, int]:
        """
        This function takes a step in the environment, return the corresponding reward.
        Difference with step() function above is that the action here is in text, typically in
        COLOR format,
        typically the output of an LLM that we are evaluating.

        Input:
            arm_in_text (str):
                The arm chosen for the current time step, in text. Typically has the format:
                COLOR

                NOTE: We do not restrict the text action to have this format, as long as it contains
                the proper arm name.

        Output:
            reward (float or int):
                The reward obtained from this step, typically sampled from the
                independent bernoulli distribution associated with the chosen arm.

                This is typically either 0 or 1,
                but we leave the option of it being other real numbers as well.
        """
        self.text_actions.append(text_action)
        return super().step(
            arm=self.extract_arm_from_text_action(text_action=text_action),
        )

    def step(self, arm: int) -> Union[float, int]:
        """
        Takes one step in the environment, and returns the corresponding reward.
        Takes an integer index to the arms_list, and not a text version of the arm.
        See step_text_action() function for the other version.

        NOTE: If used for data generation, it also uses a rationale for
        choosing a particular arm, intended to train a language model.

        NOTE: This rationale is written for the UCB algorithm, and not

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
        if self.include_text_explanation:
            # arm_string = (
            #     self.arms_list[arm]
            #     if not self.include_special_tokens
            #     else f"<Answer>{self.arms_list[arm]}</Answer>"
            # )
            # text_action = f"I choose {arm_string}"
            # self.text_actions.append(text_action)
            self.text_actions.append(self.arms_list[arm])

        return super().step(arm=arm)

    def generate_history(
        self,
        use_text_actions: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Generates the history of the bandit environment so far.

        Input:
            use_text_actions (bool):
                Whether to use self.history, which is a list of arms
                chosen at each step, vs self.text_actions,
                which is a list containing the exact text actions generated
                by an LLM.

                Default: False

        Output:
            History (List[Dict[str, str]]):
                Typically has the following format:
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

        """
        history = []

        if self.include_system_prompt_in_conversation:
            system_prompt = {
                "role": "system",
                "content": self.get_system_prompt(),
            }
            history.append(system_prompt)

        for i in range(len(self.history)):
            user_prompt = self.get_user_prompt(time_step=i)
            if not self.include_system_prompt_in_conversation and i == 0:
                system_prompt = self.get_system_prompt()
                user_prompt = system_prompt + " " + user_prompt

            context = {"role": "user", "content": user_prompt}

            action = {
                "role": "assistant",
                "content": (
                    f"{self.text_actions[i]}" if use_text_actions else f"{self.history[i]}"
                ),
            }

            history.extend([context, action])

        return history

    def get_average_reward(
        self,
        arm: int,
    ) -> float:
        """
        Returns the average reward for a given arm.
        If this arm has not been tried yet, it returns average reward of 0.
        If this arm has been tried, it returns the average reward.

        Input:
            arm (int):
                Integer index of self.arms_list, used to identify the accurate arm.

        Output:
            average_reward (float):
                The average reward for the given arm.
        """
        if arm == self.default_not_valid_action:
            average_reward = 0.0

        elif self.counts[arm] == 0:
            if self.reward_history_type == "ucb":
                average_reward = 1.0
            else:
                average_reward = 0.0

        else:
            average_reward = float(self.values[arm]) / self.counts[arm]

            if self.reward_history_type == "ucb":
                total_counts = sum(self.counts)
                exploration_bonus = self.alpha * math.sqrt(
                    (2 * math.log(total_counts)) / float(self.counts[arm])
                )
                average_reward += exploration_bonus

        return round(average_reward, 2)

    def get_all_text_actions(self) -> List[str]:
        """
        Returns the actions in text/string format,
        in most cases generated by an LLM policy.

        Input:
            None

        Output:
            self.text_actions (List[str]):
                self.text_actions[i] = the action generated by LLM policy,
                                       in text format.
        """
        return self.text_actions

    def get_num_invalid_actions(self) -> int:
        """
        Returns the number of actions that are invalid in the entire
        trajectory so far.

        Input:
            None

        Output:
            num_invalid_actions (int):
                Number of invalid actions in the trajectory
        """
        num_invalid_actions = 0
        for i in range(len(self.history)):
            if self.history[i] == self.not_valid_action_string:
                num_invalid_actions += 1

        return num_invalid_actions

    def should_include_system_prompt_in_conversation(self) -> bool:
        """
        Returns whether or not the generated trajectory begins
        with a system prompt, or the problem description is provided
        in the first user prompt instead.

        Input:
            None

        Output:
            self.include_system_prompt_in_conversation (bool):
                If True, conversation/generated history begins with
                a system prompt, else begins with an user prompt.
        """
        return self.include_system_prompt_in_conversation


class LLMBanditExtended(LLMBandit):
    """
    Extension of the LLMBandit class.

    This contains the options for different variations in the bandit prompts
    or problem descriptions.

    Mainly used for data generation for training a LLM
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes an object of the class,
        with a particular system prompt
        """
        self.system_prompt = kwargs.pop("system_prompt")
        self.answer = kwargs.pop("answer")
        self.verbose = kwargs.pop("verbose", False)

        super().__init__(*args, **kwargs)

        if not self.include_special_tokens:
            self.system_prompt = self.system_prompt.replace("<Answer>", "").replace(
                "</Answer>", ""
            )
            self.answer = self.answer.replace("<Answer>", "").replace("</Answer>", "")

        if self.verbose:
            print(
                "Since this is for data generation for training LLMs"
                "We don't include system prompt in conversation."
            )
        self.include_system_prompt_in_conversation = False

    def get_system_prompt(self) -> str:
        """
        Returns the given system prompt/problem description.

        NOTE: Overrides the parent class function.

        Input:
            None

        Output:
            system_prompt (str):
                System prompt for the model
        """
        return self.system_prompt

    def get_user_prompt(
        self,
        time_step: int,
    ) -> str:
        """
        Returns the user prompt.

        NOTE: Overrides the parent class's functionalities.

        Input:
            time_step (int):
                which time step it is to generate the user prompt for.

        Output:
            user_prompt (str):
                Typically the input to the language model.
        """
        if time_step == 0:
            return self.get_system_prompt()

        else:
            len(self.all_rewards) >= time_step

            if self.reward_history_type == "original":
                reward_statement = f"You got reward {self.all_rewards[time_step - 1]}. "

            else:
                reward_statement = f"Here is your interaction history so far: \n"

                for arm in range(len(self.arms_list)):
                    average_reward = self.get_average_reward(
                        arm=arm,
                    )

                    if self.reward_history_type == "summary":
                        arm_statement = (
                            f"You have tried the {self.arms_list[arm]} button "
                            f"{self.counts[arm]} times, and got average reward "
                            f"{average_reward} \n"
                        )
                    else:
                        arm_statement = (
                            f"Average reward for {self.arms_list[arm]} "
                            f"is {average_reward} \n"
                        )

                    reward_statement += arm_statement

        arm_string = self.answer

        if self.include_text_explanation:
            user_prompt = (
                f"Which action will you choose next? "
                f"YOU MUST provide your final answer as {arm_string}"
            )

        else:
            user_prompt = (
                f"Which action will you choose next? "
                f"YOU MUST provide your final answer as "
                f"{arm_string} where COLOR is one of {self.all_arms}, "
                f"with no text explanation."
            )

        return reward_statement + user_prompt
