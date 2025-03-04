from typing import (
    List,
    Dict,
    Optional,
    Any,
)
from abc import ABC, abstractmethod


class LLMInferenceEngine(ABC):
    """
    Class Template for running inference on a LLM.

    This class actually does not implement any of the necessary methods,
    rather only defines the functions that subsequent classes inheriting
    from it should implement.
    """

    @abstractmethod
    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        """
        Generates responses for a batch of prompts using the given language model.
        Given a list of conversations, generate the model's response for each conversation
        in a batched fashion, and returns the model's responses.

        Input:
            convs (List[List[Dict]]):
                The list of conversations. It looks like below:
                convs = [
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}, ...
                    ],  # 1st conversation
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}, ...
                    ], # 2nd conversation
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}, ...
                    ], # 3rd conversation
                    ....
                ]
                where len(convs) = batch size

            max_new_tokens (int):
                Maximum number of new tokens to generate per each response.

            temperature (float):
                The temperature parameter for LLM generation.
                Higher temperature means more variability in the generations,
                whereas lower temperature means more deterministic answers.

                See the following link for documentation:
                https://platform.openai.com/docs/api-reference/introduction

            top_p (float):
                An alternative to sampling with temperature, called nucleus sampling,
                where the model considers the results of the tokens with top_p probability mass.
                So 0.1 means only the tokens comprising the top 10% probability mass are considered.

                See the following link for documentation:
                https://platform.openai.com/docs/api-reference/introduction

            min_p (float):
                Alternative way of sampling/decoding from the LLM
                Please see this paper for more details: https://arxiv.org/abs/2407.01082
                Values must be between 0 and 1, with typically 0.01-0.2 being preferred.

                NOTE: Typically useful when one wants to sample at high temperatures,
                e.g., temperature > 1

                NOTE: Also currenlty only supported in huggingface, and not in OpenAI

                Default: None, in which case it is not used

        Output:
            model_generations (List[str]):
                List of length = batch size
                model_generations[i] = response from the LLM for convs[i]
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        """
        Given a particular conversation, generates the response.
        Main difference with batched_generate: this works only with batch size = 1,
        i.e., for a single input.

        Input:
            conv (List[Dict]):
                List of dictionaries, OpenAI API format conversation.
                The format looks as follows:
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        ...
                    ]

            max_n_tokens (int):
                max number of tokens to generate

            temperature (float):
                The temperature parameter for LLM generation.
                Higher temperature means more variability in the generations,
                whereas lower temperature means more deterministic answers.

                See the following link for documentation:
                https://platform.openai.com/docs/api-reference/introduction

            top_p (float):
                An alternative to sampling with temperature, called nucleus sampling,
                where the model considers the results of the tokens with top_p probability mass.
                So 0.1 means only the tokens comprising the top 10% probability mass are considered.

                See the following link for documentation:
                https://platform.openai.com/docs/api-reference/introduction

            min_p (float):
                Alternative way of sampling/decoding from the LLM
                Please see this paper for more details: https://arxiv.org/abs/2407.01082
                Values must be between 0 and 1, with typically 0.01-0.2 being preferred.

                NOTE: Typically useful when one wants to sample at high temperatures,
                e.g., temperature > 1

                NOTE: Also currenlty only supported in huggingface, and not in OpenAI

                Default: None, in which case it is not used

        Returns:
            generated_response (str):
                Response generated from the model.
        """
        raise NotImplementedError

    def validate_conversation(
        self,
        conv: List[Dict[str, str]],
    ) -> None:
        """
        Validate whether a given conversation has the correct form,
        which looks like below:
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                ...
            ]

        Input:
            conv (List[Dict[str, str]]):
                A conversation object

        Output:
            None
        """
        assert isinstance(conv, list)
        for item in conv:
            assert isinstance(item, dict)
            for key in item:
                assert key in ["role", "content"]
                assert isinstance(item[key], str)

            assert item.get("role") in ["system", "user", "assistant"]

    def reset(self) -> None:
        """
        For certain games, the inference engine needs to be reset to the
        start state, in order for the LLM to be able to play
        the game again.

        Input:
            None

        Output:
            None
        """
        print("The inference engine has been reset to the start state!")

    def soft_reset(self) -> None:
        """
        For certain games, the inference engine needs to be reset to the
        start state, in order for the LLM to be able to play
        the game again.

        NOTE: for the soft reset, as opposed to reset, we only reset part of
        the game states, and keep the other states same.

        Input:
            None

        Output:
            None
        """
        print("The inference engine has been reset to the start state!")

    def get_rewards(self) -> Any:
        """
        For certain games, the inference engine, if used as the environment,
        needs to keep track of the rewards earned in each time step.
        This function returns rewards per time step if the environment
        keeps track of reward, or otherwise returns None

        Input:
            None

        Output:
            rewards (Optional[List[float]]):
                per step reward earned in the game environment
                If the environment does not have this, then None
                is returned

        NOTE: Individual inference engines should implement/override
        this function
        """
        return None

    def validate_batch_of_conversations(
        self,
        convs: List[List[Dict[str, str]]],
    ) -> None:
        """
        Validate whether a given batch of conversations has
        the correct form, which looks like below:
            [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}, ...
                ],  # 1st conversation
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}, ...
                ], # 2nd conversation
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}, ...
                ], # 3rd conversation
                ....
            ]

        Input:
            convs (List[List[Dict[str, str]]]):
                A batch of conversation. Each element is a separate conversation,
                that looks like:
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        ...
                    ]

        Output:
            None
        """
        assert isinstance(convs, list)
        for conv in convs:
            self.validate_conversation(conv=conv)
