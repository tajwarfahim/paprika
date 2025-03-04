from typing import (
    List,
    Dict,
    Optional,
)
from openai import (
    OpenAI,
    OpenAIError,
)
import time
import os
from llm_exploration.inference.inference_engine import LLMInferenceEngine


class OpenAIInferenceEngine(LLMInferenceEngine):
    """
    Inference Engine for running inference on GPT-4 and similar OpenAI models.

    Example Usage:
        llm = OpenAIInferenceEngine()

        convs = [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Who is the first president of the US?",
                },
            ]
        ]

        outputs = llm.batched_generate(
            convs=convs,
            max_n_tokens=128,
            temperature=1.0,
            top_p=1.0,
        )

        print(outputs)
        # ["George Washington."]
    """

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
    ):
        """
        Instantiates the inference engine for OpenAI API models.

        Input:
            model_name (str):
                Name of the model to be used.
                For example, "gpt-4" or "gpt-4o"

            api_key (str):
                The api_key to use for billing purposes.
                If api_key is not None, then this is used as the api_key
                Else, api_key is collected from the OS
        """
        self.model_name = model_name
        if api_key is None and os.getenv("OAI_KEY") is None:
            raise ValueError("No valid OpenAI API key can be found.")

        self.client = OpenAI(
            api_key=api_key if api_key is not None else os.getenv("OAI_KEY"),
        )

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            except OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)

        return output

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        return [self.generate(conv, max_n_tokens, temperature, top_p, min_p) for conv in convs]
