from vllm import (
    LLM,
    SamplingParams,
)
from typing import (
    List,
    Dict,
    Optional,
)

from llm_exploration.inference.inference_engine import LLMInferenceEngine


class VLLMInferenceEngine(LLMInferenceEngine):
    """
    Class that handles running inference, with possible prefix caching,
    using VLLM.

    This is an alternative to using HuggingFaceLLMInferenceEngine,
    which does not support prefix caching. Especially for a multi-turn conversation setting,
    prefix caching is supposed to increase inference speed by multiple factors.
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int,
        dtype: str = "bfloat16",
    ):
        """
        Instantiates an object of class VLLMInferenceEngine,
        to handle inference on an LLM using VLLM API.

        Input:
            model_name (str):
                Name of the model that is being used

            max_model_len (int):
                Maximum length supported by the inference engine.

            dtype (str):
                The data type for the model weights and activations.
                Currently, we support float32, float16, and bfloat16.
                If auto, we use the torch_dtype attribute specified in the model config file.
                However, if the torch_dtype in the config is float32, we will use float16 instead.

                Default: bfloat16
        """
        self.llm = LLM(
            model=model_name,
            enable_prefix_caching=True,
            max_model_len=max_model_len,
            dtype=dtype,
        )

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            max_tokens=max_n_tokens,
        )

        llm_responses = self.llm.chat(
            messages=convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        generated_texts = []
        for index in range(len(llm_responses)):
            generated_text = llm_responses[index].outputs[0].text
            generated_texts.append(generated_text)

        return generated_texts

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        return self.batched_generate(
            convs=[conv],
            max_n_tokens=max_n_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
        )[0]
