from typing import (
    List,
    Dict,
    Union,
    Optional,
)
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from llm_exploration.inference.inference_engine import LLMInferenceEngine


class HuggingFaceLLMInferenceEngine(LLMInferenceEngine):
    """
    Class for running inference on a huggingface LLM.
    Gives a simple interface to run generations from an LLM.

    Example Usage:
        import transformers

        config = transformers.AutoConfig.from_pretrained(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            trust_remote_code=True,
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            config=config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32
        ).to("cuda:0")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            config=config,
            trust_remote_code=True,
            model_max_length=8192,
            padding_side="left",
            use_fast=False,
        )

        llm = HuggingFaceLLMInferenceEngine(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            model=model,
            tokenizer=tokenizer,
        )

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

    def __init__(
        self,
        model_name: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        """
        Instantiates an object of class HuggingFaceLLMInferenceEngine

        Input:
            model_name (str):
                Name of the model that is being used

            model (AutoModelForCausalLM):
                Underlying LLM that will be used for inference

            tokenizer (AutoTokenizer):
                tokenizer for the underlying LLM
                Needs to be padding from the left, since this is for generation
        """
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.padding_side == "right":
            print("Padding side for generation should be left.")
            print("Setting padding side to left.")
            self.tokenizer.padding_side = "left"

    def batched_generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> List[str]:
        self.validate_batch_of_conversations(convs=convs)

        return self.generate_helper(
            max_n_tokens=max_n_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            convs=convs,
            is_batched=True,
        )

    def process_outputs_for_llama3(
        self,
        outputs_list: List[str],
    ) -> List[str]:
        """
        Outputs from LLama-3.1 starts contains "assistant\n\n"
        This function replaces them to ensure the same format between
        outputs from different models.

        Input:
            outputs_list (List[str]):
                outputs_list[i] = model's generation for i'th conversation

        Output:
            modified_outputs_list (List[str]):
                If model is LLaMa-3.1, then it removes "assistant\n\n"
                from the model response.
                Otherwise, it returns the same output
        """
        if "meta-llama/Meta-Llama-3.1-8B-Instruct" not in self.model_name:
            return outputs_list
        else:
            return [output.replace("assistant\n\n", "") for output in outputs_list]

    def generate(
        self,
        conv: List[Dict],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float] = None,
    ) -> str:
        self.validate_conversation(conv=conv)

        return self.generate_helper(
            max_n_tokens=max_n_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            convs=conv,
            is_batched=False,
        )

    def generate_helper(
        self,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        min_p: Optional[float],
        convs: Union[List[List[Dict]], List[Dict]],
        is_batched: bool = False,
    ) -> Union[str, List[str]]:
        """
        Helper function that unifies the generation,
        either for a batch of inputs or for a single conversation.

        Input:
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

            convs (List[Dict] or List[List[Dict]]):
                Either single conversation of the following format:
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        ...
                    ]

                or a batch of conversation of the following format:
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

            is_batched (bool):
                Whether or not the generation should be batched or not.

        Output:
            output (str or List[str]):
                If a single conv is given as input, we generate the output
                for it, which is a single string.

                If a batch of conv is given as input, we generate a list
                of strings as output, where
                output[i] = model's generation for convs[i]
        """
        # Prepare the input to the LLM
        # We only do padding if the generation is batched, otherwise we don't do.
        inputs = self.prepare_model_input(convs=convs, is_batched=is_batched)

        # Put the inputs to the correct device
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        # Generation
        with torch.no_grad():
            if temperature > 0:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_n_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                )
            else:
                # No sampling as temperature is 0
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_n_tokens,
                    do_sample=False,
                    top_p=1,
                    temperature=1,
                )

        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1] :]

        output = self.decode_helper(
            output_ids=output_ids,
            is_batched=is_batched,
        )

        # moving inputs/outputs to cpu, then deleting them
        # this cleans both cpu and gpu memory, to prevent OOM issues.
        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return output

    def decode_helper(
        self,
        output_ids: torch.Tensor,
        is_batched: bool,
    ) -> Union[str, List[str]]:
        """
        Helper function to decode the outputs.

        Input:
            output_ids (torch.Tensor):
                The output_ids from the model's generation.

            is_batched (bool):
                Whether the outputs are for a batch of data
                or a single datapoint.

        Output:
            output (str or List[str]):
                If a single conv is given as input, we generate the output
                for it, which is a single string.

                If a batch of conv is given as input, we generate a list
                of strings as output, where
                output[i] = model's generation for convs[i]
        """
        if is_batched:
            output = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            output = self.process_outputs_for_llama3(
                outputs_list=output,
            )
        else:
            output = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            )
            output = self.process_outputs_for_llama3(
                outputs_list=[output],
            )[0]

        return output

    def prepare_model_input(
        self,
        convs: Union[List[List[Dict]], List[Dict]],
        is_batched: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Creates input_ids and attention_mask, that will
        then be used by the model to generate responses.

        Input:
            convs (List[Dict] or List[List[Dict]]):
                Either single conversation of the following format:
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        ...
                    ]

                or a batch of conversation of the following format:
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

            is_batched (bool):
                Whether or not the generation should be batched or not.

        Output:
            input_for_model (Dict[str, torch.Tensor]):
                A dictionary that has the following format:
                {
                    "input_ids": input_ids (torch.Tensor),
                    "attention_mask": attention_mask (torch.Tensor),
                }
        """
        input_ids = self.tokenizer.apply_chat_template(
            convs,
            return_tensors="pt",
            padding=is_batched,
            add_generation_prompt=True,
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given input_ids, labels and attention mask,
        calculates the log probs (per token) from the auto-regressive generation process

        Input:
            input_ids (torch.Tensor):
                input ids for the sequence

                Shape: (batch size, sequence length)

            labels (torch.Tensor):
                Labels, typically a sequence of tokens that the
                next token prediction in an auto-regressive generation process should
                generate

                Shape: (batch size, sequence length)

            attention_mask (torch.Tensor):
                Attention mask to be used

                Shape: (batch size, sequence length)

        Output:
            log_probs (torch.Tensor):
                Computed log probabilities (per token) for the given sequences

                Shape: (batch size, sequence length - 1)

                NOTE: the -1 comes because we have to shift by 1
        """
        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
            inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

            model_outputs = self.model(**inputs)

            logits = (
                model_outputs["logits"]
                if isinstance(model_outputs, dict)
                else model_outputs[0]
            )

            # shift by 1, since we only consider Causal models
            logits = logits[:, :-1, :]
            labels = labels[:, 1:].clone().to(self.model.device.index)

            # calculate log probs
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # labels can contain ignore_token_id, typically -100.
            # we clamp them to 0, to avoid indexing errors
            # These tokens will be ignored later
            labels_clamped = torch.clamp(labels, min=0)
            log_probs = torch.gather(
                log_probs,
                dim=2,
                index=labels_clamped.unsqueeze(2),
            )

        for key in inputs:
            inputs[key].to("cpu")
        logits.to("cpu")

        log_probs = log_probs.to("cpu")

        del inputs, logits
        gc.collect()
        torch.cuda.empty_cache()

        return log_probs
