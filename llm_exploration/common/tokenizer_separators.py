from typing import Tuple, List, Optional
from transformers import AutoTokenizer
from dataclasses import dataclass


@dataclass
class TokenizerSeparators:
    assistant_prefix: str = ("",)
    assistant_suffix: str = ("",)
    prefix_offset: int = (0,)  # Offset to add for masking; Ugly single space with mistral.
    assistant_prefix_ids: List[int] = None
    assistant_suffix_ids: List[int] = None

    def set_assistant_prefix_ids(self, tokenizer: AutoTokenizer) -> None:
        self.assistant_prefix_ids = tokenizer.encode(
            self.assistant_prefix,
            add_special_tokens=False,
        )

    def set_assistant_suffix_ids(self, tokenizer: AutoTokenizer) -> None:
        self.assistant_suffix_ids = tokenizer.encode(
            self.assistant_suffix,
            add_special_tokens=False,
        )


def get_tokenizer_separators(
    tokenizer: AutoTokenizer,
    tokenizer_name: Optional[str] = None,
) -> Tuple[AutoTokenizer, TokenizerSeparators]:
    """
    Set padding tokens and unk for standard tokenizer. [modified in place; should be used for training]
    Set the tokenizer separators for the assistant tokens.
    """
    tokenizer_name_or_path = (
        tokenizer_name if tokenizer_name is not None else tokenizer.name_or_path
    )
    if tokenizer_name_or_path in [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Magpie-Align/Llama-3-8B-WildChat",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Meta-Llama-3.1-8B",
    ]:
        tokenizer.unk_token_id = 128004  # Use the finetune right padding token
        tokenizer.pad_token_id = 128004  # Use the finetune right padding token

        tokenizer_separator = TokenizerSeparators(
            assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_suffix="<|eot_id|>",
            prefix_offset=0,
        )

        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        tokenizer_separator.set_assistant_suffix_ids(tokenizer)

        return tokenizer, tokenizer_separator

    elif tokenizer_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        tokenizer.unk_token_id = 128004  # Use the finetune right padding token
        tokenizer.pad_token_id = 128004  # Use the finetune right padding token

        tokenizer_separator = TokenizerSeparators(
            assistant_prefix="<｜Assistant｜>",
            assistant_suffix="<｜end▁of▁sentence｜>",
            prefix_offset=0,
        )

        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        tokenizer_separator.set_assistant_suffix_ids(tokenizer)

        return tokenizer, tokenizer_separator

    elif tokenizer_name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer_separator = TokenizerSeparators(
            assistant_prefix=" [/INST] ",
            assistant_suffix="</s>",
            prefix_offset=-1,
        )

        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        tokenizer_separator.set_assistant_suffix_ids(tokenizer)

        return tokenizer, tokenizer_separator

    elif tokenizer_name_or_path == "Qwen/Qwen2.5-7B-Instruct":
        tokenizer.unk_token_id = tokenizer.pad_token_id

        tokenizer_separator = TokenizerSeparators(
            assistant_prefix="<|im_start|>assistant\n",
            assistant_suffix="<|im_end|>",
            prefix_offset=0,
        )

        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        tokenizer_separator.set_assistant_suffix_ids(tokenizer)

        return tokenizer, tokenizer_separator

    elif tokenizer_name_or_path == "google/gemma-2-9b-it":
        tokenizer_separator = TokenizerSeparators(
            assistant_prefix="<start_of_turn>model\n",
            assistant_suffix="<end_of_turn>",
            prefix_offset=0,
        )

        tokenizer_separator.set_assistant_prefix_ids(tokenizer)
        tokenizer_separator.set_assistant_suffix_ids(tokenizer)

        return tokenizer, tokenizer_separator

    else:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported")
