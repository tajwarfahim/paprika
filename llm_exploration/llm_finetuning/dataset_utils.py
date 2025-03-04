import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer

from llm_exploration.common.tokenizer_separators import TokenizerSeparators


def mask_non_assistant_tokens(
    tokenizer: AutoTokenizer,
    conversation: List[Dict[str, str]],
    tokenizer_separator: TokenizerSeparators,
    ignore_token_id: int,
    max_token_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Mask all tokens that are not part of the assistant's outputs.
    Note: We expect the end of assistant's output to be marked by the assistant_suffix token.

    The labels per token looks like the following:
     <u>  <u>  ...   <u>  <a> ... <a>  <u> ...  <u>  <a> ... <a>
    -100 -100  ...  -100   1  ...  1  -100 ... -100   1  ...  1

    (Assistant tokens are not really 1, they just remain whatever they
    were in the first place, non-assistant tokens become -100,
    or in general, ignore_token_id)

    Finally, for DPO loss, we also mask the assistant prefix and assistant suffix
    tokens, following the LLama-3 Tech report: https://arxiv.org/abs/2407.21783
    This is done to stabilize DPO loss, since these tokens are in both preferred
    and dispreferred trajectores.

    For example, for LLama-3,
    assistant_prefix = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    assistant_suffix = '<|eot_id|>'

    Input:
        tokenizer (Tokenizer):
            tokenizer for a particular method

        conversation (List[Dict[str, Any]]):
            Conversation that needs to tokenized/user tokens getting masked.
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": assistant_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": assistant_prompt,
                    }, ....
                ]

        tokenizer_separator (TokenizerSeparators):
            The tokenizer separator for assistant/user special tokens.

        ignore_token_id (int):
            The token that should be ignored. We replace all tokens on which
            there should be no loss, with this.

        max_token_length (int):
            Maximum length that all trajectories should be padded to.

            Default is None, in which case we pad to the max length supported by
            the tokenizer/model.

    Output:
        datapoint_dict (Dict[str, torch.Tensor]):
        The datapoint in the following dictionary format:
            {
                "input_ids": input_id (tensor),
                "labels": labels (tensor),
                "attention_mask": attention_mask (tensor),
                "labels_dpo": labels_dpo (tensor)
            }
    """
    # Stripping is necessary because some tokenizer will by default strip
    # beginning and end empty space.
    for i in range(len(conversation)):
        conversation[i]["content"] = conversation[i]["content"].strip()

    tokens = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        truncation=False,
        padding="max_length",
        max_length=max_token_length,
    )  # Applies chat template and tokenizes

    # Start with all tokens masked
    masked_tokens = [ignore_token_id] * len(tokens)
    masked_tokens_dpo = [ignore_token_id] * len(tokens)

    assert tokenizer_separator.assistant_prefix_ids is not None, "Assistant prefix ids not set"
    assert tokenizer_separator.assistant_suffix_ids is not None, "Assistant suffix ids not set"

    # Used to find starting of assistant turn
    assistant_prefix_ids = tokenizer_separator.assistant_prefix_ids

    # Used to find the end of assistant turn
    assistant_suffix_ids = tokenizer_separator.assistant_suffix_ids

    # <assistant_prefix> msg["content"] </assistant_suffix>
    assistant_contents = [
        tokenizer_separator.assistant_prefix
        + msg["content"]
        + tokenizer_separator.assistant_suffix
        for msg in conversation
        if msg["role"] == "assistant"
    ]

    start_index = -1
    for content in assistant_contents:
        # Assistant content for each turn - encode using the tokenizer.
        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        # Linear search for assistant content.
        while True:
            try:
                start_index += 1
                # Finds the next place in the entire sequence where you find the assistant content
                start_index = tokens.index(content_tokens[0], start_index)
                if tokens[start_index : start_index + len(content_tokens)] == content_tokens:
                    # fill in masked tokens with assistant/model generated tokens
                    masked_tokens[
                        start_index
                        + len(assistant_prefix_ids)
                        + tokenizer_separator.prefix_offset : start_index
                        + len(content_tokens)
                    ] = tokens[
                        start_index
                        + len(assistant_prefix_ids)
                        + tokenizer_separator.prefix_offset : start_index
                        + len(content_tokens)
                    ]

                    # do the same for DPO, except we mask out the prefix and suffix
                    # NOTE: In practice, we don't use these. These was implemented
                    # only as an ablation.
                    masked_tokens_dpo[
                        start_index
                        + len(assistant_prefix_ids)
                        + tokenizer_separator.prefix_offset : start_index
                        + len(content_tokens)
                        - len(assistant_suffix_ids)
                    ] = tokens[
                        start_index
                        + len(assistant_prefix_ids)
                        + tokenizer_separator.prefix_offset : start_index
                        + len(content_tokens)
                        - len(assistant_suffix_ids)
                    ]
                    break
            except Exception as e:
                break

    input_ids = torch.tensor(tokens)
    labels = torch.tensor(masked_tokens)
    labels_dpo = torch.tensor(masked_tokens_dpo)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        labels_dpo=labels_dpo,
    )


def stripped_decode(
    tokenizer: AutoTokenizer,
    masked_tokens: torch.Tensor,
    ignore_token_id: int,
) -> str:
    """
    Given a tokenizer and masked tokens,
    it gives the decoded version while ignoring the ignore_token_id.
    This is used to check if the decoded strings only contain
    the assistant tokens.

    Input:
        tokenizer (Tokenizer):
            tokenizer with which we will do the decoding.

        masked_tokens (torch.Tensor):
            tokens where non-assistant tokens are masked with
            ignore_token_id
            Also, this function only handles one tokenized sequence
            and outputs one string, does not handle the batched case.

        ignore_token_id (int):
            The token that replaces ignored tokens, usually -100

    Output:
        decoded_string (str):
            The decoded string
    """
    return tokenizer.decode(
        torch.where(
            masked_tokens == ignore_token_id,
            tokenizer.unk_token_id,
            masked_tokens,
        )
    ).replace(tokenizer.unk_token, "")
