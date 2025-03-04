import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from transformers import AutoTokenizer

from llm_exploration.common.tokenizer_separators import TokenizerSeparators
from llm_exploration.llm_finetuning.dataset_utils import mask_non_assistant_tokens
from llm_exploration.utils.data_utils import write_json


class MultiturnSFTDataset(Dataset):
    """
    Dataset Class to handle the data serving for multi-turn SFT training.

    In single turn conversation training, one usually has a prompt and a response:
        input = "<user> prompt </user> <assistant> response </assistant>"

    During the forward pass, the entire sequence is put through causal attention,
    but one discards the user tokens from the loss function.

    In other words, let the token-wise log probs be:
        log_probs = model.forward(input)
        log_probs = log_probs[len(user_input):]

    And this log_probs is used for loss calculation.
    This is done so that the model learns p(y|x), not p(x, y)

    For multi-turn conversations, one has multiple rounds of user and assistant
    sequences, interleaved with each other:
        <system> system </system>
        <user> question 1 </user>
        <assistant> answer 1 </assistant>
        <user> question 2 </user>
        <assistant> answer 2 </assistant>
        .
        .
        .
        <user> question K </user>
        <assistant> answer K </assistant>

    When calculating loss on this, we take a forward pass over the entire sequence
    to calculate log probabilities of every token in an autoregressive manner.

    HOWEVER, we need to discard the log-probabilities of the user tokens to
    ignore them from the loss. Otherwise, the model will learn to generate the next
    round of user tokens/questions as well, instead of waiting for the
    questions from the user (i.e., the model won't stop with the end of assistant tokens).

    This means the labels per token looks like the following:
     <u>  <u>  ...   <u>  <a> ... <a>  <u> ...  <u>  <a> ... <a>
    -100 -100  ...  -100   1  ...  1  -100 ... -100   1  ...  1

    (Assistant tokens are not really 1, they just remain whatever they
    were in the first place, non-assistant tokens become -100,
    or in general, ignore_token_id)

    This class tokenizes data using a particular model's tokenizer,
    and handles the masking/labeling properly.
    """

    def __init__(
        self,
        conversations: List[List[Dict[str, str]]],
        tokenizer: AutoTokenizer,
        tokenizer_separator: TokenizerSeparators,
        ignore_token_id: int,
        max_token_length: Optional[int] = None,
    ):
        """
        Initializes an object of this class.

        Input:
            conversations (List[List[Dict[str, Any]]]):
                List of conversations that will be in this dataset.
                [
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        } ....
                    ], # 1st conversation
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        } ....
                    ], # 2nd conversation
                    .... (More conversations like this)
                ]

            tokenizer (Tokenizer):
                tokenizer for the model to be trained.

            tokenizer_separator (TokenizerSeparators):
                The tokenizer separator for assistant/user special tokens.

            ignore_token_id (int):
                The token that should be ignored from loss calculations.

            max_token_length (int):
                Maximum length that all trajectories should be padded to.

                Default is None, in which case we pad to the max length supported by
                the tokenizer/model.
        """
        self.input_ids, self.labels, self.attention_mask, self.labels_dpo = [], [], [], []
        self.all_conversations = []
        for _, conversation in tqdm(
            enumerate(conversations), desc="Masking Conversations - MT-SFT"
        ):
            self.all_conversations.append(conversation)
            data_dict = mask_non_assistant_tokens(
                tokenizer=tokenizer,
                conversation=conversation,
                tokenizer_separator=tokenizer_separator,
                ignore_token_id=ignore_token_id,
                max_token_length=max_token_length,
            )
            self.input_ids.append(data_dict["input_ids"])
            self.labels.append(data_dict["labels"])
            self.attention_mask.append(data_dict["attention_mask"])
            self.labels_dpo.append(data_dict["labels_dpo"])

        self.input_ids = torch.stack(self.input_ids, dim=0)
        self.labels = torch.stack(self.labels, dim=0)
        self.attention_mask = torch.stack(self.attention_mask, dim=0)
        self.labels_dpo = torch.stack(self.labels_dpo, dim=0)

        self._validate_tensor_shapes()
        self._print_dataset_information()

    def _validate_tensor_shapes(self) -> None:
        """
        Performs various checks on the tensors,
        to make sure they have the correct shapes.

        Input:
            None

        Output:
            None
        """
        num_data = self.input_ids.shape[0]
        assert (
            self.labels.shape[0] == num_data
            and self.attention_mask.shape[0] == num_data
            and self.labels_dpo.shape[0] == num_data
        )

        sequence_length = self.input_ids.shape[1]
        assert (
            self.labels.shape[1] == sequence_length
            and self.attention_mask.shape[1] == sequence_length
            and self.labels_dpo.shape[1] == sequence_length
        )

    def _print_dataset_information(self) -> None:
        """
        Prints the number of datapoints, sequence length of chosen trajectories
        and rejected trajectories.

        Input:
            None

        Output:
            None
        """
        assert (
            self.input_ids.shape[0] == self.labels.shape[0]
            and self.input_ids.shape[0] == self.attention_mask.shape[0]
        )

        print("\nDataset loaded...")
        print("Input ID shape: ", self.input_ids.shape)
        print("Labels shape: ", self.labels.shape)
        print("Labels DPO shape: ", self.labels_dpo.shape)
        print("Attention mask shape: ", self.attention_mask.shape, "\n")

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the i-th datapoint in a particular format.

        Input:
            idx (int):
                The index of the datapoint that needs to be returned.

        Output:
            datapoint_dict (Dict[str, torch.Tensor]):
                The i-th datapoint in the following dictionary format:
                    {
                        "input_ids": input_id (tensor),
                        "labels": labels (tensor),
                        "attention_mask": attention_mask (tensor),
                        "labels_dpo": labels_dpo (tensor),
                    }
        """
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
            "labels_dpo": self.labels_dpo[idx],
        }


class MultiturnSFTDatasetFromTensors(MultiturnSFTDataset):
    """
    We would often want to compute the logits/input_ids from beforehand,
    and would want to use it directly.

    This is a wrapper around the base MultiturnSFTDataset
    class, that allows that.
    """

    def __init__(
        self,
        dataset_path: str,
    ):
        """
        Input:
            dataset_path (str):
                The path to the dataset (in ".pt" format)
                that contains the dataset
        """
        data = torch.load(dataset_path, weights_only=False)
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]
        self.attention_mask = data["attention_mask"]
        self.labels_dpo = data["labels_dpo"]

        self._validate_tensor_shapes()
        self._print_dataset_information()


class MultiturnSFTDatasetWithoutTokenization(MultiturnSFTDataset):
    """
    A variant of the multi-turn SFT dataset class, where we do not perform tokenization
    of all conversations at the very beginning, since it can be very
    expensive to store in memory.
    """

    def __init__(
        self,
        conversations: List[List[Dict[str, str]]],
        tokenizer: AutoTokenizer,
        tokenizer_separator: TokenizerSeparators,
        ignore_token_id: int,
        save_dataset_path: Optional[str] = None,
        max_token_length: Optional[int] = None,
        save_dataset_to_disk: bool = False,
    ):
        """
        Initializes an object of this class.

        Input:
            conversations (List[List[Dict[str, Any]]]):
                List of conversations that will be in this dataset.
                [
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        } ....
                    ], # 1st conversation
                    [
                        {
                            "role": "user",
                            "content": user_prompt,
                        } ....
                    ], # 2nd conversation
                    .... (More conversations like this)
                ]

            tokenizer (Tokenizer):
                tokenizer for the model to be trained.

            tokenizer_separator (TokenizerSeparators):
                The tokenizer separator for assistant/user special tokens.

            ignore_token_id (int):
                The token that should be ignored from loss calculations.

            save_dataset_path (str):
                The path to which the conversations should be saved to

            max_token_length (int):
                Maximum length that all trajectories should be padded to.

                Default is None, in which case we pad to the max length supported by
                the tokenizer/model.

            save_dataset_to_disk (bool):
                Whether the dataset should be saved to disk
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.tokenizer_separator = tokenizer_separator
        self.ignore_token_id = ignore_token_id
        self.max_token_length = max_token_length

        print("Num data: ", len(self.conversations))

        if save_dataset_to_disk:
            assert save_dataset_path is not None
            write_json(
                data={
                    "trajectories": self.conversations,
                },
                fname=save_dataset_path,
            )

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the i-th datapoint in a particular format.

        Input:
            idx (int):
                The index of the datapoint that needs to be returned.

        Output:
            datapoint_dict (Dict[str, torch.Tensor]):
                The i-th datapoint in the following dictionary format:
                    {
                        "input_ids": input_id (tensor),
                        "labels": labels (tensor),
                        "attention_mask": attention_mask (tensor),
                        "labels_dpo": labels_dpo (tensor),
                    }
        """
        data_dict = mask_non_assistant_tokens(
            tokenizer=self.tokenizer,
            conversation=self.conversations[idx],
            tokenizer_separator=self.tokenizer_separator,
            ignore_token_id=self.ignore_token_id,
            max_token_length=self.max_token_length,
        )

        return {
            "input_ids": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "attention_mask": data_dict["attention_mask"],
            "labels_dpo": data_dict["labels_dpo"],
        }
