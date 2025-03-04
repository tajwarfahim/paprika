import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer,
)

from llm_exploration.common.tokenizer_separators import TokenizerSeparators
from llm_exploration.llm_finetuning.sft_dataset import MultiturnSFTDataset


class MultiturnDPODataset(Dataset):
    """
    Dataset class for DPO, in the multi-turn setting

    Each datapoint contains two elements:
        (chosen conversation, rejected_conversation)

    NOTE: This variation of the DPO class works from conversation in
    string/dict format
    """

    def __init__(
        self,
        chosen_conversations: List[List[Dict[str, str]]],
        rejected_conversations: List[List[Dict[str, str]]],
        tokenizer: AutoTokenizer,
        tokenizer_separator: TokenizerSeparators,
        ignore_token_id: int,
        max_token_length: Optional[int] = None,
    ):
        """
        Input:
            chosen_conversations (List[List[Dict[str, str]]]):
                chosen_conversations[i] = the i-th datapoint's chosen conversation

                chosen_conversations looks like:
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

            rejected_conversations (List[List[Dict[str, str]]]):
                rejected_conversations[i] = the i-th datapoint's rejected conversation

                Has the same format as chosen_conversations

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
        self.chosen_conversations = chosen_conversations
        self.rejected_conversations = rejected_conversations

        self.chosen_dataset = MultiturnSFTDataset(
            conversations=chosen_conversations,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
            max_token_length=max_token_length,
        )

        self.rejected_dataset = MultiturnSFTDataset(
            conversations=rejected_conversations,
            tokenizer=tokenizer,
            tokenizer_separator=tokenizer_separator,
            ignore_token_id=ignore_token_id,
            max_token_length=max_token_length,
        )

        assert len(self.chosen_dataset) == len(self.rejected_dataset)
        self.length = len(self.chosen_dataset)

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return self.length

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
                        "chosen_input_ids": input_id (tensor),
                        "chosen_labels": label (tensor),
                        "chosen_attention_mask": attention_mask (tensor),
                        "chosen_labels_sft": label (tensor),
                        "rejected_input_ids": input_id (tensor),
                        "rejected_labels": label (tensor),
                        "rejected_attention_mask": attention_mask (tensor),
                    }

                    i.e., the input_ids, labels and attention_mask for
                    the i-th chosen conversation, and the i-th rejected
                    conversation
        """
        chosen_datapoint = self.chosen_dataset[idx]
        rejected_datapoint = self.rejected_dataset[idx]

        return {
            "chosen_input_ids": chosen_datapoint["input_ids"],
            "chosen_labels": chosen_datapoint["labels"],
            "chosen_attention_mask": chosen_datapoint["attention_mask"],
            "chosen_labels_sft": chosen_datapoint["labels"],
            "rejected_input_ids": rejected_datapoint["input_ids"],
            "rejected_labels": rejected_datapoint["labels"],
            "rejected_attention_mask": rejected_datapoint["attention_mask"],
        }


class MultiturnDPODatasetFromTensors(Dataset):
    """
    A variation of the DPO dataset, that uses precomputed tensors,
    specially precomputed log_probs, saved in ".pt" format.

    This is unlike MultiturnDPODataset, that uses conversations in
    dict/str format.
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
        assert dataset_path.endswith(".pt")
        data = torch.load(dataset_path, weights_only=False)

        self.chosen_input_ids = data["chosen_input_ids"]
        self.chosen_labels = data["chosen_labels"]
        self.chosen_attention_mask = data["chosen_attention_mask"]
        self.chosen_ref_log_probs = data["chosen_log_probs"]
        self.chosen_labels_sft = data["chosen_labels_sft"]

        self.rejected_input_ids = data["rejected_input_ids"]
        self.rejected_labels = data["rejected_labels"]
        self.rejected_attention_mask = data["rejected_attention_mask"]
        self.rejected_ref_log_probs = data["rejected_log_probs"]

        self._validate_tensor_shapes()
        self._print_dataset_information()

    def _validate_tensor_shapes(self) -> None:
        """
        Performs various checks on the loaded tensors,
        to make sure they have the correct shapes.

        Input:
            None

        Output:
            None
        """
        num_data = self.chosen_input_ids.shape[0]
        assert (
            self.chosen_labels.shape[0] == num_data
            and self.chosen_attention_mask.shape[0] == num_data
            and self.chosen_ref_log_probs.shape[0] == num_data
            and self.rejected_input_ids.shape[0] == num_data
            and self.rejected_labels.shape[0] == num_data
            and self.rejected_ref_log_probs.shape[0] == num_data
            and self.chosen_labels_sft.shape[0] == num_data
        )

        chosen_sequence_length = self.chosen_input_ids.shape[1]
        assert (
            self.chosen_attention_mask.shape[1] == chosen_sequence_length
            and self.chosen_labels.shape[1] == chosen_sequence_length
            and self.chosen_labels_sft.shape[1] == chosen_sequence_length
            and self.chosen_ref_log_probs.shape[1] == chosen_sequence_length - 1
        )

        rejected_sequence_length = self.rejected_input_ids.shape[1]
        assert (
            self.rejected_attention_mask.shape[1] == rejected_sequence_length
            and self.rejected_labels.shape[1] == rejected_sequence_length
            and self.rejected_ref_log_probs.shape[1] == rejected_sequence_length - 1
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
        num_data = self.chosen_input_ids.shape[0]
        chosen_sequence_length = self.chosen_input_ids.shape[1]
        rejected_sequence_length = self.rejected_input_ids.shape[1]

        print("\n Dataset information:")
        print("Num data: ", num_data)
        print("Chosen sequence length: ", chosen_sequence_length)
        print("Rejected sequence length: ", rejected_sequence_length, "\n")

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.

        Input:
            None

        Output:
            length: Number of datapoints in the dataset
        """
        return self.chosen_input_ids.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns the i-th datapoint in a particular format.

        Input:
            idx (int):
                The index of the datapoint that needs to be returned.

        Output:
            datapoint_dict (Dict[str, torch.Tensor]):
                The i-th datapoint in the following dictionary format:
                    {
                        "input_ids": chosen_input_id (tensor),
                        "labels": chosen_label (tensor),
                        "labels_sft": chosen_label_sft (tensor),
                        "attention_mask": chosen_attention_mask (tensor),
                        "ref_log_probs": chosen_ref_log_probs (tensor),
                        "rejected_input_ids": rejected_input_id (tensor),
                        "rejected_labels": rejected_label (tensor),
                        "rejected_attention_mask": rejected_attention_mask (tensor),
                        "rejected_ref_log_probs": rejected_ref_log_probs (tensor),
                    }

                    i.e., the input_ids, labels, attention_mask and precomputed ref log probs
                    for the i-th chosen conversation, and the i-th rejected
                    conversation
        """
        return {
            "input_ids": self.chosen_input_ids[idx],
            "labels": self.chosen_labels[idx],
            "labels_sft": self.chosen_labels_sft[idx],
            "attention_mask": self.chosen_attention_mask[idx],
            "ref_log_probs": self.chosen_ref_log_probs[idx],
            "rejected_input_ids": self.rejected_input_ids[idx],
            "rejected_labels": self.rejected_labels[idx],
            "rejected_attention_mask": self.rejected_attention_mask[idx],
            "rejected_ref_log_probs": self.rejected_ref_log_probs[idx],
        }
